""" CLIP Model

Adapted from ***. Originally *** License, Copyright (c) 2021 ***.
"""
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from open_clip.hf_model import HFTextEncoder
from open_clip.modified_resnet import ModifiedResNet
from open_clip.timm_model import TimmModel
from open_clip.transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, WindowVisionTransformer, VisionTransformer_Mul
from open_clip.vp import (
    PadPrompter,
    RandomPatchPrompter,
    FixedPatchPrompter
)


from torch.autograd import Variable, grad

PROMPT_TYPES = {
    "padding": PadPrompter,
    "random_patch": RandomPatchPrompter,
    "fixed_patch": FixedPatchPrompter
}


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = True

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype

state_level = {
               "normal":["{}", "flawless {}", "perfect {}", "unblemished {}",
                         "{} without flaw", "{} without defect", "{} without damage"],
                "anomaly":["damaged {}", "{} with flaw", "{} with defect", "{} with damage"]
}
template_level = [
                  "a cropped photo of the {}.",
                  "a cropped photo of a {}.",
                  "a close-up photo of a {}.",
                  "a close-up photo of the {}.",
                  "a bright photo of a {}.",
                  "a bright photo of the {}.",
                  "a dark photo of a {}.",
                  "a dark photo of the {}.",
                  "a jpeg corrupted photo of a {}.",
                  "a jpeg corrupted photo of the {}.",
                  "a blurry photo of the {}.",
                  "a blurry photo of a {}.",
                  "a photo of the {}.",
                  "a photo of a {}.",
                  "a photo of a small {}.",
                  "a photo of the small {}.",
                  "a photo of a large {}.",
                  "a photo of the large {}.",
                  "a photo of a {} for visual inspection.",
                  "a photo of the {} for visual inspection.",
                  "a photo of a {} for anomaly detection.",
                  "a photo of the {} for anomaly detection."
]

def get_texts(obj_name):

    l = ["airplane", "automobile", "bird",
         "cat", "deer", "dog", "frog", "horse", "ship", "truck", "animal"]

    if obj_name in l:
        normal_texts = []
        anomaly_texts = []
        normal = "a photo of " + obj_name + " for anomaly detection."
        normal_texts.append(normal)
        anomaly = "a photo without " + obj_name + " for anomaly detection."
        anomaly_texts.append(anomaly)
    else:
        normal_states = [s.format(obj_name) for s in state_level["normal"]]
        anomaly_states = [s.format(obj_name) for s in state_level["anomaly"]]

        normal_texts = [t.format(state) for state in normal_states for t in template_level]
        anomaly_texts = [t.format(state) for state in anomaly_states for t in template_level]

    return normal_texts, anomaly_texts


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # *** models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_vision_tower_Mul(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # *** models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer_Mul(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return visual

def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            output_tokens=text_cfg.output_tokens,
            pad_id=text_cfg.pad_id,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.view(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.view(s1, s2, s3)

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()


class Detector(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        super(Detector, self).__init__()
        self.projection1 = nn.Linear(dim_in, dim_in * 2, bias=True)
        self.projection2 = nn.Linear(dim_in * 2, dim_in, bias=True)
        self.projection3 = nn.Linear(dim_in, num_classes, bias=True)
        # self.projection4 = nn.Linear(32, num_classes, bias=True)
        self.bn1 = nn.BatchNorm1d(dim_in * 2)
        self.bn2 = nn.BatchNorm1d(dim_in)


    def forward(self, x):
        x = self.projection1(x)
        x = F.relu(x, inplace=True)
        x = self.bn1(x)
        x = self.projection2(x)
        x = F.relu(x, inplace=True)
        x = self.bn2(x)
        x = self.projection3(x)

        return torch.sigmoid(x)

def detector_function(input, w1, w1_b, w2, w2_b, w3, w3_b, w_bn1=None, b_bn1=None,w_bn2=None, b_bn2=None):
    x = F.linear(input, w1, w1_b)
    x = F.relu(x, inplace=True)
    running_mean_bn1 = nn.Parameter(torch.zeros(x.shape[1]), requires_grad=False).cuda()
    running_var_bn1 = nn.Parameter(torch.ones(x.shape[1]), requires_grad=False).cuda()
    x = F.batch_norm(x, running_mean=running_mean_bn1, running_var=running_var_bn1, weight=w_bn1, bias=b_bn1, training=True)
    x = F.linear(x, w2, w2_b)
    x = F.relu(x, inplace=True)
    running_mean_bn2 = nn.Parameter(torch.zeros(x.shape[1]), requires_grad=False).cuda()
    running_var_bn2 = nn.Parameter(torch.ones(x.shape[1]), requires_grad=False).cuda()
    x = F.batch_norm(x, running_mean=running_mean_bn2, running_var=running_var_bn2, weight=w_bn2, bias=b_bn2, training=True)
    x = F.linear(x, w3, w3_b)
    return torch.sigmoid(x)


class Projection(nn.Module):
    def __init__(self, c_in, c_out,reduction=4):
        super(Projection, self).__init__()
        self.linear1 = nn.Linear(c_in, 1024, bias=False)
        self.linear2 = nn.Linear(1024, c_out, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        embeds = self.relu2(x)
        return embeds

def projection_function(input, w1, w2):
    x = F.linear(input, w1)
    x = F.relu(x, inplace=True)
    x = F.linear(x, w2)
    x = F.relu(x, inplace=True)
    return x




class SAN(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower_Mul(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.projection = Projection(896, 640, 4)
        self.diff_head = Detector(225, 1)

        for p in self.visual.parameters():
            p.requires_grad = False

        for p in text.parameters():
            p.requires_grad = False

    def encode_image(self, image, out_layers: list = [1, 7, 12], normalize: bool = False):
        features = self.visual.forward(image, out_layers)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None, labels = None, path_list = None):
        if normal_list == None:
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.cuda(non_blocking=True)
            # normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)
        else:
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.cuda(non_blocking=True)
            # normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)


        token, Fp_list, _ = self.encode_image(img, normalize=False)
        token_list_n = []
        Fp_list_n = []
        for n in range(shot):
            token_n, Fp_n, _ = self.encode_image(normal_image[n,:,:,:,:], normalize=False)
            Fp_n = torch.stack(Fp_n, dim=1) #[b, 3, 226, 896]
            token_list_n.append(token_n)
            Fp_list_n.append(Fp_n[:, :, 1:, :]) #[b, 3, 225, 896]

        Fp_list = torch.stack(Fp_list)
        Fp_list = Fp_list.permute(1, 0, 2, 3)
        # Fp_list = [3, 8, 226, 896]
        # Fp_list_n = torch.stack(Fp_list_n)
        # Fp_list_n = Fp_list_n.permute(1, 0, 2, 3)
        # Fp_list = [3, 16, 226, 896]

        Fp_list = Fp_list[:, :, 1:, :]
        # Fp_list_n = Fp_list_n[:, :, 1:, :]


        max_diff_score = []
        text_score = []
        vit_map = []
        text_score_predict = []
        first_max_diff_score = []
        middle_max_diff_score = []
        last_max_diff_score = []
        same_map_list = []
        data_list = []

        for i in range(len(token)):
            Fp = Fp_list[i, :, :, :]
            temp_list = []
            for n in range(shot):
                temp_list.append(Fp_list_n[n][i, :, :, :])
            Fp_n = torch.cat(temp_list, dim=1)

            dict = {}

            # dict['image_path'] = path_list[i]
            # dict['label'] = labels[i].item()

            Fp_map = list()
            for j in range(3):
                tmp_x = Fp[j, :, :]
                tmp_n = Fp_n[j, :, :]
                am_fp = list()
                same_map = list()
                for k in range(len(tmp_x)):
                    # print(tmp_x.shape) [225, 896]
                    tmp = tmp_x[k]
                    # print(tmp_n.shape) [450, 896]
                    tmp = tmp.unsqueeze(0)
                    tmp_n = tmp_n / tmp_n.norm(dim=-1, keepdim=True)
                    tmp = tmp / tmp.norm(dim=-1, keepdim=True)
                    # print(tmp_n.shape) [450, 896]
                    # print(tmp.shape) [1, 896]
                    # kl = F.kl_div(tmp.softmax(dim=-1).log(), tmp_n.softmax(dim=-1), reduction='mean')
                    # s, indexs = torch.min(0.5 * (1 - (tmp @ tmp_n.T)), dim=1)

                    # indexs = indexs.item()
                    # same_map.append(Fp_n[j, indexs, :])
                    # s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
                    s, _ = (0.5 * (1 - (tmp @ tmp_n.T))).topk(shot, largest=False, sorted=True)
                    s = torch.mean(s).unsqueeze(0)
                    am_fp.append(s)

                am_fp = torch.stack(am_fp)
                Fp_map.append(am_fp)
                if j == 0:
                    frist_map = am_fp

                    frist_map = frist_map.squeeze(1)

                    # scores = frist_map.tolist()
                    # rounded_scores = [round(score, 4) for score in scores]
                    # dict['frist_map'] = rounded_scores
                    # s, _ = frist_map.topk(shot, largest=True, sorted=True)
                    # first_score = torch.mean(s)

                    first_score = frist_map.max(dim=0).values
                    first_max_diff_score.append(first_score)
                if j == 1:
                    middle_map = am_fp
                    middle_map = middle_map.squeeze(1)

                    # scores = middle_map.tolist()
                    # rounded_scores = [round(score, 4) for score in scores]
                    # dict['middle_map'] = rounded_scores

                    middle_score = middle_map.max(dim=0).values
                    middle_max_diff_score.append(middle_score)
                if j == 2:
                    last_map = am_fp
                    last_map = last_map.squeeze(1)

                    # scores = last_map.tolist()
                    # rounded_scores = [round(score, 4) for score in scores]
                    # dict['last_map'] = rounded_scores
                    # s, _ = last_map.topk(shot, largest=True, sorted=True)
                    # last_score = torch.mean(s)

                    last_score = last_map.max(dim=0).values
                    last_max_diff_score.append(last_score)

            data_list.append(dict)

        # 读取现有的JSON文件
        # with open('data.json', 'r') as file:
        #     data = json.load(file)

        # data.extend(data_list)
        # 写入更新后的数据到JSON文件
        # with open('data.json', 'w') as file:
        #     json.dump(data, file, indent=4)

        Fp_visual_token = Fp_list[:, 2, :, :]

        temp = 0
        for n in range(shot):
            temp = temp + Fp_list_n[n][:, 2, :, :]
        Fp_visual_token_n = temp / shot

        # temp_list = []
        # for n in range(shot):
        #     temp_list.append(Fp_list_n[n][:, 2, :, :])
        # Fp_visual_token_n_list = torch.stack(temp_list)
        # Fp_visual_token_n = torch.mean(Fp_visual_token_n_list, dim=0)

        # Fp_visual_token_n = torch.stack(same_map_list)
        Diff_visual_token = Fp_visual_token - Fp_visual_token_n
        # Diff_visual_token = F.cosine_similarity(Fp_visual_token, Fp_visual_token_n, dim=2)
        # Diff_visual_token = 1 - (Diff_visual_token + 1) / 2
        # Fp_visual_token_n = torch.stack(same_map_list)
        # Diff_visual_token = torch.cat([Fp_visual_token, Fp_visual_token_n], dim=1)

        Diff_visual_token = self.projection(Diff_visual_token)

        image_text_list = []
        for i in range(len(token)):
            dict = {}
            image_feature = Diff_visual_token[i]
            # image_feature = image_feature.unsqueeze(0)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            obj_type = text[i]
            # normal_texts, anomaly_texts = get_texts(obj_type.replace('_', " "))
            normal_texts, anomaly_texts = get_texts('object')
            pos_features = tokenizer(normal_texts).cuda()
            neg_features = tokenizer(anomaly_texts).cuda()
            pos_features = self.encode_text(pos_features)
            neg_features = self.encode_text(neg_features)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            text_features = torch.cat([pos_features, neg_features], dim=0)
            # print(text_features.shape)#[2,640]
            # print(image_feature.shape)#[1,640]
            score = (100 * image_feature @ text_features.T).softmax(dim=-1)
            tmp = score[:, 1]

            # scores = tmp.tolist()
            # rounded_scores = [round(score, 4) for score in scores]
            # dict['map'] = rounded_scores
            # dict['image_path'] = path_list[i]
            # dict['label'] = labels[i].item()
            # image_text_list.append(dict)


            text_score.append(tmp)


        text_score = torch.stack(text_score)

        loss_it = torch.zeros(1).cuda()


        holistic_map = text_score

        hl_score = self.diff_head(holistic_map)
        hl_score = hl_score.squeeze(1)

        # for index, image_text in enumerate(image_text_list):
        #     score = hl_score[index].item()
        #     rounded_score = round(score, 4)
        #     image_text['score'] = rounded_score
        #     # 读取现有的JSON文件
        # with open('Image-Text.json', 'r') as file:
        #     data = json.load(file)
        #
        # data.extend(image_text_list)
        # # 写入更新后的数据到JSON文件
        # with open('Image-Text.json', 'w') as file:
        #     json.dump(data, file, indent=4)
        # fg_score = torch.stack(max_diff_score)
        # final_score = hl_score
        # final_score = (torch.stack(first_max_diff_score) + torch.stack(middle_max_diff_score) + torch.stack(last_max_diff_score))/3
        final_score = (hl_score + torch.stack(first_max_diff_score) + torch.stack(middle_max_diff_score) + torch.stack(last_max_diff_score)) / 4
        # final_score = torch.stack(middle_max_diff_score)
        # final_score = hl_score
        return final_score

    def functional_forward(self, tokenizer, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None, normal_list = None, params= None, labels = None):
        if normal_list == None:
            img = image[0].cuda(non_blocking=True)
            normal_image = image[1:]
            normal_image = torch.stack(normal_image)
            shot, b, _, _, _ = normal_image.shape
            normal_image = normal_image.cuda(non_blocking=True)
            # normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)
        else:
            img = image[0].cuda(non_blocking=True)
            normal_image = normal_list
            normal_image = torch.stack(normal_image)
            normal_image = normal_image.unsqueeze(1)
            b = len(img)
            normal_image = normal_image.repeat(1, b, 1, 1, 1)
            shot, _, _, _, _ = normal_image.shape
            normal_image = normal_image.cuda(non_blocking=True)
            # normal_image = normal_image.reshape(-1, 3, 240, 240).cuda(non_blocking=True)


        token, Fp_list, _ = self.encode_image(img, normalize=False)
        token_list_n = []
        Fp_list_n = []
        for n in range(shot):
            token_n, Fp_n, _ = self.encode_image(normal_image[n,:,:,:,:], normalize=False)
            Fp_n = torch.stack(Fp_n, dim=1) #[b, 3, 226, 896]
            token_list_n.append(token_n)
            Fp_list_n.append(Fp_n[:, :, 1:, :]) #[b, 3, 225, 896]


        Fp_list = torch.stack(Fp_list)
        Fp_list = Fp_list.permute(1, 0, 2, 3)
        Fp_list = Fp_list[:, :, 1:, :]
        # Fp_list = [3, 8, 226, 896]
        # Fp_list_n = torch.stack(Fp_list_n)
        # Fp_list_n = Fp_list_n.permute(1, 0, 2, 3)
        # Fp_list = [3, 16, 226, 896]
        # Fp_list_n = Fp_list_n[:, :, 1:, :]


        max_diff_score = []
        text_score = []
        vit_map = []
        text_score_predict = []
        first_max_diff_score = []
        middle_max_diff_score = []
        last_max_diff_score = []
        same_map_list = []


        for i in range(len(token)):
            Fp = Fp_list[i, :, :, :]
            temp_list = []
            for n in range(shot):
                temp_list.append(Fp_list_n[n][i, :, :, :])
            Fp_n = torch.cat(temp_list, dim=1)

            Fp_map = list()
            for j in range(3):
                tmp_x = Fp[j, :, :]
                tmp_n = Fp_n[j, :, :]
                same_map = list()
                am_fp = list()
                for k in range(len(tmp_x)):
                    # print(tmp_x.shape) [225, 896]
                    tmp = tmp_x[k]
                    # print(tmp_n.shape) [450, 896]
                    tmp = tmp.unsqueeze(0)
                    tmp_n = tmp_n / tmp_n.norm(dim=-1, keepdim=True)
                    tmp = tmp / tmp.norm(dim=-1, keepdim=True)
                    # print(tmp_n.shape) [450, 896]
                    # print(tmp.shape) [1, 896]
                    # kl = F.kl_div(tmp.softmax(dim=-1).log(), tmp_n.softmax(dim=-1), reduction='mean')
                    # s = (0.5 * (1 - (tmp @ tmp_n.T))).min(dim=1).values
                    # s, indexs = torch.min(0.5 * (1 - (tmp @ tmp_n.T)), dim=1)
                    # indexs = indexs.item()
                    # same_map.append(Fp_n[j, indexs, :])

                    s, _ = (0.5 * (1 - (tmp @ tmp_n.T))).topk(shot, largest=False, sorted=True)
                    s = torch.mean(s).unsqueeze(0)
                    am_fp.append(s)

                am_fp = torch.stack(am_fp)
                Fp_map.append(am_fp)
                if j == 0:
                    frist_map = am_fp
                    frist_map = frist_map.squeeze(1)
                    first_score = frist_map.max(dim=0).values
                    first_max_diff_score.append(first_score)
                if j == 1:
                    middle_map = am_fp
                    middle_map = middle_map.squeeze(1)
                    middle_score = middle_map.max(dim=0).values
                    middle_max_diff_score.append(middle_score)
                if j == 2:
                    last_map = am_fp
                    last_map = last_map.squeeze(1)
                    last_score = last_map.max(dim=0).values
                    last_max_diff_score.append(last_score)

            # Fp_map = torch.stack(Fp_map)
            # print(Fp_map.shape) [3, 255, 1]
            # Fp_map = torch.mean(Fp_map.squeeze(2), dim=0)
            # print(Fp_map.shape) [225]
            # Fp_map = Fp_map.squeeze(2)
            # vit_map.append(Fp_map)
            # score = Fp_map.max(dim=1).values
            # max_diff_score.append(score)

        Fp_visual_token = Fp_list[:, 2, :, :]

        temp = 0
        for n in range(shot):
            temp = temp + Fp_list_n[n][:, 2, :, :]
        Fp_visual_token_n = temp / shot
        # for n in range(shot):
        #     temp_list.append(Fp_list_n[n][:, 2, :, :])
        # Fp_visual_token_n_list = torch.stack(temp_list)
        # Fp_visual_token_n = torch.mean(Fp_visual_token_n_list, dim=0)

        # Fp_visual_token_n = torch.stack(same_map_list)
        # Diff_visual_token = F.cosine_similarity(Fp_visual_token, Fp_visual_token_n, dim=2)

        # Diff_visual_token = torch.cat([Fp_visual_token, Fp_visual_token_n], dim=1)

        Diff_visual_token = Fp_visual_token - Fp_visual_token_n


        Diff_visual_token = projection_function(Diff_visual_token, params[f'projection.linear1.weight'], params[f'projection.linear2.weight'] )
        # Diff_visual_token = projection_function(Diff_visual_token, params[f'projection.linear1.weight'],params[f'projection.linear1.bias'],
        #                                         params[f'projection.linear2.weight'], params[f'projection.linear2.bias'], )

        for i in range(len(token)):
            image_feature = Diff_visual_token[i]
            # image_feature = image_feature.unsqueeze(0)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            obj_type = text[i]
            # normal_texts, anomaly_texts = get_texts(obj_type.replace('_', " "))
            normal_texts, anomaly_texts = get_texts('object')
            pos_features = tokenizer(normal_texts).cuda()
            neg_features = tokenizer(anomaly_texts).cuda()
            pos_features = self.encode_text(pos_features)
            neg_features = self.encode_text(neg_features)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            pos_features = torch.mean(pos_features, dim=0, keepdim=True)
            neg_features = torch.mean(neg_features, dim=0, keepdim=True)
            pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            text_features = torch.cat([pos_features, neg_features], dim=0)
            # print(text_features.shape)#[2,640]
            # print(image_feature.shape)#[1,640]
            score = (100 * image_feature @ text_features.T).softmax(dim=-1)
            text_score_predict.append(score[0])
            tmp = score[:, 1]

            text_score.append(tmp)

        text_score = torch.stack(text_score)

        loss_it = torch.zeros(1).cuda()



        holistic_map = text_score

        hl_score = detector_function(holistic_map, params[f'diff_head.projection1.weight'], params[f'diff_head.projection1.bias'],
                                                params['diff_head.projection2.weight'], params['diff_head.projection2.bias'],
                                                params['diff_head.projection3.weight'], params['diff_head.projection3.bias'],
                                                 params['diff_head.bn1.weight'], params['diff_head.bn1.bias'],
                                                 params['diff_head.bn2.weight'], params['diff_head.bn2.bias'],)

        # hl_score = detector_function(holistic_map, params[f'diff_head.projection1.weight'], params[f'diff_head.projection1.bias'],
        #                                          params['diff_head.projection2.weight'], params['diff_head.projection2.bias'],
        #                                          params['diff_head.bn1.weight'], params['diff_head.bn1.bias'],)

        hl_score = hl_score.squeeze(1)
        # fg_score = torch.stack(max_diff_score)
        # final_score = hl_score
        # final_score = (hl_score + (fg_score[:,0]+ fg_score[:,1])/2) / 2
        final_score = (hl_score + torch.stack(first_max_diff_score) + torch.stack(middle_max_diff_score) + torch.stack(last_max_diff_score)) / 4
        # final_score = torch.stack(last_max_diff_score)
        return final_score

