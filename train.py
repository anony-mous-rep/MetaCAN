# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Wrapper to train/test models."""
import os
# os environ setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import sys
from open_clip.utils.misc import launch_job
from open_clip.config.defaults import assert_and_infer_cfg, get_cfg
import random
import json
import open_clip
from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import open_clip.utils.misc as misc
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
from binary_focal_loss import BinaryFocalLoss
from open_clip.model import get_cast_dtype
from open_clip.utils.env import checkpoint_pathmgr as pathmgr
import torch.optim as optim
from datasets.loader import construct_meta
import collections
from model import SAN


logger = logging.get_logger(__name__)




def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:8888",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--model",
        help="model_name",
        default="ViT-B-16-plus-240",
        type=str,
    )
    parser.add_argument(
        "--pretrained",
        help="whether use pretarined model",
        default=None,
        type=str
    )
    parser.add_argument('--normal_json_path', default='./datasets/AD_json/hyperkvasir_normal.json', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--outlier_json_path', default='./datasets/AD_json/hyperkvasir_outlier.json', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--val_normal_json_path', default='./datasets/AD_json/elpv_normal.json', nargs='+', type=str,
                        help='json path')
    parser.add_argument('--val_outlier_json_path', default='./datasets/AD_json/elpv_outlier.json', nargs='+', type=str,
                        help='json path')
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="the number of batches per epoch")
    parser.add_argument(
        "--shot", type=int, default=2, help="size for visual prompts"
    )
    parser.add_argument("--image_size", type=int, default=240, help="image size")

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.cd vis
    """
    # Setup cfg.
    cfg = get_cfg()

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "normal_json_path"):
        cfg.normal_json_path = args.normal_json_path
    if hasattr(args, "outlier_json_path"):
        cfg.outlier_json_path = args.outlier_json_path
    if hasattr(args, "val_normal_json_path"):
        cfg.val_normal_json_path = args.val_normal_json_path
    if hasattr(args, "val_outlier_json_path"):
        cfg.val_outlier_json_path = args.val_outlier_json_path
    if hasattr(args, "steps_per_epoch"):
        cfg.steps_per_epoch = args.steps_per_epoch

    if hasattr(args, "local_rank"):
        cfg.local_rank = args.local_rank

    if hasattr(args, "model"):
        cfg.model = args.model

    if hasattr(args, "pretrained"):
        cfg.pretrained = args.pretrained

    if hasattr(args, "shot"):
        cfg.shot = args.shot

    if hasattr(args, "image_size"):
        cfg.image_size = args.image_size

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

# MAML算法
class MetaSAN:
    def __init__(self, model):
        self.model = model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=0.001)  # meta optimizer

    def meta_update(self, tasks, tokenizer):
        losses_q = [0 for _ in range(len(tasks))]  # losses_q[i] is the loss on step i
        index = 0
        for task in tasks:
            inner_lr = 0.01

            train_query_image, train_query_label, train_few_normal_image, train_sample_type, test_query_image, test_query_label, test_few_normal_image, test_sample_type = task
            train_query_image = torch.stack(tuple(train_query_image))
            b, c, h, w = train_query_image.shape

            train_image = list()
            train_image.append(train_query_image)
            train_image.append(train_few_normal_image[0].unsqueeze(0).repeat(b, 1, 1, 1))
            train_image.append(train_few_normal_image[1].unsqueeze(0).repeat(b, 1, 1, 1))

            fast_weights = collections.OrderedDict(
                [(name, param) for (name, param) in self.model.named_parameters() if param.requires_grad], )

            for _ in range(1):
                # inner-update
                preds = self.model.functional_forward(tokenizer, train_image, train_sample_type, None,
                                                               fast_weights, train_query_label)
                loss_fun = BinaryFocalLoss()
                loss_fun = loss_fun.cuda()

                # Compute the loss.
                loss = loss_fun(preds, torch.Tensor(train_query_label).cuda())
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                fast_weights = collections.OrderedDict((name, param - inner_lr * grads)
                                                       for ((name, param), grads) in
                                                       zip(fast_weights.items(), grads))

            test_query_image = torch.stack(tuple(test_query_image))
            b, c, h, w = test_query_image.shape
            test_image = list()
            test_image.append(test_query_image)
            test_image.append(test_few_normal_image[0].unsqueeze(0).repeat(b, 1, 1, 1))
            test_image.append(test_few_normal_image[1].unsqueeze(0).repeat(b, 1, 1, 1))

            # Compute the loss.
            preds = self.model.functional_forward(tokenizer, test_image, test_sample_type, None, fast_weights,
                                                           test_query_label)
            loss_fun = BinaryFocalLoss()
            loss_fun = loss_fun.cuda()
            losses_q[index] = loss_fun(preds, torch.Tensor(test_query_label).cuda())


        meta_loss = 0
        for loss in losses_q:
            meta_loss = meta_loss + loss
        meta_loss = meta_loss / len(tasks)

        # outer-update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()


def _convert_to_rgb(image):
    return image.convert('RGB')


# meta-training
def train_epoch(
        train_dataset,
        model,
        tokenizer,
):
    """
    Perform the training for one epoch.
    Args:
        train_dataset (auxiliary train dataset): to construct meta training task.
        model (model): the model to train.
    """
    # Enable train mode.
    model.train()
    metasan = MetaSAN(model)

    tasks = train_dataset.create_task()
    loss = metasan.meta_update(tasks, tokenizer=tokenizer)
    return loss


# no_meta training
# def train_epoch(
#     train_loader,
#     model,
#     optimizer,
#     tokenizer,
#     cfg
# ):
#     """
#     Perform the training for one epoch.
#     Args:
#         train_loader (loader): training loader.
#         model (model): the model to train.
#         optimizer (optim): the optimizer to perform optimization on the model's
#             parameters.
#         scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
#         train_meter (TrainMeter): training meters to log the training performance.
#         cur_epoch (int): current epoch of training.
#         cfg (CfgNode): configs. Details can be found in
#             open_clip/config/defaults.py
#     """
#     # Enable train mode.
#     model.train()
#
#     all_loss = 0.0
#     for cur_iter, (inputs, types, labels, path) in enumerate(train_loader):
#
#         if cfg.NUM_GPUS:
#             labels = labels.cuda()
#
#
#         preds, preds2 = model(tokenizer, inputs, types, None, None, None)
#         loss_fun = BinaryFocalLoss()
#         loss_fun = loss_fun.cuda()
#
#         # Compute the loss.
#         # loss = loss_fun(preds, labels.float()) + loss_fun(preds2, labels.float())
#         loss = loss_fun(preds, labels.float())
#
#         # check Nan Loss.
#         misc.check_nan_losses(loss)
#
#         # Perform the backward pass.
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Update the parameters.
#         optimizer.step()
#
#         # dist.all_reduce(loss)
#         loss_value = loss.item()
#         all_loss = all_loss + loss_value
#
#     all_loss = all_loss / (cur_iter + 1)
#     print("train_loss: ", all_loss)
#     return all_loss

def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    if cfg.NUM_GPUS:
        device = torch.cuda.current_device()

    # Build the model and print model statistics.
    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = SAN(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)

    if torch.cuda.is_available():
        assert (
                cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
                cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        # Transfer the model to the current GPU device
        model = model.cuda(device=device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[device], output_device=device
        )

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Load a checkpoint to resume training if applicable.
    with pathmgr.open("./vit_b_16_plus_240-laion400m_e32-699c4b84.pt", "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    start_epoch = 0
    # model = model.module
    model.load_state_dict(checkpoint, strict=False)

    # Create the train and val loaders.
    # train_loader = loader.construct_loader(cfg, "train", transform)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=[0.9, 0.999])

    # create meta dataset
    train_dataset = construct_meta(cfg, 'train', transform, 100)
    # test_loader = loader.construct_loader(cfg, "test", transform)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    epoch_losses = []

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, 15):
        print("Epoch: ", cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        epoch_loss = train_epoch(
            train_dataset,
            model,
            tokenizer,
        )
        # epoch_loss = train_epoch(
        #     train_loader,
        #     model,
        #     optimizer,
        #     tokenizer,
        #     cfg
        # )
        epoch_losses.append(epoch_loss)
        epoch_timer.epoch_toc()
        print(f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s, loss: {epoch_loss:.4f}.")

        save_path = './tmp/1712_image_map_text+3chen+2dim+top'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = save_path + "/checkpoint_" + str(cur_epoch + 1) + ".pyth"
        torch.save(model.state_dict(), path)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)
    # train(cfg)



if __name__ == "__main__":
    main()
