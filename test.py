# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Wrapper to train/test models."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import sys
from open_clip.utils.misc import launch_job
from open_clip.config.defaults import assert_and_infer_cfg, get_cfg
import random
import json
import open_clip
import open_clip.utils.checkpoint as cu
import open_clip.utils.distributed as du
import open_clip.utils.logging as logging
import numpy as np
import torch
from datasets import loader
from torchvision import transforms
from open_clip.utils.meters import EpochTimer, TrainMeter, ValMeter
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from open_clip.model import get_cast_dtype
from PIL import Image
from model import SAN


logger = logging.get_logger(__name__)

def _convert_to_rgb(image):
    return image.convert('RGB')

@torch.no_grad()
def eval_epoch(val_loader, model, cfg, tokenizer, normal_list, mode=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            open_clip/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()

    total_label = torch.Tensor([]).cuda()
    total_pred = torch.Tensor([]).cuda()

    for cur_iter, (inputs, types, labels, paths) in enumerate(val_loader):

        if cfg.NUM_GPUS:
            labels = labels.cuda()

        preds, _ = model(tokenizer, inputs, types, normal_list, labels, paths)

        total_pred = torch.cat((total_pred, preds), 0)
        total_label = torch.cat((total_label, labels), 0)



    total_pred = total_pred.cpu().numpy()  #.squeeze()
    total_label = total_label.cpu().numpy()

    print("Predict " + mode + " set: ")

    total_roc, total_pr = aucPerformance(total_pred, total_label)

    return total_roc

def aucPerformance(mse, labels, prt=True):

    roc_auc = roc_auc_score(labels, mse)

    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def drawing(cfg, data, xlabel, ylabel, dir):
    plt.switch_backend('Agg')
    plt.figure()
    plt.plot(data, 'b', label='loss')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, dir))


def test(cfg, load=None, mode = None):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in open_clip/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    device = torch.cuda.current_device()

    transform = transforms.Compose([
        transforms.Resize(size=240, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size=(240, 240)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    cf = './open_clip/model_configs/ViT-B-16-plus-240.json'
    with open(cf, 'r') as f:
        model_cfg = json.load(f)
    embed_dim = model_cfg["embed_dim"]
    vision_cfg = model_cfg["vision_cfg"]
    text_cfg = model_cfg["text_cfg"]
    cast_dtype = get_cast_dtype('fp32')
    quick_gelu = False

    model = SAN(cfg, embed_dim, vision_cfg, text_cfg, quick_gelu, cast_dtype=cast_dtype)
    model = model.cuda(device=device)

    cu.load_test_checkpoint(cfg, model)

    tokenizer = open_clip.get_tokenizer('ViT-B-16-plus-240')

    if load == None:
        load = loader.construct_loader(cfg, "test", transform)
        mode = "test"

    few_shot_path = os.path.join(cfg.few_shot_dir, cfg.category+".pt")
    normal_list = torch.load(few_shot_path)

    total_roc = eval_epoch(load, model, cfg, tokenizer, normal_list, mode)

    return total_roc

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
    parser.add_argument("--dataset_dir", type=str, default="./", help="the number of batches per epoch")
    parser.add_argument("--category", type=str, default="mvtecad", help="")
    parser.add_argument(
        "--shot", type=int, default=2, help="size for visual prompts"
    )
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--few_shot_dir", type=str, default="./visa", help="path to few shot sample prompts")

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
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
    if hasattr(args, "dataset_dir"):
        cfg.dataset_dir = args.dataset_dir
    if hasattr(args, "category"):
        cfg.category = args.category

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

    if hasattr(args, "few_shot_dir"):
        cfg.few_shot_dir = args.few_shot_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
