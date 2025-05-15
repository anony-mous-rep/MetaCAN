#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from .Sampler import DistributedSamplerWrapper
from .new_utlis import worker_init_fn_seed, BalancedBatchSampler

from .build import build_dataset
import open_clip.utils.misc as misc
import numpy as np
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torch.utils.data.dataset import Dataset
import json
import random

class meta_dataset(Dataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        normal_json_path_list: list,
        outlier_json_path_list: list,
        transform: Optional[Callable] = None,
        shot = None,
        steps_per_epoch = 100
    ) -> None:

        self.steps_per_epoch = steps_per_epoch
        self.normal_samples = []
        self.outlier_samples = []
        self.image = []
        self.total_n = 0
        self.total_o = 0
        self.shot = shot
        self.sample_type_list = []
        self.normal_samples_dicts = {}
        self.outlier_samples_dicts = {}
        for idx, json_path in enumerate(normal_json_path_list):
            cur_normal = json.load(open(json_path))
            sample_type = cur_normal[0]['type']
            self.sample_type_list.append(sample_type)
            self.normal_samples_dicts[sample_type] = cur_normal
            # self.normal_samples_dicts[sample_type].append(cur_normal)

        for idx, json_path in enumerate(outlier_json_path_list):
            cur_normal = json.load(open(json_path))
            sample_type = cur_normal[0]['type']
            self.outlier_samples_dicts[sample_type] = cur_normal
            # self.outlier_samples_dicts[sample_type].append(cur_normal)

        # if len(normal_json_path_list) == 1:
        #     cur_normal = json.load(open(normal_json_path_list[0]))
        #     self.normal_samples.extend(cur_normal)
        # else:
        #     for idx, json_path in enumerate(normal_json_path_list):
        #         cur_normal = json.load(open(json_path))
        #         sample_type = cur_normal[0]['type']
        #         self.sample_type_list.append(sample_type)
        #         self.normal_samples.extend(cur_normal)
        #
        # if len(outlier_json_path_list) == 1:
        #     cur_outlier = json.load(open(outlier_json_path_list[0]))
        #     self.outlier_samples.extend(cur_outlier)
        # else:
        #     for idx, json_path in enumerate(outlier_json_path_list):
        #         cur_outlier = json.load(open(json_path))
        #         self.outlier_samples.extend(cur_outlier)

        self.transform = transform
        self.total_n, self.total_o = len(self.normal_samples), len(self.outlier_samples)
        self.image = self.normal_samples + self.outlier_samples

    def _load_image(self, path: str) -> Image.Image:
        if 'npy' in path[-3:]:
            img = np.load(path)
            return img
        return Image.open(path)    #.convert('RGB')

    def _combine_images(self, image, image2):
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    # def get_one_task_data(self):
    #     """
    #     Get ones task maml data, include one batch support images and labels, one batch query images and labels.
    #     Returns: support_data, query_data
    #
    #     """
    #     cur_transforms = self.transform
    #     sample_type = random.choice(self.sample_type_list)
    #     same_normal_samples = self.normal_samples_dicts[sample_type]
    #     same_outlier_samples = self.outlier_samples_dicts[sample_type]
    #     normal_index = random.sample(same_normal_samples, self.shot)  # change to normal json file
    #     image_list = list()
    #     for i in range(0, len(normal_index)):
    #         assert normal_index[i]['type'] == sample_type
    #         n_img = self._load_image(normal_index[i]['image_path'])
    #         n_img = cur_transforms(n_img)
    #         image_list.append(n_img)
    #     train_few_normal_image = image_list
    #     test_few_normal_image = image_list
    #     image_list = list()
    #     label_list = list()
    #     sample_type_list = list()
    #     same_samples = same_normal_samples + same_outlier_samples
    #     normal_samples = random.sample(same_normal_samples, 2)
    #     outlier_samples = random.sample(same_outlier_samples, 2)
    #     train_query_samples = normal_samples + outlier_samples
    #     for i in range(0, 4):
    #         assert train_query_samples[i]['type'] == sample_type
    #         n_img = self._load_image(train_query_samples[i]['image_path'])
    #         n_img = cur_transforms(n_img)
    #         image_list.append(n_img)
    #         label = train_query_samples[i]['target']
    #         label_list.append(label)
    #         sample_type_list.append(sample_type)
    #     train_query_image = image_list
    #     train_query_label = label_list
    #     train_sample_type = sample_type_list
    #
    #     image_list = list()
    #     label_list = list()
    #     sample_type_list = list()
    #     normal_samples = random.sample(same_normal_samples, 16)
    #     outlier_samples = random.sample(same_outlier_samples, 16)
    #     test_query_samples = normal_samples + outlier_samples
    #     for i in range(0, 32):
    #         assert test_query_samples[i]['type'] == sample_type
    #         n_img = self._load_image(test_query_samples[i]['image_path'])
    #         n_img = cur_transforms(n_img)
    #         image_list.append(n_img)
    #         label = test_query_samples[i]['target']
    #         label_list.append(label)
    #         sample_type_list.append(sample_type)
    #     test_query_image = image_list
    #     test_query_label = label_list
    #     test_sample_type = sample_type_list
    #
    #     return np.array(train_query_image), np.array(train_query_label), np.array(train_few_normal_image), train_sample_type, \
    #         np.array(test_query_image), np.array(test_query_label), np.array(test_few_normal_image), test_sample_type

    def __len__(self):
        return  self.steps_per_epoch

    def __getitem__(self, index):
        return self.get_one_task_data()


    def create_task(self):
        num_tasks = 10
        cur_transforms = self.transform
        tasks = []
        for sample_type in self.sample_type_list:
            # same_normal_samples = [i for i in self.normal_samples if i['type'] == sample_type]
            # same_outlier_samples = [i for i in self.outlier_samples if i['type'] == sample_type]
            same_normal_samples = self.normal_samples_dicts[sample_type]
            same_outlier_samples = self.outlier_samples_dicts[sample_type]
            normal_index = random.sample(same_normal_samples, self.shot)  # change to normal json file
            image_list = list()
            for i in range(0, len(normal_index)):
                assert normal_index[i]['type'] == sample_type
                n_img = self._load_image(normal_index[i]['image_path'])
                n_img = cur_transforms(n_img)
                image_list.append(n_img)
            train_few_normal_image = image_list
            test_few_normal_image = image_list
            image_list = list()
            label_list = list()
            sample_type_list = list()
            same_samples = same_normal_samples + same_outlier_samples
            normal_samples = random.sample(same_normal_samples, 4)
            outlier_samples = random.sample(same_outlier_samples, 4)
            train_query_samples_tmp = normal_samples + outlier_samples
            # 生成一个随机打乱的索引
            random_indices = torch.randperm(len(train_query_samples_tmp))
            # 使用索引打乱Tensor
            train_query_samples = [train_query_samples_tmp[i] for i in random_indices]
            for i in range(0, 8):
                assert train_query_samples[i]['type'] == sample_type
                n_img = self._load_image(train_query_samples[i]['image_path'])
                n_img = cur_transforms(n_img)
                image_list.append(n_img)
                label = train_query_samples[i]['target']
                label_list.append(label)
                sample_type_list.append(sample_type)
            train_query_image = image_list
            train_query_label = label_list
            train_sample_type = sample_type_list

            image_list = list()
            label_list = list()
            sample_type_list = list()
            normal_samples = random.sample(same_normal_samples, 16)
            outlier_samples = random.sample(same_outlier_samples, 16)
            test_query_samples_tmp = normal_samples + outlier_samples
            # 生成一个随机打乱的索引
            random_indices = torch.randperm(len(test_query_samples_tmp))
            # 使用索引打乱Tensor
            test_query_samples = [test_query_samples_tmp[i] for i in random_indices]
            for i in range(0, 32):
                assert test_query_samples[i]['type'] == sample_type
                n_img = self._load_image(test_query_samples[i]['image_path'])
                n_img = cur_transforms(n_img)
                image_list.append(n_img)
                label = test_query_samples[i]['target']
                label_list.append(label)
                sample_type_list.append(sample_type)
            test_query_image = image_list
            test_query_label = label_list
            test_sample_type = sample_type_list
            tasks.append((train_query_image, train_query_label, train_few_normal_image, train_sample_type, test_query_image, test_query_label, test_few_normal_image, test_sample_type))

        return tasks



def construct_meta(cfg, split, transform, steps):
    assert split in ["train", "val", "test"]
    data_path = cfg.DATA_LOADER.data_path
    data_name = cfg.TRAIN.DATASET
    shot = cfg.shot
    transform = transform

    normal_json_path = None
    outlier_json_path = None
    if split in ["train"]:
        normal_json_path = cfg.normal_json_path
        outlier_json_path = cfg.outlier_json_path
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    elif split in ["test"]:
        normal_json_path = cfg.val_normal_json_path
        outlier_json_path = cfg.val_outlier_json_path
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    # Construct the dataset
    # dataset = build_dataset(data_name, data_path, normal_json_path, outlier_json_path, transform, shot)
    # Construct meta dataset
    dataset = meta_dataset(data_path, normal_json_path, outlier_json_path, transform, shot, steps)

    return dataset


def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, targets, labels = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    targets = [item for sublist in targets for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    inputs, targets, labels = default_collate(inputs), default_collate(targets), default_collate(labels)

    return inputs, targets, labels




def construct_loader(cfg, split, transform):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """

    assert split in ["train", "val", "test"]
    data_path = cfg.DATA_LOADER.data_path
    data_name = cfg.TRAIN.DATASET
    shot = cfg.shot
    transform = transform

    normal_json_path = None
    outlier_json_path = None
    if split in ["train"]:
        normal_json_path = cfg.normal_json_path
        outlier_json_path = cfg.outlier_json_path
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    elif split in ["test"]:
        normal_json_path = cfg.val_normal_json_path
        outlier_json_path = cfg.val_outlier_json_path
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    # Construct the dataset
    dataset = build_dataset(data_name, data_path, normal_json_path, outlier_json_path, transform, shot)
    print(len(dataset))
    # Create a sampler for multi-process training
    if cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
        collate_func = multiple_samples_collate
    else:
        collate_func = None

    # Create a loader
    if split in ["train"]:
        loader = torch.utils.data.DataLoader(
            dataset,
            worker_init_fn=worker_init_fn_seed,
            batch_sampler = BalancedBatchSampler(cfg, dataset),  # sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
            collate_fn=collate_func,
        )

    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the dataset.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler.py type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
