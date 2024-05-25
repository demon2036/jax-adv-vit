# Copyright 2024 Jungwoo Park (affjljoo3581)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import copy
import itertools
import math
from collections.abc import Iterator
from functools import partial
from typing import Any

import jax
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import webdataset as wds
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from torch.utils.data import DataLoader, default_collate, IterableDataset
from torchvision.transforms import Compose, ToTensor

# from baseline.model import MAE_ViT_2_T

IMAGENET_DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_DEFAULT_STD = np.array([0.229, 0.224, 0.225])


def auto_augment_factory(args: argparse.Namespace) -> T.Transform:
    aa_hparams = {
        "translate_const": int(args.image_size * 0.45),
        "img_mean": tuple((IMAGENET_DEFAULT_MEAN * 0xFF).astype(int)),
    }
    if args.auto_augment == "none":
        return T.Identity()
    if args.auto_augment.startswith("rand"):
        return rand_augment_transform(args.auto_augment, aa_hparams)
    if args.auto_augment.startswith("augmix"):
        aa_hparams["translate_pct"] = 0.3
        return augment_and_mix_transform(args.auto_augment, aa_hparams)
    return auto_augment_transform(args.auto_augment, aa_hparams)


def create_transforms() -> tuple[nn.Module, nn.Module]:
    train_transforms = [
        T.RandomHorizontalFlip(),
        # T.Resize(224, interpolation=3),
        # T.CenterCrop(224),
        T.PILToTensor(),
    ]

    test_transforms = [
        T.ToTensor()
    ]

    return T.Compose(train_transforms), T.Compose(test_transforms)


def repeat_samples(samples: Iterator[Any], repeats: int = 1) -> Iterator[Any]:
    for sample in samples:
        for _ in range(repeats):
            yield copy.deepcopy(sample)


def collate_and_shuffle(batch: list[Any], repeats: int = 1) -> Any:
    return default_collate(sum([batch[i::repeats] for i in range(repeats)], []))


def collate_and_pad(batch: list[Any], batch_size: int = 1) -> Any:
    pad = tuple(torch.full_like(x, fill_value=-1) for x in batch[0])
    # print(batch_size, len(batch))
    return default_collate(batch + [pad] * (batch_size - len(batch)))


class SampleEqually(wds.DataPipeline, wds.compat.FluidInterface ):
    def __init__(self, gen, ori,batch_size=1024):
        super().__init__()
        self.gen = iter(gen)
        self.ori = iter(ori)
        self.batch_size=batch_size

        # self.datasets = datasets

    def __iter__(self):
        batch = []
        while True:
            try:
                if np.random.uniform(0, 1) < 0.2:
                    batch.append(next(self.ori))
                else:
                    batch.append(next(self.gen))
                if len(batch) == self.batch_size:  # Check if batch is full
                    yield batch
                    batch = []  # Reset batch
            except StopIteration:
                if batch:
                    yield batch  # Yield the remaining samples
                return


def get_train_dataloader(batch_size=1024,
                         shard_path='gs://caster-us-central-2b/cifar10-20m-wds/shards-{00000..01290}.tar',
                         test_shard_path='gs://caster-us-central-2b/cifar10-test-wds/shards-{00000..00078}.tar',
                         shard_path_origin='gs://caster-us-central-2b/cifar10-train-wds/shards-{00000..00390}.tar',
                         origin_to_generate_ratio=0.2

                         ):
    # shard_path = './shards_01/shards-00040.tar'

    origin_batch_size = int(origin_to_generate_ratio * batch_size)
    generate_batch_size = batch_size - origin_batch_size

    process_count = jax.process_count()
    local_device_count = jax.local_device_count()

    origin_batch_size = math.ceil(
        origin_batch_size / process_count / local_device_count) * process_count * local_device_count
    generate_batch_size = math.ceil(
        generate_batch_size / process_count / local_device_count) * process_count * local_device_count

    train_transform, test_transform = create_transforms()
    ops = [
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        # # wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # partial(repeat_samples, repeats=3),
        wds.map_tuple(train_transform, torch.tensor),
    ]

    dataset = wds.WebDataset(urls=shard_path, handler=wds.ignore_and_continue)
    for op in ops:
        dataset = dataset.compose(op)

    train_dataloader = DataLoader(
        dataset,
        batch_size=generate_batch_size // jax.process_count(),
        num_workers=32,
        # collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
        drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )

    dataset = wds.WebDataset(urls=shard_path_origin, handler=wds.ignore_and_continue)
    for op in ops:
        dataset = dataset.compose(op)

    train_origin_dataloader = DataLoader(
        dataset,
        batch_size=origin_batch_size // jax.process_count(),
        num_workers=32,
        # collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
        drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )

    ops = [
        # wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        # wds.split_by_worker,
        # # wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        # wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        wds.map_tuple(test_transform, torch.tensor),

    ]

    test_dataset = wds.WebDataset(urls=test_shard_path, handler=wds.ignore_and_continue).mcached()

    for op in ops:
        test_dataset = test_dataset.compose(op)
    #
    test_batch_size = 1024
    num_workers = 32

    # count=8

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size // process_count,
        num_workers=num_workers,
        collate_fn=partial(collate_and_pad, batch_size=test_batch_size // process_count),
        drop_last=False,
        prefetch_factor=10,
        persistent_workers=True,
    )

    return train_dataloader, test_dataloader, train_origin_dataloader


def get_train_dataloader2(batch_size=1024,
                          shard_path='gs://caster-us-central-2b/cifar10-20m-wds/shards-{00000..01290}.tar',
                          test_shard_path='gs://caster-us-central-2b/cifar10-test-wds/shards-{00000..00078}.tar',
                          shard_path_origin='gs://caster-us-central-2b/cifar10-train-wds/shards-{00000..00390}.tar',
                          origin_to_generate_ratio=0.2

                          ):
    # shard_path = './shards_01/shards-00040.tar'

    origin_batch_size = int(origin_to_generate_ratio * batch_size)
    generate_batch_size = batch_size - origin_batch_size

    process_count = jax.process_count()
    local_device_count = jax.local_device_count()

    origin_batch_size = math.ceil(
        origin_batch_size / process_count / local_device_count) * process_count * local_device_count
    generate_batch_size = math.ceil(
        generate_batch_size / process_count / local_device_count) * process_count * local_device_count

    train_transform, test_transform = create_transforms()
    ops = [
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        # # wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # partial(repeat_samples, repeats=3),
        wds.map_tuple(train_transform, torch.tensor),
    ]

    gen_dataset = wds.WebDataset(urls=shard_path, handler=wds.ignore_and_continue)
    for op in ops:
        gen_dataset = gen_dataset.compose(op)

    ori_dataset = wds.WebDataset(urls=shard_path_origin, handler=wds.ignore_and_continue)
    for op in ops:
        ori_dataset = ori_dataset.compose(op)

    dataset = SampleEqually(gen_dataset, ori_dataset)

    train_dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        # batch_size=generate_batch_size // jax.process_count(),
        num_workers=32,
        # collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
        # drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )

    for data in train_dataloader:
        print(data)
        img, _ = data
        print(img.shape,_)
        break

    while True:
        pass

    ops = [
        # wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        # wds.split_by_worker,
        # # wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        # wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        wds.map_tuple(test_transform, torch.tensor),

    ]

    test_dataset = wds.WebDataset(urls=test_shard_path, handler=wds.ignore_and_continue).mcached()

    for op in ops:
        test_dataset = test_dataset.compose(op)
    #
    test_batch_size = 1024
    num_workers = 32

    # count=8

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size // process_count,
        num_workers=num_workers,
        collate_fn=partial(collate_and_pad, batch_size=test_batch_size // process_count),
        drop_last=False,
        prefetch_factor=10,
        persistent_workers=True,
    )

    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = get_train_dataloader2(
        test_shard_path='./cifar10-train-wds/shards-{00000..00078}.tar',
        shard_path='./cifar10-train-wds/shards-{00000..00078}.tar',
        shard_path_origin='./cifar10-train-wds/shards-{00000..00078}.tar')

    for data in train_dataloader:
        img, _ = data
        print(img.shape)
        break

# if __name__ == "__main__":
#     model = MAE_ViT_2_T()
#     test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
#                                                 transform=Compose(
#                                                     [ToTensor(), ]))  # 0.5, 0.5
#     test_dataloader = DataLoader(test_dataset, 128, shuffle=False, num_workers=1, )
#     count = 0
#     for data in train_dataloader:
#         print(data)
#         count += 1
#
#         if count == 1000:
#             break