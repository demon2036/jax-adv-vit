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
from collections.abc import Iterator
from functools import partial
from typing import Any

import PIL
import einops
import jax
import numpy as np
import optax.losses
import timm
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import webdataset as wds
from timm.data import create_transform
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import Compose, ToTensor

# from baseline.model import MAE_ViT_2_T

IMAGENET_DEFAULT_MEAN = np.array([0.4914, 0.4822, 0.4465])
IMAGENET_DEFAULT_STD = np.array([0.2471, 0.2435, 0.2616])


def auto_augment_factory(auto_augment) -> T.Transform:
    aa_hparams = {
        "translate_const": int(32 * 0.45),
        "img_mean": tuple((IMAGENET_DEFAULT_MEAN * 0xFF).astype(int)),
    }

    if auto_augment == "none":
        return T.Identity()
    if auto_augment.startswith("rand"):
        return rand_augment_transform(auto_augment, aa_hparams)
    if auto_augment.startswith("augmix"):
        aa_hparams["translate_pct"] = 0.3
        return augment_and_mix_transform(auto_augment, aa_hparams)
    return auto_augment_transform(auto_augment, aa_hparams)


def create_transforms(image_size):
    train_transforms = [
        T.ToPILImage(),
        T.Resize(image_size, interpolation=3),
        T.RandomCrop(image_size, padding=4, fill=128),
        # T.RandomResizedCrop(image_size, scale=(0.6, 1)),
        T.RandomHorizontalFlip(),
        # T.RandomErasing(0.25, value="random"),
        # T.Resize(224, interpolation=3),
        # T.CenterCrop(224),
        T.PILToTensor(),
    ]

    train_strong_transforms = [
        T.ToPILImage(),
        T.Resize(image_size, interpolation=3),
        T.RandomCrop(image_size, padding=4, fill=128),
        T.RandomHorizontalFlip(),
        auto_augment_factory('rand-m9-mstd0.5-inc1'),
        T.PILToTensor(),
        # test,
    ]

    test_transforms = [
        T.ToPILImage(),
        T.Resize(image_size, interpolation=3),
        T.ToTensor()
    ]

    return T.Compose(train_transforms), T.Compose(train_strong_transforms), T.Compose(test_transforms)


def repeat_samples(samples: Iterator[Any], repeats: int = 1) -> Iterator[Any]:
    for sample in samples:
        for _ in range(repeats):
            yield copy.deepcopy(sample)


def collate_and_shuffle(batch: list[Any], repeats: int = 1) -> Any:
    return default_collate(sum([batch[i::repeats] for i in range(repeats)], []))


def collate_and_pad(batch: list[Any], batch_size: int = 1) -> Any:
    pad = tuple(torch.full_like(x, fill_value=-1) for x in batch[0])
    # print(batch, len(batch))
    return default_collate(batch + [pad] * (batch_size - len(batch)))


def mix_dataloader_iter(train_dataloader, train_origin_dataloader):
    train_dataloader_iter = iter(train_dataloader)
    train_origin_dataloader_iter = iter(train_origin_dataloader)

    while True:
        yield [torch.cat([x, y], dim=0) for x, y in
               zip(next(train_dataloader_iter), next(train_origin_dataloader_iter))]


def get_train_dataloader(batch_size=1024,
                         shard_path='gs://caster-us-central-2b/cifar10-20m-wds/shards-{00000..01290}.tar',
                         test_shard_path='gs://caster-us-central-2b/cifar10-test-wds/shards-{00000..00078}.tar',
                         origin_shard_path='gs://fbs0_dl_bucket/cifar100-train-wds/shards-{00000..00099}.tar',
                         image_size=32,
                         ):
    print(batch_size)

    total_batch_size = batch_size // jax.process_count()
    train_batch_size = int(total_batch_size * 0.8)
    train_origin_batch_size = total_batch_size - train_batch_size

    train_transform, train_strong_transforms, test_transform = create_transforms(image_size)

    # train_transform = [
    #     T.ToPILImage(),
    #     create_transform(
    #         input_size=image_size,
    #         is_training=True,
    #         color_jitter=0,
    #         # auto_augment=args.auto_augment,
    #         interpolation='bicubic',
    #         re_prob=0.25,
    #         re_mode='pixel',
    #         re_count=1, mean=(0, 0, 0), std=(1, 1, 1))
    #
    # ]
    # test_transform = [
    #     T.ToPILImage(),
    #     T.Resize(256, interpolation=PIL.Image.BICUBIC),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ]

    def test(x):
        print(type(x))
        return x

    test_transform2 = [
        T.Resize(256, interpolation=PIL.Image.BICUBIC, antialias=False),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]

    train_transform = T.Compose(train_transform)
    test_transform = T.Compose(test_transform)
    test_transform2 = T.Compose(test_transform2)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shard_path, seed=1),
        itertools.cycle,
        wds.detshuffle(seed=100),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(seed=100),
        wds.decode("pil", handler=wds.ignore_and_continue),
        # wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # # partial(repeat_samples, repeats=args.augment_repeats),
        # wds.map_tuple(train_transform, torch.tensor),

        wds.to_tuple("jpg.pyd", "jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # partial(repeat_samples, repeats=args.augment_repeats),
        wds.map_tuple(train_transform, train_strong_transforms, torch.tensor),

    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=32,
        # collate_fn=partial(collate_and_shuffle, repeats=args.augment_repeats),
        drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )

    dataset = wds.DataPipeline(
        wds.SimpleShardList(origin_shard_path, seed=1),
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        # wds.to_tuple("jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # # partial(repeat_samples, repeats=args.augment_repeats),
        # wds.map_tuple(train_transform, torch.tensor),
        wds.to_tuple("jpg.pyd", "jpg.pyd", "cls", handler=wds.ignore_and_continue),
        # partial(repeat_samples, repeats=args.augment_repeats),
        wds.map_tuple(train_transform, train_strong_transforms, torch.tensor),
    )

    train_origin_dataloader = DataLoader(
        dataset,
        batch_size=train_origin_batch_size,
        num_workers=4,
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
        wds.to_tuple("jpg", "jpg", "cls", handler=wds.ignore_and_continue),
        wds.map_tuple(test_transform, test_transform2, torch.tensor),

    ]

    test_dataset = wds.WebDataset(urls=test_shard_path, handler=wds.ignore_and_continue).mcached()

    for op in ops:
        test_dataset = test_dataset.compose(op)
    #
    test_batch_size = 16
    num_workers = 32

    count = jax.process_count()
    # count=8

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size // count,
        num_workers=num_workers,
        # collate_fn=partial(collate_and_pad, batch_size=test_batch_size // count),
        drop_last=False,
        prefetch_factor=10,
        persistent_workers=True,
    )

    # for data in test_dataloader:
    #     print(data[0].shape)

    return mix_dataloader_iter(train_dataloader, train_origin_dataloader), test_dataloader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_dataloader, test_dataloader = get_train_dataloader(
        test_shard_path='imagenet1k-validation-00.tar',
        shard_path='imagenet1k-validation-00.tar',
        origin_shard_path='imagenet1k-validation-00.tar', image_size=16, batch_size=16)


    # def patchify(imgs):
    #     """
    #     imgs: (N, H, W, 3)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = 4
    #     assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0
    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape((imgs.shape[0], h, p, w, p, 3))
    #     x = torch.einsum('nhpwqc->nhwpqc', x)
    #     x = x.reshape((imgs.shape[0], h * w, p ** 2 * 3))
    #     return x

    def patchify(imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 4
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


    for data in test_dataloader:
        img, *_ = data
        img, img2, _ = data

        # print(img - img2)
        # print(img.shape)

        # x=einops.rearrange(img,'b c h w->b h w c')
        #
        # x1=x.reshape(x.shape[0], -1, x.shape[-1])
        # x2=einops.rearrange(x,'b h w c->b (h w) c')
        # print(x1-x2)

        # x = einops.rearrange(img, 'b c h w->b h w c')
        x=img
        x1 = patchify(x)
        # x2 = einops.rearrange(x, 'b (h k1) (w k2) c->b (h w) (k1 k2 c)', k1=4, k2=4)
        x2 = einops.rearrange(x, 'b c (h k1) (w k2) ->b (h w) (c k1 k2 )', k1=4, k2=4)
        x3= x.flatten(2).transpose(1, 2)
        print(x3.shape)

        # temp=


        # if img.shape[1] == 3:
        #     img = einops.rearrange(img, '(b k) c h w -> (b h) (k w) c', k=4)
        # else:
        #     img = einops.rearrange(img, '(b k)  h w c -> (b h) (k w) c', k=4)
        #
        # for i in range(100):
        #     # plt.imshow(img[i])
        #     plt.imshow(x1[0])
        #     plt.show()
        #
        #     break


        break

    while True:
        pass

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
