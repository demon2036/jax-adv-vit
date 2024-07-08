import argparse
import math
from pathlib import Path

import einops
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor
import webdataset as wds
from tqdm import tqdm


def write_shard(shard_path='./cifar10-l2-test-wds', dataset=None):
    shard_dir_path = Path(shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')

    with wds.ShardWriter(
            shard_filename, maxcount=math.floor(len(dataset) / 100),
    ) as sink, tqdm(dataset) as pbar:
        for i, (img, label) in enumerate(pbar):
            # print(np.array(img).shape)

            temp = np.array(img)
            temp = einops.rearrange(temp, 'c h w->h w c')
            # print(temp.shape)

            sink.write({
                "__key__": str(i),
                "jpg.pyd": temp,
                "cls": np.array(label),
                # "json": label,
            })


def get_dataset(dataset_name):
    transform_test = [PILToTensor()]
    if dataset_name == 'cifar10-l2':
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                                    transform=Compose(
                                                        transform_test))

        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                     transform=Compose(
                                                         transform_test))
    elif dataset_name == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                     transform=Compose(
                                                         transform_test))

        train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                      transform=Compose(
                                                          transform_test))

    else:
        raise NotImplemented()

    return train_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--shard-path", type=str, default='./cifar10-l2-test-wds')

    args = parser.parse_args()
    train_dataset, test_dataset = get_dataset(args.dataset_name)

    write_shard(args.shard_path, train_dataset)

    # test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=16, drop_last=False)
