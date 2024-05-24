from pathlib import Path

import einops
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor
import webdataset as wds
from tqdm import tqdm


def write_shard(shard_path='./cifar10-test-wds', dataset=None):
    shard_dir_path = Path(shard_path)
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')

    with wds.ShardWriter(
            shard_filename, maxcount=128,
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


if __name__ == '__main__':
    transform_test = [PILToTensor()]
    ransform_test = [ToTensor()]

    test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
                                                transform=Compose(
                                                    transform_test))

    train_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=True, download=True,
                                                transform=Compose(
                                                    transform_test))

    write_shard('./cifar10-train-wds',train_dataset)

    # test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=16, drop_last=False)
