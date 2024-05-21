from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor
import webdataset as wds
from tqdm import tqdm

transform_test = [PILToTensor()]

test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
                                            transform=Compose(
                                                transform_test))

test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=16, drop_last=False)



shard_path = './cifar10-test-wds'
shard_dir_path = Path(shard_path)
shard_dir_path.mkdir(exist_ok=True)
shard_filename = str(shard_dir_path / 'shards-%05d.tar')

with wds.ShardWriter(
        shard_filename, maxcount=128,
) as sink, tqdm(test_dataset) as pbar:
    for i, (img, label) in enumerate(pbar):
        sink.write({
            "__key__": str(i),
            "jpg.pyd": img,
            "cls": int(label),
            # "json": label,
        })
