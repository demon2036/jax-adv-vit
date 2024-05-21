import copy

from accelerate import Accelerator
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize

# from autoaugment import CIFAR10Policy
from utils import setup_seed
import torchvision
import torch
from normal_utils import CIFAR10_MEAN, CIFAR10_STD
from torch.utils.data import DataLoader
from utils import get_obj_from_str, json_print
from functools import partial

from main import get_train_dataloader


class BaseTrainer:
    def __init__(self,
                 seed: int,
                 batch_size,
                 max_device_batch_size,
                 total_epoch,
                 mixed_precision,
                 use_aux_dataset=False,
                 unsup_fraction=0.9,
                 aux_data_filename='/home/jtitor/Downloads/1m.npz',
                 save_every=500,
                 transform=True,
                 ):
        setup_seed(int(seed))

        batch_size = batch_size
        load_batch_size = min(max_device_batch_size, batch_size)
        self.total_epoch = total_epoch

        self.batch_size = batch_size
        self.load_batch_size = load_batch_size

        assert batch_size % load_batch_size == 0
        self.steps_per_update = batch_size // load_batch_size

        if not use_aux_dataset:
            print('Using Common DataSet')
            train_dataset,train_dataloader = get_train_dataloader()

            test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
                                                        transform=Compose(
                                                            [ToTensor(), ]))  # 0.5, 0.5

            train_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=True, download=True,
                                                        transform=Compose(
                                                            [ToTensor(), ]))  # 0.5, 0.5

            # train_dataloader = DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=6,
            #                               drop_last=True, pin_memory=True, persistent_workers=True)
            test_dataloader = DataLoader(test_dataset, load_batch_size, shuffle=False, num_workers=1, )

        self.train_dataset = train_dataset
        self.val_dataset = test_dataset

        self.train_dataloader = train_dataloader

        self.val_dataloader = test_dataloader

        self.save_every = save_every


if __name__ == "__main__":
    pass
