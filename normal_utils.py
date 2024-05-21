import numpy as np
import torch
import torch.nn as nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class Normalize(nn.Module):
    def __init__(self, mean, std, preprocess=True, ):
        super().__init__()
        mean = np.array(mean,dtype=np.float32)
        std = np.array(std,dtype=np.float32)

        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

        if preprocess:
            self.mean = self.mean.reshape(3, 1, 1)
            self.std = self.std.reshape(3, 1, 1)
        else:
            self.mean = self.mean.reshape(1, 3, 1, 1)
            self.std = self.std.reshape(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class DeNormalize(nn.Module):
    def __init__(self, mean, std, preprocess=False, ):
        super().__init__()
        mean = np.array(mean)
        std = np.array(std)

        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

        if preprocess:
            self.mean = self.mean.reshape(3, 1, 1)
            self.std = self.std.reshape(3, 1, 1)
        else:
            self.mean = self.mean.reshape(1, 3, 1, 1)
            self.std = self.std.reshape(1, 3, 1, 1)

    def forward(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)
