from typing import Any

import einops
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy
import numpy as np

from convert_flax_to_pytorch_wideresnet import convert_flax_to_pytorch_wideresnet

CIFAR10_MEAN = jnp.array((0.4914, 0.4822, 0.4465))
CIFAR10_STD = jnp.array((0.2471, 0.2435, 0.2616))
CIFAR100_MEAN = jnp.array((0.5071, 0.4865, 0.4409))
CIFAR100_STD = jnp.array((0.2673, 0.2564, 0.2762))
SVHN_MEAN = jnp.array((0.5, 0.5, 0.5))
SVHN_STD = jnp.array((0.5, 0.5, 0.5))

class Block(nn.Module):
    filters: int
    strides: int
    proj_shortcut: bool = False
    act: Any = nn.silu
    precision:Any=jax.lax.Precision.DEFAULT

    def setup(self) -> None:
        self.batchnorm_0 = nn.BatchNorm()
        self.conv_0 = nn.Conv(self.filters, (3, 3), (self.strides, self.strides), padding="SAME", use_bias=False,precision=self.precision)
        self.batchnorm_1 = nn.BatchNorm()
        self.conv_1 = nn.Conv(self.filters, (3, 3), (1, 1), use_bias=False,precision=self.precision)

        if self.proj_shortcut:
            self.shortcut = nn.Conv(self.filters, (1, 1), (self.strides, self.strides), padding="SAME", use_bias=False,precision=self.precision)

    def __call__(self, x, use_running_average=True):

        if self.proj_shortcut:
            x = self.act(self.batchnorm_0(x, use_running_average=use_running_average))
        else:
            out = self.act(self.batchnorm_0(x, use_running_average=use_running_average))
        v = x if self.proj_shortcut else out

        out = self.conv_0(v)
        out = self.act(self.batchnorm_1(out, use_running_average=use_running_average))
        out = self.conv_1(out)

        out = jnp.add(self.shortcut(x) if self.proj_shortcut else x, out)

        return out

    def test(self, x, use_running_average=True):
        out = self.act(self.batchnorm_0(x, use_running_average=use_running_average))
        v = out
        out = self.conv_0(v)
        return out


class BlockGroup(nn.Module):
    num_blocks: int
    filters: int
    strides: int
    proj_shortcut: bool = False
    act: Any = nn.silu

    def setup(self):
        block = []
        for i in range(self.num_blocks):
            block.append(
                Block(self.filters,
                      i == 0 and self.strides or 1, act=self.act, proj_shortcut=i == 0 and self.proj_shortcut)
            )
        self.block = nn.Sequential(block)

    def __call__(self, x, use_running_average=True):
        x = self.block(x, use_running_average)
        print(x.shape)
        return x


class WideResNetJAX(nn.Module):
    num_classes: int = 10
    depth: int = 28
    width: int = 10
    act: Any = nn.silu

    mean: Any = CIFAR10_MEAN
    std: Any = CIFAR10_STD

    def setup(self):
        num_channels = [16, 16 * self.width, 32 * self.width, 64 * self.width]
        assert (self.depth - 4) % 6 == 0
        num_blocks = (self.depth - 4) // 6
        self.init_conv = nn.Conv(num_channels[0], (3, 3), (1, 1), (1, 1), use_bias=False)
        self.layer = nn.Sequential([
            BlockGroup(num_blocks, num_channels[1], 1, proj_shortcut=num_channels[0] != num_channels[1],
                       act=self.act),
            BlockGroup(num_blocks, num_channels[2], 2, proj_shortcut=num_channels[1] != num_channels[2],
                       act=self.act),
            BlockGroup(num_blocks, num_channels[3], 2, proj_shortcut=num_channels[2] != num_channels[3],
                       act=self.act)
        ])

        self.batchnorm = nn.BatchNorm()
        self.logits = nn.Dense(self.num_classes)

    def __call__(self, x, use_running_average=True):
        x = (x - self.mean) / self.std
        out = self.init_conv(x)
        out = self.layer(out)
        out = self.act(self.batchnorm(out, use_running_average=use_running_average))
        out = jnp.mean(out, axis=(1, 2))
        out = self.logits(out)
        return out

    def test(self, x, use_running_average=True):
        out = self.init_conv(x)
        out = self.layer(out)
        out = self.act(self.batchnorm(out, use_running_average=use_running_average))
        out = jnp.mean(out, axis=(1, 2))
        out = self.logits(out)
        return out


