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

from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal

import einops
import flax.linen as nn
import flax.linen.initializers as init
import jax.experimental.pallas.ops.tpu.flash_attention
import jax.numpy as jnp
import numpy as np
from chex import Array

from datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils2 import fixed_sincos2d_embeddings

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))


def normalize_jax(x, dim=None, eps=1e-4):
    if dim is None:
        dim = tuple(range(1, x.ndim))
    norm = jnp.linalg.vector_norm(x, axis=dim, keepdims=True, )
    norm = jnp.add(eps, norm, ) * np.sqrt(norm.size / x.size)
    return x / norm


def mp_silu(x):
    return nn.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return (a + t * (b - a)) / np.sqrt((1 - t) ** 2 + t ** 2)

    # return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    patch_size: int = 2
    image_size: int = 32
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "gap"
    qk_norm: bool = False
    use_fc_norm: bool = True
    reduce_include_prefix: bool = False

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    use_kan: bool = False
    polynomial_degree: int = 8

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ViTBase)}

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def num_patches(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size,) * 2


class MPDense(ViTBase, nn.Module):
    kernel_init: Any = init.truncated_normal(0.02)

    @nn.compact
    def __call__(self, x, gain=1):
        w = self.param(
            'kernel',
            self.kernel_init,
            (jnp.shape(x)[-1], self.dim),

        )

        w = normalize_jax(w, dim=0)
        w = w * (gain / np.sqrt(w[:, 0].size))
        return x @ w


class PatchEmbed(ViTBase, nn.Module):
    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )
        # if self.pooling == "cls":
        self.cls_token = self.param(
            "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
        )

        if self.posemb == "learnable":
            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            )
        elif self.posemb == "sincos2d":
            self.wpe = fixed_sincos2d_embeddings(*self.num_patches, self.dim)

    def __call__(self, x: Array) -> Array:
        x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        # if self.pooling == "cls":
        cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
        x = jnp.concatenate((cls_token, x), axis=1)
        return x


class Identity(nn.Module):
    def __call__(self, x):
        return x


# class Attention(ViTBase, nn.Module):
#     def setup(self):
#         self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()
#         self.wq = DenseGeneral((self.heads, self.head_dim))
#         self.wk = DenseGeneral((self.heads, self.head_dim))
#         self.wv = DenseGeneral((self.heads, self.head_dim))
#         self.wo = DenseGeneral(self.dim, axis=(-2, -1))
#         self.drop = nn.Dropout(self.dropout)
#
#     def __call__(self, x: Array, det: bool = True) -> Array:
#         z = jnp.einsum("bqhd,bkhd->bhqk", self.q_norm(self.wq(x)) / self.head_dim ** 0.5, self.k_norm(self.wk(x)))
#         z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x))
#         return self.drop(self.wo(z), det)


class Attention(ViTBase, nn.Module):
    def setup(self):
        self.wq = MPDense(self.dim, )
        self.wk = MPDense(self.dim, )
        self.wv = MPDense(self.dim)
        self.wo = MPDense(self.dim, )

    def __call__(self, x, det: bool = True):
        q = einops.rearrange(self.wq(x), 'b q (h d)-> b h q d', h=self.heads)
        k = einops.rearrange(self.wk(x), 'b q (h d)-> b h q d', h=self.heads)
        v = einops.rearrange(self.wv(x), 'b q (h d)-> b h q d', h=self.heads)
        z = jnp.einsum("bhqd,bhkd->bhqk", q / self.head_dim ** 0.5, k)

        # print(z.shape,v.shape)

        z = jnp.einsum("bhqk,bhkd->bhqd", nn.softmax(z), v)
        z = einops.rearrange(z, 'b h q d ->  b q (h d) ')

        return self.wo(z)


class FeedForward(ViTBase, nn.Module):
    def setup(self):
        self.w1 = MPDense(dim=self.hidden_dim)
        self.w2 = MPDense(dim=self.dim)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.w2(mp_silu(self.w1(x)))

        # return self.drop(self.w2(self.drop(nn.gelu(self.w1(x)), det)), det)


class ViTLayer(ViTBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        if self.use_kan:
            self.ff = KANLayer(self.polynomial_degree)
        else:
            self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.drop = nn.Dropout(self.droppath, broadcast_dims=(1, 2))

        self.scale1 = self.scale2 = 1.0
        if self.layerscale:
            self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
            self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))
            # self.scale1 = self.param("scale1", init.constant(1e-6), (self.dim,))
            # self.scale2 = self.param("scale2", init.constant(1e-6), (self.dim,))

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = x + self.drop(self.scale1 * self.attn(self.norm1(x), det), det)
        x = x + self.drop(self.scale2 * self.ff(self.norm2(x), det), det)

        # x = mp_sum(x, self.attn(self.norm1(x)), 0.3)
        # x = mp_sum(x, self.ff(self.norm2(x)), 0.3)

        return x


class ViT(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(**self.kwargs)
        self.drop = nn.Dropout(self.dropout)

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer
        self.layer = [layer_fn(**self.kwargs) for _ in range(self.layers)]

        # self.norm = nn.LayerNorm()

        self.norm = nn.LayerNorm() if not self.use_fc_norm else Identity()
        self.fc_norm = nn.LayerNorm() if self.use_fc_norm else Identity()

        print(self.kwargs)

        self.head = nn.Dense(self.labels) if self.labels is not None else None

    def __call__(self, x: Array, det: bool = True) -> Array:
        # x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        x = self.drop(self.embed(x), det)
        for layer in self.layer:
            x = layer(x, det)
        x = self.norm(x)

        # If the classification head is not defined, then return the output of all
        # tokens instead of pooling to a single vector and then calculate class logits.
        if self.head is None:
            return x
        """
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x[:, 0:].mean(1)
        return self.head(x)
        """

        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "gap":
            x = x if self.reduce_include_prefix else x[:, 1:]
            x = x.mean(1)
        else:
            raise NotImplemented()

        x = self.fc_norm(x)

        return self.head(x)
