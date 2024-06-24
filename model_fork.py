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
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey, ArrayTree
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from flax.training.train_state import TrainState


import optax
import jax
import flax

from datasets_fork import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.truncated_normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.truncated_normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.truncated_normal(0.02))


def sincos_pos_embed_init(key, shape, cls_token=True):
    grid_size, embed_dim = shape

    pos_embed = jnp.array(get_2d_sincos_pos_embed(embed_dim, grid_size,
                                                  cls_token=cls_token))

    return jnp.expand_dims(pos_embed, 0)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@dataclass
class ViTBase:
    layers: int = 12
    dim: int = 768
    heads: int = 12
    labels: int | None = 1000
    layerscale: bool = False

    use_cls_token: bool = True

    patch_size: int = 16
    image_size: int = 224
    posemb: Literal["learnable", "sincos2d"] = "learnable"
    pooling: Literal["cls", "gap"] = "cls"

    dropout: float = 0.0
    droppath: float = 0.0
    grad_ckpt: bool = False
    use_kan: bool = False
    polynomial_degree: int = 8
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT
    use_fast_variance: bool = True

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


class PatchEmbed(ViTBase, nn.Module):
    stop_gradient_wpe: bool = True

    def setup(self):
        self.wte = Conv(
            self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID", dtype=self.dtype, precision=self.precision
        )
        # if self.pooling == "cls":
        #     self.cls_token = self.param(
        #         "cls_token", init.truncated_normal(0.02), (1, 1, self.dim)
        #     )

        if self.use_cls_token:
            self.cls_token = self.param(
                "cls_token", init.truncated_normal(0.02), (1, 1, self.dim), dtype=self.dtype
            )

        if self.posemb == "learnable":
            # self.wpe = self.param(
            #     "wpe", init.truncated_normal(0.02), (*self.num_patches, self.dim)
            # )

            # self.wpe = self.param(
            #     "wpe", init.truncated_normal(0.02), (1,self.num_patches[0]*self.num_patches[1]+1, self.dim),dtype=self.dtype
            # )

            self.wpe = self.param(
                "wpe", init.truncated_normal(0.02), (1, self.num_patches[0] * self.num_patches[1] + 1, self.dim),
                dtype=self.dtype
            )


        elif self.posemb == "sincos2d":
            self.wpe = self.param("wpe", sincos_pos_embed_init,
                                  (self.num_patches[0], self.dim))
            # self.wpe = get_2d_sincos_pos_embed(self.dim, self.num_patches[0], cls_token=True)

    def __call__(self, x: Array) -> Array:
        # x = (self.wte(x) + self.wpe).reshape(x.shape[0], -1, self.dim)
        x = (self.wte(x)).reshape(x.shape[0], -1, self.dim)
        # if self.pooling == "cls":
        #     cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
        #     x = jnp.concatenate((cls_token, x), axis=1)

        if self.use_cls_token:
            cls_token = jnp.repeat(self.cls_token, x.shape[0], axis=0)
            x = jnp.concatenate((cls_token, x), axis=1)

        if self.stop_gradient_wpe:
            x = x + jax.lax.stop_gradient(self.wpe)
        else:
            x = x + self.wpe

        return x


class Attention(ViTBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim), dtype=self.dtype, precision=self.precision)
        self.wk = DenseGeneral((self.heads, self.head_dim), dtype=self.dtype, precision=self.precision)
        self.wv = DenseGeneral((self.heads, self.head_dim), dtype=self.dtype, precision=self.precision)
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        # z = jnp.einsum("bqhd,bkhd->bhqk", self.wq(x) / self.head_dim ** 0.5, self.wk(x), precision=self.precision)
        # z = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(z), det), self.wv(x), precision=self.precision)

        z = nn.dot_product_attention(self.wq(x), self.wk(x), self.wv(x), precision=self.precision)

        return self.drop(self.wo(z), det)


class FeedForward(ViTBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim, dtype=self.dtype, precision=self.precision)
        self.w2 = Dense(self.dim, dtype=self.dtype, precision=self.precision)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(self.drop(nn.gelu(self.w1(x), approximate=True), det)), det)


class ViTLayer(ViTBase, nn.Module):
    drop_path_prob: float = 0.0

    def setup(self):
        self.attn = Attention(**self.kwargs)
        if self.use_kan:
            self.ff = KANLayer(self.polynomial_degree)
        else:
            self.ff = FeedForward(**self.kwargs)

        self.norm1 = nn.LayerNorm(dtype=self.dtype, use_fast_variance=self.use_fast_variance)
        self.norm2 = nn.LayerNorm(dtype=self.dtype, use_fast_variance=self.use_fast_variance)
        self.drop1 = nn.Dropout(self.drop_path_prob, broadcast_dims=(1, 2))
        self.drop2 = nn.Dropout(self.drop_path_prob, broadcast_dims=(1, 2))

        self.scale1 = self.scale2 = 1.0
        if self.layerscale:
            self.scale1 = self.param("scale1", init.constant(1e-4), (self.dim,))
            self.scale2 = self.param("scale2", init.constant(1e-4), (self.dim,))

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = x + self.drop1(self.scale1 * self.attn(self.norm1(x), det), det)
        x = x + self.drop2(self.scale2 * self.ff(self.norm2(x), det), det)
        return x


class ViT(ViTBase, nn.Module):
    def setup(self):
        self.embed = PatchEmbed(stop_gradient_wpe=False, **self.kwargs, )
        self.drop = nn.Dropout(self.dropout)

        # The layer class should be wrapped with `nn.remat` if `grad_ckpt` is enabled.
        layer_fn = nn.remat(ViTLayer) if self.grad_ckpt else ViTLayer

        dpr = [x.item() for x in np.linspace(0, self.droppath, self.layers)]
        self.layer = [layer_fn(**self.kwargs, drop_path_prob=dpr[i]) for i in range(self.layers)]
        # self.layer = [layer_fn(**self.kwargs, drop_path_prob=self.droppath) for i in range(self.layers)]

        self.norm = nn.LayerNorm(dtype=self.dtype, use_fast_variance=self.use_fast_variance)
        self.head = Dense(self.labels, dtype=self.dtype, precision=self.precision) if self.labels is not None else None

    def __call__(self, x: Array, det: bool = True) -> Array:
        x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        x = self.drop(self.embed(x), det)
        for layer in self.layer:
            x = layer(x, det)
        # x = self.norm(x)

        # If the classification head is not defined, then return the output of all
        # tokens instead of pooling to a single vector and then calculate class logits.
        if self.head is None:
            return x

        if self.pooling == "cls":
            x = self.norm(x)
            x = x[:, 0]
        elif self.pooling == "gap":
            # x = x.mean(1)
            x = x[:, 1:, :].mean(1)
            x = self.norm(x)
        x = self.head(x)
        print(x.dtype)
        return x
