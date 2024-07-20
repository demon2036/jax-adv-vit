import argparse
import json
import math
import os.path
import re

import jax
import torch
from flax.serialization import msgpack_serialize

# jax.distributed.initialize()

from functools import partial

from timm.models import VisionTransformer
from torch.utils.data import DataLoader
import einops
import flax.jax_utils

import jax.numpy as jnp
import numpy as np
import orbax
import orbax.checkpoint as ocp


def convert(params, checkpoint):
    params = {'model': params}
    pos_embed = params["model"]["embed"]["wpe"]
    pos_embed = pos_embed.reshape(1, -1, pos_embed.shape[-1])
    wte = params["model"]["embed"]["wte"]["kernel"].transpose(3, 2, 0, 1)

    state_dict = {
        "patch_embed.proj.weight": wte,
        "patch_embed.proj.bias": params["model"]["embed"]["wte"]["bias"],
    }

    if "cls_token" in params["model"]["embed"]:
        print("Use cls token")
        state_dict["cls_token"] = params["model"]["embed"]["cls_token"]
        state_dict["pos_embed"] = jnp.pad(pos_embed, ((0, 0), (1, 0), (0, 0)))
    else:
        state_dict["pos_embed"] = pos_embed
        print("Use Mean Pooling")

    print(pos_embed.shape, state_dict["pos_embed"].shape)

    if "norm" in params["model"]:
        print('use norm')
        state_dict["norm.weight"] = params["model"]["norm"]["scale"]
        state_dict["norm.bias"] = params["model"]["norm"]["bias"]

    if "fc_norm" in params["model"]:
        print('use fc_norm')
        state_dict["fc_norm.weight"] = params["model"]["fc_norm"]["scale"]
        state_dict["fc_norm.bias"] = params["model"]["fc_norm"]["bias"]
    # if "head" in params["model"] and not args.exclude_heads:
    #     state_dict["head.weight"] = params["model"]["head"]["kernel"].transpose(1, 0)
    #     state_dict["head.bias"] = params["model"]["head"]["bias"]

    if "head" in params["model"]:
        state_dict["head.weight"] = params["model"]["head"]["kernel"].transpose(1, 0)
        state_dict["head.bias"] = params["model"]["head"]["bias"]
        print(params["model"]["head"]["kernel"].shape)

    for name, layer in params["model"].items():
        if not name.startswith("layer_"):
            continue
        layer_idx = int(name[6:])

        if "q_norm" in layer["attn"]:
            print('use qk_norm')
            state_dict[f"blocks.{layer_idx}.attn.q_norm.weight"] = layer["attn"]["q_norm"]["scale"]
            state_dict[f"blocks.{layer_idx}.attn.q_norm.bias"] = layer["attn"]["q_norm"]["bias"]
            state_dict[f"blocks.{layer_idx}.attn.k_norm.weight"] = layer["attn"]["k_norm"]["scale"]
            state_dict[f"blocks.{layer_idx}.attn.k_norm.bias"] = layer["attn"]["k_norm"]["bias"]
        else:
            print('Do not use qk_norm')

        wq = layer["attn"]["wq"]["kernel"]
        wk = layer["attn"]["wk"]["kernel"]
        wv = layer["attn"]["wv"]["kernel"]
        wo = layer["attn"]["wo"]["kernel"]

        wq = wq.reshape(wq.shape[0], -1)
        wk = wk.reshape(wk.shape[0], -1)
        wv = wv.reshape(wv.shape[0], -1)
        wo = wo.reshape(wv.shape[0], -1)
        qkv = jnp.concatenate((wq, wk, wv), axis=1).transpose(1, 0)

        state_dict[f"blocks.{layer_idx}.attn.qkv.weight"] = qkv
        state_dict[f"blocks.{layer_idx}.attn.qkv.bias"] = jnp.concatenate(
            (
                layer["attn"]["wq"]["bias"].reshape(-1),
                layer["attn"]["wk"]["bias"].reshape(-1),
                layer["attn"]["wv"]["bias"].reshape(-1),
            ),
        )
        state_dict[f"blocks.{layer_idx}.attn.proj.weight"] = wo.transpose(1, 0)
        state_dict[f"blocks.{layer_idx}.attn.proj.bias"] = layer["attn"]["wo"]["bias"]

        fc1 = layer["ff"]["w1"]["kernel"].transpose(1, 0)
        fc2 = layer["ff"]["w2"]["kernel"].transpose(1, 0)
        state_dict[f"blocks.{layer_idx}.mlp.fc1.weight"] = fc1
        state_dict[f"blocks.{layer_idx}.mlp.fc1.bias"] = layer["ff"]["w1"]["bias"]
        state_dict[f"blocks.{layer_idx}.mlp.fc2.weight"] = fc2
        state_dict[f"blocks.{layer_idx}.mlp.fc2.bias"] = layer["ff"]["w2"]["bias"]

        state_dict[f"blocks.{layer_idx}.norm1.weight"] = layer["norm1"]["scale"]
        state_dict[f"blocks.{layer_idx}.norm1.bias"] = layer["norm1"]["bias"]
        state_dict[f"blocks.{layer_idx}.norm2.weight"] = layer["norm2"]["scale"]
        state_dict[f"blocks.{layer_idx}.norm2.bias"] = layer["norm2"]["bias"]

        if "scale1" in layer:
            state_dict[f"blocks.{layer_idx}.ls1.gamma"] = layer["scale1"]
        if "scale2" in layer:
            state_dict[f"blocks.{layer_idx}.ls2.gamma"] = layer["scale2"]

    state_dict = {k: torch.tensor(np.asarray(v)) for k, v in state_dict.items()}

    print(state_dict.keys())

    torch.save(state_dict, checkpoint)  #args.checkpoint.replace(".msgpack", ".pth")
    return state_dict


def train_and_evaluate(args):
    """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """

    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    # checkpointer = ocp.PyTreeCheckpointer()
    print(os.getcwd()+'/'+args.pretrained_model)
    if args.restore_type == 'orbax':
        state = checkpointer.restore(os.getcwd()+'/'+args.pretrained_model, )['model']
        params = state['ema_params']
    else:
        with open(args.pretrained_model, "rb") as fp:
            params = flax.serialization.msgpack_restore(fp.read())

    state_dict = convert(params, args.checkpoint)

    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model", type=str,
                        default='checkpoint/jax_model/best/cifar10/vit-l2-32-cifar10-2000ep-ls0.4-gap-beta3-ema')
    parser.add_argument("--checkpoint", type=str,
                        default='checkpoint/pytorch_model/best/cifar10/vit-l2-32-cifar10-2000ep-ls0.4-gap-beta3-ema.pth')

    parser.add_argument("--restore-type", type=str,
                        default='orbax',
                        # default='no_orbax',
                        )

    args = parser.parse_args()
    train_and_evaluate(args)
