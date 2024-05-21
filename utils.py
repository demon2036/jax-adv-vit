import importlib
import random

import timm.optim
import torch
import numpy as np
import yaml
import json
import torch.nn as nn

acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())


def features_kl_adv_loss(model,
                         x_natural, x_adv, loss_start=0
                         ):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    feats = model.forward_feat(x_natural)[loss_start:]

    losses = 0
    feats_adv = model.forward_feat(x_adv)[loss_start:]
    for feat, feat_adv in zip(feats, feats_adv):
        loss = criterion_kl(torch.log_softmax(feat_adv, -1), torch.softmax(feat, -1))
        losses += loss / len(feats_adv)

    return losses, feats, feats_adv


def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_obj_from_str(string: str):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)


def json_print(json_data, indent=5):
    print(json.dumps(json_data, indent=indent))


def read_yaml(yaml_file_path: str = 'configs/mod_custom/tiny.yaml'):
    print_with_seperator(f'Read yaml From {yaml_file_path}')
    with open(yaml_file_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
        json_print(yaml_data)
        print()
    return yaml_data


def get_config(args):
    args_dict = {key: value for key, value in vars(args).items() if value is not None and key != 'yaml_path'}
    print('Using Default Config From Yaml')
    yaml_data = read_yaml(args.yaml_path)

    if len(args_dict) > 0:
        print_with_seperator("Using Custom Setting")
        json_print(args_dict)
        yaml_data.update(args_dict)
        print_with_seperator('Now Setting')
        json_print(yaml_data)
    return yaml_data


def print_with_seperator(str_to_print, seperator='*', multi=None):
    seperator_length = min(100, int(len(str_to_print) * 2)) if multi is None else multi

    print()
    print(f'{seperator}' * seperator_length)
    print(' ' * 2 + str_to_print)
    print(f'{seperator}' * seperator_length)
    print()
