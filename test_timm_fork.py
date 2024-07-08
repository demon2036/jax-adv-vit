import argparse
import functools

# import accelerate
import numpy as np
import torch

from functools import partial

from timm.models import VisionTransformer
from torch.utils.data import DataLoader

import torchvision

from torchvision.transforms import Compose, ToTensor, Normalize

import os

EPOCHS = 1000  # @param{type:"integer"}
# @markdown Number of samples for each batch in the training set:
TRAIN_BATCH_SIZE = 1024  # @param{type:"integer"}
# @markdown Number of samples for each batch in the test set:
TEST_BATCH_SIZE = 64  # @param{type:"integer"}
# @markdown Learning rate for the optimizer:
LEARNING_RATE = 1e-3  # @param{type:"number"}
# @markdown The dataset to use.
DATASET = "cifar10-l2"  # @param{type:"string"}
# @markdown The amount of L2 regularization to use:
L2_REG = 0.0001  # @param{type:"number"}
# @markdown Adversarial perturbations lie within the infinity-ball of radius epsilon.
EPSILON = 8 / 255  # @param{type:"number"}

os.environ['WANDB_API_KEY'] = 'ec6aa52f09f51468ca407c0c00e136aaaa18a445'

IMAGENET_DEFAULT_MEAN = np.array([0.4914, 0.4822, 0.4465])
IMAGENET_DEFAULT_STD = np.array([0.2471, 0.2435, 0.2616])


def train_and_evaluate(args):
    print(args)

    model = VisionTransformer(img_size=32, patch_size=2, num_classes=args.labels, init_values=1e-4, embed_dim=args.dim,
                              qk_norm=False,
                              num_heads=args.heads, depth=args.layers,
                              global_pool=args.class_token,
                              class_token=False if args.use_class_token is None else True,
                              fc_norm=False, act_layer=functools.partial(torch.nn.GELU, approximate='tanh'))

    # model = timm.create_model("vit_tiny_patch16_224", init_values=1e-4)
    print(type(model))
    state_dict=torch.load(args.checkpoint)
    # state_dict['head.weight']=state_dict['head.weight'][:10]
    # state_dict['head.bias'] = state_dict['head.bias'][:10]
    print(model.load_state_dict(state_dict))
    # model.num_prefix_tokens = 0
    # model = torch.nn.Sequential(Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), model)
    model = model.cuda()
    import foolbox as fb
    model.eval()
    # from accelerate import Accelerator
    # model = Accelerator(mixed_precision='bf16').prepare_model(model)

    from autoattack import AutoAttack
    # adversary = AutoAttack(model, norm='Linf', eps=8 / 255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary = AutoAttack(model, norm='Linf', eps=8 / 255, )
    # adversary.apgd.n_restarts = 1

    if args.dataset == 'cifar10-l2':

        test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
                                                    transform=Compose(
                                                        [ToTensor()]))  # 0.5, 0.5
    elif args.dataset == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                     transform=Compose(
                                                         [ToTensor()]))  # 0.5, 0.5
    else:
        raise NotImplemented()

    test_dataloader = DataLoader(test_dataset, 128, shuffle=False, num_workers=4, drop_last=False)
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)

    # x_test = x_test.cuda()
    # y_test = y_test.cuda()

    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    
    
    """
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    for data in test_dataloader:
        x_test, y_test = data
        # x_test = x_test.cuda()
        # y_test = y_test.cuda()

        _, advs, success = fb.attacks.LinfPGD( random_start=True)(fmodel, x_test.to('cuda:0'),
                                                                           y_test.to('cuda:0'), epsilons=[8 / 255])

        # _, advs, success = fb.attacks.LinfPGD(steps=10,rel_stepsize=2/255,random_start=True)(fmodel, x_test.to('cuda:0'),
        #                                                          y_test.to('cuda:0'), epsilons=[8 / 255])
        # print(success.shape)
        print(success[success == True].sum() / success.shape[1])
     """



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--labels", type=int, default=10)
    parser.add_argument("--qk-norm", type=bool, )
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--use-class-token', type=str, default=None)
    parser.add_argument("--class-token", type=str)

    train_and_evaluate(parser.parse_args())
