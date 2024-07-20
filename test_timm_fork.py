import argparse
import functools
# import accelerate
import numpy as np
import torch
from timm.models import VisionTransformer
from torch.utils.data import DataLoader
import torchvision

from torchvision.transforms import Compose, ToTensor, Normalize

IMAGENET_DEFAULT_MEAN = np.array([0.4914, 0.4822, 0.4465])
IMAGENET_DEFAULT_STD = np.array([0.2471, 0.2435, 0.2616])


def train_and_evaluate(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(args)

    model = VisionTransformer(img_size=32, patch_size=2, num_classes=args.labels, init_values=1e-4, embed_dim=args.dim,
                              qk_norm=False,
                              num_heads=args.heads, depth=args.layers,
                              global_pool=args.class_token,
                              class_token=False if args.use_class_token is None else True,
                              fc_norm=args.fc_norm, act_layer=functools.partial(torch.nn.GELU, approximate='tanh'))

    print(type(model))
    state_dict = torch.load(args.checkpoint)
    print(state_dict['head.weight'].shape)
    print(model.load_state_dict(state_dict))

    # model.num_prefix_tokens = 0
    # model = torch.nn.Sequential(Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD), model)
    model = model.cuda()

    model.eval()
    # from accelerate import Accelerator
    # model = Accelerator(mixed_precision='bf16').prepare_model(model)

    from autoattack import AutoAttack
    # adversary = AutoAttack(model, norm='Linf', eps=8 / 255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    # adversary = AutoAttack(model, norm='Linf', eps=8 / 255, )
    if args.norm == 'l2':
        adversary = AutoAttack(model, norm='L2', eps=0.5, )
    elif args.norm == 'linf':
        adversary = AutoAttack(model, norm='Linf', eps=8 / 255, )
    else:
        raise NotImplemented()
    # adversary.apgd.n_restarts = 1

    if args.dataset == 'cifar10':

        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
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
    import foolbox as fb
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    for data in test_dataloader:
        x_test, y_test = data
        # x_test = x_test.cuda()
        # y_test = y_test.cuda()

        # _, advs, success = fb.attacks.LinfPGD( random_start=True)(fmodel, x_test.to('cuda:0'),
        #                                                                    y_test.to('cuda:0'), epsilons=[8 / 255])
        _, advs, success = fb.attacks.L2PGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[0.5])

        # _, advs, success = fb.attacks.LinfPGD(steps=10,rel_stepsize=2/255,random_start=True)(fmodel, x_test.to('cuda:0'),
        #                                                          y_test.to('cuda:0'), epsilons=[8 / 255])
        # print(success.shape)
        print(success[success == True].sum() / success.shape[1])
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--norm", type=str, default='linf')
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--labels", type=int, default=10)
    parser.add_argument("--qk-norm", type=bool, )
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--use-class-token', type=str, default=None)
    parser.add_argument("--class-token", type=str)
    parser.add_argument("--fc-norm", type=bool,default=False)

    train_and_evaluate(parser.parse_args())
