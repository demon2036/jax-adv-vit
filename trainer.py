import collections
import os
import argparse
import math

import einops
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from base_trainer import BaseTrainer
from baseline.model import ViT_Classifier
from utils import setup_seed, get_obj_from_str, acc_fn, get_config
from accelerate import Accelerator


def get_model_finetune(model: str = None, pretrained_model_path: str = None, is_mae=True):
    assert model is not None
    print(model)
    mae_model = get_obj_from_str(model)()

    if is_mae:

        if pretrained_model_path is not None:
            data = torch.load(pretrained_model_path)
            # print(data)

            # for key in data.keys():
            #     print(key)
            # while True:
            #     pass

            # data= collections.OrderedDict([(k.replace('_orig_mod.',''), v)  for k, v in data.items()])
            # print(type(data))

            # while True:
            #     pass

            mae_model.load_state_dict(data)

        model = ViT_Classifier(mae_model.encoder, num_classes=10)
    else:
        model = mae_model
    return model


class MaeFinetuneTrainer(BaseTrainer):
    def __init__(self,
                 seed=2022,
                 batch_size=128,
                 max_device_batch_size=256,
                 base_learning_rate=1e-3,
                 weight_decay=0.05,
                 total_epoch=100,
                 warmup_epoch=5,
                 pretrained_model_path=None,
                 save_model_path=None,
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 model_instant_function=get_model_finetune,
                 model_target: str = None,
                 save_model_name: str = None,
                 mixed_precision='fp16',
                 save_every=500,
                 compile=False,
                 ):
        super().__init__(seed, batch_size, max_device_batch_size, total_epoch, mixed_precision, save_every=save_every)

        self.loss_fn = loss_fn
        self.model = model_instant_function(model_target, pretrained_model_path)
        self.optim = torch.optim.AdamW(self.model.parameters(),
                                       lr=base_learning_rate * batch_size / 256,
                                       betas=(0.9, 0.999), weight_decay=weight_decay
                                       )
        # configs = {
        #     'lr': base_learning_rate * batch_size / 256,
        #     'betas': (0.9, 0.999),
        #     weight_decay: weight_decay
        # }
        # self.optim = torch.optim.AdamW([
        #     {'params': self.model.head.parameters(), **configs},
        #     {'params': self.model.layer_norm.parameters(), **configs},
        #
        # ],
        # )

        # summary(self.model, (1, 3, 32, 32), )
        if compile:
            self.model = torch.compile(self.model, fullgraph=False, )  # mode='max-autotune'

        lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))

        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_func, verbose=True)

        self.model, \
            self.optim, \
            self.train_dataloader, \
            self.val_dataloader, \
            self.lr_scheduler = self.accelerator.prepare(self.model, self.optim, self.train_dataloader,
                                                         self.val_dataloader,
                                                         self.lr_scheduler)

        self.save_model_path = save_model_path
        self.save_model_name = save_model_name
        # self.pretrained_model_path = pretrained_model_path

    def train(self):
        best_val_acc = 0
        step_count = 0
        self.optim.zero_grad()

        train_dataloader_iter = iter(self.train_dataloader)

        for e in range(self.total_epoch):
            self.model.train()
            losses = []
            acces = []
            train_step = 50000 // 64
            with tqdm(total=train_step, desc=f'Train Epoch {e + 1}/{self.total_epoch}', postfix=dict,
                      mininterval=0.3) as pbar:
                for _ in range(train_step):
                    img, label = next(train_dataloader_iter)
                    img = einops.rearrange(img, 'b h w c->b c h w')

                    print(img)

                    img = img.float()/255

                    # print(img)

                    # print(img.shape)

                    with self.accelerator.autocast():
                        step_count += 1

                        # img = Normalize(CIFAR10_MEAN, CIFAR10_STD)(img)

                        # print(img,label)

                        logits = self.model(img)

                        # print(logits)

                        loss = self.loss_fn(logits, label)  # F.softmax(logits, -1)

                        # for transformer in self.model.transformer:
                        #     if not transformer.skip:
                        #         z_router_losses.append(transformer.entropy_loss)
                        #
                        # z_router_losses = torch.stack(z_router_losses, dim=0).mean()

                    acc = acc_fn(logits, label)
                    # loss.backward()
                    # accelerator.backward(loss+1e-4 * z_router_losses)  #  +1e-2 * z_router_losses
                    self.accelerator.backward(loss)

                    if step_count % self.steps_per_update == 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optim.step()
                        self.optim.zero_grad()
                    losses.append(loss.item())
                    acces.append(acc.item())

                    pbar.set_postfix(**{'Train Loss': np.mean(losses),
                                        'Tran accs': np.mean(acces),
                                        # 'z_router_losses': np.mean(z_router_losses.item())
                                        })
                    pbar.update(1)

            self.lr_scheduler.step()
            avg_train_loss = sum(losses) / len(losses)
            avg_train_acc = sum(acces) / len(acces)
            # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

            avg_val_acc = self.eval(e)

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                print(f'saving best model with acc {best_val_acc} at {e} epoch!')
                self.save()

    def eval(self, epoch):
        self.model.eval()
        # adversary = AutoAttack(self.model, norm='Linf', eps=8 / 255, version='custom', verbose=True,
        #                        attacks_to_run=['apgd-ce', 'apgd-dlr'])

        with torch.no_grad():
            losses = []
            acces = []
            val_step = len(self.val_dataloader)
            with tqdm(total=val_step, desc=f'Val Epoch {epoch + 1}/{self.total_epoch}', postfix=dict,
                      mininterval=0.3) as pbar2:
                for img, label in iter(self.val_dataloader):
                    # adv_img = adversary.run_standard_evaluation(img, label)
                    # logits, feats = self.model(img, return_feats=True)
                    # logits_adv, feats_adv = self.model(adv_img, return_feats=True)
                    #
                    # import torch.nn.functional as F
                    # for i, (feat, feta_adv) in enumerate(zip(feats, feats_adv)):
                    #     js_div = 0.5 * F.kl_div(torch.log_softmax(feta_adv, -1), torch.softmax(feat, -1))
                    #     js_div += 0.5 * F.kl_div(torch.log_softmax(feat, -1), torch.softmax(feta_adv, -1))
                    #
                    #     print(js_div)
                    #
                    # print(F.softmax(logits, -1))
                    # print(F.softmax(logits_adv, -1))
                    logits = self.model(img)

                    loss = self.loss_fn(logits, label)
                    acc = acc_fn(logits, label)
                    losses.append(loss.item())
                    acces.append(acc.item())

                    pbar2.set_postfix(**{'Val Loss': np.mean(losses),
                                         'Val accs': np.mean(acces)})
                    pbar2.update(1)
                    # break
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(
                f'In epoch {epoch}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')
        return avg_val_acc

    def save(self):
        assert self.save_model_path is not None and self.save_model_name is not None
        os.makedirs(self.save_model_path, exist_ok=True)
        torch.save(self.accelerator.get_state_dict(self.model), f'{self.save_model_path}/{self.save_model_name}.pt')
        # torch.save(self.model, args.output_model_path)

    def load(self, path='save_model_path/mae/baseline/baseline_tiny.pt'):
        self.model.load_state_dict(torch.load(path), strict=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, )  #default=2022
    parser.add_argument('--batch_size', type=int, )  #default=128
    parser.add_argument('--max_device_batch_size', type=int, )  # default=256
    parser.add_argument('--base_learning_rate', type=float, )  #default=1e-3
    parser.add_argument('--weight_decay', type=float, )  #default=0.05
    parser.add_argument('--total_epoch', type=int, )  #default=100
    parser.add_argument('--warmup_epoch', type=int, )  #default=5
    parser.add_argument('--pretrained_model_path', type=str, )
    parser.add_argument('--save_every', type=int, )
    parser.add_argument('--yaml_path', type=str,
                        default='configs/vit/baseline/tiny.yaml')  #'configs/vit/baseline/tiny.yaml'

    args = parser.parse_args()
    # print('Using Default Config From Yaml')
    # yaml_data = read_yaml(args.yaml_path)
    # yaml_data.update({'pretrained_model_path': 'mod_mae_custom.pt'})

    yaml_data = get_config(args)

    trainer = MaeFinetuneTrainer(**yaml_data, )
    # trainer.load('save_model_path/vit/baseline_aux/baseline_aux09_tiny.pt')
    trainer.train()

    # trainer.load('save_model_path/vit/baseline_aux/baseline_aux09_tiny.pt')
    # trainer.eval(0)
