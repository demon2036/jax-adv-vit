import wandb
import os

from utils_log import AverageMeter

os.environ['WANDB_API_KEY'] = 'ec6aa52f09f51468ca407c0c00e136aaaa18a445'

wandb.init(name='vit-t2', project='cifar10-20m')

prefix = 'train/'

average_meter = AverageMeter(use_latest=["learning_rate"])

metrics = {'loss1': 1, 'loss2': 10}

average_meter.update(**metrics)

metrics = average_meter.summary('train/')

wandb.log(metrics, 1)
