import jax
from torch.utils.data import DataLoader

jax.distributed.initialize()

from functools import partial

import einops
import flax.jax_utils
import torchvision
import tqdm
from flax.training.common_utils import shard, shard_prng_key
# See issue #620.
# pytype: disable=wrong-keyword-args

from absl import logging
from flax import linen as nn
# from flax.metrics import tensorboard
from flax.training import train_state

import jax.numpy as jnp
import numpy as np
import optax
from optax.losses import softmax_cross_entropy_with_integer_labels
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor

# from auto_augment import AutoAugment, Cutout
from main import get_train_dataloader
from model import ViT
import os
import wandb
from utils2 import AverageMeter

EPOCHS = 1000  # @param{type:"integer"}
# @markdown Number of samples for each batch in the training set:
TRAIN_BATCH_SIZE = 1024  # @param{type:"integer"}
# @markdown Number of samples for each batch in the test set:
TEST_BATCH_SIZE = 64  # @param{type:"integer"}
# @markdown Learning rate for the optimizer:
LEARNING_RATE = 0.001  # @param{type:"number"}
# @markdown The dataset to use.
DATASET = "cifar10"  # @param{type:"string"}
# @markdown The amount of L2 regularization to use:
L2_REG = 0.0001  # @param{type:"number"}
# @markdown Adversarial perturbations lie within the infinity-ball of radius epsilon.
EPSILON = 8 / 255  # @param{type:"number"}

os.environ['WANDB_API_KEY'] = 'ec6aa52f09f51468ca407c0c00e136aaaa18a445'


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


# @jax.jit
# def loss_fun(params, l2reg, data):
#     """Compute the loss of the network."""
#     inputs, labels = data
#     x = inputs.astype(jnp.float32)
#     logits = net.apply({"params": params}, x)
#     sqnorm = tree_l2_norm(params, squared=True)
#     loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
#     return loss_value + 0.5 * l2reg * sqnorm


def pgd_attack3(image, label, state, epsilon=8 / 255, step_size=2 / 255, maxiter=10):
    """PGD attack on the L-infinity ball with radius epsilon.

  Args:
    image: array-like, input data for the CNN
    label: integer, class label corresponding to image
    params: tree, parameters of the model to attack
    epsilon: float, radius of the L-infinity ball.
    maxiter: int, number of iterations of this algorithm.

  Returns:
    perturbed_image: Adversarial image on the boundary of the L-infinity ball
      of radius epsilon and centered at image.

  Notes:
    PGD attack is described in (Madry et al. 2017),
    https://arxiv.org/pdf/1706.06083.pdf
    :param step_size:
  """
    image_perturbation = jnp.zeros_like(image)

    def adversarial_loss(perturbation):
        logits = state.apply_fn({"params": state.params}, image + perturbation)
        # sqnorm = tree_l2_norm(params, squared=True)
        loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, label))
        # return loss_value + 0.5 * l2reg * sqnorm
        return loss_value

        # return loss_fun(params, 0, (image + perturbation, label))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(grad_adversarial(image_perturbation))

        # heuristic step-size 2 eps / maxiter
        image_perturbation += step_size * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jnp.clip(image + image_perturbation, 0, 1)


@partial(jax.pmap, axis_name="batch", )
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    adv_image = pgd_attack3(images, labels, state, )

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, adv_image)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)

    return new_state, grads, loss, accuracy


def loss_fun_trade(state, data):
    """Compute the loss of the network."""
    image, inputs, labels = data
    x_adv = inputs.astype(jnp.float32)

    x = image.astype(jnp.float32)

    logits = state.apply_fn({"params": state.params}, x)
    logits_adv = state.apply_fn({"params": state.params}, x_adv)

    return optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()


def trade(image, label, state, epsilon=0.1, maxiter=10, step_size=0.007, key=None):
    """PGD attack on the L-infinity ball with radius epsilon.

  Args:
    image: array-like, input data for the CNN
    label: integer, class label corresponding to image
    params: tree, parameters of the model to attack
    epsilon: float, radius of the L-infinity ball.
    maxiter: int, number of iterations of this algorithm.

  Returns:
    perturbed_image: Adversarial image on the boundary of the L-infinity ball
      of radius epsilon and centered at image.

  Notes:
    PGD attack is described in (Madry et al. 2017),
    https://arxiv.org/pdf/1706.06083.pdf

    # image_perturbation = jnp.zeros_like(image)
    image_perturbation = 0.001 * jax.random.normal(key, shape=image.shape)

    def adversarial_loss(perturbation):
        return loss_fun_trade(params, (image, image + perturbation, label))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(grad_adversarial(image_perturbation))

        # heuristic step-size 2 eps / maxiter
        # image_perturbation += (2 * epsilon / maxiter) * sign_grad

        image_perturbation += step_size * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jnp.clip(image + image_perturbation, 0, 1)

     """

    x_adv = 0.001 * jax.random.normal(key, shape=image.shape) + image

    def adversarial_loss(adv_image, image):
        return loss_fun_trade(state, (image, adv_image, label))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image

        sign_grad = jnp.sign(jax.lax.stop_gradient(grad_adversarial(x_adv, image)))
        # heuristic step-size 2 eps / maxiter
        # image_perturbation += step_size * sign_grad

        # delta = jnp.clip(image_perturbation - image, min=-epsilon, max=epsilon)

        x_adv = jax.lax.stop_gradient(x_adv) + step_size * sign_grad
        r1 = jnp.where(x_adv > image - epsilon, x_adv, image - epsilon)
        x_adv = jnp.where(r1 < image + epsilon, r1, image + epsilon)

        x_adv = jnp.clip(x_adv, min=0, max=1)

        # projection step onto the L-infinity ball centered at image
        # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return x_adv


# @jax.jit
# def loss_fun_trade_train(params, data):
#     """Compute the loss of the network."""
#     image, inputs, labels = data
#     x_adv = inputs.astype(jnp.float32)
#
#     x = image.astype(jnp.float32)
#
#     logits = net.apply({"params": params}, x)
#     logits_adv = net.apply({"params": params}, x_adv)
#
#     loss_natural = softmax_cross_entropy_with_integer_labels(logits, labels)
#
#     return (loss_natural + 5 * optax.kl_divergence(nn.log_softmax(logits_adv, axis=1),
#                                                    nn.softmax(logits, axis=1))).mean()


@partial(jax.pmap, axis_name="batch", )
def apply_model_trade(state, images, labels, key):
    """Computes gradients, loss and accuracy for a single batch."""
    adv_image = trade(images, labels, state, key=key)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        logits_adv = state.apply_fn({'params': params}, adv_image)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
        metrics = {'loss': loss, 'trade_loss': trade_loss, 'logits': logits, 'logits_adv': logits_adv}

        return loss + 5 * trade_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    accuracy_std = jnp.mean(jnp.argmax(metrics['logits'], -1) == labels)
    accuracy_adv = jnp.mean(jnp.argmax(metrics['logits_adv'], -1) == labels)

    metrics['accuracy'] = accuracy_std
    metrics['adversarial accuracy'] = accuracy_adv

    metrics = jax.lax.pmean(metrics, axis_name="batch")

    grads = jax.lax.pmean(grads, axis_name="batch")

    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics  #, grads, loss, metrics  #| state.opt_state.hyperparams


# @jax.jit
# def update_model(state, grads):
#     return state.apply_gradients(grads=grads)


def train_epoch(state, train_dataloader, rng, step=0):
    """Train for a single epoch."""
    average_meter = AverageMeter(use_latest=["learning_rate"])
    for data in tqdm.tqdm(train_dataloader):
        step += 1
        data = jax.tree_map(np.asarray, data)
        batch_images, batch_labels = data

        batch_images = einops.rearrange(batch_images, 'b c h w->b h w c')

        rng, train_step_key = jax.random.split(rng, num=2)
        train_step_key = shard_prng_key(train_step_key)

        batch_images = shard(batch_images)
        batch_labels = shard(batch_labels)

        state, grads, loss, metrics = apply_model_trade(state, batch_images, batch_labels, train_step_key)
        # state = update_model(state, grads)

        average_meter.update(**metrics)

        metrics = average_meter.summary('train/')

        wandb.log(metrics, step)

    return state, step


def create_train_state(rng):
    """Creates initial `TrainState`."""

    factor = 2
    cnn = ViT(
        layers=12,
        dim=192 * factor ** 2,
        heads=3 * factor ** 2,
        labels=10,
        layerscale=True,
        patch_size=2 * factor,
        image_size=32,
        posemb='learnable',
        pooling='cls',
        dropout=0.0,
        droppath=0.0,
        # grad_ckpt=args.grad_ckpt,
        use_kan=False,
        # polynomial_degree=args.polynomial_degree,
    )

    cnn = CNN()

    image_shape = [1, 28, 28, 1]
    image_shape = [1, 32, 32, 3]

    params = cnn.init(rng, jnp.ones(image_shape))['params']

    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            learning_rate=learning_rate,

            # eps=args.adam_eps,
            weight_decay=0.05,
            # mask=partial(tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        # if args.lr_decay < 1.0:
        #     layerwise_scales = {
        #         i: optax.scale(args.lr_decay ** (args.layers - i))
        #         for i in range(args.layers + 1)
        #     }
        #     label_fn = partial(get_layer_index_fn, num_layers=args.layers)
        #     label_fn = partial(tree_map_with_path, label_fn)
        #     tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        # if args.clip_grad > 0:
        #     tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=1e-3,
        warmup_steps=50000 * 5 // TRAIN_BATCH_SIZE,
        decay_steps=50000 * EPOCHS // TRAIN_BATCH_SIZE,
        end_value=1e-5,
    )

    tx = create_optimizer_fn(learning_rate)

    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


@partial(jax.pmap, axis_name="batch", )
def accuracy(state, data):
    inputs, labels = data
    logits = state.apply_fn({"params": state.params}, inputs)
    clean_accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)

    adversarial_images = pgd_attack3(inputs, labels, state, )
    logits_adv = state.apply_fn({"params": state.params}, adversarial_images)
    adversarial_accuracy = jnp.mean(jnp.argmax(logits_adv, axis=-1) == labels)

    metrics = {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy}
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics


def dataset_stats(state, data_loader, iter_per_epoch, ):
    """Computes accuracy on clean and adversarial images."""
    adversarial_accuracy = 0.
    clean_accuracy = 0.

    pmap_pgd = jax.pmap(pgd_attack3)

    for batch in data_loader.as_numpy_iterator():
        images, labels = batch
        images = images.astype(jnp.float32) / 255

        images = shard(images)
        labels = shard(labels)

        clean_accuracy += jnp.mean(accuracy(state, (images, labels))) / iter_per_epoch
        # adversarial_images = pgd_attack(images, labels, params, epsilon=EPSILON)
        adversarial_images = pmap_pgd(images, labels, state, )

        adversarial_accuracy += jnp.mean(accuracy(state, (adversarial_images, labels))) / iter_per_epoch
    return {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy}


def eval(test_dataloader, state, ):
    average_meter = AverageMeter(use_latest=["learning_rate"])
    for data in test_dataloader:
        data = jax.tree_util.tree_map(np.asarray, data)
        images, labels = data
        images = images.astype(jnp.float32)
        labels = labels.astype(jnp.int64)
        images = einops.rearrange(images, 'b c h w->b h w c')
        images = shard(images)
        labels = shard(labels)
        metrics = accuracy(state, (images, labels))

        average_meter.update(**metrics)
    if jax.process_index() == 0:
        metrics = average_meter.summary('val/')
        wandb.log(metrics, 1)


def train_and_evaluate(
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """

    if jax.process_index() == 0:
        wandb.init(name='vit-b4', project='cifar10-20m')
        average_meter = AverageMeter(use_latest=["learning_rate"])

    transform_train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                       # AutoAugment(), Cutout(),
                       ToTensor()]

    transform_test = [ToTensor()]

    # train_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=True, download=True,
    #                                              transform=Compose(
    #                                                  transform_train))  # 0.5, 0.5

    # train_dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True)

    _, train_dataloader, test_dataloader = get_train_dataloader(TRAIN_BATCH_SIZE)

    rng = jax.random.key(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, )

    state = flax.jax_utils.replicate(state)

    train_dataloader_iter = iter(train_dataloader)

    test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
                                                transform=Compose(
                                                    transform_test))  # 0.5, 0.5

    # test_dataloader = DataLoader(test_dataset, TRAIN_BATCH_SIZE, shuffle=False, num_workers=16, drop_last=False)

    log_interval = 100

    for step in tqdm.tqdm(range(1, 50000 * EPOCHS // TRAIN_BATCH_SIZE)):
        rng, input_rng = jax.random.split(rng)
        # state, step = train_epoch(
        #     state, train_dataloader, input_rng, step
        # )
        # for data in tqdm.tqdm(train_dataloader):
        """
        data = next(train_dataloader_iter)

        data = jax.tree_util.tree_map(np.asarray, data)
        batch_images, batch_labels = data
        batch_images = batch_images.astype(jnp.float32) / 255
        batch_labels = batch_labels.astype(jnp.float32)
        # batch_images = einops.rearrange(batch_images, 'b c h w->b h w c')

        rng, train_step_key = jax.random.split(rng, num=2)
        train_step_key = shard_prng_key(train_step_key)

        batch_images = shard(batch_images)
        batch_labels = shard(batch_labels)

        state, metrics = apply_model_trade(state, batch_images, batch_labels, train_step_key)
       
        # state = update_model(state, grads)
        if jax.process_index() == 0:
            average_meter.update(**metrics)
            metrics = average_meter.summary('train/')
            wandb.log(metrics, step)
        """
        if step % log_interval == 0:
            # eval(test_dataloader, state)
            for data in test_dataloader:
                data = jax.tree_util.tree_map(np.asarray, data)
                images, labels = data
                images = images.astype(jnp.float32)
                labels = labels.astype(jnp.int64)

                # print(images)
                # while True:
                #     pass

                images = einops.rearrange(images, 'b c h w->b h w c')
                images = shard(images)
                labels = shard(labels)
                metrics = accuracy(state, (images, labels))

                if jax.process_index() == 0:
                    average_meter.update(**metrics)
            if jax.process_index() == 0:
                metrics = average_meter.summary('val/')
                print(metrics)
                wandb.log(metrics, step)

    return state


if __name__ == "__main__":
    # _, train_dataloader, test_dataloader = get_train_dataloader(TRAIN_BATCH_SIZE)
    # train_dataloader_iter = iter(train_dataloader)
    #
    # for _ in range(100):
    #     data=next(train_dataloader_iter)

    train_and_evaluate()
