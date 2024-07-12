import jax

jax.distributed.initialize()

import argparse
from typing import Any
from flax.serialization import msgpack_serialize
from attacks.pgd import pgd_attack_l2
from functools import partial
from torch.utils.data import DataLoader
import einops
import flax.jax_utils
import torchvision
import tqdm
from flax.training.common_utils import shard, shard_prng_key
from flax import linen as nn
from flax.training import train_state

import jax.numpy as jnp
import numpy as np
import optax
from optax.losses import softmax_cross_entropy_with_integer_labels

from datasets_fork import get_train_dataloader
from model import ViT
import os
import wandb
from utils2 import AverageMeter, save_checkpoint_in_background, load_pretrained_params

EPSILON = 128 / 255  # @param{type:"number"}

os.environ['WANDB_API_KEY'] = 'ec6aa52f09f51468ca407c0c00e136aaaa18a445'


def trade(image, label, state, epsilon=128 / 255, maxiter=10, key=None):
    delta = 0.001 * jax.random.normal(key, image.shape)
    optimizer = optax.sgd(epsilon / maxiter * 2)
    opt_state = optimizer.init(delta)
    p_natural = state.apply_fn({'params': state.params}, image)
    p_natural = jax.lax.stop_gradient(p_natural)

    def jax_re_norm(delta, max_norm):
        b, h, w, c = delta.shape
        norms = jnp.linalg.norm(delta.reshape(b, -1), ord=2, axis=1, keepdims=True).reshape(b, 1, 1, 1)
        desired = jnp.clip(norms, a_min=None, a_max=max_norm)
        scale = desired / (1e-6 + norms)
        return delta * scale

    def grad_fn(delta, x):
        adv = x + delta
        model_out = state.apply_fn({'params': state.params}, adv)
        loss_value = -1 * optax.losses.kl_divergence(nn.log_softmax(model_out), p_natural)
        # loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(model_out, label))
        return loss_value.mean()

    for _ in range(maxiter):
        grad = jax.grad(grad_fn)(delta, image)
        grad_norm = jnp.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1)
        grad = grad / grad_norm.reshape(-1, 1, 1, 1)
        updates, opt_state = optimizer.update(grad, opt_state, delta)
        delta = optax.apply_updates(delta, updates)
        delta = delta + image
        delta = jnp.clip(delta, 0, 1) - image
        delta = jax_re_norm(delta, max_norm=epsilon)

    return jnp.clip(image + delta, 0, 1)


@partial(jax.pmap, axis_name="batch")
def apply_model_trade(state, data, key):
    images, labels = data

    images = einops.rearrange(images, 'b c h w->b h w c')

    images = images.astype(jnp.float32) / 255
    labels = labels.astype(jnp.float32)

    print(images.shape)

    """Computes gradients, loss and accuracy for a single batch."""
    adv_image = trade(images, labels, state, key=key, epsilon=EPSILON, )

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        logits_adv = state.apply_fn({'params': params}, adv_image)
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        one_hot = optax.smooth_labels(one_hot, state.label_smoothing)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        trade_loss = optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()
        metrics = {'loss': loss, 'trade_loss': trade_loss, 'logits': logits, 'logits_adv': logits_adv}

        return loss + state.trade_beta * trade_loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    accuracy_std = jnp.mean(jnp.argmax(metrics['logits'], -1) == labels)
    accuracy_adv = jnp.mean(jnp.argmax(metrics['logits_adv'], -1) == labels)

    metrics['accuracy'] = accuracy_std
    metrics['adversarial accuracy'] = accuracy_adv

    metrics = jax.lax.pmean(metrics, axis_name="batch")

    grads = jax.lax.pmean(grads, axis_name="batch")

    state = state.apply_gradients(grads=grads)

    new_ema_params = jax.tree_util.tree_map(
        lambda ema, normal: ema * state.ema_decay + (1 - state.ema_decay) * normal,
        state.ema_params, state.params)
    state = state.replace(ema_params=new_ema_params)

    return state, metrics | state.opt_state.hyperparams


class EMATrainState(flax.training.train_state.TrainState):
    label_smoothing: int
    trade_beta: int
    ema_decay: int = 0.995
    ema_params: Any = None


def create_train_state(rng,
                       layers=12,
                       dim=192,
                       heads=3,
                       labels=10,
                       layerscale=True,
                       patch_size=2,
                       image_size=32,
                       posemb="learnable",
                       pooling='cls',
                       dropout=0.0,
                       droppath=0.0,
                       clip_grad=1.0,
                       warmup_steps=None,
                       training_steps=None,
                       learning_rate=None,
                       weight_decay=None,
                       ema_decay=0.9999,
                       trade_beta=5.0,
                       label_smoothing=0.1,
                       use_fc_norm: bool = True,
                       reduce_include_prefix: bool = False,
                       b1=0.95,
                       b2=0.98, pretrained_ckpt=None

                       ):
    """Creates initial `TrainState`."""

    cnn = ViT(
        layers=layers,
        dim=dim,
        heads=heads,
        labels=labels,
        layerscale=layerscale,
        patch_size=patch_size,
        image_size=image_size,
        posemb=posemb,
        pooling=pooling,
        dropout=dropout,
        droppath=droppath,
        use_fc_norm=use_fc_norm,
        reduce_include_prefix=reduce_include_prefix
    )

    # cnn = CNN()

    # image_shape = [1, 28, 28, 1]
    image_shape = [1, 32, 32, 3]

    params = cnn.init(rng, jnp.ones(image_shape))['params']

    if pretrained_ckpt is not None:
        params = load_pretrained_params(pretrained_ckpt, params)

    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = optax.lion(
            learning_rate=learning_rate,
            b1=b1, b2=b2,
            # eps=args.adam_eps,
            weight_decay=weight_decay,
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        # if args.lr_decay < 1.0:
        #     layerwise_scales = {
        #         i: optax.scale(args.lr_decay ** (args.layers - i))
        #         for i in range(args.layers + 1)
        #     }
        #     label_fn = partial(get_layer_index_fn, num_layers=args.layers)
        #     label_fn = partial(tree_map_with_path, label_fn)
        #     tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=training_steps,
        end_value=1e-6,
    )

    # learning_rate = optax.warmup_cosine_decay_schedule(
    #     init_value=1e-7,
    #     peak_value=LEARNING_RATE,
    #     warmup_steps=50000 * 5 // TRAIN_BATCH_SIZE,
    #     decay_steps=50000 * EPOCHS // TRAIN_BATCH_SIZE,
    #     end_value=1e-6,
    # )

    tx = create_optimizer_fn(learning_rate)

    return EMATrainState.create(apply_fn=cnn.apply, params=params, tx=tx, ema_params=params, ema_decay=ema_decay,
                                trade_beta=trade_beta, label_smoothing=label_smoothing)


@partial(jax.pmap, axis_name="batch", )
def accuracy(state, data):
    # inputs, labels = data

    inputs, labels = data
    inputs = inputs.astype(jnp.float32)
    labels = labels.astype(jnp.int64)

    inputs = einops.rearrange(inputs, 'b c h w->b h w c')

    logits = state.apply_fn({"params": state.ema_params}, inputs)
    clean_accuracy = jnp.argmax(logits, axis=-1) == labels

    maxiter = 20
    adversarial_images = pgd_attack_l2(inputs, labels, state, epsilon=EPSILON, maxiter=maxiter,
                                       key=jax.random.PRNGKey(0))
    logits_adv = state.apply_fn({"params": state.ema_params}, adversarial_images)
    adversarial_accuracy = jnp.argmax(logits_adv, axis=-1) == labels
    metrics = {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy, "num_samples": labels != -1}
    metrics = jax.tree_util.tree_map(lambda x: (x * (labels != -1)).sum(), metrics)
    metrics = jax.lax.psum(metrics, axis_name='batch')
    return metrics


def train_and_evaluate(args
                       ) -> train_state.TrainState:
    """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """

    if jax.process_index() == 0:
        wandb.init(name=args.name, project=args.project, config=args.__dict__,
                   settings=wandb.Settings(_disable_stats=True),
                   config_exclude_keys=['train_dataset_shards', 'valid_dataset_shards', 'train_origin_dataset_shards'])
        average_meter = AverageMeter(use_latest=["learning_rate"])

    rng = jax.random.key(0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng,
                               layers=args.layers,
                               dim=args.dim,
                               heads=args.heads,
                               labels=args.labels,
                               layerscale=args.layerscale,
                               patch_size=args.patch_size,
                               image_size=args.image_size,
                               posemb=args.posemb,
                               pooling=args.pooling,
                               dropout=args.dropout,
                               droppath=args.droppath,
                               warmup_steps=args.warmup_steps,
                               training_steps=args.training_steps,
                               learning_rate=args.learning_rate,
                               weight_decay=args.weight_decay,
                               ema_decay=args.ema_decay,
                               trade_beta=args.beta,
                               label_smoothing=args.label_smoothing,
                               use_fc_norm=args.use_fc_norm,
                               reduce_include_prefix=args.reduce_include_prefix,
                               b1=args.adam_b1,
                               b2=args.adam_b2,
                               clip_grad=0.0,
                               pretrained_ckpt=args.pretrained_ckpt
                               )

    if 'step' in state.params:
        init_step = state.params['step']
    else:
        init_step = 1
        raise NotImplementedError()

    state = flax.jax_utils.replicate(state)

    train_dataloader_iter, test_dataloader = get_train_dataloader(args.train_batch_size,
                                                                  shard_path=args.train_dataset_shards,
                                                                  test_shard_path=args.valid_dataset_shards,
                                                                  origin_shard_path=args.train_origin_dataset_shards)

    def prepare_tf_data(xs):
        """Convert a input batch from tf Tensors to numpy arrays."""
        local_device_count = jax.local_device_count()

        def _prepare(x):
            # Use _numpy() for zero-copy conversion between TF and NumPy.
            # x = {'img': x['img'], 'cls': x['cls']}
            x = np.asarray(x)
            # x = x._numpy()  # pylint: disable=protected-access

            # reshape (host_batch_size, height, width, 3) to
            # (local_devices, device_batch_size, height, width, 3)
            return x.reshape((local_device_count, -1) + x.shape[1:])

        return jax.tree_util.tree_map(_prepare, xs)

    train_dataloader_iter = map(prepare_tf_data, train_dataloader_iter)

    train_dataloader_iter = flax.jax_utils.prefetch_to_device(train_dataloader_iter, 2)

    for step in tqdm.tqdm(range(init_step, args.training_steps)):
        rng, input_rng = jax.random.split(rng)
        data = next(train_dataloader_iter)

        rng, train_step_key = jax.random.split(rng, num=2)
        train_step_key = shard_prng_key(train_step_key)

        state, metrics = apply_model_trade(state, data, train_step_key)

        if jax.process_index() == 0 and step % args.log_interval == 0:
            average_meter.update(**flax.jax_utils.unreplicate(metrics))
            metrics = average_meter.summary('train/')
            # print(metrics)
            wandb.log(metrics, step)

        if step % args.eval_interval == 0:
            for data in tqdm.tqdm(test_dataloader, leave=False, dynamic_ncols=True):
                data = shard(jax.tree_util.tree_map(np.asarray, data))
                metrics = accuracy(state, data)

                if jax.process_index() == 0:
                    average_meter.update(**jax.device_get(flax.jax_utils.unreplicate(metrics)))
            if jax.process_index() == 0:
                metrics = average_meter.summary("val/")
                num_samples = metrics.pop("val/num_samples")
                metrics = jax.tree_util.tree_map(lambda x: x / num_samples, metrics)
                wandb.log(metrics, step)

                # params = flax.jax_utils.unreplicate(state.params)
                # params_bytes = msgpack_serialize(params)
                # save_checkpoint_in_background(params_bytes=params_bytes, postfix="last", name=args.name,
                #                               output_dir=os.getenv('GCS_DATASET_DIR'))

                params = flax.jax_utils.unreplicate(state.ema_params)
                params_bytes = msgpack_serialize(params | {'step': step})
                save_checkpoint_in_background(params_bytes=params_bytes, postfix="ema", name=args.name,
                                              output_dir=args.output_dir)

    return state


if __name__ == "__main__":
    # _, train_dataloader, test_dataloader = get_train_dataloader(TRAIN_BATCH_SIZE)
    # train_dataloader_iter = iter(train_dataloader)
    #
    # for _ in range(100):
    #     data=next(train_dataloader_iter)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-shards")
    parser.add_argument("--train-origin-dataset-shards")
    parser.add_argument("--valid-dataset-shards")
    parser.add_argument("--train-batch-size", type=int, default=2048)
    # parser.add_argument("--valid-batch-size", type=int, default=256)
    # parser.add_argument("--train-loader-workers", type=int, default=40)
    # parser.add_argument("--valid-loader-workers", type=int, default=5)

    # parser.add_argument("--random-crop", default="rrc")
    # parser.add_argument("--color-jitter", type=float, default=0.0)
    # parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    # parser.add_argument("--random-erasing", type=float, default=0.25)
    # parser.add_argument("--augment-repeats", type=int, default=3)
    # parser.add_argument("--test-crop-ratio", type=float, default=0.875)

    # parser.add_argument("--mixup", type=float, default=0.8)
    # parser.add_argument("--cutmix", type=float, default=1.0)
    # parser.add_argument("--criterion", default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=5)

    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--labels", type=int, default=10)
    parser.add_argument("--layerscale", action="store_true", default=False)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--posemb", default="learnable")
    parser.add_argument("--pooling", default="cls")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--droppath", type=float, default=0.1)
    parser.add_argument("--grad-ckpt", action="store_true", default=False)
    parser.add_argument("--use-fc-norm", action="store_true", default=False)
    parser.add_argument("--reduce_include_prefix", action="store_true", default=False)

    # parser.add_argument("--init-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--mixup-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--dropout-seed", type=int, default=random.randint(0, 1000000))
    # parser.add_argument("--shuffle-seed", type=int, default=random.randint(0, 1000000))
    parser.add_argument("--pretrained-ckpt")
    # parser.add_argument("--label-mapping")
    #
    # parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adam-b1", type=float, default=0.95)
    parser.add_argument("--adam-b2", type=float, default=0.98)
    # parser.add_argument("--adam-eps", type=float, default=1e-8)
    # parser.add_argument("--lr-decay", type=float, default=1.0)
    # parser.add_argument("--clip-grad", type=float, default=0.0)
    # parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    #
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--training-steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    #
    parser.add_argument("--project")
    parser.add_argument("--name")
    # parser.add_argument("--ipaddr")
    # parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default=".")
    train_and_evaluate(parser.parse_args())
