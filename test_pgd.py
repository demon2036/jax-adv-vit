import datetime

import jax
import tqdm
from jax import numpy as jnp
from flax import linen as nn

import optax
from optax.losses import softmax_cross_entropy_with_integer_labels
from optax.tree_utils import tree_l2_norm

from matplotlib import pyplot as plt

from model import ViT

plt.rcParams.update({"font.size": 22})

import tensorflow as tf
import tensorflow_datasets as tfds

# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")

# Show on which platform JAX is running.
print("JAX running on", jax.devices()[0].platform.upper())

# @markdown Total number of epochs to train for:
EPOCHS = 100  # @param{type:"integer"}
# @markdown Number of samples for each batch in the training set:
TRAIN_BATCH_SIZE = 16  # @param{type:"integer"}
# @markdown Number of samples for each batch in the test set:
TEST_BATCH_SIZE = 64  # @param{type:"integer"}
# @markdown Learning rate for the optimizer:
LEARNING_RATE = 0.001  # @param{type:"number"}
# @markdown The dataset to use.
DATASET = "mnist"  # @param{type:"string"}
# @markdown The amount of L2 regularization to use:
L2_REG = 0.0001  # @param{type:"number"}
# @markdown Adversarial perturbations lie within the infinity-ball of radius epsilon.
EPSILON = 8 / 255  # @param{type:"number"}


class CNN(nn.Module):
    """A simple CNN model."""
    num_classes: int

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
        x = nn.Dense(features=self.num_classes)(x)
        return x


(train_loader, test_loader), mnist_info = tfds.load(
    "cifar10", split=["train", "test"], as_supervised=True, with_info=True
)

train_loader_batched = train_loader.shuffle(
    10 * TRAIN_BATCH_SIZE, seed=0
).batch(TRAIN_BATCH_SIZE, drop_remainder=True)
test_loader_batched = test_loader.batch(TEST_BATCH_SIZE, drop_remainder=True)

input_shape = (1,) + mnist_info.features["image"].shape
num_classes = mnist_info.features["label"].num_classes
iter_per_epoch_train = (
        mnist_info.splits["train"].num_examples // TRAIN_BATCH_SIZE
)
iter_per_epoch_test = mnist_info.splits["test"].num_examples // TEST_BATCH_SIZE

# net = CNN(num_classes)
net = ViT(
    layers=12,
    dim=192,
    heads=3,
    labels=10,
    layerscale=True,
    patch_size=2,
    image_size=32,
    posemb='learnable',
    pooling='cls',
    dropout=0.0,
    droppath=0.0,
    # grad_ckpt=args.grad_ckpt,
    use_kan=False,
    # polynomial_degree=args.polynomial_degree,
)


@jax.jit
def accuracy(params, data):
    inputs, labels = data
    logits = net.apply({"params": params}, inputs)
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


@jax.jit
def loss_fun(params, l2reg, data):
    """Compute the loss of the network."""
    inputs, labels = data
    x = inputs.astype(jnp.float32)
    logits = net.apply({"params": params}, x)
    sqnorm = tree_l2_norm(params, squared=True)
    loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, labels))
    return loss_value + 0.5 * l2reg * sqnorm


@jax.jit
def pgd_attack(image, label, params, epsilon=0.1, maxiter=10):
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
  """
    image_perturbation = jnp.zeros_like(image)

    def adversarial_loss(perturbation):
        return loss_fun(params, 0, (image + perturbation, label))

    grad_adversarial = jax.grad(adversarial_loss)
    for _ in range(maxiter):
        # compute gradient of the loss wrt to the image
        sign_grad = jnp.sign(grad_adversarial(image_perturbation))

        # heuristic step-size 2 eps / maxiter
        image_perturbation += (2 * epsilon / maxiter) * sign_grad
        # projection step onto the L-infinity ball centered at image
        image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)

    # clip the image to ensure pixels are between 0 and 1
    return jnp.clip(image + image_perturbation, 0, 1)


@jax.jit
def loss_fun_trade(params, data):
    """Compute the loss of the network."""
    image, inputs, labels = data
    x_adv = inputs.astype(jnp.float32)

    x = image.astype(jnp.float32)

    logits = net.apply({"params": params}, x)
    logits_adv = net.apply({"params": params}, x_adv)

    return optax.kl_divergence(nn.log_softmax(logits_adv, axis=1), nn.softmax(logits, axis=1)).mean()


@jax.jit
def trade(image, label, params, epsilon=0.1, maxiter=10, step_size=0.007, key=None):
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
        return loss_fun_trade(params, (image, adv_image, label))

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


@jax.jit
def loss_fun_trade_train(params, data):
    """Compute the loss of the network."""
    image, inputs, labels = data
    x_adv = inputs.astype(jnp.float32)

    x = image.astype(jnp.float32)

    logits = net.apply({"params": params}, x)
    logits_adv = net.apply({"params": params}, x_adv)

    loss_natural = softmax_cross_entropy_with_integer_labels(logits, labels)

    return (loss_natural + 5 * optax.kl_divergence(nn.log_softmax(logits_adv, axis=1),
                                                   nn.softmax(logits, axis=1))).mean()


@jax.jit
def train_step_trade(params, opt_state, batch, key):
    images, labels = batch
    # convert images to float as attack requires to take gradients wrt to them
    images = images.astype(jnp.float32) / 255
    adversarial_images_train = trade(images, labels, params, epsilon=EPSILON, key=key)
    # train on adversarial images
    loss_grad_fun = jax.grad(loss_fun_trade_train)
    grads = loss_grad_fun(params, (images, adversarial_images_train, labels))
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# @jax.jit
# def pgd_attack3(image, label, params, epsilon=0.1, step_size=2 / 255, maxiter=10):
#     """PGD attack on the L-infinity ball with radius epsilon.
#
#   Args:
#     image: array-like, input data for the CNN
#     label: integer, class label corresponding to image
#     params: tree, parameters of the model to attack
#     epsilon: float, radius of the L-infinity ball.
#     maxiter: int, number of iterations of this algorithm.
#
#   Returns:
#     perturbed_image: Adversarial image on the boundary of the L-infinity ball
#       of radius epsilon and centered at image.
#
#   Notes:
#     PGD attack is described in (Madry et al. 2017),
#     https://arxiv.org/pdf/1706.06083.pdf
#     :param step_size:
#   """
#     image_perturbation = jnp.zeros_like(image)
#
#     def adversarial_loss(adv_image):
#         return loss_fun(params, 0, (adv_image, label))
#
#     grad_adversarial = jax.grad(adversarial_loss)
#     for _ in range(maxiter):
#         # compute gradient of the loss wrt to the image
#
#         sign_grad = jnp.sign(grad_adversarial(image_perturbation))
#         # heuristic step-size 2 eps / maxiter
#         image_perturbation += step_size * sign_grad
#
#         delta = jnp.clip(image_perturbation - image, min=-epsilon, max=epsilon)
#         image_perturbation = jnp.clip(image + delta, min=0, max=1)
#
#         # projection step onto the L-infinity ball centered at image
#         # image_perturbation = jnp.clip(image_perturbation, - epsilon, epsilon)
#
#     # clip the image to ensure pixels are between 0 and 1
#     return image_perturbation


@jax.jit
def pgd_attack3(image, label, params, epsilon=0.1, step_size=2 / 255, maxiter=20):
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
        return loss_fun(params, 0, (image + perturbation, label))

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


def dataset_stats(params, data_loader, iter_per_epoch):
    """Computes accuracy on clean and adversarial images."""
    adversarial_accuracy = 0.
    clean_accuracy = 0.
    for batch in data_loader.as_numpy_iterator():
        images, labels = batch
        images = images.astype(jnp.float32) / 255
        clean_accuracy += jnp.mean(accuracy(params, (images, labels))) / iter_per_epoch
        # adversarial_images = pgd_attack(images, labels, params, epsilon=EPSILON)
        adversarial_images = pgd_attack3(images, labels, params, epsilon=EPSILON)
        adversarial_accuracy += jnp.mean(accuracy(params, (adversarial_images, labels))) / iter_per_epoch
    return {"adversarial accuracy": adversarial_accuracy, "accuracy": clean_accuracy}


@jax.jit
def train_step(params, opt_state, batch):
    images, labels = batch
    # convert images to float as attack requires to take gradients wrt to them
    images = images.astype(jnp.float32) / 255
    adversarial_images_train = pgd_attack(images, labels, params, epsilon=EPSILON)
    # train on adversarial images
    loss_grad_fun = jax.grad(loss_fun)
    grads = loss_grad_fun(params, L2_REG, (adversarial_images_train, labels))
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


# Initialize parameters.
key = jax.random.PRNGKey(0)
var_params = net.init(key, jnp.zeros(input_shape))["params"]

# Initialize the optimizer.
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(var_params)

start = datetime.datetime.now().replace(microsecond=0)

accuracy_train = []
accuracy_test = []
adversarial_accuracy_train = []
adversarial_accuracy_test = []
for epoch in range(EPOCHS):
    for train_batch in tqdm.tqdm(train_loader_batched.as_numpy_iterator()):
        key, train_step_key = jax.random.split(key, num=2)
        var_params, opt_state = train_step_trade(var_params, opt_state, train_batch, key)
        # var_params, opt_state = train_step(var_params, opt_state, train_batch,)

    # compute train set accuracy, both on clean and adversarial images
    train_stats = dataset_stats(var_params, train_loader_batched, iter_per_epoch_train)
    accuracy_train.append(train_stats["accuracy"])
    adversarial_accuracy_train.append(train_stats["adversarial accuracy"])

    # compute test set accuracy, both on clean and adversarial images
    test_stats = dataset_stats(var_params, test_loader_batched, iter_per_epoch_test)
    accuracy_test.append(test_stats["accuracy"])
    adversarial_accuracy_test.append(test_stats["adversarial accuracy"])

    time_elapsed = (datetime.datetime.now().replace(microsecond=0) - start)
    print(f"Epoch {epoch} out of {EPOCHS}")
    print(f"Accuracy on train set: {accuracy_train[-1]:.3f}")
    print(f"Accuracy on test set: {accuracy_test[-1]:.3f}")
    print(f"Adversarial accuracy on train set: {adversarial_accuracy_train[-1]:.3f}")
    print(f"Adversarial accuracy on test set: {adversarial_accuracy_test[-1]:.3f}")
    print(f"Time elapsed: {time_elapsed}\n")
