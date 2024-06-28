import jax.numpy as jnp
from optax.losses import softmax_cross_entropy_with_integer_labels
import jax


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
        logits = state.apply_fn({"params": state.ema_params,'batch_stats':state.batch_stats}, image + perturbation)
        loss_value = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, label))
        return loss_value

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


