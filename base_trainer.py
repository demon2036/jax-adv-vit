import jax
import jax.numpy as jnp
import flax
import numpy as np
import optax
import flax.linen as nn
import torch

from model import ViT

model = ViT()
rng = jax.random.PRNGKey(0)
shape = (2, 32, 32, 3)
x = jnp.ones(shape)
# x = jax.random.normal(rng, shape)
params = model.init(rng, x)['params']

"""
@jax.jit
def trade_l2(epsilon=128 / 255, perturb_steps=10):
    delta = 0.001 * jax.random.normal(rng, x.shape)
    optimizer = optax.adamw(epsilon / perturb_steps * 2)
    opt_state = optimizer.init(delta)
    print(opt_state)
    p_natural = model.apply({'params': params}, x)

    def jax_re_norm(delta, max_norm):
        b, h, w, c = delta.shape
        norms = jnp.linalg.norm(delta.reshape(b, -1), ord=2, axis=1, keepdims=True)
        desired = jnp.clip(norms, a_min=None, a_max=max_norm)
        scale = desired / (1e-6 + norms)
        return delta * scale

    def grad_fn(delta, x):
        adv = x + delta
        model_out = model.apply({'params': params}, adv)
        loss = -1 * optax.losses.kl_divergence(nn.log_softmax(model_out), p_natural)
        return loss.mean()

    for _ in range(perturb_steps):
        grad = jax.grad(grad_fn)(delta, x)
        grad_norm = jnp.linalg.norm(grad.reshape(grad.shape[0], -1), axis=1)
        grad = grad / grad_norm.reshape(-1, 1, 1, 1)
        updates, opt_state = optimizer.update(grad, opt_state, delta)
        delta = optax.apply_updates(delta, updates)
        delta = delta + x
        delta = jnp.clip(delta, 0, 1) - x
        # delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        delta = jax_re_norm(delta, max_norm=epsilon)

    return jnp.clip(x + delta, 0, 1)


x_adv = trade_l2()
print(x_adv)
"""

x_torch = torch.from_numpy(np.array(x))
x_torch_out = torch.renorm(x_torch, p=2, dim=0, maxnorm=5)


#1 0 5
# torch.renorm()


def clip_grad_norm(grad, max_norm=5):
    b, h, w, c = grad.shape
    norms = jnp.linalg.norm(grad.reshape(b, -1), ord=2, axis=1, keepdims=True).reshape(b, 1, 1, 1)
    desired = jnp.clip(norms, a_min=None, a_max=max_norm)
    scale = desired / (1e-7 + norms)
    return grad * scale


np_torch_out = np.array(x_torch_out)
np_jax_out = np.array(clip_grad_norm(x))
print(np_jax_out - np_torch_out)
""""""
#         for _ in range(perturb_steps):
#             adv = x_natural + delta
#
#             # optimize
#             optimizer_delta.zero_grad()
#             with torch.enable_grad():
#                 loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
#             loss.backward()
#             # renorming gradient
#             grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
#             delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
#             # avoid nan or inf if gradient is 0
#             if (grad_norms == 0).any():
#                 delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
#             optimizer_delta.step()
#
#             # projection
#             delta.data.add_(x_natural)
#             delta.data.clamp_(0, 1).sub_(x_natural)
#             delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
