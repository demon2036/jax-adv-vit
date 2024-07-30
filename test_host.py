import functools
import time

import jax
import flax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.linen.linear import default_kernel_init
from flax.training.common_utils import shard
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import flax.linen as nn


def block_all(xs):
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
    return xs


def case1():
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=('data',))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    class DPDense(nn.Module):
        dim: int

        @nn.compact
        def __call__(self, x, *args, **kwargs):
            # for i in range(12):
            x = nn.Dense(self.dim, )(x)

            x = jax.lax.with_sharding_constraint(x, mesh_sharding(PartitionSpec('data', )))
            return x

    shape = (128, 256, 384)
    x = jnp.ones(shape)
    x_sharding = mesh_sharding(PartitionSpec('data'))
    # x = jax.device_put(x, x_sharding)

    # global_batch_shape = (128*jax.process_count(), 256, 384)
    #
    # per_replica_batches = np.split(x, jax.local_device_count())
    #
    # global_batch_array = jax.make_array_from_single_device_arrays(
    #     global_batch_shape, sharding=x_sharding,
    #     arrays=[
    #         jax.device_put(batch, device)
    #         for batch, device in zip(per_replica_batches, x_sharding.addressable_devices)
    #     ]
    # )
    """"""

    global_batch_array = jax.device_put(x, x_sharding)

    rng = jax.random.PRNGKey(1)
    model = DPDense(384)

    def init_fn(x, model):
        variables = model.init(rng, x)
        return variables['params']

    abstract_variables = jax.eval_shape(
        functools.partial(init_fn, model=model, ), global_batch_array)

    state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_init_fn = jax.jit(init_fn, static_argnums=(1,),
                          in_shardings=x_sharding,  # PRNG key and x
                          out_shardings=state_sharding)

    params = jit_init_fn(global_batch_array, model)

    def train_step(x, params):
        out = model.apply({'params': params}, x)
        return out

        def loss_fn(params):
            out = model.apply({'params': params}, x)
            loss = (jnp.zeros_like(out) - out).mean()
            return loss

        grad = jax.grad(loss_fn)(params)

        return out

    train_step_jit = jax.jit(train_step, in_shardings=(x_sharding, state_sharding), out_shardings=(x_sharding), )

    #
    # start = time.time()
    # print(abs(global_batch_array[-1]))
    # end = time.time()
    # print(end - start)

    with mesh:

        out = block_all(train_step_jit(global_batch_array, params))

        for i in range(100):
            out = block_all(train_step_jit(global_batch_array, params))

        start = time.time()
        for i in range(1000):
            out = block_all(train_step_jit(global_batch_array, params))
        end = time.time()

        if jax.process_index() == 0:
            print(device_mesh)
            print(x_sharding.addressable_devices)
            # print()
            # print(mesh)
            # jax.debug.visualize_sharding((shape[0], shape[1]), sharding=x_sharding)
            # jax.debug.visualize_array_sharding(global_batch_array[:, :, 0])
            #
            # print(x_sharding.addressable_devices)
            # print(state_sharding)
            # jax.debug.visualize_array_sharding(params['Dense_0']['kernel'])
            # print(global_batch_array.shape)
            print(end - start)

    return out


def case2():
    class DPDense(nn.Module):
        dim: int

        @nn.compact
        def __call__(self, x, *args, **kwargs):
            # for i in range(12):
            x = nn.Dense(self.dim, )(x)

            return x

    shape = (128, 256, 384)
    x = jnp.ones(shape)

    """"""
    rng = jax.random.PRNGKey(1)
    model = DPDense(384)

    def init_fn(x, model):
        variables = model.init(rng, x)
        return variables['params']

    params = init_fn(x, model)

    def train_step(x, params):
        out = model.apply({'params': params}, x)
        return out
        def loss_fn(params):
            out = model.apply({'params': params}, x)
            loss = (jnp.zeros_like(out) - out).mean()
            return loss

        grad = jax.grad(loss_fn)(params)
        grad = jax.lax.pmean(grad, axis_name='batch')

        return grad

    train_step_pmap = jax.pmap(train_step, axis_name='batch')

    #
    # start = time.time()
    # print(abs(global_batch_array[-1]))
    # end = time.time()
    # print(end - start)

    global_batch_array = shard(x)
    params = replicate(params)

    def block_all(xs):
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
        return xs

    out = block_all(train_step_pmap(global_batch_array, params))

    for i in range(100):
        out = block_all(train_step_pmap(global_batch_array, params))

    start = time.time()
    for i in range(1000):
        out = block_all(train_step_pmap(global_batch_array, params))
    end = time.time()

    if jax.process_index() == 0:
        print(end - start)

    return out


if __name__ == "__main__":
    jax.distributed.initialize()

    if jax.process_index() == 0:
        print(jax.devices())

    out1=case1()
    out2=case2()

    out2=out2.reshape(out1.shape)

    print(out1-out2)