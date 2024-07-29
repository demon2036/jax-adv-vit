import jax
import flax
import jax.numpy as jnp
from flax.linen.linear import default_kernel_init
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import flax.linen as nn


def case1():
    device_mesh = mesh_utils.create_device_mesh((8, 1))
    mesh = Mesh(device_mesh, axis_names=('data', 'model'))

    def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
        return NamedSharding(mesh, pspec)

    class DPDense(nn.Module):
        dim: int

        @nn.compact
        def __call__(self, x, *args, **kwargs):
            x = nn.Dense(self.dim,

                         )(x)

            return x


    shape=(128,256,384)
    x=jnp.ones(shape)
    x_sharding=mesh_sharding(PartitionSpec('data'))



    if jax.process_index() == 0:
        print(device_mesh)
        print()
        print(mesh)
        jax.debug.visualize_sharding((shape[0],shape[1]),sharding=x_sharding)
        # jax.debug.visualize_array_sharding(x_sharding)


if __name__ == "__main__":
    jax.distributed.initialize()

    if jax.process_index() == 0:
        print(jax.devices())

    case1()
