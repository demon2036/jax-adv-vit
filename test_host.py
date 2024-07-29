import jax
import flax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding


def case1():
    device_mesh = mesh_utils.create_device_mesh((2, 8))
    if jax.process_index() == 0:
        print(device_mesh)


if __name__ == "__main__":
    jax.distributed.initialize()

    if jax.process_index() == 0:
        print(jax.devices())

    case1()
