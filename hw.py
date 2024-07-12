import numpy as np
import orbax.checkpoint as ocp
import jax

path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')


my_tree = {
    'a': np.arange(8),
    'b': {
        'c': 42,
        'd': np.arange(16),
    },
}
abstract_my_tree = jax.tree_util.tree_map(
    ocp.utils.to_shape_dtype_struct, my_tree)


checkpointer = ocp.StandardCheckpointer()
# 'checkpoint_name' must not already exist.
checkpointer.save(path / 'checkpoint_name', my_tree)
print(checkpointer.restore(
    path / 'checkpoint_name/',
    args=ocp.args.StandardRestore(abstract_my_tree)
))