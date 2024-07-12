import numpy as np
import orbax.checkpoint as ocp
import jax
from flax.jax_utils import replicate,unreplicate

jax.distributed.initialize()


# path = ocp.test_utils.erase_and_create_empty('gs://caster-us-central-2b-2/test')
path='gs://caster-us-central-2b-2/test'
state = {
    'a': np.arange(8),
    'b': np.arange(16),
}
extra_params = [42, 43]

options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(
    path, options=options, item_names=('state', 'extra_params')
)
print(path)



state=replicate(state)

def p(state):
    return jax.tree_util.tree_map(lambda x:x+1,state)

# state=jax.pmap(p,axis_name='batch')(state)

state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
# state=unreplicate(state)




for step in range(11):  # [0, 1, ..., 10]
    mngr.save(
        step,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            extra_params=ocp.args.JsonSave(extra_params),
        ),force=True
    )






mngr.wait_until_finished()
restored = mngr.restore(10)
restored_state, restored_extra_params = restored.state, restored.extra_params
print(restored)
