import numpy as np
import orbax.checkpoint as ocp
import jax
jax.distributed.initialize()

path = ocp.test_utils.erase_and_create_empty('gs://caster-us-central-2b-2/test')
state = {
    'a': np.arange(8),
    'b': np.arange(16),
}
extra_params = [42, 43]


options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(
    path, options=options, item_names=('state', 'extra_params')
)

for step in range(11):  # [0, 1, ..., 10]
  mngr.save(
      step,
      args=ocp.args.Composite(
          state=ocp.args.StandardSave(state),
          extra_params=ocp.args.JsonSave(extra_params),
      ),
  )
mngr.wait_until_finished()
restored = mngr.restore(10)
restored_state, restored_extra_params = restored.state, restored.extra_params