import jax
import optax
from flax.training.train_state import TrainState
import jax.numpy as jnp
from model import ViT


def create_train_state(args: argparse.Namespace) -> TrainState:
    model = ViT(
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
    # module = TrainModule(
    #     model=model,
    #     mixup=Mixup(args.mixup, args.cutmix),
    #     label_smoothing=args.label_smoothing if args.criterion == "ce" else 0,
    #     criterion=CRITERION_COLLECTION[args.criterion],
    # )

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of model and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    # example_inputs = {
    #     "images": jnp.zeros((1, 3, args.image_size, args.image_size), dtype=jnp.uint8),
    #     "labels": jnp.zeros((1,), dtype=jnp.int32),
    # }
    # init_rngs = {"params": jax.random.PRNGKey(42)}
    # print(module.tabulate(init_rngs, **example_inputs))

    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 32, 32, 3)))["params"]

    # params = module.init(init_rngs, **example_inputs)["params"]

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.
    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:
        tx = OPTIMIZER_COLLECTION[args.optimizer](
            learning_rate=learning_rate,
            b1=args.adam_b1,
            b2=args.adam_b2,
            # eps=args.adam_eps,
            weight_decay=args.weight_decay,
            mask=partial(tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        if args.lr_decay < 1.0:
            layerwise_scales = {
                i: optax.scale(args.lr_decay ** (args.layers - i))
                for i in range(args.layers + 1)
            }
            label_fn = partial(get_layer_index_fn, num_layers=args.layers)
            label_fn = partial(tree_map_with_path, label_fn)
            tx = optax.chain(tx, optax.multi_transform(layerwise_scales, label_fn))
        if args.clip_grad > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.clip_grad), tx)
        return tx

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.training_steps,
        end_value=1e-5,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer_fn(learning_rate),
        mixup_rng=jax.random.PRNGKey(args.mixup_seed + jax.process_index()),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed + jax.process_index()),
        micro_step=0,
        micro_in_mini=args.grad_accum,
        grad_accum=grad_accum if args.grad_accum > 1 else None,
    )
