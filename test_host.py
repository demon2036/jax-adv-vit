import jax

if __name__ == "__main__":
    jax.distributed.initialize()
    print(jax.devices())
