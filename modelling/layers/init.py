import jax


def get_initializers(strategy, hidden_dim):
    if strategy == "nanochat":
        # Uniform with std = 1/sqrt(hidden_dim), bound = sqrt(3) * std
        bound = (3 ** 0.5) * (hidden_dim ** -0.5)
        return {
            "embed": jax.nn.initializers.normal(stddev=1.0),
            "lm_head": jax.nn.initializers.normal(stddev=0.001),
            "qkv": jax.nn.initializers.uniform(scale=bound),
            "o_proj": jax.nn.initializers.zeros,
            "mlp_up": jax.nn.initializers.uniform(scale=bound),
            "mlp_down": jax.nn.initializers.zeros,
            "bias": jax.nn.initializers.zeros,
        }
    elif strategy == "default":
        default = jax.nn.initializers.lecun_normal()
        return {
            "embed": jax.nn.initializers.normal(stddev=0.02),
            "lm_head": default,
            "qkv": default,
            "o_proj": default,
            "mlp_up": default,
            "mlp_down": default,
            "bias": jax.nn.initializers.zeros,
        }
    else:
        raise ValueError(f"Unknown init_strategy: {strategy}")
