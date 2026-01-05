from flax import nnx


def get_initializers(strategy, hidden_dim):
    if strategy == "nanochat":
        # Uniform with std = 1/sqrt(hidden_dim), bound = sqrt(3) * std
        bound = (3 ** 0.5) * (hidden_dim ** -0.5)
        return {
            "embed": nnx.initializers.normal(stddev=1.0),
            "lm_head": nnx.initializers.normal(stddev=0.001),
            "qkv": nnx.initializers.uniform(scale=bound),
            "o_proj": nnx.initializers.zeros,
            "mlp_up": nnx.initializers.uniform(scale=bound),
            "mlp_down": nnx.initializers.zeros,
            "bias": nnx.initializers.zeros,
        }
    elif strategy == "default":
        default = nnx.initializers.lecun_normal()
        return {
            "embed": nnx.initializers.normal(stddev=0.02),
            "lm_head": default,
            "qkv": default,
            "o_proj": default,
            "mlp_up": default,
            "mlp_down": default,
            "bias": nnx.initializers.zeros,
        }
    else:
        raise ValueError(f"Unknown init_strategy: {strategy}")

