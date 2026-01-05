from jax.sharding import PartitionSpec as P

SHARDING_RULES = {
    "dp": {
        "batch": "data",
        "vocab": None,
        "seq": None,
        "embed": None,
        "q_heads": None,
        "kv_heads": None,
        "head_dim": None,
        "intermediate": None,
    },
    # "fsdp": {
    #     "batch": "data",
    #     "vocab": "fsdp",
    #     "embed": "fsdp",
    #     "seq": None,
    #     "heads": None,
    #     "head_dim": None,
    #     "intermediate": "fsdp",
    # },
}

_current_strategy = "dp"


def set_sharding_strategy(strategy: str):
    global _current_strategy
    if strategy not in SHARDING_RULES:
        raise ValueError(f"Unknown sharding strategy: {strategy}. Must be one of {list(SHARDING_RULES.keys())}")
    _current_strategy = strategy

def logical_to_physical(logical_axes):
    rules = SHARDING_RULES[_current_strategy]
    return P(*[rules.get(axis, None) for axis in logical_axes])


def shard_init(init, logical_axes):
    def init_fn(*args, **kwargs):
        return init(*args, **kwargs, out_sharding=logical_to_physical(logical_axes))
    return init_fn