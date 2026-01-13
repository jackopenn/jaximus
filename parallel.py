from jax.sharding import PartitionSpec as P

SHARDING_RULES = {
    "dp": {
        "batch": "data",
        "seq": None,
        "model_vocab": None,
        "model_embed": None,
        "model_intermediate": None,
        "model_q": None,
        "model_kv": None,
        "model_head": None,
        "head_embed": None,
    },
    "fsdp": {
        "batch": "data",
        "seq": None,
        "model_vocab": "data",
        "model_embed": None,
        "model_intermediate": "data",
        "model_q": "data",
        "model_kv": "data",
        "model_head": "data",
        "head_embed": None,
    },
}

_current_rules = SHARDING_RULES["dp"]


def set_sharding_strategy(strategy: str):
    """Sets the sharding strategy by name ('dp' or 'fsdp')."""
    global _current_rules
    if strategy not in SHARDING_RULES:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of {list(SHARDING_RULES.keys())}")
    _current_rules = SHARDING_RULES[strategy]


def logical_to_physical(logical_axes):
    """Convert logical axes to physical PartitionSpec using current rules."""
    return P(*[_current_rules.get(axis) for axis in logical_axes])
