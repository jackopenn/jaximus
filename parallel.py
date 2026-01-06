from jax.sharding import PartitionSpec as P

# Embedding: (model_vocab, model_embed)
# Pos Embedding: (model_seq, model_embed)
# Attention: 
#   - q_proj: (model_embed, model_q)
#   - k_proj: (model_embed, model_kv)
#   - v_proj: (model_embed, model_kv)
#   - o_proj: (model_q, model_embed)
# MLP: 
#   - up_proj: (model_embed, model_intermediate)
#   - down_proj: (model_intermediate, model_embed)
# LM Head: 
#   - lm_head: (model_embed, model_vocab)

SHARDING_RULES = {
    "dp": {
        "batch": "data",
        "act_seq": None,
        "act_vocab": None,
        "act_embed": None,
        "act_intermediate": None,
        "act_q": None,
        "act_kv": None,
        "model_seq": None,
        "model_vocab": None,
        "model_embed": None,
        "model_intermediate": None,
        "model_q": None,
        "model_kv": None,
    },
    "fsdp": {
        "batch": "data",
        "act_seq": None,
        "act_vocab": None,
        "act_embed": None,
        "act_intermediate": None,
        "act_q": None,
        "act_kv": None,
        "model_seq": None,
        "model_vocab": "data",
        "model_embed": None,
        "model_intermediate": "data",
        "model_q": "data",
        "model_kv": "data",
    },

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