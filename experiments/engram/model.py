from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, attention
from modelling.layers.mlp import MLPWeights, mlp
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p


def attention_with_top_k(x, weights, rope_cos, rope_sin, eps, num_heads, num_kv_heads, top_k, exclude_self):
    """Attention that returns top_k attended position indices for engram."""
    B, L, _ = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads

    q = jnp.matmul(x, weights.q_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_q")))
    q = q.reshape(B, L, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    k = jnp.matmul(x, weights.k_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    k = k.reshape(B, L, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))
    v = jnp.matmul(x, weights.v_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    v = v.reshape(B, L, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    q, k = apply_rope(q, rope_cos[:, :L], rope_sin[:, :L]), apply_rope(k, rope_cos[:, :L], rope_sin[:, :L])
    q, k = rms_norm(q, None, eps).astype(jnp.bfloat16), rms_norm(k, None, eps).astype(jnp.bfloat16)

    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim**-0.5)
    scores = jnp.where(jnp.tril(jnp.ones((L, L), dtype=jnp.bool_)), scores, -1e9)
    attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)

    att = jnp.einsum("bhqk,bkhd->bqhd", attn_weights.astype(jnp.bfloat16), v)
    att = att.reshape(B, L, num_heads * head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    out = jnp.matmul(att, weights.o_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_embed")))

    attn_avg = attn_weights.mean(axis=1)
    if exclude_self:
        attn_avg = attn_avg * (1.0 - jnp.eye(L, dtype=attn_avg.dtype))
    return out, jax.lax.top_k(attn_avg, top_k)[1]


@jax.tree_util.register_dataclass
@dataclass
class EngramWeights:
    embedding: jax.Array
    key_proj: jax.Array
    value_proj: jax.Array
    conv_weight: jax.Array


@dataclass
class EngramHashConfig:
    multipliers: Dict[int, jax.Array]
    prime_vocab_sizes: Dict[int, List[List[int]]]
    offsets: Dict[int, jax.Array]


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    mlp_weights: MLPWeights
    engram_weights: Optional[EngramWeights] = None


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array


def _next_prime(n):
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    while not is_prime(n):
        n += 1
    return n


def _generate_distinct_primes(base_size, count, start_offset=0):
    primes, candidate = [], base_size + start_offset
    while len(primes) < count:
        candidate = _next_prime(candidate)
        primes.append(candidate)
        candidate += 1
    return primes


def _make_hash_config(engram_cfg, layer_ids, seed):
    rng = jax.random.PRNGKey(seed)
    n_ngram_types = engram_cfg.max_ngram_size - 1
    total_heads = n_ngram_types * engram_cfg.n_head_per_ngram
    multipliers, prime_vocab_sizes, offsets = {}, {}, {}

    for layer_idx, layer_id in enumerate(layer_ids):
        rng, key = jax.random.split(rng)
        multipliers[layer_id] = jax.random.randint(key, (engram_cfg.max_ngram_size,), 1, 2**30, dtype=jnp.int32)

        layer_primes = [
            _generate_distinct_primes(
                engram_cfg.vocab_size_per_ngram[i], engram_cfg.n_head_per_ngram,
                start_offset=layer_idx * total_heads * 100 + i * engram_cfg.n_head_per_ngram * 10
            )
            for i in range(n_ngram_types)
        ]
        prime_vocab_sizes[layer_id] = layer_primes

        all_primes = [p for primes in layer_primes for p in primes]
        offsets[layer_id] = jnp.array([0] + list(jnp.cumsum(jnp.array(all_primes[:-1]))), dtype=jnp.int32)

    return EngramHashConfig(multipliers=multipliers, prime_vocab_sizes=prime_vocab_sizes, offsets=offsets)


def compute_hash_ids(input_ids, layer_id, hash_config, engram_cfg, top_k_indices=None):
    """Compute hash indices for n-gram embeddings. Uses top_k_indices if provided, else consecutive positions."""
    B, L = input_ids.shape
    n_ngram_types = engram_cfg.max_ngram_size - 1
    multipliers = hash_config.multipliers[layer_id]
    input_ids_i32 = input_ids.astype(jnp.int32)
    batch_idx = jnp.broadcast_to(jnp.arange(B)[:, None], (B, L))

    all_hash_ids = []
    for ngram_idx in range(n_ngram_types):
        k = ngram_idx + 1
        mix = input_ids_i32 * multipliers[0]
        for i in range(k):
            if top_k_indices is not None:
                ctx = input_ids_i32.at[batch_idx, top_k_indices[:, :, i]].get(out_sharding=l2p(("batch", "act_seq")))
            else:
                ctx = jnp.pad(input_ids_i32[:, :-(i + 1)], ((0, 0), (i + 1, 0)), constant_values=0)
            mix = mix ^ (ctx * multipliers[i + 1])

        for prime in hash_config.prime_vocab_sizes[layer_id][ngram_idx]:
            all_hash_ids.append(jnp.abs(mix) % prime)

    return jnp.stack(all_hash_ids, axis=-1)


def engram_forward(hidden_states, input_ids, engram_weights, layer_id, hash_config, engram_cfg, eps, top_k_indices=None):
    B, L, D = hidden_states.shape
    engram_hidden_size = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram

    hash_input_ids = compute_hash_ids(input_ids, layer_id, hash_config, engram_cfg, top_k_indices)
    hash_input_ids = hash_input_ids + hash_config.offsets[layer_id][None, None, :]
    embeddings = engram_weights.embedding.at[hash_input_ids].get(out_sharding=l2p(("batch", "act_seq", None, None)))
    embeddings = embeddings.reshape(B, L, engram_hidden_size).astype(jnp.bfloat16)

    key = jnp.matmul(embeddings, engram_weights.key_proj.astype(jnp.bfloat16))
    normed_key = rms_norm(key, None, eps)
    query = hidden_states
    normed_query = rms_norm(query, None, eps)
    gate = jnp.sum(normed_key * normed_query, axis=-1, keepdims=True) / jnp.sqrt(D)
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate)
    gate = jax.nn.sigmoid(gate)

    value = jnp.matmul(embeddings, engram_weights.value_proj.astype(jnp.bfloat16))
    gated_value = gate * value

    conv_weight = engram_weights.conv_weight.astype(jnp.bfloat16).T[:, None, :]
    padding = (engram_cfg.kernel_size - 1) * engram_cfg.max_ngram_size
    conv_out = jax.lax.conv_general_dilated(
        gated_value.transpose(0, 2, 1), conv_weight, window_strides=(1,), padding=[(padding, 0)],
        rhs_dilation=(engram_cfg.max_ngram_size,), feature_group_count=D, dimension_numbers=("NCH", "OIH", "NCH")
    ).transpose(0, 2, 1)

    return gated_value + jax.nn.silu(rms_norm(conv_out, None, eps))


def _init_weight(key, init_fn, shape, sharding):
    return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))


def _init_attention_weights(config, keys):
    D, N, K, H = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    bound = (3**0.5) * (D**-0.5)
    return AttentionWeights(
        q_proj=_init_weight(next(keys), jax.nn.initializers.uniform(scale=bound), (D, N * H), ("model_embed", "model_q")),
        k_proj=_init_weight(next(keys), jax.nn.initializers.uniform(scale=bound), (D, K * H), ("model_embed", "model_kv")),
        v_proj=_init_weight(next(keys), jax.nn.initializers.uniform(scale=bound), (D, K * H), ("model_embed", "model_kv")),
        o_proj=_init_weight(next(keys), jax.nn.initializers.zeros, (N * H, D), ("model_q", "model_embed")),
    )


def _init_mlp_weights(config, keys):
    D, I = config.hidden_dim, config.intermediate_dim
    return MLPWeights(
        up_proj=_init_weight(next(keys), jax.nn.initializers.uniform(scale=(3**0.5) * (D**-0.5)), (D, I), ("model_embed", "model_intermediate")),
        down_proj=_init_weight(next(keys), jax.nn.initializers.zeros, (I, D), ("model_intermediate", "model_embed")),
    )


def _init_engram_weights(config, layer_id, hash_config, keys):
    cfg = config.engram
    engram_hidden = (cfg.max_ngram_size - 1) * cfg.n_embed_per_ngram
    embed_per_head = cfg.n_embed_per_ngram // cfg.n_head_per_ngram
    total_vocab = sum(sum(primes) for primes in hash_config.prime_vocab_sizes[layer_id])
    return EngramWeights(
        embedding=_init_weight(next(keys), jax.nn.initializers.normal(stddev=0.02), (total_vocab, embed_per_head), ("model_engram_vocab", "model_engram_embed")),
        key_proj=_init_weight(next(keys), jax.nn.initializers.uniform(scale=(3**0.5) * (engram_hidden**-0.5)), (engram_hidden, config.hidden_dim), ("model_engram_hidden", "model_embed")),
        value_proj=_init_weight(next(keys), jax.nn.initializers.zeros, (engram_hidden, config.hidden_dim), ("model_engram_hidden", "model_embed")),
        conv_weight=_init_weight(next(keys), jax.nn.initializers.normal(stddev=0.01), (cfg.kernel_size, config.hidden_dim), (None, "model_embed")),
    )


def _init_layer_weights(config, layer_idx, hash_config, keys):
    engram_weights = None
    if hasattr(config, "engram") and config.engram.enabled and layer_idx in config.engram.layer_ids:
        engram_weights = _init_engram_weights(config, layer_idx, hash_config, keys)
    return LayerWeights(
        attention_weights=_init_attention_weights(config, keys),
        mlp_weights=_init_mlp_weights(config, keys),
        engram_weights=engram_weights,
    )


_hash_config_cache = {}


def _get_hash_config(engram_cfg):
    cache_key = (engram_cfg.seed, tuple(engram_cfg.layer_ids))
    if cache_key not in _hash_config_cache:
        _hash_config_cache[cache_key] = _make_hash_config(engram_cfg, engram_cfg.layer_ids, engram_cfg.seed)
    return _hash_config_cache[cache_key]


def init_model_weights(config, key):
    engram_enabled = hasattr(config, "engram") and config.engram.enabled
    hash_config = _get_hash_config(config.engram) if engram_enabled else None
    num_engram_layers = len(config.engram.layer_ids) if engram_enabled else 0
    keys = iter(jax.random.split(key, 2 + config.num_layers * 6 + num_engram_layers * 4))

    return ModelWeights(
        embed=_init_weight(next(keys), jax.nn.initializers.normal(stddev=1.0), (config.vocab_size, config.hidden_dim), ("model_vocab", "model_embed")),
        layer_weights=[_init_layer_weights(config, i, hash_config, keys) for i in range(config.num_layers)],
        unembed=_init_weight(next(keys), jax.nn.initializers.normal(stddev=0.001), (config.hidden_dim, config.vocab_size), ("model_embed", "model_vocab")),
    )


def model_forward(x, weights, config, rope_cos=None, rope_sin=None, mask=None):
    eps = config.norm_epsilon
    engram_enabled = hasattr(config, "engram") and config.engram.enabled
    hash_config = _get_hash_config(config.engram) if engram_enabled else None
    need_attn_top_k = engram_enabled and getattr(config.engram, "mode", "consecutive") == "attention"
    attn_lag = getattr(config.engram, "attn_lag", 0) if engram_enabled else 0

    input_ids = x
    x = rms_norm(weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16), None, eps)

    attention_fn = partial(
        attention, rope_cos=rope_cos, rope_sin=rope_sin, qk_norm=True, qk_norm_type="rms", qk_norm_epsilon=eps,
        sliding_window=None, dtype="bfloat16", num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads,
    )
    mlp_fn = partial(mlp, act_fn="relu_squared", dtype="bfloat16")
    top_k_history = []

    for layer_idx, layer_weights in enumerate(weights.layer_weights):
        if engram_enabled and layer_weights.engram_weights is not None:
            top_k_indices = top_k_history[layer_idx - 1 - attn_lag] if need_attn_top_k else None
            x = x + engram_forward(x, input_ids, layer_weights.engram_weights, layer_idx, hash_config, config.engram, eps, top_k_indices)

        x_norm = rms_norm(x, None, eps)
        if need_attn_top_k:
            attn_out, top_k = attention_with_top_k(
                x_norm, layer_weights.attention_weights, rope_cos, rope_sin, eps,
                config.num_attention_heads, config.num_key_value_heads,
                config.engram.max_ngram_size - 1, getattr(config.engram, "attn_exclude_self", True)
            )
            top_k_history.append(top_k)
            x = x + attn_out
        else:
            x = x + attention_fn(x_norm, layer_weights.attention_weights, mask=mask)

        x = x + mlp_fn(rms_norm(x, None, eps), layer_weights.mlp_weights)

    logits = jnp.matmul(rms_norm(x, None, eps), weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)
