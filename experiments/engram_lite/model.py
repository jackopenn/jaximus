from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, attention
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import precompute_rope_embeddings
from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class EngramWeights:
    embeddings: jax.Array  # (vocab_size * table_multiplier, hidden_dim)
    lambdas: jax.Array  # (num_layers,) - per-layer ngram scaling


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    glu_weights: GLUWeights


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array
    engram: Optional[EngramWeights] = None


def _init_engram_weights(config, key):
    """Initialize engram-lite weights."""
    table_size = config.vocab_size * config.engram.table_multiplier
    D, L = config.hidden_dim, config.num_layers

    def w(init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    zeros = jax.nn.initializers.zeros
    lambda_init = jax.nn.initializers.constant(config.engram.lambda_init)

    return EngramWeights(
        embeddings=w(zeros, (table_size, D), ("model_engram_vocab", "model_embed")),
        lambdas=w(lambda_init, (L,), ()),
    )


def init_model_weights(config, key):
    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    keys = iter(jax.random.split(key, 3 + config.num_layers * 7 + 1))

    V, D, N, K, H, I = (
        config.vocab_size,
        config.hidden_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_dim,
    )
    zeros, normal = jax.nn.initializers.zeros, jax.nn.initializers.lecun_normal()

    layer_weights = []
    for _ in range(config.num_layers):
        layer_weights.append(
            LayerWeights(
                attention_weights=AttentionWeights(
                    q_proj=w(next(keys), normal, (D, N * H), ("model_embed", "model_q")),
                    k_proj=w(next(keys), normal, (D, K * H), ("model_embed", "model_kv")),
                    v_proj=w(next(keys), normal, (D, K * H), ("model_embed", "model_kv")),
                    o_proj=w(next(keys), zeros, (N * H, D), ("model_q", "model_embed")),
                ),
                glu_weights=GLUWeights(
                    gate_proj=w(next(keys), normal, (D, I), ("model_embed", "model_intermediate")),
                    up_proj=w(next(keys), normal, (D, I), ("model_embed", "model_intermediate")),
                    down_proj=w(next(keys), zeros, (I, D), ("model_intermediate", "model_embed")),
                ),
            )
        )

    engram_weights = None
    if getattr(config, "engram", None) and config.engram.enabled:
        engram_weights = _init_engram_weights(config, next(keys))

    return ModelWeights(
        embed=w(next(keys), normal, (V, D), ("model_vocab", "model_embed")),
        layer_weights=layer_weights,
        unembed=w(next(keys), normal, (D, V), ("model_embed", "model_vocab")),
        engram=engram_weights,
    )


def trigram_hash(input_ids, table_size):
    """Trigram hash: (36313*curr) ^ (27191*prev) ^ (17093*prev2) % (table_size - 1)."""
    curr = input_ids[:, 2:]
    prev = input_ids[:, 1:-1]
    prev2 = input_ids[:, :-2]
    h = (36313 * curr) ^ (27191 * prev) ^ (17093 * prev2)
    return h % (table_size - 1)


def make_model_forward(config):
    """Factory that returns forward function with precomputed rope embeddings."""
    rope_cos, rope_sin = precompute_rope_embeddings(
        config.max_seq_len, config.head_dim, config.rope_theta, "bfloat16", sharding=l2p(())
    )

    if getattr(config, "engram", None) and config.engram.enabled:
        table_size = config.vocab_size * config.engram.table_multiplier
        return partial(
            _model_forward_with_engram_lite, config=config, rope_cos=rope_cos, rope_sin=rope_sin, table_size=table_size
        )

    return partial(_model_forward, config=config, rope_cos=rope_cos, rope_sin=rope_sin)


def _model_forward(x, weights, config, rope_cos, rope_sin, mask=None):
    """Base forward pass."""
    norm_fn = partial(rms_norm, weights=None, eps=config.norm_epsilon)
    attention_fn = partial(
        attention,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm=True,
        qk_norm_type="rms",
        qk_norm_epsilon=config.norm_epsilon,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        dtype="bfloat16",
        sliding_window=None,
    )
    mlp_fn = partial(glu, act_fn="silu", dtype="bfloat16")

    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
    x = norm_fn(x)

    for layer_weights in weights.layer_weights:
        residual = x
        x = norm_fn(x)
        x = attention_fn(x, layer_weights.attention_weights, mask=mask)
        x = x + residual

        residual = x
        x = norm_fn(x)
        x = mlp_fn(x, layer_weights.glu_weights)
        x = x + residual

    x = norm_fn(x)
    return jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))


def _model_forward_with_engram_lite(x, weights, config, rope_cos, rope_sin, table_size, mask=None):
    """Forward pass with engram-lite: trigram hash + per-layer lambdas."""
    norm_fn = partial(rms_norm, weights=None, eps=config.norm_epsilon)
    attention_fn = partial(
        attention,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm=True,
        qk_norm_type="rms",
        qk_norm_epsilon=config.norm_epsilon,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        dtype="bfloat16",
        sliding_window=None,
    )
    mlp_fn = partial(glu, act_fn="silu", dtype="bfloat16")

    input_ids = x
    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)

    # Compute trigram embeddings once (pad first two positions with zeros)
    hash_indices = trigram_hash(input_ids, table_size)
    hash_indices = jnp.pad(hash_indices, ((0, 0), (2, 0)), constant_values=0)
    ngram_embed = (
        weights.engram.embeddings.at[hash_indices]
        .get(out_sharding=l2p(("batch", "act_seq", "act_embed")))
        .astype(jnp.bfloat16)
    )

    x = norm_fn(x)

    for layer_idx, layer_weights in enumerate(weights.layer_weights):
        # Trigram injection before each block
        lam = weights.engram.lambdas[layer_idx].astype(jnp.bfloat16)
        x = x + lam * ngram_embed

        residual = x
        x = norm_fn(x)
        x = attention_fn(x, layer_weights.attention_weights, mask=mask)
        x = x + residual

        residual = x
        x = norm_fn(x)
        x = mlp_fn(x, layer_weights.glu_weights)
        x = x + residual

    x = norm_fn(x)
    return jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
