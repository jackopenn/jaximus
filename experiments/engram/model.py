from dataclasses import dataclass
from typing import List

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p


def attention(x, weights, rope_cos, rope_sin, eps, num_heads, num_kv_heads):
    B, L, D = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads
    q = jnp.matmul(x, weights.q_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_q")))
    q = q.reshape(B, L, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    k = jnp.matmul(x, weights.k_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    k = k.reshape(B, L, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))
    v = jnp.matmul(x, weights.v_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    v = v.reshape(B, L, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))
    q, k = apply_rope(q, rope_cos[:, :L], rope_sin[:, :L]), apply_rope(k, rope_cos[:, :L], rope_sin[:, :L])
    q, k = rms_norm(q, None, eps).astype(jnp.bfloat16), rms_norm(k, None, eps).astype(jnp.bfloat16)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim**-0.5)
    logits = jnp.where(jnp.tril(jnp.ones((L, L), dtype=jnp.bool_)), logits, -1e9)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
    encoded = jnp.einsum("bhqk,bkhd->bqhd", probs, v)
    encoded = jnp.reshape(encoded, (B, L, D), out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    out = jnp.matmul(encoded, weights.o_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_embed")))
    return out


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


def init_model_weights(config, key):
    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    keys = iter(jax.random.split(key, 2 + config.num_layers * 7))

    D, N, K, H, I = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.intermediate_dim
    bound = (3**0.5) * (D**-0.5)
    uniform, zeros, normal = jax.nn.initializers.uniform, jax.nn.initializers.zeros, jax.nn.initializers.normal

    layer_weights = []
    for _ in range(config.num_layers):
        layer_weights.append(LayerWeights(
            attention_weights=AttentionWeights(
                q_proj=w(next(keys), uniform(scale=bound), (D, N * H), ("model_embed", "model_q")),
                k_proj=w(next(keys), uniform(scale=bound), (D, K * H), ("model_embed", "model_kv")),
                v_proj=w(next(keys), uniform(scale=bound), (D, K * H), ("model_embed", "model_kv")),
                o_proj=w(next(keys), zeros, (N * H, D), ("model_q", "model_embed")),
            ),
            glu_weights=GLUWeights(
                gate_proj=w(next(keys), uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")),
                up_proj=w(next(keys), uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")),
                down_proj=w(next(keys), zeros, (I, D), ("model_intermediate", "model_embed")),
            ),
        ))

    return ModelWeights(
        embed=w(next(keys), normal(stddev=1.0), (config.vocab_size, D), ("model_vocab", "model_embed")),
        layer_weights=layer_weights,
        unembed=w(next(keys), normal(stddev=0.001), (D, config.vocab_size), ("model_embed", "model_vocab")),
    )


def model_forward(x, weights, config, rope_cos=None, rope_sin=None, mask=None):
    eps = config.norm_epsilon
    x = rms_norm(weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16), None, eps)

    for layer_weights in weights.layer_weights:
        x = x + attention(rms_norm(x, None, eps), layer_weights.attention_weights, rope_cos, rope_sin, eps, config.num_attention_heads, config.num_key_value_heads)
        x = x + glu(rms_norm(x, None, eps), layer_weights.glu_weights, act_fn="silu", dtype="bfloat16")

    x = rms_norm(x, None, eps)
    logits = jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return logits
