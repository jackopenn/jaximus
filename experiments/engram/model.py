from dataclasses import dataclass
from functools import partial
from typing import List

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, attention
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p



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

    D, N, K, H, I = (
        config.hidden_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_dim
    )
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
        sliding_window=None
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
    logits = jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return logits
