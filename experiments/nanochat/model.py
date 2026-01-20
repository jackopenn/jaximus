from dataclasses import dataclass
from functools import partial
from typing import List

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, attention
from modelling.layers.mlp import MLPWeights, mlp
from modelling.layers.norm import rms_norm
from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    mlp_weights: MLPWeights


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array


def _init_weight(key, init_fn, shape, sharding):
    return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))


def _init_attention_weights(config, keys):
    D, N, K, H = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    bound = (3**0.5) * (D**-0.5)
    qkv_init, zero_init = jax.nn.initializers.uniform(scale=bound), jax.nn.initializers.zeros
    return AttentionWeights(
        q_proj=_init_weight(next(keys), qkv_init, (D, N * H), ("model_embed", "model_q")),
        k_proj=_init_weight(next(keys), qkv_init, (D, K * H), ("model_embed", "model_kv")),
        v_proj=_init_weight(next(keys), qkv_init, (D, K * H), ("model_embed", "model_kv")),
        o_proj=_init_weight(next(keys), zero_init, (N * H, D), ("model_q", "model_embed")),
    )


def _init_mlp_weights(config, keys):
    D, I = config.hidden_dim, config.intermediate_dim
    bound = (3**0.5) * (D**-0.5)
    return MLPWeights(
        up_proj=_init_weight(
            next(keys), jax.nn.initializers.uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")
        ),
        down_proj=_init_weight(next(keys), jax.nn.initializers.zeros, (I, D), ("model_intermediate", "model_embed")),
    )


def _init_layer_weights(config, keys):
    return LayerWeights(
        attention_weights=_init_attention_weights(config, keys),
        mlp_weights=_init_mlp_weights(config, keys),
    )


def init_model_weights(config, key):
    keys = iter(jax.random.split(key, 2 + config.num_layers * 6))
    return ModelWeights(
        embed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=1.0),
            (config.vocab_size, config.hidden_dim),
            ("model_vocab", "model_embed"),
        ),
        layer_weights=[_init_layer_weights(config, keys) for _ in range(config.num_layers)],
        unembed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=0.001),
            (config.hidden_dim, config.vocab_size),
            ("model_embed", "model_vocab"),
        ),
    )


def model_forward(x, weights, config, rope_cos=None, rope_sin=None, mask=None):
    """Forward pass: pre-norm RMSNorm, RoPE, QK norm, relu_squared MLP, softcap=15.0, bfloat16."""
    eps = config.norm_epsilon

    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
    x = rms_norm(x, None, eps)

    attention_fn = partial(
        attention,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm=True,
        qk_norm_type="rms",
        qk_norm_epsilon=eps,
        sliding_window=None,
        dtype="bfloat16",
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )
    mlp_fn = partial(mlp, act_fn="relu_squared", dtype="bfloat16")

    for layer_weights in weights.layer_weights:
        residual = x
        x = rms_norm(x, None, eps)
        x = attention_fn(x, layer_weights.attention_weights, mask=mask)
        x = x + residual

        residual = x
        x = rms_norm(x, None, eps)
        x = mlp_fn(x, layer_weights.mlp_weights)
        x = x + residual

    x = rms_norm(x, None, eps)
    logits = jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)
