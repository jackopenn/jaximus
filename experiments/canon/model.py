from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, make_attention_mask
from modelling.layers.mlp import MLPWeights, resolve_act_fn
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class CanonWeights:
    kernel: jax.Array  # (4, dim) where rows are [w3, w2, w1, w0]


@jax.tree_util.register_dataclass
@dataclass
class CanonBlockWeights:
    canon_a: Optional[CanonWeights]
    canon_b_q: Optional[CanonWeights]
    canon_b_k: Optional[CanonWeights]
    canon_b_v: Optional[CanonWeights]
    canon_c: Optional[CanonWeights]
    canon_d: Optional[CanonWeights]


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    mlp_weights: MLPWeights
    canon: Optional[CanonBlockWeights]


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array


def canon_layer(x, weights):
    """h'_t = h_t + sum_i(w_i * h_{t-i}) via depthwise 1D conv."""
    dtype = x.dtype
    dim = x.shape[-1]
    depth = weights.kernel.shape[0]
    kernel = weights.kernel.astype(dtype).T[:, None, :]  # (depth, dim) -> (dim, 1, depth)
    x_t = x.transpose(0, 2, 1)
    conv_out = jax.lax.conv_general_dilated(x_t, kernel, window_strides=(1,), padding=((depth - 1, 0),), feature_group_count=dim)
    return x + conv_out.transpose(0, 2, 1)


def _init_weight(key, init_fn, shape, sharding):
    return init_fn(key, shape, dtype=jnp.float32, out_sharding=None if sharding is None else l2p(sharding))


def _init_canon_weights(dim, depth, init, sharding, key):
    if init == "zeros":
        init_fn = jax.nn.initializers.zeros
    elif init == "normal":
        init_fn = jax.nn.initializers.normal(stddev=0.02)
    else:
        raise ValueError(f"Unknown canon init: {init}")
    return CanonWeights(kernel=_init_weight(key, init_fn, (depth, dim), sharding))


def _init_canon_block_weights(config, key):
    D, N, K, H, I = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.intermediate_dim
    depth, init = config.canon_depth, config.canon_init
    keys = iter(jax.random.split(key, 6))
    return CanonBlockWeights(
        canon_a=_init_canon_weights(D, depth, init, None, next(keys)) if config.canon_a else None,
        canon_b_q=_init_canon_weights(N * H, depth, init, None, next(keys)) if config.canon_b else None,
        canon_b_k=_init_canon_weights(K * H, depth, init, None, next(keys)) if config.canon_b else None,
        canon_b_v=_init_canon_weights(K * H, depth, init, None, next(keys)) if config.canon_b else None,
        canon_c=_init_canon_weights(D, depth, init, None, next(keys)) if config.canon_c else None,
        canon_d=_init_canon_weights(I, depth, init, None, next(keys)) if config.canon_d else None,
    )


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
        up_proj=_init_weight(next(keys), jax.nn.initializers.uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")),
        down_proj=_init_weight(next(keys), jax.nn.initializers.zeros, (I, D), ("model_intermediate", "model_embed")),
    )


def _init_layer_weights(config, keys, canon_key):
    return LayerWeights(
        attention_weights=_init_attention_weights(config, keys),
        mlp_weights=_init_mlp_weights(config, keys),
        canon=_init_canon_block_weights(config, canon_key) if (config.canon_a or config.canon_b or config.canon_c or config.canon_d) else None,
    )


def init_model_weights(config, key):
    keys = iter(jax.random.split(key, 2 + config.num_layers * 7))
    return ModelWeights(
        embed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=1.0),
            (config.vocab_size, config.hidden_dim),
            ("model_vocab", "model_embed"),
        ),
        layer_weights=[_init_layer_weights(config, keys, next(keys)) for _ in range(config.num_layers)],
        unembed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=0.001),
            (config.hidden_dim, config.vocab_size),
            ("model_embed", "model_vocab"),
        ),
    )


def _attention_with_canon_b(x, weights, canon, rope_cos, rope_sin, qk_norm_epsilon, num_heads, num_kv_heads, mask, dtype):
    """Attention with Canon-B applied after Q/K/V projections, before reshape."""
    batch, seq_len, _ = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads

    q = jnp.matmul(x, weights.q_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q")))
    if canon is not None and canon.canon_b_q is not None:
        q = canon_layer(q, canon.canon_b_q)
    q = q.reshape(batch, seq_len, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))

    k = jnp.matmul(x, weights.k_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
    if canon is not None and canon.canon_b_k is not None:
        k = canon_layer(k, canon.canon_b_k)
    k = k.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    v = jnp.matmul(x, weights.v_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
    if canon is not None and canon.canon_b_v is not None:
        v = canon_layer(v, canon.canon_b_v)
    v = v.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    if rope_cos is not None and rope_sin is not None:
        cos, sin = rope_cos[:, :seq_len, :, :], rope_sin[:, :seq_len, :, :]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    norm_fn = partial(rms_norm, eps=qk_norm_epsilon)
    q = norm_fn(q, None).astype(dtype)
    k = norm_fn(k, None).astype(dtype)

    if mask is not None:
        mask = make_attention_mask(mask)

    att = jax.nn.dot_product_attention(
        query=q, key=k, value=v, is_causal=True,
        implementation="cudnn" if jax.default_backend() == "gpu" else "xla", mask=mask,
    )

    att = att.reshape(batch, seq_len, num_heads * head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    return jnp.matmul(att, weights.o_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_embed")))


def _mlp_with_canon_d(x, weights, canon, act_fn, dtype):
    """MLP with Canon-D applied after up_proj, before activation."""
    h = jnp.matmul(x, weights.up_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
    if canon is not None and canon.canon_d is not None:
        h = canon_layer(h, canon.canon_d)
    h = resolve_act_fn(act_fn)(h)
    return jnp.matmul(h, weights.down_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_embed")))


def model_forward(x, weights, config, rope_cos=None, rope_sin=None, mask=None):
    """Forward pass: pre-norm RMSNorm, RoPE, QK norm, relu_squared MLP, softcap=15.0, bfloat16, Canon layers."""
    eps = config.norm_epsilon
    dtype = jnp.bfloat16

    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(dtype)
    x = rms_norm(x, None, eps)

    for layer_weights in weights.layer_weights:
        canon = layer_weights.canon

        residual = x
        x = rms_norm(x, None, eps)
        if canon is not None and canon.canon_a is not None:
            x = canon_layer(x, canon.canon_a)
        x = _attention_with_canon_b(
            x, layer_weights.attention_weights, canon,
            rope_cos=rope_cos, rope_sin=rope_sin, qk_norm_epsilon=eps,
            num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, mask=mask, dtype=dtype,
        )
        x = x + residual

        residual = x
        x = rms_norm(x, None, eps)
        if canon is not None and canon.canon_c is not None:
            x = canon_layer(x, canon.canon_c)
        x = _mlp_with_canon_d(x, layer_weights.mlp_weights, canon, act_fn="relu_squared", dtype=dtype)
        x = x + residual

    x = rms_norm(x, None, eps)
    logits = jnp.matmul(x, weights.unembed.astype(dtype), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)
