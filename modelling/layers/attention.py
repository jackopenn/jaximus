from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import jax
from jax import numpy as jnp

from modelling.layers.norm import LayerNormWeights, RMSNormWeights, layer_norm, rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class AttentionWeights:
    q_proj: jax.Array
    k_proj: jax.Array
    v_proj: jax.Array
    o_proj: jax.Array
    q_bias: Optional[jax.Array] = None
    k_bias: Optional[jax.Array] = None
    v_bias: Optional[jax.Array] = None
    o_bias: Optional[jax.Array] = None
    q_norm: Optional[Union[RMSNormWeights, LayerNormWeights]] = None
    k_norm: Optional[Union[RMSNormWeights, LayerNormWeights]] = None


def make_attention_mask(mask):
    return (mask[:, None, None, :] & mask[:, None, :, None]).astype(jnp.bool_)


def attention(
    x,
    weights,
    rope_cos,
    rope_sin,
    qk_norm,
    qk_norm_type,
    qk_norm_epsilon,
    sliding_window,
    dtype,
    num_heads,
    num_kv_heads,
    mask=None,
):
    # Weights shapes (2D):
    #   q_proj: (D, N*H) where D=embed, N=num_heads, H=head_dim
    #   k_proj: (D, K*H) where K=num_kv_heads
    #   v_proj: (D, K*H)
    #   o_proj: (N*H, D)
    dtype = getattr(jnp, dtype)
    batch, seq_len, _ = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads

    with jax.named_scope("q_proj"):
        q = jnp.matmul(x, weights.q_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q")))
        if weights.q_bias is not None:
            q = q + weights.q_bias.astype(dtype)
        q = q.reshape(batch, seq_len, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))

    with jax.named_scope("k_proj"):
        k = jnp.matmul(x, weights.k_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
        if weights.k_bias is not None:
            k = k + weights.k_bias.astype(dtype)
        k = k.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    with jax.named_scope("v_proj"):
        v = jnp.matmul(x, weights.v_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
        if weights.v_bias is not None:
            v = v + weights.v_bias.astype(dtype)
        v = v.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    if rope_cos is not None and rope_sin is not None:
        with jax.named_scope("apply_rope"):
            cos = rope_cos[:, :seq_len, :, :]
            sin = rope_sin[:, :seq_len, :, :]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

    if qk_norm:
        norm_factory = partial(
            rms_norm if qk_norm_type == "rms" else layer_norm,
            eps=qk_norm_epsilon,
        )
        with jax.named_scope("q_norm"):
            q = norm_factory(q, weights.q_norm).astype(dtype)
        with jax.named_scope("k_norm"):
            k = norm_factory(k, weights.k_norm).astype(dtype)

    if mask is not None:
        mask = make_attention_mask(mask)

    with jax.named_scope("dot_product_attention"):
        att = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=True,
            implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
            mask=mask,
            local_window_size=(sliding_window, 0) if sliding_window else None,
        )

    with jax.named_scope("o_proj"):
        att = att.reshape(batch, seq_len, num_heads * head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
        out = jnp.matmul(att, weights.o_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_embed")))
        if weights.o_bias is not None:
            out = out + weights.o_bias.astype(dtype)

    return out
