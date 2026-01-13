from dataclasses import dataclass
from functools import partial
from typing import Optional, Callable, Union

import jax
from jax import numpy as jnp

from modelling.layers.position import apply_rope
from parallel import logical_to_physical


@jax.tree_util.register_dataclass
@dataclass
class RMSNormWeights:
    scale: Optional[jax.Array] = None


@jax.tree_util.register_dataclass
@dataclass
class LayerNormWeights:
    scale: jax.Array
    bias: Optional[jax.Array] = None


# Union type for norm weights
NormWeights = Union[RMSNormWeights, LayerNormWeights]


@jax.tree_util.register_dataclass
@dataclass
class MLPWeights:
    up_proj: jax.Array
    down_proj: jax.Array
    up_bias: Optional[jax.Array] = None
    down_bias: Optional[jax.Array] = None


@jax.tree_util.register_dataclass
@dataclass
class GLUWeights:
    gate_proj: jax.Array
    up_proj: jax.Array
    down_proj: jax.Array
    gate_bias: Optional[jax.Array] = None
    up_bias: Optional[jax.Array] = None
    down_bias: Optional[jax.Array] = None


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


def rms_norm(x: jax.Array, weights: Optional[RMSNormWeights], eps: float) -> jax.Array:
    """RMSNorm with optional learned scale parameter."""
    x_f32 = x.astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps)
    out = x_f32 * rms
    if weights is not None and weights.scale is not None:
        out = out * weights.scale
    return out.astype(x.dtype)


def layer_norm(x: jax.Array, weights: LayerNormWeights, eps: float) -> jax.Array:
    """LayerNorm with learned scale and optional bias parameters."""
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.var(x_f32, axis=-1, keepdims=True)
    normalized = (x_f32 - mean) * jax.lax.rsqrt(var + eps)
    normalized = normalized * weights.scale
    if weights.bias is not None:
        normalized = normalized + weights.bias
    return normalized.astype(x.dtype)


def resolve_act_fn(act_fn_name: str) -> Callable:
    if act_fn_name == "relu_squared":
        return lambda x: jnp.square(jax.nn.relu(x))
    return getattr(jax.nn, act_fn_name)


def mlp(
    x: jax.Array,
    weights: MLPWeights,
    act_fn: str,
    dtype: str,
) -> jax.Array:
    dtype = getattr(jnp, dtype)
    with jax.named_scope("up_proj"):
        h = jnp.matmul(
            x, weights.up_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_intermediate"))
        )
        if weights.up_bias is not None:
            h = h + weights.up_bias.astype(dtype)
    
    with jax.named_scope("act_fn"):
        h = resolve_act_fn(act_fn)(h)
    
    with jax.named_scope("down_proj"):
        out = jnp.matmul(
            h, weights.down_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_embed"))
        )
        if weights.down_bias is not None:
            out = out + weights.down_bias.astype(dtype)
    
    return out


def glu(
    x: jax.Array,
    weights: GLUWeights,
    act_fn: str,
    dtype: str,
) -> jax.Array:
    dtype = getattr(jnp, dtype)
    with jax.named_scope("up_proj"):
        up = jnp.matmul(
            x, weights.up_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_intermediate"))
        )
        if weights.up_bias is not None:
            up = up + weights.up_bias.astype(dtype)
    
    with jax.named_scope("gate_proj"):
        gate = jnp.matmul(
            x, weights.gate_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_intermediate"))
        )
        if weights.gate_bias is not None:
            gate = gate + weights.gate_bias.astype(dtype)
    
    with jax.named_scope("down_proj"):
        out = jnp.matmul(
            resolve_act_fn(act_fn)(gate) * up, weights.down_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_embed"))
        )
        if weights.down_bias is not None:
            out = out + weights.down_bias.astype(dtype)
    
    return out


def make_attention_mask(mask: jax.Array) -> jax.Array:
    return (mask[:, None, None, :] & mask[:, None, :, None]).astype(jnp.bool_)


def attention(
    x: jax.Array,
    weights: AttentionWeights,
    rope_cos: Optional[jax.Array],
    rope_sin: Optional[jax.Array],
    qk_norm: bool,
    qk_norm_type: Optional[str],
    qk_norm_epsilon: Optional[float],
    sliding_window: Optional[int],
    dtype: str,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    # Weights shapes:
    #   q_proj: (D, N, H) where D=embed, N=num_heads, H=head_dim
    #   k_proj: (D, K, H) where K=num_kv_heads
    #   v_proj: (D, K, H)
    #   o_proj: (N, H, D)
    dtype = getattr(jnp, dtype)
    
    with jax.named_scope("q_proj"):
        q = jnp.einsum(
            "BSD, DNH -> BSNH", x, weights.q_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_q", "head_embed"))
        )
        if weights.q_bias is not None:
            q = q + weights.q_bias.astype(dtype)
    
    with jax.named_scope("k_proj"):
        k = jnp.einsum(
            "BSD, DKH -> BSKH", x, weights.k_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_kv", "head_embed"))
        )
        if weights.k_bias is not None:
            k = k + weights.k_bias.astype(dtype)
    
    with jax.named_scope("v_proj"):
        v = jnp.einsum(
            "BSD, DKH -> BSKH", x, weights.v_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_kv", "head_embed"))
        )
        if weights.v_bias is not None:
            v = v + weights.v_bias.astype(dtype)
    
    if rope_cos is not None and rope_sin is not None:
        with jax.named_scope("apply_rope"):
            seq_len = x.shape[1]
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
            query=q, key=k, value=v,
            is_causal=True,
            implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
            mask=mask,
            local_window_size=(sliding_window, 0) if sliding_window else None
        )
    
    with jax.named_scope("o_proj"):
        out = jnp.einsum(
            "BSNH, NHD -> BSD", att, weights.o_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_embed"))
        )
        if weights.o_bias is not None:
            out = out + weights.o_bias.astype(dtype)
    
    return out
