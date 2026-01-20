from dataclasses import dataclass
from typing import Optional

import jax
from jax import numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class RMSNormWeights:
    scale: Optional[jax.Array] = None


@jax.tree_util.register_dataclass
@dataclass
class LayerNormWeights:
    scale: jax.Array
    bias: Optional[jax.Array] = None


def rms_norm(x, weights, eps):
    """RMSNorm with optional learned scale parameter."""
    x_f32 = x.astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + eps)
    out = x_f32 * rms
    if weights is not None and weights.scale is not None:
        out = out * weights.scale
    return out.astype(x.dtype)


def layer_norm(x, weights, eps):
    """LayerNorm with learned scale and optional bias parameters."""
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.var(x_f32, axis=-1, keepdims=True)
    normalized = (x_f32 - mean) * jax.lax.rsqrt(var + eps)
    normalized = normalized * weights.scale
    if weights.bias is not None:
        normalized = normalized + weights.bias
    return normalized.astype(x.dtype)


def norm(x, weights, eps):
    """Unified norm dispatch based on weights type."""
    if isinstance(weights, RMSNormWeights):
        return rms_norm(x, weights, eps)
    return layer_norm(x, weights, eps)
