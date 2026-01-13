import jax
from jax import numpy as jnp


def precompute_rope_embeddings(seq_len: int, head_dim: int, base: float, dtype: str):
    dtype = getattr(jnp, dtype)
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    cos, sin = cos.astype(dtype), sin.astype(dtype)
    # Shape: [1, seq_len, 1, head_dim//2] for broadcasting with [B, L, N, H]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin


def apply_rope(inputs: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    H = inputs.shape[-1] // 2
    x1, x2 = inputs[..., :H], inputs[..., H:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1).astype(inputs.dtype)
