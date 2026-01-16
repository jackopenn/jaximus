from dataclasses import dataclass

import jax
from jax import numpy as jnp

from parallel import logical_to_physical


@jax.tree_util.register_dataclass
@dataclass
class CanonWeights:
    """Weights for a Canon layer with window size 4.

    Each weight has shape (m,) where m is the feature dimension.
    """
    w0: jax.Array  # weight for h_t (current position)
    w1: jax.Array  # weight for h_{t-1}
    w2: jax.Array  # weight for h_{t-2}
    w3: jax.Array  # weight for h_{t-3}


def canon_layer(x: jax.Array, weights: CanonWeights) -> jax.Array:
    """Apply Canon layer with window size 4 and residual connection.

    Computes: h'_t = h_t + conv1d([h_t, h_{t-1}, h_{t-2}, h_{t-3}])
            = h_t + (w0 ⊙ h_t + w1 ⊙ h_{t-1} + w2 ⊙ h_{t-2} + w3 ⊙ h_{t-3})

    Args:
        x: Input tensor of shape (batch, seq_len, m)
        weights: CanonWeights with w0, w1, w2, w3 each of shape (m,)

    Returns:
        Output tensor of shape (batch, seq_len, m)
    """
    dtype = x.dtype
    w0 = weights.w0.astype(dtype)
    w1 = weights.w1.astype(dtype)
    w2 = weights.w2.astype(dtype)
    w3 = weights.w3.astype(dtype)

    # Pad with zeros at the start for boundary conditions
    # x_padded: (batch, seq_len + 3, m)
    x_padded = jnp.pad(x, ((0, 0), (3, 0), (0, 0)), mode='constant')

    # Compute weighted sum using slices
    # h0 corresponds to positions 3: (current h_t)
    # h1 corresponds to positions 2:-1 (h_{t-1})
    # h2 corresponds to positions 1:-2 (h_{t-2})
    # h3 corresponds to positions :-3 (h_{t-3})
    h0 = x_padded[:, 3:, :] * w0
    h1 = x_padded[:, 2:-1, :] * w1
    h2 = x_padded[:, 1:-2, :] * w2
    h3 = x_padded[:, :-3, :] * w3

    # Residual connection
    return x + h0 + h1 + h2 + h3


def init_canon_weights(
    dim: int,
    init_strategy: str,
    sharding: tuple,
    key: jax.random.PRNGKey,
) -> CanonWeights:
    """Initialize Canon weights.

    Args:
        dim: Feature dimension (m)
        init_strategy: One of "identity", "ones", "normal"
        sharding: Logical sharding tuple for the weights
        key: Random key for initialization

    Returns:
        Initialized CanonWeights
    """
    physical_sharding = logical_to_physical(sharding)

    if init_strategy == "identity":
        # All zeros makes canon_layer an identity function (x + 0 = x)
        w0 = jnp.zeros((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
        w1 = jnp.zeros((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
        w2 = jnp.zeros((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
        w3 = jnp.zeros((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
    elif init_strategy == "ones":
        w0 = jnp.ones((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
        w1 = jnp.ones((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
        w2 = jnp.ones((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
        w3 = jnp.ones((dim,), dtype=jnp.float32, out_sharding=physical_sharding)
    elif init_strategy == "normal":
        keys = jax.random.split(key, 4)
        w0 = jax.random.normal(keys[0], (dim,), dtype=jnp.float32) * 0.02
        w1 = jax.random.normal(keys[1], (dim,), dtype=jnp.float32) * 0.02
        w2 = jax.random.normal(keys[2], (dim,), dtype=jnp.float32) * 0.02
        w3 = jax.random.normal(keys[3], (dim,), dtype=jnp.float32) * 0.02
        # Apply sharding after random init
        w0 = jax.lax.with_sharding_constraint(w0, physical_sharding)
        w1 = jax.lax.with_sharding_constraint(w1, physical_sharding)
        w2 = jax.lax.with_sharding_constraint(w2, physical_sharding)
        w3 = jax.lax.with_sharding_constraint(w3, physical_sharding)
    else:
        raise ValueError(f"Unknown canon init strategy: {init_strategy}")

    return CanonWeights(w0=w0, w1=w1, w2=w2, w3=w3)
