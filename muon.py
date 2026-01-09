"""Simplified Muon optimizer for 2D matrices only.

Based on the Muon optimizer by Keller Jordan:
https://github.com/KellerJordan/modded-nanogpt

This is a minimal implementation that assumes all parameters are 2D matrices.
"""
from typing import NamedTuple, Union
import math

import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics, transform, utils


class MuonState(NamedTuple):
    """State for the Muon algorithm."""
    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array


def _newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    """Single Newton-Schulz iteration.
    
    Computes: f(X) = c0*X + c1*(XX^T)*X + c2*(XX^T)^2*X
    Rewritten as: f(X) = c0*X + (c1*A + c2*A@A)@X where A = XX^T
    """
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


from jax.sharding import auto_axes
@auto_axes
def orthogonalize(x: jax.Array, ns_coeffs: jax.Array, ns_steps: int = 5, eps: float = 1e-8) -> jax.Array:
    """Orthogonalize a 2D matrix via Newton-Schulz iteration.
    
    Args:
        x: 2D matrix to orthogonalize
        ns_coeffs: Newton-Schulz coefficients (3,)
        ns_steps: Number of NS iterations
        eps: Small constant for numerical stability
        
    Returns:
        Orthogonalized matrix with same shape as input
    """
    # Transpose if rows > cols for efficiency (smaller intermediate matrices)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    
    # Normalize to ensure spectral norm <= 1
    x = x / (jnp.linalg.norm(x) + eps)
    
    # Newton-Schulz iterations
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(
        0, ns_steps, 
        lambda _, x: _newton_schulz_iteration(x, ns_coeffs), 
        x,
        unroll=True
    )
    
    return x.T if transposed else x


def scale_by_muon(
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
) -> base.GradientTransformation:
    """Scale updates by the Muon algorithm.
    
    Args:
        ns_coeffs: Coefficients for Newton-Schulz iteration
        ns_steps: Number of NS iterations
        beta: Momentum decay rate
        eps: Small constant for numerical stability
        nesterov: Whether to use Nesterov momentum
        
    Returns:
        A GradientTransformation
    """
    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=jnp.asarray(ns_coeffs),
        )
    
    def update_fn(updates, state, params=None):
        del params
        
        # Update momentum: mu = beta * mu + (1 - beta) * g
        mu = jax.tree.map(
            lambda m, g: beta * m + (1 - beta) * g,
            state.mu, updates
        )
        count_inc = numerics.safe_increment(state.count)
        
        # Bias correction
        bias_correction = 1 - beta ** count_inc
        
        if nesterov:
            # Nesterov momentum: use interpolation of corrected mu and g
            mu_hat = jax.tree.map(
                lambda m, g: beta * (m / bias_correction) + (1 - beta) * (g / bias_correction),
                mu, updates
            )
        else:
            mu_hat = jax.tree.map(lambda m: m / bias_correction, mu)
        
        # Apply Newton-Schulz orthogonalization to each 2D matrix
        orthogonalized = jax.tree.map(
            lambda x: orthogonalize(x, state.ns_coeffs, ns_steps, eps, out_sharding=jax.typeof(x).sharding),
            mu_hat
        )
        
        # Scale by sqrt(max(1, out_features / in_features))
        def apply_shape_factor(x):
            factor = math.sqrt(max(1, x.shape[1] / x.shape[0]))
            return x * factor
        
        updates = jax.tree.map(apply_shape_factor, orthogonalized)
        
        return updates, MuonState(
            count=count_inc,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )
    
    return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: Union[float, base.Schedule],
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    nesterov: bool = True,
) -> base.GradientTransformation:
    """Muon optimizer: Momentum Orthogonalized by Newton-Schulz.
    
    A simplified optimizer for 2D matrices that applies Newton-Schulz
    orthogonalization to the momentum.
    
    Args:
        learning_rate: Learning rate (float or schedule)
        ns_coeffs: Coefficients for Newton-Schulz iteration
        ns_steps: Number of NS iterations (default 5)
        beta: Momentum decay rate (default 0.95)
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient
        nesterov: Whether to use Nesterov momentum
        
    Returns:
        A GradientTransformation
        
    Example:
        >>> tx = muon(learning_rate=0.02)
        >>> state = tx.init(params)
        >>> updates, state = tx.update(grads, state, params)
        >>> params = optax.apply_updates(params, updates)
    """
    return optax.chain(
        scale_by_muon(
            ns_coeffs=ns_coeffs,
            ns_steps=ns_steps,
            beta=beta,
            eps=eps,
            nesterov=nesterov,
        ),
        transform.add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(learning_rate),
    )
