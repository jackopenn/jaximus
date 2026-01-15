from typing import NamedTuple, Union
import math

import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics, transform


class MuonState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array


def _newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


from jax.sharding import auto_axes
@auto_axes
def orthogonalize(x: jax.Array, ns_coeffs: jax.Array, ns_steps: int = 5, eps: float = 1e-8) -> jax.Array:
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: _newton_schulz_iteration(x, ns_coeffs), x, unroll=True)
    return x.T if transposed else x


def scale_by_muon(
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
) -> base.GradientTransformation:
    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        return MuonState(count=jnp.zeros([], jnp.int32), mu=mu, ns_coeffs=jnp.asarray(ns_coeffs))
    
    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, state.mu, updates)
        count_inc = numerics.safe_increment(state.count)
        bias_correction = 1 - beta ** count_inc
        if nesterov:
            mu_hat = jax.tree.map(lambda m, g: beta * (m / bias_correction) + (1 - beta) * (g / bias_correction), mu, updates)
        else:
            mu_hat = jax.tree.map(lambda m: m / bias_correction, mu)
        # out sharding needed for 'Explicit sharding' since X @ X.T is ambiguous
        orthogonalized = jax.tree.map(lambda x: orthogonalize(x, state.ns_coeffs, ns_steps, eps, out_sharding=jax.typeof(x).sharding), mu_hat)
        # apply shape factor
        updates = jax.tree.map(lambda x: x * math.sqrt(max(1, x.shape[1] / x.shape[0])), orthogonalized)
        return updates, MuonState(count=count_inc, mu=mu, ns_coeffs=state.ns_coeffs)
    
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
