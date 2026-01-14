"""Layer-sharded Muon optimizer for 2D matrices.

Implements Strategy #3 from Essential AI: layer sharding via all_to_all
to efficiently parallelize Newton-Schulz iterations across devices.
"""
from typing import NamedTuple, Union
from functools import partial
import math

import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics, transform
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map


class MuonState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array


def _newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    a = x @ x.swapaxes(-2, -1)
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


def _orthogonalize_single(x: jax.Array, ns_coeffs: jax.Array, ns_steps: int, eps: float) -> jax.Array:
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.swapaxes(-2, -1)
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: _newton_schulz_iteration(x, ns_coeffs), x, unroll=True)
    return x.swapaxes(-2, -1) if transposed else x


def _group_by_shape_and_sharding(params):
    """Group 2D arrays by (shape, sharding_spec). Returns dict: group_key -> [(path, array)]."""
    groups = {}
    
    def collect(path, leaf):
        key = (leaf.shape, jax.typeof(leaf).sharding)
        if key not in groups:
            groups[key] = []
        groups[key].append((path, leaf))
    
    jax.tree_util.tree_map_with_path(collect, params)
    return groups


def _get_sharded_axis(spec):
    """Find which axis (0 or 1) is sharded, returns None if neither."""
    if spec is None:
        return None
    for i, s in enumerate(spec):
        if s is not None:
            return i
    return None


def _layer_shard_orthogonalize(stacked: jax.Array, ns_coeffs: jax.Array, ns_steps: int, eps: float) -> jax.Array:
    """Apply layer-sharded NS orthogonalization to stacked tensor (L, P, Q)."""
    sharding = jax.typeof(stacked).sharding
    mesh = sharding.mesh
    spec = sharding.spec
    sharded_axis = _get_sharded_axis(spec[1:])  # Check P, Q dims (skip L)
    
    if sharded_axis is None:
        return _orthogonalize_single(stacked, ns_coeffs, ns_steps, eps)
    
    sharded_axis += 1  # Adjust for L dimension at front
    axis_name = spec[sharded_axis]
    axis_size = mesh.shape[axis_name]
    num_layers = stacked.shape[0]
    
    padded_layers = ((num_layers + axis_size - 1) // axis_size) * axis_size
    pad_size = padded_layers - num_layers
    if pad_size > 0:
        pad_width = [(0, pad_size)] + [(0, 0)] * (stacked.ndim - 1)
        stacked = jnp.pad(stacked, pad_width)
    
    in_spec = list(spec)
    out_spec = list(spec)
    in_spec[0] = None
    out_spec[0] = None
    
    def ns_kernel(x, ns_coeffs):
        x = jax.lax.all_to_all(x, axis_name, split_axis=0, concat_axis=sharded_axis, tiled=True)
        
        transposed = x.shape[-2] > x.shape[-1]
        if transposed:
            x = x.swapaxes(-2, -1)
        x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
        ns_coeffs_typed = ns_coeffs.astype(x.dtype)
        x = jax.lax.fori_loop(0, ns_steps, lambda _, x: _newton_schulz_iteration(x, ns_coeffs_typed), x, unroll=True)
        if transposed:
            x = x.swapaxes(-2, -1)
        
        x = jax.lax.all_to_all(x, axis_name, split_axis=sharded_axis, concat_axis=0, tiled=True)
        return x
    
    sharded_ns = shard_map(
        ns_kernel,
        mesh=mesh,
        in_specs=(P(*in_spec), P()),
        out_specs=P(*out_spec),
        check_rep=False,
    )
    
    result = sharded_ns(stacked, ns_coeffs)
    return result[:num_layers] if pad_size > 0 else result


def orthogonalize_layer_sharded(params, ns_coeffs: jax.Array, ns_steps: int, eps: float):
    """Apply layer-sharded orthogonalization to all 2D params."""
    groups = _group_by_shape_and_sharding(params)
    
    if not groups:
        return params
    
    results = {}
    for group_key, items in groups.items():
        paths, arrays = zip(*items)
        stacked = jnp.stack(arrays, axis=0)
        orthogonalized = _layer_shard_orthogonalize(stacked, ns_coeffs, ns_steps, eps)
        for i, path in enumerate(paths):
            results[path] = orthogonalized[i]
    
    def reconstruct(path, leaf):
        if path in results:
            return results[path]
        return leaf
    
    return jax.tree_util.tree_map_with_path(reconstruct, params)


def scale_by_muon(
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
    layer_sharding: bool = True,
) -> base.GradientTransformation:
    
    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=jnp.asarray(ns_coeffs),
        )
    
    def update_fn(updates, state, params=None):
        del params
        
        mu = jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, state.mu, updates)
        count_inc = numerics.safe_increment(state.count)
        bias_correction = 1 - beta ** count_inc
        
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: beta * (m / bias_correction) + (1 - beta) * (g / bias_correction),
                mu, updates
            )
        else:
            mu_hat = jax.tree.map(lambda m: m / bias_correction, mu)
        
        if layer_sharding:
            orthogonalized = orthogonalize_layer_sharded(mu_hat, state.ns_coeffs, ns_steps, eps)
        else:
            from jax.sharding import auto_axes
            @auto_axes
            def ortho_single(x, ns_coeffs):
                return _orthogonalize_single(x, ns_coeffs, ns_steps, eps)
            orthogonalized = jax.tree.map(
                lambda x: ortho_single(x, state.ns_coeffs, out_sharding=jax.typeof(x).sharding),
                mu_hat
            )
        
        def apply_shape_factor(x):
            factor = math.sqrt(max(1, x.shape[1] / x.shape[0]))
            return x * factor
        
        updates = jax.tree.map(apply_shape_factor, orthogonalized)
        
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
    layer_sharding: bool = True,
) -> base.GradientTransformation:
    return optax.chain(
        scale_by_muon(
            ns_coeffs=ns_coeffs,
            ns_steps=ns_steps,
            beta=beta,
            eps=eps,
            nesterov=nesterov,
            layer_sharding=layer_sharding,
        ),
        transform.add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(learning_rate),
    )
