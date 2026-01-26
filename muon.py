import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P
from optax._src import base, numerics, transform


class MuonState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array


def _newton_schulz_iteration(x, coeffs):
    a = x @ x.swapaxes(-2, -1)
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


def orthogonalize(x, ns_coeffs, ns_steps=5, eps=1e-8):
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.swapaxes(-2, -1)
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: _newton_schulz_iteration(x, ns_coeffs), x, unroll=True)
    return x.swapaxes(-2, -1) if transposed else x


def layer_shard_orthogonalize(stacked, ns_coeffs, ns_steps, eps):
    # stacked: (L, P, Q) either P or Q is sharded
    # find sharded axis
    # if no sharded axis, orthogonalize locally
    # if sharded axis, pad group to nearest multiple of axis size (L)
    # orthogonalize:
    #    all-to-all (L, P@X, Q) -> (L@X, P, Q)
    #    orthogonalize (L@X, P, Q) -> (L@X, P, Q)
    #    all-to-all (L@X, P, Q) -> (L, P@X, Q)
    # unpad after orthogonalization
    sharding = jax.typeof(stacked).sharding
    mesh, spec = sharding.mesh, sharding.spec
    sharded_axis = next((i + 1 for i, s in enumerate(spec[1:]) if s is not None), None)
    if sharded_axis is None:
        return orthogonalize(stacked, ns_coeffs, ns_steps, eps)

    axis_name, axis_size = spec[sharded_axis], mesh.shape[spec[sharded_axis]]
    num_layers = stacked.shape[0]
    pad_size = ((num_layers + axis_size - 1) // axis_size) * axis_size - num_layers
    if pad_size > 0:
        stacked = jnp.pad(stacked, [(0, pad_size)] + [(0, 0)] * (stacked.ndim - 1))

    spec_list = list(spec)
    spec_list[0] = None

    @jax.shard_map(in_specs=(P(*spec_list), P(), P(), P()), out_specs=P(*spec_list))
    def orthgonalise_group(x, ns_coeffs, ns_steps, eps):
        # all to all (sharded on layer axis)
        x = jax.lax.all_to_all(x, axis_name, split_axis=0, concat_axis=sharded_axis, tiled=True)
        # orthogonalize (local)
        x = orthogonalize(x, ns_coeffs, ns_steps, eps)
        # all to all (sharded on original axis)
        x = jax.lax.all_to_all(x, axis_name, split_axis=sharded_axis, concat_axis=0, tiled=True)
        return x

    result = orthgonalise_group(stacked, jnp.asarray(ns_coeffs), jnp.asarray(ns_steps), jnp.asarray(eps))
    return result[:num_layers] if pad_size > 0 else result


def orthogonalize_layer_sharded(params, ns_coeffs, ns_steps, eps):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    # group parameters by shape and sharding
    groups = {}
    for pos, leaf in enumerate(leaves):
        key = (leaf.shape, jax.typeof(leaf).sharding)
        if key not in groups:
            groups[key] = []
        groups[key].append((pos, leaf))
    # for each group, stack, orthogonalize and put back at original positions
    results = [None] * len(leaves)
    for _, items in groups.items():
        positions, arrays = zip(*items)
        stacked = jnp.stack(arrays, axis=0)
        orthogonalized = layer_shard_orthogonalize(stacked, ns_coeffs, ns_steps, eps)
        for i, pos in enumerate(positions):
            results[pos] = orthogonalized[i]
    return treedef.unflatten(results)


def scale_by_muon(
    ns_coeffs=(3.4445, -4.7750, 2.0315), ns_steps=5, beta=0.95, eps=1e-8, nesterov=True, layer_sharding=True, adjust_lr_fn="original"
):
    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)
        return MuonState(count=jnp.zeros([], jnp.int32), mu=mu, ns_coeffs=jnp.asarray(ns_coeffs))

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree.map(lambda m, g: beta * m + (1 - beta) * g, state.mu, updates)
        count_inc = numerics.safe_increment(state.count)
        bias_correction = 1 - beta**count_inc
        # apply momentum
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: beta * (m / bias_correction) + (1 - beta) * (g / bias_correction), mu, updates
            )
        else:
            mu_hat = jax.tree.map(lambda m: m / bias_correction, mu)
        # orthogonalize
        if layer_sharding:
            orthogonalized = orthogonalize_layer_sharded(mu_hat, state.ns_coeffs, ns_steps, eps)
        else:
            # use auto_axes since X @ X.T is ambiguous
            # but XLA does this inefficiently with multiple all gathers for each matrix and iteration
            # hence layer sharding approach above
            orthogonalized = jax.tree.map(
                lambda x: jax.sharding.auto_axes(orthogonalize)(
                    x, state.ns_coeffs, ns_steps, eps, out_sharding=jax.typeof(x).sharding
                ),
                mu_hat,
            )
        # apply shape factor
        if adjust_lr_fn == "original":
            updates = jax.tree.map(lambda x: x * math.sqrt(max(1, x.shape[1] / x.shape[0])), orthogonalized)
        elif adjust_lr_fn == "match_rms_adamw":
            # https://arxiv.org/pdf/2502.16982v1
            # transfer adamw tuned lr
            updates = jax.tree.map(lambda x: x * 0.2 * math.sqrt(max(x.shape[0], x.shape[1])), orthogonalized)
        return updates, MuonState(count=count_inc, mu=mu, ns_coeffs=state.ns_coeffs)

    return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate,
    ns_coeffs=(3.4445, -4.7750, 2.0315),
    ns_steps=5,
    beta=0.95,
    eps=1e-8,
    weight_decay=0.0,
    nesterov=True,
    layer_sharding=True,
    adjust_lr_fn="original"
):
    """Muon optimizer."""
    return optax.chain(
        scale_by_muon(
            ns_coeffs=ns_coeffs,
            ns_steps=ns_steps,
            beta=beta,
            eps=eps,
            nesterov=nesterov,
            layer_sharding=layer_sharding,
            adjust_lr_fn=adjust_lr_fn
        ),
        transform.add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(learning_rate),
    )
