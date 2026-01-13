"""Muon optimizer with layer sharding for efficient large-scale training.

Based on the Muon optimizer by Keller Jordan and the layer sharding strategy
described by Essential AI for scaling Muon to large training runs.

Layer sharding exploits the fact that Newton-Schulz iterations have no inter-layer
dependencies. By resharding from (L, P/S, Q) to (L/S, P, Q) using all-to-all,
we reduce communication from ~6 all-gathers to just 2 all-to-alls.

Reference: https://essential.ai/blog/muon-layer-sharding
"""
from typing import NamedTuple, Union, Dict, List, Tuple, Any, Optional
import math

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P, Mesh
import optax
from optax._src import base, numerics, transform


class MuonState(NamedTuple):
    """State for the Muon algorithm."""
    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array


def _newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    """Single Newton-Schulz iteration for 2D matrix.
    
    Computes: f(X) = c0*X + c1*(XX^T)*X + c2*(XX^T)^2*X
    Rewritten as: f(X) = c0*X + (c1*A + c2*A@A)@X where A = XX^T
    """
    a = x @ x.T
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x


def _newton_schulz_iteration_batched(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    """Single Newton-Schulz iteration for batched 3D tensor.
    
    Args:
        x: Shape (L, P, Q) - batched matrices
        coeffs: Shape (3,) - NS coefficients
    
    Returns:
        Shape (L, P, Q) - updated matrices
    """
    # A = X @ X^T: (L, P, Q) @ (L, Q, P) -> (L, P, P)
    # einsum: contract over q (the Q dimension)
    a = jnp.einsum('lpq,lkq->lpk', x, x)
    # A @ A: (L, P, P) @ (L, P, P) -> (L, P, P)
    aa = jnp.einsum('lpk,lkr->lpr', a, a)
    # B = c1*A + c2*A@A
    b = coeffs[1] * a + coeffs[2] * aa
    # B @ X: (L, P, P) @ (L, P, Q) -> (L, P, Q)
    # einsum: contract over the inner P dimension
    bx = jnp.einsum('lpk,lkq->lpq', b, x)
    return coeffs[0] * x + bx


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


def _orthogonalize_layer_sharded_inner(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int,
    eps: float,
    mesh_axis: str,
) -> jax.Array:
    """Inner function for layer-sharded orthogonalization (runs inside shard_map).
    
    This function is called from within shard_map where the mesh axis is bound.
    """
    L, P, Q = x.shape
    
    # Transpose if P > Q for efficiency (smaller intermediate matrices)
    transposed = P > Q
    if transposed:
        x = jnp.swapaxes(x, -2, -1)  # (L, Q, P)
        P, Q = Q, P
    
    # Normalize each matrix to ensure spectral norm <= 1
    # Compute Frobenius norm per matrix
    norms = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
    x = x / (norms + eps)
    
    # All-to-all: reshard from P-sharding to L-sharding
    # (L, P/S, Q) -> (L/S, P, Q)
    # split_axis=1 splits along P, concat_axis=0 concatenates along L
    x = jax.lax.all_to_all(x, mesh_axis, split_axis=1, concat_axis=0, tiled=True)
    
    # Newton-Schulz iterations (now fully local compute on each device)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(
        0, ns_steps,
        lambda _, x: _newton_schulz_iteration_batched(x, ns_coeffs),
        x,
        unroll=True
    )
    
    # All-to-all: reshard back from L-sharding to P-sharding
    # (L/S, P, Q) -> (L, P/S, Q)
    # split_axis=0 splits along L, concat_axis=1 concatenates along P
    x = jax.lax.all_to_all(x, mesh_axis, split_axis=0, concat_axis=1, tiled=True)
    
    if transposed:
        x = jnp.swapaxes(x, -2, -1)  # (L, P, Q)
    
    return x


def _orthogonalize_batched_no_alltoall(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int,
    eps: float,
) -> jax.Array:
    """Batched orthogonalization without all-to-all (for single device)."""
    L, P, Q = x.shape
    
    # Transpose if P > Q for efficiency
    transposed = P > Q
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
        P, Q = Q, P
    
    # Normalize each matrix
    norms = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
    x = x / (norms + eps)
    
    # Newton-Schulz iterations
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(
        0, ns_steps,
        lambda _, x: _newton_schulz_iteration_batched(x, ns_coeffs),
        x,
        unroll=True
    )
    
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    
    return x


def orthogonalize_layer_sharded(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
    mesh: Optional[Mesh] = None,
    mesh_axis: str = "data",
) -> jax.Array:
    """Orthogonalize batched matrices using layer sharding strategy.
    
    This function uses shard_map to reshard from FSDP's P-sharding to layer-sharding
    using all-to-all, performs Newton-Schulz iterations locally, then reshards back.
    
    Input sharding: (L, P/S, Q) - P dimension sharded across S devices
    After first all-to-all: (L/S, P, Q) - L dimension sharded, P is full
    After NS iterations: (L/S, P, Q)
    After second all-to-all: (L, P/S, Q) - back to original sharding
    
    On single-device or when mesh is None, falls back to batched orthogonalization
    without all-to-all.
    
    Args:
        x: 3D tensor of shape (L, P, Q) - batched matrices across layers
        ns_coeffs: Newton-Schulz coefficients (3,)
        ns_steps: Number of NS iterations
        eps: Small constant for numerical stability
        mesh: JAX Mesh for distributed computation. If None, uses batched computation
              without all-to-all.
        mesh_axis: Name of the mesh axis for all-to-all communication
        
    Returns:
        Orthogonalized tensor with same shape as input
    """
    # On single device or no mesh, skip shard_map and just do batched computation
    if mesh is None or jax.device_count() == 1:
        return _orthogonalize_batched_no_alltoall(x, ns_coeffs, ns_steps, eps)
    
    # Use shard_map to properly bind the mesh axis for all_to_all
    # Input is sharded on P dimension (axis 1), output same sharding
    sharded_fn = shard_map(
        lambda x: _orthogonalize_layer_sharded_inner(x, ns_coeffs, ns_steps, eps, mesh_axis),
        mesh=mesh,
        in_specs=(P(None, mesh_axis, None),),  # (L, P/S, Q) - P sharded
        out_specs=P(None, mesh_axis, None),     # (L, P/S, Q) - P sharded
        check_rep=False,  # Allow replicated intermediates
    )
    
    return sharded_fn(x)


def _get_flat_params(params: base.Params) -> Tuple[List[Tuple[Any, jax.Array]], Any]:
    """Flatten params tree to list of (path, array) pairs.
    
    Returns:
        Tuple of (flat list of (path, array), tree structure for reconstruction)
    """
    flat_with_paths = list(jax.tree_util.tree_leaves_with_path(params))
    tree_struct = jax.tree_util.tree_structure(params)
    return flat_with_paths, tree_struct


def _reconstruct_tree(flat_values: List[jax.Array], paths: List[Any], tree_struct: Any) -> base.Params:
    """Reconstruct params tree from flat values and paths."""
    # Create a dict mapping paths to values
    path_to_value = dict(zip(paths, flat_values))
    # Use tree_unflatten with the values in the correct order
    leaves = [path_to_value[p] for p in paths]
    return jax.tree_util.tree_unflatten(tree_struct, leaves)


def batch_params_by_shape(
    params: base.Updates
) -> Tuple[Dict[Tuple[int, int], jax.Array], Dict[Tuple[int, int], List[Any]], List[Any], Any]:
    """Group parameters by shape and stack them along a layer axis.
    
    Args:
        params: Parameter tree (typically momentum values)
        
    Returns:
        Tuple of:
        - batched: Dict mapping (P, Q) shape to stacked tensor of shape (L, P, Q)
        - shape_to_paths: Dict mapping (P, Q) shape to list of tree paths
        - original_paths: List of paths in original tree traversal order
        - tree_struct: Tree structure for reconstruction
    """
    flat_with_paths, tree_struct = _get_flat_params(params)
    
    # Save original path order for reconstruction
    original_paths = [path for path, _ in flat_with_paths]
    
    # Group by shape
    shape_groups: Dict[Tuple[int, int], List[Tuple[Any, jax.Array]]] = {}
    for path, arr in flat_with_paths:
        if arr.ndim != 2:
            # Non-2D arrays get their own "group" with shape (1, *original_shape)
            # We'll handle them separately
            key = ("non2d", arr.shape)
        else:
            key = arr.shape
        
        if key not in shape_groups:
            shape_groups[key] = []
        shape_groups[key].append((path, arr))
    
    # Stack each group
    batched = {}
    shape_to_paths = {}
    for shape_key, group in shape_groups.items():
        paths = [p for p, _ in group]
        arrays = [a for _, a in group]
        
        if isinstance(shape_key, tuple) and shape_key[0] == "non2d":
            # Keep non-2D arrays as-is (wrapped in list for consistency)
            batched[shape_key] = arrays
        else:
            # Stack 2D arrays along new axis 0
            batched[shape_key] = jnp.stack(arrays, axis=0)
        
        shape_to_paths[shape_key] = paths
    
    return batched, shape_to_paths, original_paths, tree_struct


def unbatch_params(
    batched: Dict[Tuple[int, int], jax.Array],
    shape_to_paths: Dict[Tuple[int, int], List[Any]],
    original_paths: List[Any],
    tree_struct: Any,
) -> base.Updates:
    """Reconstruct parameter tree from batched tensors.
    
    Args:
        batched: Dict mapping shape to batched tensor
        shape_to_paths: Dict mapping shape to list of tree paths
        original_paths: List of paths in original tree traversal order
        tree_struct: Original tree structure
        
    Returns:
        Reconstructed parameter tree
    """
    # Collect all (path, value) pairs
    path_value_pairs = []
    
    for shape_key, stacked in batched.items():
        paths = shape_to_paths[shape_key]
        
        if isinstance(shape_key, tuple) and shape_key[0] == "non2d":
            # Non-2D arrays stored as list
            for path, arr in zip(paths, stacked):
                path_value_pairs.append((path, arr))
        else:
            # Unstack 2D arrays
            unstacked = [stacked[i] for i in range(stacked.shape[0])]
            for path, arr in zip(paths, unstacked):
                path_value_pairs.append((path, arr))
    
    # Create mapping and unflatten using original path order
    path_to_value = dict(path_value_pairs)
    ordered_values = [path_to_value[p] for p in original_paths]
    
    return jax.tree_util.tree_unflatten(tree_struct, ordered_values)


def scale_by_muon(
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
    layer_sharding: bool = False,
    mesh: Optional[Mesh] = None,
    mesh_axis: str = "data",
) -> base.GradientTransformation:
    """Scale updates by the Muon algorithm.
    
    Args:
        ns_coeffs: Coefficients for Newton-Schulz iteration
        ns_steps: Number of NS iterations
        beta: Momentum decay rate
        eps: Small constant for numerical stability
        nesterov: Whether to use Nesterov momentum
        layer_sharding: Whether to use layer sharding strategy for efficient
            distributed computation. When True, parameters are batched by shape
            and shard_map with all-to-all is used to reshard from FSDP to layer sharding.
        mesh: JAX Mesh for distributed computation. Required when layer_sharding=True
            and running on multiple devices.
        mesh_axis: Name of the mesh axis for all-to-all communication
            (only used when layer_sharding=True)
        
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
        
        if layer_sharding:
            # Layer sharding path: batch by shape, apply batched orthogonalization
            orthogonalized = _apply_orthogonalize_layer_sharded(
                mu_hat, state.ns_coeffs, ns_steps, eps, mesh, mesh_axis
            )
        else:
            # Standard path: orthogonalize each matrix independently
            orthogonalized = jax.tree.map(
                lambda x: orthogonalize(x, state.ns_coeffs, ns_steps, eps) if x.ndim == 2 else x,
                mu_hat
            )
        
        # Scale by sqrt(max(1, out_features / in_features))
        def apply_shape_factor(x):
            if x.ndim != 2:
                return x
            factor = math.sqrt(max(1, x.shape[1] / x.shape[0]))
            return x * factor
        
        updates = jax.tree.map(apply_shape_factor, orthogonalized)
        
        return updates, MuonState(
            count=count_inc,
            mu=mu,
            ns_coeffs=state.ns_coeffs,
        )
    
    return base.GradientTransformation(init_fn, update_fn)


def _apply_orthogonalize_layer_sharded(
    params: base.Updates,
    ns_coeffs: jax.Array,
    ns_steps: int,
    eps: float,
    mesh: Optional[Mesh],
    mesh_axis: str,
) -> base.Updates:
    """Apply layer-sharded orthogonalization to all 2D params.
    
    Groups params by shape, applies batched orthogonalization with layer sharding,
    then reconstructs the tree.
    """
    # Batch params by shape
    batched, shape_to_paths, original_paths, tree_struct = batch_params_by_shape(params)
    
    # Apply orthogonalization to each batch
    orthogonalized_batched = {}
    for shape_key, stacked in batched.items():
        if isinstance(shape_key, tuple) and shape_key[0] == "non2d":
            # Skip non-2D arrays
            orthogonalized_batched[shape_key] = stacked
        else:
            # Apply layer-sharded orthogonalization
            orthogonalized_batched[shape_key] = orthogonalize_layer_sharded(
                stacked, ns_coeffs, ns_steps, eps, mesh, mesh_axis
            )
    
    # Reconstruct tree
    return unbatch_params(orthogonalized_batched, shape_to_paths, original_paths, tree_struct)


def muon(
    learning_rate: Union[float, base.Schedule],
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    nesterov: bool = True,
    layer_sharding: bool = False,
    mesh: Optional[Mesh] = None,
    mesh_axis: str = "data",
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
        layer_sharding: Whether to use layer sharding strategy for efficient
            distributed computation. When True, parameters are batched by shape
            and shard_map with all-to-all is used to reshard from FSDP to layer sharding.
            This significantly reduces communication overhead at large scale.
        mesh: JAX Mesh for distributed computation. Required when layer_sharding=True
            and running on multiple devices.
        mesh_axis: Name of the mesh axis for all-to-all communication
            (only used when layer_sharding=True)
        
    Returns:
        A GradientTransformation
        
    Example:
        >>> # Standard usage
        >>> tx = muon(learning_rate=0.02)
        
        >>> # With layer sharding for distributed training
        >>> mesh = jax.make_mesh((8,), ("data",))
        >>> tx = muon(learning_rate=0.02, layer_sharding=True, mesh=mesh, mesh_axis="data")
        
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
            layer_sharding=layer_sharding,
            mesh=mesh,
            mesh_axis=mesh_axis,
        ),
        transform.add_decayed_weights(weight_decay),
        transform.scale_by_learning_rate(learning_rate),
    )
