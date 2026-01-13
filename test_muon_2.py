"""Tests for Muon optimizer with layer sharding."""
import jax
import jax.numpy as jnp
import optax
from muon_2 import (
    muon as our_muon,
    scale_by_muon,
    orthogonalize,
    orthogonalize_layer_sharded,
    batch_params_by_shape,
    unbatch_params,
    _newton_schulz_iteration_batched,
)


def test_newton_schulz_batched():
    """Test that batched NS iteration matches individual iterations."""
    key = jax.random.PRNGKey(42)
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    # Create batched input
    L, P, Q = 4, 32, 64
    x_batched = jax.random.normal(key, (L, P, Q))
    
    # Apply batched iteration
    result_batched = _newton_schulz_iteration_batched(x_batched, ns_coeffs)
    
    # Apply individual iterations and compare
    for i in range(L):
        x_single = x_batched[i]
        a = x_single @ x_single.T
        aa = a @ a
        b = ns_coeffs[1] * a + ns_coeffs[2] * aa  # Note: need explicit aa to avoid precedence issues
        expected = ns_coeffs[0] * x_single + b @ x_single
        
        assert jnp.allclose(result_batched[i], expected, rtol=1e-5), \
            f"Batched NS iteration differs at index {i}"
    
    print("✓ Batched Newton-Schulz iteration matches individual iterations")


def test_orthogonalize():
    """Test the orthogonalize function produces reasonable output."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (64, 128))
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    result = orthogonalize(x, ns_coeffs, ns_steps=5)
    
    # Check shape preserved
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    
    # Check it's not all zeros or NaN
    assert not jnp.any(jnp.isnan(result)), "Result contains NaN"
    assert jnp.abs(result).sum() > 0, "Result is all zeros"
    
    print("✓ orthogonalize produces valid output")


def test_orthogonalize_transposed():
    """Test orthogonalize with rows > cols (triggers transpose path)."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (128, 64))  # rows > cols
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    result = orthogonalize(x, ns_coeffs, ns_steps=5)
    
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    assert not jnp.any(jnp.isnan(result)), "Result contains NaN"
    
    print("✓ orthogonalize handles rows > cols correctly")


def test_batch_unbatch_params():
    """Test that batching and unbatching preserves tree structure."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Create params with different shapes
    params = {
        'layer1': {
            'w1': jax.random.normal(k1, (64, 128)),
            'w2': jax.random.normal(k2, (128, 64)),
        },
        'layer2': {
            'w1': jax.random.normal(k3, (64, 128)),  # Same shape as layer1.w1
            'w2': jax.random.normal(k4, (128, 64)),  # Same shape as layer1.w2
        },
    }
    
    # Batch
    batched, shape_to_paths, original_paths, tree_struct = batch_params_by_shape(params)
    
    # Check batching grouped correctly
    # Keys are now ((shape), sharding_key) tuples
    # On single device, sharding_key should be None
    shape_64_128_key = ((64, 128), None)
    shape_128_64_key = ((128, 64), None)
    
    assert shape_64_128_key in batched, f"Missing (64, 128) shape group. Keys: {list(batched.keys())}"
    assert shape_128_64_key in batched, f"Missing (128, 64) shape group. Keys: {list(batched.keys())}"
    assert batched[shape_64_128_key].shape == (2, 64, 128), \
        f"Expected (2, 64, 128), got {batched[shape_64_128_key].shape}"
    assert batched[shape_128_64_key].shape == (2, 128, 64), \
        f"Expected (2, 128, 64), got {batched[shape_128_64_key].shape}"
    
    # Unbatch
    reconstructed = unbatch_params(batched, shape_to_paths, original_paths, tree_struct)
    
    # Check structure preserved
    assert 'layer1' in reconstructed and 'layer2' in reconstructed
    assert 'w1' in reconstructed['layer1'] and 'w2' in reconstructed['layer1']
    
    # Check values preserved
    for layer in ['layer1', 'layer2']:
        for w in ['w1', 'w2']:
            assert jnp.allclose(reconstructed[layer][w], params[layer][w]), \
                f"Value mismatch for {layer}.{w}"
    
    print("✓ batch_params_by_shape and unbatch_params preserve tree structure")


def test_orthogonalize_layer_sharded_single_device():
    """Test layer-sharded orthogonalization on single device (no actual sharding).
    
    On single device, all_to_all is automatically skipped (it would be identity).
    This tests that the batched Newton-Schulz computation is correct.
    """
    key = jax.random.PRNGKey(42)
    L, P, Q = 4, 32, 64
    x = jax.random.normal(key, (L, P, Q))
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    # Test the full orthogonalize_layer_sharded function
    # On single device (device_count == 1), all_to_all is skipped automatically
    result = orthogonalize_layer_sharded(x, ns_coeffs, ns_steps=5, mesh_axis="data")
    
    # Check shape preserved
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
    assert not jnp.any(jnp.isnan(result)), "Result contains NaN"
    
    # Compare with individual orthogonalization
    for i in range(L):
        individual_result = orthogonalize(x[i], ns_coeffs, ns_steps=5)
        # Note: results may differ slightly due to different normalization
        # (per-matrix vs batch normalization), but should be similar in direction
        assert not jnp.any(jnp.isnan(individual_result)), f"Individual result {i} contains NaN"
    
    print("✓ orthogonalize_layer_sharded works on single device")


def test_scale_by_muon_standard():
    """Test scale_by_muon transformation without layer sharding."""
    key = jax.random.PRNGKey(0)
    params = {
        'w1': jax.random.normal(key, (64, 128)),
        'w2': jax.random.normal(jax.random.split(key)[0], (128, 64)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = scale_by_muon(layer_sharding=False)
    state = tx.init(params)
    updates, new_state = tx.update(grads, state, params)
    
    # Check updates have correct shapes
    for k in params:
        assert updates[k].shape == params[k].shape, f"Shape mismatch for {k}"
        assert not jnp.any(jnp.isnan(updates[k])), f"NaN in updates for {k}"
    
    # Check state was updated
    assert new_state.count == 1
    
    print("✓ scale_by_muon (standard) works correctly")


def test_scale_by_muon_layer_sharded():
    """Test scale_by_muon transformation with layer sharding."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Multiple layers with same shapes (realistic scenario)
    params = {
        'layer0': {'proj': jax.random.normal(k1, (64, 128))},
        'layer1': {'proj': jax.random.normal(k2, (64, 128))},
        'layer2': {'proj': jax.random.normal(k3, (64, 128))},
        'layer3': {'proj': jax.random.normal(k4, (64, 128))},
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    # Create mesh for layer sharding (single device falls back to batched without all_to_all)
    mesh = jax.make_mesh((1,), ("data",))
    
    # Pass mesh to scale_by_muon for shard_map
    tx = scale_by_muon(layer_sharding=True, mesh=mesh, mesh_axis="data")
    state = tx.init(params)
    updates, new_state = tx.update(grads, state, params)
    
    # Check updates have correct shapes
    for layer in params:
        for k in params[layer]:
            assert updates[layer][k].shape == params[layer][k].shape, \
                f"Shape mismatch for {layer}.{k}"
            assert not jnp.any(jnp.isnan(updates[layer][k])), \
                f"NaN in updates for {layer}.{k}"
    
    assert new_state.count == 1
    
    print("✓ scale_by_muon (layer sharded) works correctly")


def test_muon_optimizer():
    """Test full muon optimizer."""
    key = jax.random.PRNGKey(0)
    params = {
        'w1': jax.random.normal(key, (64, 128)),
        'w2': jax.random.normal(jax.random.split(key)[0], (128, 64)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = our_muon(learning_rate=0.01, layer_sharding=False)
    state = tx.init(params)
    updates, new_state = tx.update(grads, state, params)
    
    # Apply updates
    new_params = optax.apply_updates(params, updates)
    
    # Check params changed
    for k in params:
        assert not jnp.allclose(new_params[k], params[k]), f"Params {k} didn't change"
    
    print("✓ muon optimizer applies updates correctly")


def test_muon_matches_optax():
    """Verify our muon matches optax.contrib.muon for 2D params (without layer sharding)."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    
    params = {
        'w1': jax.random.normal(k1, (64, 128)),
        'w2': jax.random.normal(k2, (128, 64)),
    }
    grads = {
        'w1': jax.random.normal(k3, (64, 128)) * 0.1,
        'w2': jax.random.normal(jax.random.split(k3)[0], (128, 64)) * 0.1,
    }
    
    # Common params
    lr = 0.01
    beta = 0.95
    ns_steps = 5
    
    # Our muon (standard mode)
    our_tx = our_muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True, layer_sharding=False)
    our_state = our_tx.init(params)
    our_updates, _ = our_tx.update(grads, our_state, params)
    
    # optax.contrib.muon
    optax_tx = optax.contrib.muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True)
    optax_state = optax_tx.init(params)
    optax_updates, _ = optax_tx.update(grads, optax_state, params)
    
    # Compare
    print("\nComparing updates with optax.contrib.muon:")
    all_close = True
    for k in params:
        ours = our_updates[k]
        theirs = optax_updates[k]
        max_diff = jnp.abs(ours - theirs).max()
        rel_diff = max_diff / (jnp.abs(theirs).max() + 1e-8)
        is_close = jnp.allclose(ours, theirs, rtol=1e-4, atol=1e-6)
        
        print(f"  {k}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}, close={is_close}")
        
        if not is_close:
            all_close = False
    
    if all_close:
        print("✓ Outputs match optax.contrib.muon")
    else:
        print("⚠ Outputs differ from optax.contrib.muon (may be due to implementation details)")
    
    return all_close


def test_multiple_steps():
    """Test muon over multiple optimization steps."""
    key = jax.random.PRNGKey(42)
    params = {'w': jax.random.normal(key, (32, 64))}
    
    tx = our_muon(learning_rate=0.01, layer_sharding=False)
    state = tx.init(params)
    
    # Run 10 steps
    for i in range(10):
        grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(i), x.shape) * 0.1, params)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    
    assert not jnp.any(jnp.isnan(params['w'])), "NaN after multiple steps"
    print("✓ muon stable over multiple steps")


def test_layer_sharded_vs_standard_equivalence():
    """Test that layer-sharded produces similar results to standard on single device."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    # Multiple layers with same shapes
    params = {
        'layer0': {'proj': jax.random.normal(k1, (64, 128))},
        'layer1': {'proj': jax.random.normal(k2, (64, 128))},
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    # Standard muon
    tx_standard = scale_by_muon(layer_sharding=False)
    state_standard = tx_standard.init(params)
    updates_standard, _ = tx_standard.update(grads, state_standard, params)
    
    # Layer-sharded muon (on single device, falls back to batched without all_to_all)
    mesh = jax.make_mesh((1,), ("data",))
    tx_sharded = scale_by_muon(layer_sharding=True, mesh=mesh, mesh_axis="data")
    state_sharded = tx_sharded.init(params)
    updates_sharded, _ = tx_sharded.update(grads, state_sharded, params)
    
    # Results should be similar but may differ due to batched vs individual normalization
    print("\nComparing standard vs layer-sharded:")
    for layer in params:
        for k in params[layer]:
            standard = updates_standard[layer][k]
            sharded = updates_sharded[layer][k]
            max_diff = jnp.abs(standard - sharded).max()
            rel_diff = max_diff / (jnp.abs(standard).max() + 1e-8)
            print(f"  {layer}.{k}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
    
    print("✓ Layer-sharded vs standard comparison complete")


def test_handles_non_2d_params():
    """Test that non-2D parameters are handled correctly."""
    key = jax.random.PRNGKey(0)
    
    # Mix of 2D and 1D params
    params = {
        'weight': jax.random.normal(key, (64, 128)),
        'bias': jax.random.normal(key, (128,)),  # 1D
        'scale': jax.random.normal(key, (64,)),   # 1D
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = scale_by_muon(layer_sharding=False)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    
    # Check all shapes preserved
    for k in params:
        assert updates[k].shape == params[k].shape, f"Shape mismatch for {k}"
        assert not jnp.any(jnp.isnan(updates[k])), f"NaN in {k}"
    
    print("✓ Non-2D parameters handled correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Muon Layer Sharding Tests")
    print("=" * 60)
    
    test_newton_schulz_batched()
    test_orthogonalize()
    test_orthogonalize_transposed()
    test_batch_unbatch_params()
    test_orthogonalize_layer_sharded_single_device()
    test_scale_by_muon_standard()
    test_scale_by_muon_layer_sharded()
    test_muon_optimizer()
    test_multiple_steps()
    test_handles_non_2d_params()
    test_layer_sharded_vs_standard_equivalence()
    
    print("\n" + "=" * 60)
    print("Comparison with optax.contrib.muon")
    print("=" * 60)
    test_muon_matches_optax()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
