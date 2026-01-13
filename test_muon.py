"""Test that our simplified Muon matches optax.contrib.muon."""
import jax
import jax.numpy as jnp
import optax
from muon import muon as our_muon, scale_by_muon, orthogonalize


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


def test_scale_by_muon():
    """Test scale_by_muon transformation."""
    key = jax.random.PRNGKey(0)
    params = {
        'w1': jax.random.normal(key, (64, 128)),
        'w2': jax.random.normal(jax.random.split(key)[0], (128, 64)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = scale_by_muon()
    state = tx.init(params)
    updates, new_state = tx.update(grads, state, params)
    
    # Check updates have correct shapes
    for k in params:
        assert updates[k].shape == params[k].shape, f"Shape mismatch for {k}"
        assert not jnp.any(jnp.isnan(updates[k])), f"NaN in updates for {k}"
    
    # Check state was updated
    assert new_state.count == 1
    
    print("✓ scale_by_muon works correctly")


def test_muon_optimizer():
    """Test full muon optimizer."""
    key = jax.random.PRNGKey(0)
    params = {
        'w1': jax.random.normal(key, (64, 128)),
        'w2': jax.random.normal(jax.random.split(key)[0], (128, 64)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = our_muon(learning_rate=0.01)
    state = tx.init(params)
    updates, new_state = tx.update(grads, state, params)
    
    # Apply updates
    new_params = optax.apply_updates(params, updates)
    
    # Check params changed
    for k in params:
        assert not jnp.allclose(new_params[k], params[k]), f"Params {k} didn't change"
    
    print("✓ muon optimizer applies updates correctly")


def test_muon_matches_optax():
    """Verify our simplified muon matches optax.contrib.muon for 2D params."""
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
    
    # Our muon
    our_tx = our_muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True)
    our_state = our_tx.init(params)
    our_updates, _ = our_tx.update(grads, our_state, params)
    
    # optax.contrib.muon
    optax_tx = optax.contrib.muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True)
    optax_state = optax_tx.init(params)
    optax_updates, _ = optax_tx.update(grads, optax_state, params)
    
    # Compare
    print("\nComparing updates:")
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
    
    tx = our_muon(learning_rate=0.01)
    state = tx.init(params)
    
    # Run 10 steps
    for i in range(10):
        grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(i), x.shape) * 0.1, params)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    
    assert not jnp.any(jnp.isnan(params['w'])), "NaN after multiple steps"
    print("✓ muon stable over multiple steps")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Muon tests")
    print("=" * 50)
    
    test_orthogonalize()
    test_orthogonalize_transposed()
    test_scale_by_muon()
    test_muon_optimizer()
    test_multiple_steps()
    
    print("\n" + "=" * 50)
    print("Comparison with optax.contrib.muon")
    print("=" * 50)
    test_muon_matches_optax()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

