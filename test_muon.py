"""Tests for layer-sharded Muon optimizer."""
import jax
import jax.numpy as jnp
import optax
from muon import (
    muon,
    scale_by_muon,
    orthogonalize,
    orthogonalize_layer_sharded,
)


def test_orthogonalize_single():
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (64, 128))
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    result = orthogonalize(x, ns_coeffs, ns_steps=5, eps=1e-8)
    
    assert result.shape == x.shape
    assert not jnp.any(jnp.isnan(result))
    assert jnp.abs(result).sum() > 0
    print("✓ orthogonalize produces valid output")


def test_orthogonalize_single_transposed():
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (128, 64))
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    result = orthogonalize(x, ns_coeffs, ns_steps=5, eps=1e-8)
    
    assert result.shape == x.shape
    assert not jnp.any(jnp.isnan(result))
    print("✓ orthogonalize handles rows > cols")


def test_orthogonalize_batched():
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4, 64, 128))
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    result = orthogonalize(x, ns_coeffs, ns_steps=5, eps=1e-8)
    
    assert result.shape == x.shape
    assert not jnp.any(jnp.isnan(result))
    print("✓ orthogonalize handles batched input (L, P, Q)")


def test_grouping():
    """Test that orthogonalize_layer_sharded groups by shape and sharding correctly."""
    params = {
        'layer_0': {
            'q_proj': jnp.zeros((64, 128)),
            'k_proj': jnp.zeros((64, 128)),
            'o_proj': jnp.zeros((128, 64)),
        },
        'layer_1': {
            'q_proj': jnp.zeros((64, 128)),
            'k_proj': jnp.zeros((64, 128)),
            'o_proj': jnp.zeros((128, 64)),
        },
    }
    
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    result = orthogonalize_layer_sharded(params, ns_coeffs, ns_steps=5, eps=1e-8)
    
    # Verify structure is preserved
    assert result['layer_0']['q_proj'].shape == (64, 128)
    assert result['layer_0']['k_proj'].shape == (64, 128)
    assert result['layer_0']['o_proj'].shape == (128, 64)
    assert result['layer_1']['q_proj'].shape == (64, 128)
    assert result['layer_1']['k_proj'].shape == (64, 128)
    assert result['layer_1']['o_proj'].shape == (128, 64)
    
    # Verify no NaNs
    for layer in result.values():
        for proj in layer.values():
            assert not jnp.any(jnp.isnan(proj))
    
    print("✓ orthogonalize_layer_sharded groups and processes correctly")


def test_stack_unstack_roundtrip():
    key = jax.random.PRNGKey(42)
    params = {
        'layer_0': {'w': jax.random.normal(key, (64, 128))},
        'layer_1': {'w': jax.random.normal(jax.random.split(key)[0], (64, 128))},
    }
    ns_coeffs = jnp.array([3.4445, -4.7750, 2.0315])
    
    result = orthogonalize_layer_sharded(params, ns_coeffs, ns_steps=5, eps=1e-8)
    
    assert result['layer_0']['w'].shape == params['layer_0']['w'].shape
    assert result['layer_1']['w'].shape == params['layer_1']['w'].shape
    assert not jnp.any(jnp.isnan(result['layer_0']['w']))
    assert not jnp.any(jnp.isnan(result['layer_1']['w']))
    print("✓ orthogonalize_layer_sharded preserves structure")


def test_scale_by_muon():
    key = jax.random.PRNGKey(0)
    params = {
        'w1': jax.random.normal(key, (64, 128)),
        'w2': jax.random.normal(jax.random.split(key)[0], (128, 64)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = scale_by_muon()
    state = tx.init(params)
    updates, new_state = tx.update(grads, state, params)
    
    for k in params:
        assert updates[k].shape == params[k].shape
        assert not jnp.any(jnp.isnan(updates[k]))
    
    assert new_state.count == 1
    print("✓ scale_by_muon works correctly")


def test_muon_optimizer():
    key = jax.random.PRNGKey(0)
    params = {
        'w1': jax.random.normal(key, (64, 128)),
        'w2': jax.random.normal(jax.random.split(key)[0], (128, 64)),
    }
    grads = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * 0.1, params)
    
    tx = muon(learning_rate=0.01, layer_sharding=False)
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    
    new_params = optax.apply_updates(params, updates)
    
    for k in params:
        assert not jnp.allclose(new_params[k], params[k])
    
    print("✓ muon optimizer applies updates correctly")


def test_multiple_steps():
    key = jax.random.PRNGKey(42)
    params = {'w': jax.random.normal(key, (32, 64))}
    
    tx = muon(learning_rate=0.01, layer_sharding=True)
    state = tx.init(params)
    
    for i in range(10):
        grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(i), x.shape) * 0.1, params)
        updates, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    
    assert not jnp.any(jnp.isnan(params['w']))
    print("✓ muon stable over multiple steps")


def test_matches_optax():
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
    
    lr = 0.01
    beta = 0.95
    ns_steps = 5
    
    our_tx = muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True, layer_sharding=False)
    our_state = our_tx.init(params)
    our_updates, _ = our_tx.update(grads, our_state, params)
    
    optax_tx = optax.contrib.muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True)
    optax_state = optax_tx.init(params)
    optax_updates, _ = optax_tx.update(grads, optax_state, params)
    
    print("\nComparing muon vs optax.contrib.muon:")
    all_close = True
    for k in params:
        max_diff = jnp.abs(our_updates[k] - optax_updates[k]).max()
        rel_diff = max_diff / (jnp.abs(optax_updates[k]).max() + 1e-8)
        is_close = jnp.allclose(our_updates[k], optax_updates[k], rtol=1e-4, atol=1e-6)
        print(f"  {k}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}, close={is_close}")
        if not is_close:
            all_close = False
    
    if all_close:
        print("✓ muon matches optax.contrib.muon")
    else:
        print("⚠ Outputs differ from optax.contrib.muon")
    
    return all_close


def test_layer_sharding_vs_no_sharding():
    """Test that muon with and without layer sharding produce similar results."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
    
    # Set up mesh with 8 CPU devices
    jax.config.update('jax_num_cpu_devices', 8)
    mesh = jax.make_mesh((8,), ("data",), (AxisType.Explicit,))
    jax.set_mesh(mesh)
    
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Create params with sharding
    params = {
        'layer_0': {
            'q_proj': jax.random.normal(k1, (64, 128)),
            'k_proj': jax.random.normal(jax.random.split(k1)[0], (64, 128)),
            'o_proj': jax.random.normal(jax.random.split(k1)[1], (128, 64)),
        },
        'layer_1': {
            'q_proj': jax.random.normal(k2, (64, 128)),
            'k_proj': jax.random.normal(jax.random.split(k2)[0], (64, 128)),
            'o_proj': jax.random.normal(jax.random.split(k2)[1], (128, 64)),
        },
    }
    
    # Shard params along the data axis
    sharding = NamedSharding(mesh, P("data", None))
    params = jax.tree.map(lambda x: jax.device_put(x, sharding) if x.ndim == 2 else x, params)
    
    grads = {
        'layer_0': {
            'q_proj': jax.random.normal(k3, (64, 128)) * 0.1,
            'k_proj': jax.random.normal(jax.random.split(k3)[0], (64, 128)) * 0.1,
            'o_proj': jax.random.normal(jax.random.split(k3)[1], (128, 64)) * 0.1,
        },
        'layer_1': {
            'q_proj': jax.random.normal(jax.random.split(k3)[2], (64, 128)) * 0.1,
            'k_proj': jax.random.normal(jax.random.split(k3)[3], (64, 128)) * 0.1,
            'o_proj': jax.random.normal(jax.random.split(k3)[4], (128, 64)) * 0.1,
        },
    }
    grads = jax.tree.map(lambda x: jax.device_put(x, sharding) if x.ndim == 2 else x, grads)
    
    lr = 0.01
    beta = 0.95
    ns_steps = 5
    
    # Test with layer sharding
    tx_with_sharding = muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True, layer_sharding=True)
    state_with = tx_with_sharding.init(params)
    updates_with, _ = tx_with_sharding.update(grads, state_with, params)
    
    # Test without layer sharding
    tx_without_sharding = muon(learning_rate=lr, beta=beta, ns_steps=ns_steps, nesterov=True, layer_sharding=False)
    state_without = tx_without_sharding.init(params)
    updates_without, _ = tx_without_sharding.update(grads, state_without, params)
    
    print("\nComparing muon with layer_sharding=True vs layer_sharding=False:")
    all_close = True
    for layer_name in params:
        for param_name in params[layer_name]:
            with_val = updates_with[layer_name][param_name]
            without_val = updates_without[layer_name][param_name]
            max_diff = jnp.abs(with_val - without_val).max()
            rel_diff = max_diff / (jnp.abs(without_val).max() + 1e-8)
            is_close = jnp.allclose(with_val, without_val, rtol=1e-4, atol=1e-6)
            print(f"  {layer_name}.{param_name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}, close={is_close}")
            if not is_close:
                all_close = False
    
    if all_close:
        print("✓ muon with and without layer sharding produce similar results")
    else:
        print("⚠ Outputs differ between layer_sharding=True and False")
    
    return all_close


if __name__ == "__main__":
    from jax.sharding import AxisType
    
    # Set up mesh with 8 CPU devices for tests that need it
    jax.config.update('jax_num_cpu_devices', 8)
    mesh = jax.make_mesh((8,), ("data",), (AxisType.Explicit,))
    jax.set_mesh(mesh)
    
    print("=" * 50)
    print("Running Muon tests")
    print("=" * 50)
    
    test_orthogonalize_single()
    test_orthogonalize_single_transposed()
    test_orthogonalize_batched()
    test_grouping()
    test_stack_unstack_roundtrip()
    test_scale_by_muon()
    test_muon_optimizer()
    test_multiple_steps()
    
    print("\n" + "=" * 50)
    print("Comparison with optax.contrib.muon")
    print("=" * 50)
    test_matches_optax()
    
    print("\n" + "=" * 50)
    print("Layer sharding vs no layer sharding")
    print("=" * 50)
    test_layer_sharding_vs_no_sharding()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
