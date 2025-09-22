from functools import partial
import jax
from jax.sharding import NamedSharding, PartitionSpec
from flax import nnx


def shard_model_and_optimizer(model, optimizer, cfg, mesh):
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)

    model_sharding_spec = nnx.get_partition_spec(model_state)
    optimizer_sharding_spec = nnx.get_partition_spec(optimizer_state)
    print("model_sharding_spec")
    print(model_sharding_spec)
    with mesh:
        match cfg.zero_stage:
            case None: # no model or optimizer sharding
                sharded_model_state = jax.lax.with_sharding_constraint(model_state, replicated_sharding)
                sharded_optimizer_state = jax.lax.with_sharding_constraint(optimizer_state, replicated_sharding)
            case 1: # optimizer sharding
                sharded_model_state = jax.lax.with_sharding_constraint(model_state, replicated_sharding)
                sharded_optimizer_state = jax.lax.with_sharding_constraint(optimizer_state, optimizer_sharding_spec)
            case 2: # optimizer + gradient sharding
                pass
            case 3: # optimizer + gradient + model sharding
                sharded_model_state = jax.lax.with_sharding_constraint(model_state, model_sharding_spec)
                sharded_optimizer_state = jax.lax.with_sharding_constraint(optimizer_state, optimizer_sharding_spec)
    
    nnx.update(model, sharded_model_state)
    nnx.update(optimizer, sharded_optimizer_state)

    return model, optimizer
    
