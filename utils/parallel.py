from functools import partial
import jax
from jax.sharding import NamedSharding, PartitionSpec
from flax import nnx


REPL_SHARDING_RULES = (
    ('batch', 'data'),
    ("vocab", None),
    ('embed', None),
    ('intermediate', None),
    ("q", None),
    ("kv", None),
)


FSDP_SHARDING_RULES = (
    ('batch', 'data'),
    ("vocab", "data"),
    ('embed', None),
    ('intermediate', "data"),
    ("q", "data"),
    ("kv", "data"),
)


def make_and_set_mesh(cfg):
    mesh = jax.make_mesh((cfg.data_parallel,), ("data",))
    jax.set_mesh(mesh)
    return mesh


def init_model_and_optimizer_with_sharding(partial_model, optimizer_tx, cfg):
    if cfg.zero_stage == 0: # no sharding
        with nnx.logical_axis_rules(REPL_SHARDING_RULES):
            model = partial_model()
            optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)
    elif cfg.zero_stage in {1, 2}: # 1 = optimizer, 2 = optimizer + grads
        with nnx.logical_axis_rules(REPL_SHARDING_RULES):
            model = partial_model()
        with nnx.logical_axis_rules(FSDP_SHARDING_RULES):
            optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)
    elif cfg.zero_stage == 3: # optimizer + grads + model (fsdp)
        with nnx.logical_axis_rules(FSDP_SHARDING_RULES):
            model = partial_model()
            optimizer = nnx.Optimizer(model, optimizer_tx, wrt=nnx.Param)
    return model, optimizer
