import contextlib
import threading
import jax
from jax.sharding import PartitionSpec as P

# Logical axis rules for different sharding modes
REPLICATED_RULES = (
    # Model params fully replicated
    ("model_vocab", None),
    ("model_embed", None),
    ("model_intermediate", None),
    ("model_q", None),
    ("model_kv", None),
    ("model_seq", None),
    # Activations
    ("batch", "data"),
    ("act_seq", None),
    ("act_vocab", None),
    ("act_embed", None),
    ("act_intermediate", None),
    ("act_q", None),
    ("act_kv", None),
    ("head_embed", None),
    ("seq", None),
)

SHARDED_RULES = (
    # Model params sharded on data axis
    ("model_vocab", "data"),
    ("model_embed", None),
    ("model_intermediate", "data"),
    ("model_q", "data"),
    ("model_kv", "data"),
    ("model_seq", None),
    # Activations (same as replicated)
    ("batch", "data"),
    ("act_seq", None),
    ("act_vocab", None),
    ("act_embed", None),
    ("act_intermediate", None),
    ("act_q", None),
    ("act_kv", None),
    ("head_embed", None),
    ("seq", None),
)


class _AxisRules(threading.local):
    rules: tuple = REPLICATED_RULES

_axis_rules = _AxisRules()


def get_axis_rules():
    """Returns the current logical axis rules."""
    return _axis_rules.rules


def set_axis_rules(rules):
    """Sets the global logical axis rules."""
    _axis_rules.rules = rules


@contextlib.contextmanager
def axis_rules(rules):
    """Context manager for setting logical to physical axis bindings."""
    old_rules = _axis_rules.rules
    try:
        _axis_rules.rules = rules
        yield
    finally:
        _axis_rules.rules = old_rules


def logical_to_physical(logical_axes):
    """Convert logical axes to physical PartitionSpec using current rules."""
    rules_dict = {name: mesh_axis for name, mesh_axis in get_axis_rules()}
    return P(*[rules_dict.get(axis, None) for axis in logical_axes])


def shard_init(init, logical_axes):
    """Wrap initializer to apply sharding based on logical axes."""
    def init_fn(*args, **kwargs):
        return init(*args, **kwargs, out_sharding=logical_to_physical(logical_axes))
    return init_fn
