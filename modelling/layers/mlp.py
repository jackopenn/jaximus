from dataclasses import dataclass
from typing import Optional

import jax
from jax import numpy as jnp

from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class MLPWeights:
    up_proj: jax.Array
    down_proj: jax.Array
    up_bias: Optional[jax.Array] = None
    down_bias: Optional[jax.Array] = None


@jax.tree_util.register_dataclass
@dataclass
class GLUWeights:
    gate_proj: jax.Array
    up_proj: jax.Array
    down_proj: jax.Array
    gate_bias: Optional[jax.Array] = None
    up_bias: Optional[jax.Array] = None
    down_bias: Optional[jax.Array] = None


def resolve_act_fn(act_fn_name):
    if act_fn_name == "relu_squared":
        return lambda x: jnp.square(jax.nn.relu(x))
    return getattr(jax.nn, act_fn_name)


def mlp(x, weights, act_fn, dtype):
    dtype = getattr(jnp, dtype)
    with jax.named_scope("up_proj"):
        h = jnp.matmul(x, weights.up_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
        if weights.up_bias is not None:
            h = h + weights.up_bias.astype(dtype)

    with jax.named_scope("act_fn"):
        h = resolve_act_fn(act_fn)(h)

    with jax.named_scope("down_proj"):
        out = jnp.matmul(h, weights.down_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_embed")))
        if weights.down_bias is not None:
            out = out + weights.down_bias.astype(dtype)

    return out


def glu(x, weights, act_fn, dtype):
    dtype = getattr(jnp, dtype)
    with jax.named_scope("up_proj"):
        up = jnp.matmul(x, weights.up_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
        if weights.up_bias is not None:
            up = up + weights.up_bias.astype(dtype)

    with jax.named_scope("gate_proj"):
        gate = jnp.matmul(x, weights.gate_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
        if weights.gate_bias is not None:
            gate = gate + weights.gate_bias.astype(dtype)

    with jax.named_scope("down_proj"):
        out = jnp.matmul(
            resolve_act_fn(act_fn)(gate) * up,
            weights.down_proj.astype(dtype),
            out_sharding=l2p(("batch", "seq", "act_embed")),
        )
        if weights.down_bias is not None:
            out = out + weights.down_bias.astype(dtype)

    return out
