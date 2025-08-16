import jax
from jax import numpy as jnp
from flax import nnx
from typing import Callable


class MLP(nnx.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn: Callable, use_bias: bool, rngs: jnp.ndarray):
        super().__init__()
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=use_bias, rngs=rngs)
        self.act_fn = act_fn

    def __call__(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x
    

class GLU(nnx.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn: Callable, use_bias: bool, rngs: jnp.ndarray):
        super().__init__()
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, rngs=rngs)
        self.gate_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=use_bias, rngs=rngs)
        self.act_fn = act_fn

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nnx.Module):
    def __init__(self, hidden_dim: int, eps: float, rngs: jnp.ndarray):
        super().__init__()
        self.weight = self.param('weight', jnp.ones, (hidden_dim,), rngs=rngs)
        self.eps = eps

    def __call__(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps) * self.weight
