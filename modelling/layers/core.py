import jax
from jax import numpy as jnp
from flax import nnx
from typing import Callable


class MLP(nnx.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn: Callable, use_bias: bool, rngs: jnp.ndarray, dtype: jnp.dtype):
        super().__init__()
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=dtype, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=use_bias, dtype=dtype, rngs=rngs)
        self.act_fn = act_fn

    def __call__(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x
    

class GLU(nnx.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn: Callable, use_bias: bool, rngs: jnp.ndarray, dtype: jnp.dtype):
        super().__init__()
        self.up_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=dtype, rngs=rngs)
        self.gate_proj = nnx.Linear(hidden_dim, intermediate_dim, use_bias=use_bias, dtype=dtype, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_dim, hidden_dim, use_bias=use_bias, dtype=dtype, rngs=rngs)
        self.act_fn = act_fn

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nnx.Module):
    def __init__(self, hidden_dim: int, num_attention_heads: int, num_key_value_heads: int, head_dim: int, rngs: jnp.ndarray, dtype: jnp.dtype, qk_norm: bool):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.qk_norm = qk_norm
        self.dtype = dtype

        self.q_proj = nnx.Linear(hidden_dim, num_attention_heads * head_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_dim, num_key_value_heads * head_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_dim, num_key_value_heads * head_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.o_proj = nnx.Linear(num_attention_heads * head_dim, hidden_dim, use_bias=False, dtype=dtype, rngs=rngs)

        if qk_norm:
            self.q_norm = nnx.RMSNorm(head_dim, dtype=jnp.float32, rngs=rngs)
            self.k_norm = nnx.RMSNorm(head_dim, dtype=jnp.float32, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        B, S, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, S, self.num_attention_heads, self.head_dim)
        k = k.reshape(B, S, self.num_key_value_heads, self.head_dim)
        v = v.reshape(B, S, self.num_key_value_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q).astype(self.dtype)
            k = self.k_norm(k).astype(self.dtype)

        # this does GQA
        mask = (mask == 1.0)
        att = jax.nn.dot_product_attention(query=q, key=k, value=v, mask=mask)

        return self.o_proj(att.reshape(B, S, -1))