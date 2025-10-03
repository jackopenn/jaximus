import jax
from jax import numpy as jnp
from flax import nnx
from typing import Callable

from modelling.layers.position import apply_rope


class MLP(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: Callable,
        use_bias: bool,
        rngs: jnp.ndarray,
        dtype: jnp.dtype,
        kernel_init: nnx.Initializer = nnx.initializers.lecun_normal(),
        bias_init: nnx.Initializer = nnx.initializers.zeros_init(),
        proj_init: nnx.Initializer = nnx.initializers.lecun_normal(),
        shard_axis_name: str | None = None
    ):
        super().__init__()
        self.up_proj = nnx.Linear(
            hidden_dim,
            intermediate_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init, (shard_axis_name, None)),
            bias_init=nnx.with_partitioning(bias_init, (shard_axis_name,))
        )
        self.down_proj = nnx.Linear(
            intermediate_dim,
            hidden_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(proj_init, (None, shard_axis_name)),
            bias_init=nnx.with_partitioning(bias_init, (shard_axis_name,))
        )
        self.act_fn = act_fn

    def __call__(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x
    

class GLU(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn: Callable,
        use_bias: bool,
        rngs: jnp.ndarray,
        dtype: jnp.dtype,
        kernel_init: nnx.Initializer = nnx.initializers.lecun_normal(),
        bias_init: nnx.Initializer = nnx.initializers.zeros_init(),
        proj_init: nnx.Initializer = nnx.initializers.lecun_normal(),
    ):
        super().__init__()
        self.up_proj = nnx.Linear(
            hidden_dim,
            intermediate_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init, ("embed", "intermediate")),
            bias_init=nnx.with_partitioning(bias_init, ("embed"))
        )
        self.gate_proj = nnx.Linear(
            hidden_dim,
            intermediate_dim,
            use_bias=use_bias, 
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init, ("embed", "intermediate")),
            bias_init=nnx.with_partitioning(bias_init, ("embed"))
        )
        self.down_proj = nnx.Linear(
            intermediate_dim,
            hidden_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(proj_init, ("intermediate", "embed")),
            bias_init=nnx.with_partitioning(bias_init, ("intermediate"))
        )
        self.act_fn = act_fn

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Attention(nnx.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int, 
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: int | None, 
        qk_norm: bool,
        use_bias: bool,
        dtype: jnp.dtype, 
        rngs: jnp.ndarray,
        kernel_init: nnx.Initializer = nnx.initializers.lecun_normal(),
        bias_init: nnx.Initializer = nnx.initializers.zeros_init(),
        proj_init: nnx.Initializer = nnx.initializers.lecun_normal(),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.use_bias = use_bias
        self.dtype = dtype

        self.q_proj = nnx.Linear(
            hidden_dim,
            num_attention_heads * head_dim,
            use_bias=use_bias,
            dtype=dtype, rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init, ("embed", "q")),
            bias_init=nnx.with_partitioning(bias_init, ("embed"))
        )
        self.k_proj = nnx.Linear(
            hidden_dim,
            num_key_value_heads * head_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init, ("embed", "kv")),
            bias_init=nnx.with_partitioning(bias_init, ("embed"))
        )
        self.v_proj = nnx.Linear(
            hidden_dim,
            num_key_value_heads * head_dim, 
            use_bias=use_bias,
            dtype=dtype, 
            rngs=rngs,
            kernel_init=nnx.with_partitioning(kernel_init, ("embed", "kv")),
            bias_init=nnx.with_partitioning(bias_init, ("embed"))
        )

        self.o_proj = nnx.Linear(
            num_attention_heads * head_dim,
            hidden_dim,
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(proj_init, ("q", "embed")),
            bias_init=nnx.with_partitioning(bias_init, ("embed"))
            )

        if self.qk_norm:
            self.q_norm = nnx.RMSNorm(
                head_dim,
                dtype=jnp.float32,
                scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ("embed")),
                rngs=rngs
            )
            self.k_norm = nnx.RMSNorm(
                head_dim,
                dtype=jnp.float32,
                scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), ("embed")),
                rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
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

        if self.rope_theta:
            positions = jnp.arange(S)[None, :]
            q = apply_rope(q, positions, base_frequency=self.rope_theta)
            k = apply_rope(k, positions, base_frequency=self.rope_theta)

        if mask is not None:
            mask = nnx.make_attention_mask(mask, mask).astype(jnp.bool_)
            
        with jax.profiler.TraceAnnotation("attention"): # this does not work :/
            att = jax.nn.dot_product_attention(
                query=q, key=k, value=v,
                is_causal=True,
                implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
                mask=mask
            )

        return self.o_proj(att.reshape(B, S, -1))
    

