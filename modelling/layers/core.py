import jax
from jax import numpy as jnp
from flax import nnx
from modelling.layers.position import apply_rope
from parallel import logical_to_physical, shard_init

def create_norm(norm_type, num_features, epsilon, use_bias, rngs):
    """Factory function to create normalization layers."""
    if norm_type == "rms":
        return nnx.RMSNorm(
            num_features=num_features,
            epsilon=epsilon,
            dtype=jnp.float32,
            use_scale=False,
            rngs=rngs,
        )
    elif norm_type == "layer":
        return nnx.LayerNorm(
            num_features=num_features,
            epsilon=epsilon,
            use_bias=use_bias,
            dtype=jnp.float32,
            rngs=rngs,
        )
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class MLP(nnx.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        act_fn,
        use_bias,
        dtype,
        inits,
        rngs
    ):
        super().__init__()
        self.up_proj = nnx.Linear(
            hidden_dim,
            intermediate_dim,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=shard_init(inits["mlp_up"], ("model_embed", "model_intermediate")),
            bias_init=shard_init(inits["bias"], ("intermediate", )),
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate_dim,
            hidden_dim,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=shard_init(inits["mlp_down"], ("model_intermediate", "model_embed")),
            bias_init=shard_init(inits["bias"], ("model_embed", )),
            rngs=rngs,
        )
        self.act_fn = act_fn

    def __call__(self, x):
        x = self.up_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_intermediate")))
        x = self.act_fn(x)
        x = self.down_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_embed")))
        return x
    

class GLU(nnx.Module):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        act_fn,
        use_bias,
        dtype,
        inits,
        rngs
    ):
        super().__init__()
        self.up_proj = nnx.Linear(
            hidden_dim,
            intermediate_dim,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=shard_init(inits["mlp_up"], ("model_embed", "model_intermediate")),
            bias_init=shard_init(inits["bias"], ("model_intermediate", )),
            rngs=rngs,
        )
        self.gate_proj = nnx.Linear(
            hidden_dim,
            intermediate_dim,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=shard_init(inits["mlp_up"], ("model_embed", "model_intermediate")),
            bias_init=shard_init(inits["bias"], ("model_intermediate", )),
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate_dim,
            hidden_dim,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=shard_init(inits["mlp_down"], ("model_intermediate", "model_embed")),
            bias_init=shard_init(inits["bias"], ("model_embed", )),
            rngs=rngs,
        )
        self.act_fn = act_fn

    def __call__(self, x):
        up = self.up_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_intermediate")))
        gate = self.gate_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_intermediate")))
        out = self.down_proj(self.act_fn(gate) * up, out_sharding=logical_to_physical("batch", "seq", "act_embed"))
        return out


class Attention(nnx.Module):
    def __init__(
        self,
        hidden_dim,
        num_attention_heads, 
        num_key_value_heads,
        head_dim,
        rope_theta, 
        qk_norm,
        qk_norm_type,
        qk_norm_epsilon,
        use_bias,
        sliding_window,
        dtype, 
        inits,
        rngs
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.dtype = dtype
        self.sliding_window = sliding_window

        self.q_proj = nnx.Linear(
            hidden_dim,
            num_attention_heads * head_dim,
            use_bias=use_bias,
            kernel_init=shard_init(inits["qkv"], ("model_embed", "model_q")),
            bias_init=shard_init(inits["bias"], ("model_q",)),
            dtype=dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            hidden_dim,
            num_key_value_heads * head_dim,
            use_bias=use_bias,
            kernel_init=shard_init(inits["qkv"], ("model_embed", "model_kv")),
            bias_init=shard_init(inits["bias"], ("model_kv",)),
            dtype=dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            hidden_dim,
            num_key_value_heads * head_dim,
            use_bias=use_bias,
            kernel_init=shard_init(inits["qkv"], ("model_embed", "model_kv")),
            bias_init=shard_init(inits["bias"], ("model_kv",)),
            dtype=dtype,
            rngs=rngs,
        )

        self.o_proj = nnx.Linear(
            num_attention_heads * head_dim,
            hidden_dim,
            use_bias=use_bias,
            kernel_init=shard_init(inits["o_proj"], ("model_q", "model_embed")),
            bias_init=shard_init(inits["bias"], ("model_embed",)),
            dtype=dtype,
            rngs=rngs,
        )

        if self.qk_norm:
            if qk_norm_type is None:
                raise ValueError("qk_norm_type must be specified when qk_norm is True")
            if qk_norm_epsilon is None:
                raise ValueError("qk_norm_epsilon must be specified when qk_norm is True")
            self.q_norm = create_norm(
                norm_type=qk_norm_type,
                num_features=head_dim,
                epsilon=qk_norm_epsilon,
                use_bias=use_bias,
                rngs=rngs,
            )
            self.k_norm = create_norm(
                norm_type=qk_norm_type,
                num_features=head_dim,
                epsilon=qk_norm_epsilon,
                use_bias=use_bias,
                rngs=rngs,
            )

    def __call__(self, x, mask=None):
        batch, seq, _ = x.shape
        
        q = self.q_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_q")))
        k = self.k_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_kv")))
        v = self.v_proj(x, out_sharding=logical_to_physical(("batch", "seq", "act_kv")))
        
        q = q.reshape(batch, seq, self.num_attention_heads, self.head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_q", "head_embed")))
        k = k.reshape(batch, seq, self.num_key_value_heads, self.head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_kv", "head_embed")))
        v = v.reshape(batch, seq, self.num_key_value_heads, self.head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_kv", "head_embed")))
     
        if self.rope_theta:
            positions = jnp.arange(x.shape[1])[None, :]
            q = apply_rope(q, positions, base_frequency=self.rope_theta)
            k = apply_rope(k, positions, base_frequency=self.rope_theta)

        if self.qk_norm:
            q = self.q_norm(q).astype(self.dtype)
            k = self.k_norm(k).astype(self.dtype)

        if mask is not None:
            mask = nnx.make_attention_mask(mask, mask).astype(jnp.bool_)

        # handles repeating kv for GQA
        att = jax.nn.dot_product_attention(
            query=q, key=k, value=v,
            is_causal=True,
            implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
            mask=mask,
            local_window_size=(self.sliding_window, 0) if self.sliding_window else None
        )

        att = att.reshape(batch, seq, self.num_attention_heads * self.head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_embed")))
        out = self.o_proj(att, out_sharding=logical_to_physical(("batch", "seq", "act_embed")))
        return out
