from typing import Callable
from dataclasses import dataclass

import jax
from jax import numpy as jnp

from flax import nnx

from modelling.layers.core import GLU, Attention
import chz
from utils.configs import ModelConfig



@chz.chz
class LlamaConfig(ModelConfig):
    name: str = "llama"
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    act_fn: Callable
    intermediate_dim: int
    max_seq_len: int
    rope_theta: int
    rms_norm_eps: float
    use_attention_bias: bool
    use_mlp_bias: bool


class LlamaLayer(nnx.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_attention_heads: int,
            num_key_value_heads: int,
            head_dim: int,
            intermediate_dim: int,
            act_fn: Callable,
            rope_theta: int,
            rms_norm_eps: float,
            use_attention_bias: bool,
            use_mlp_bias: bool,
            dtype: jnp.dtype,
            rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attention = Attention(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            dtype=dtype,
            qk_norm=False,
            use_bias=use_attention_bias,
            rngs=rngs
        )
        self.norm_1 = nnx.RMSNorm(
            hidden_dim,
            dtype=jnp.float32,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                ("norm",)
            ),
            rngs=rngs
            )
        self.mlp = GLU(
            hidden_dim,
            intermediate_dim,
            act_fn,
            use_bias=use_mlp_bias,
            dtype=dtype,
            rngs=rngs
        )
        self.norm_2 = nnx.RMSNorm(
            hidden_dim,
            dtype=jnp.float32,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                ("norm",)
            ),
            rngs=rngs
        )

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        # x = x + self.attention(self.norm_1(x), mask=mask)
        # x = x + self.mlp(self.norm_2(x))
        # return x
        
        with jax.named_scope("pre_att_norm"):
            z = self.norm_1(x)
        z = self.attention(z, mask=mask)
        with jax.named_scope("residual"):
            x = x + z
        
        with jax.named_scope("pre_mlp_norm"):
            z = self.norm_2(x)
        z = self.mlp(z)
        with jax.named_scope("residual"):
            x = x + z
        return x


class Llama(nnx.Module):
    def __init__(self, config: LlamaConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_dim,
            dtype=config.dtype,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02),
                ("vocab", "vocab_embed")
            ),
            rngs=rngs,
        )
        self.layers = nnx.List([
            LlamaLayer(
                hidden_dim=config.hidden_dim,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_dim=config.intermediate_dim,
                act_fn=config.act_fn,
                rope_theta=config.rope_theta,
                dtype=config.dtype,
                rms_norm_eps=config.rms_norm_eps,
                use_attention_bias=config.use_attention_bias,
                use_mlp_bias=config.use_mlp_bias,
                rngs=rngs
            )
            for _ in range(config.num_layers)
        ])
        self.lm_norm = nnx.RMSNorm(
            config.hidden_dim,
            dtype=jnp.float32,
            scale_init=nnx.with_partitioning(
                nnx.initializers.ones_init(),
                ("norm",)
            ),
            rngs=rngs
        )
            
    def __call__(self, input_ids: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        with jax.named_scope("embed"):
            x = self.token_embedding(input_ids)
            
        for layer in self.layers:
            x = layer(x, mask)
        
        with jax.named_scope("lm_norm"):
            x = self.lm_norm(x)
        
        with jax.named_scope("attend"):
            out = self.token_embedding.attend(x)
        return out
    
