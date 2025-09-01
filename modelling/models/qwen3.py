from typing import Callable
from dataclasses import dataclass

import jax
from jax import numpy as jnp

from flax import nnx

from modelling.layers.core import GLU, Attention
import chz

@chz.chz
class Qwen3Config:
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_dim: int
    act_fn: Callable
    max_seq_len: int
    rope_theta: int


class Qwen3Layer(nnx.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_attention_heads: int,
            num_key_value_heads: int,
            head_dim: int,
            intermediate_dim: int,
            act_fn: Callable,
            rope_theta: int,
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
            qk_norm=True,
            use_bias=False,
            rngs=rngs
        )
        self.norm_1 = nnx.RMSNorm(hidden_dim, dtype=jnp.float32, rngs=rngs)
        self.mlp = GLU(hidden_dim, intermediate_dim, act_fn, use_bias=False, dtype=dtype, rngs=rngs)
        self.norm_2 = nnx.RMSNorm(hidden_dim, dtype=jnp.float32, rngs=rngs)

    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(self.norm_1(x), mask=attention_mask)
        x = x + self.mlp(self.norm_2(x))
        return x


class Qwen3(nnx.Module):
    def __init__(self, config: Qwen3Config, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_dim,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.layers = [
            Qwen3Layer(
                hidden_dim=config.hidden_dim,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_dim=config.intermediate_dim,
                act_fn=config.act_fn,
                rope_theta=config.rope_theta,
                dtype=config.dtype,
                rngs=rngs
            )
            for _ in range(config.num_layers)
        ]
        self.lm_norm = nnx.RMSNorm(config.hidden_dim, dtype=jnp.float32, rngs=rngs)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        if attention_mask is None:
            attention_mask = nnx.make_causal_mask(input_ids)
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.lm_norm(x)
        return self.token_embedding.attend(x)
    
