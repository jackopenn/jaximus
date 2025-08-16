from typing import Callable
from dataclasses import dataclass

import jax
from jax import numpy as jnp

from flax import nnx

from modelling.layers.core import GLU, RMSNorm

@dataclass
class Llama3Config:
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    act_fn: Callable
    max_seq_len: int


class Llama3Layer(nnx.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            intermediate_dim: int,
            act_fn: Callable,
            rms_eps: float,
            rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            use_bias=False,
            rngs=rngs,
        )
        self.norm_1 = RMSNorm(hidden_dim, eps=rms_eps, rngs=rngs)
        self.mlp = GLU(hidden_dim, intermediate_dim, act_fn, use_bias=False, rngs=rngs)
        self.norm_2 = RMSNorm(hidden_dim, eps=rms_eps, rngs=rngs)

    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(x, mask=attention_mask)
        x = self.norm_1(x)
        x = x + self.mlp(x)
        x = self.norm_2(x)
        return x


class Llama3(nnx.Module):
    def __init__(self, config: Llama3Config, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.token_embed = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_dim,
            rngs=rngs,
        )
        self.pos_embed = nnx.Embed(
            num_embeddings=config.max_seq_len,
            features=config.hidden_dim,
            rngs=rngs,
        )
        self.layers = [
            Llama3Layer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                intermediate_dim=config.intermediate_dim,
                act_fn=config.act_fn,
                layer_norm_epsilon=config.layer_norm_epsilon,
                use_bias=config.use_bias,
                rngs=rngs
            )
            for _ in range(config.num_layers)
        ]
        self.ln_f = nnx.LayerNorm(
            num_features=config.hidden_dim,
            epsilon=config.layer_norm_epsilon,
            rngs=rngs,
        )

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        if attention_mask is None:
            attention_mask = nnx.make_causal_mask(input_ids)
        x = self.token_embed(input_ids)
        x = x + self.pos_embed(jnp.arange(input_ids.shape[1]))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        return self.token_embed.attend(x)
    
