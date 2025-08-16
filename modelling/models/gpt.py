from typing import Callable
from dataclasses import dataclass

import jax
from jax import numpy as jnp

from flax import nnx

from modelling.layers.core import MLP


@dataclass
class GPTConfig:
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    act_fn: Callable
    max_seq_len: int
    layer_norm_epsilon: float
    use_bias: bool = False


class GPTLayer(nnx.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            intermediate_dim: int,
            act_fn: Callable,
            layer_norm_epsilon: float,
            use_bias: bool,
            rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            use_bias=use_bias,
            rngs=rngs,
        )
        self.ln_1 = nnx.LayerNorm(num_features=hidden_dim, epsilon=layer_norm_epsilon, rngs=rngs)
        self.mlp = MLP(hidden_dim, intermediate_dim, act_fn, use_bias, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(num_features=hidden_dim, epsilon=layer_norm_epsilon, rngs=rngs)

    def __call__(self, x: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attention(x, mask=attention_mask)
        x = self.ln_1(x)
        x = x + self.mlp(x)
        x = self.ln_2(x)
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_dim,
            rngs=rngs,
        )
        self.layers = [GPTLayer(config.hidden_dim, config.num_heads, config.intermediate_dim, config.act_fn, config.layer_norm_epsilon, config.use_bias, rngs) for _ in range(config.num_layers)]
        self.ln_f = nnx.LayerNorm(num_features=config.hidden_dim, epsilon=config.layer_norm_epsilon, rngs=rngs)
        self.lm_head = nnx.Linear(config.hidden_dim, config.vocab_size, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        if attention_mask is None:
            attention_mask = nnx.make_causal_mask(input_ids)
        # print(input_ids.shape, attention_mask.shape)
        x = self.embed_tokens(input_ids)
        # print(x.shape)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        return self.lm_head(x)
    
