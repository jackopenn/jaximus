from typing import Callable
from jax import numpy as jnp
from flax import nnx
from modelling.layers.core import MLP, Attention
import chz
from utils.configs import ModelConfig

@chz.chz
class GPTConfig(ModelConfig):
    name: str = "gpt"
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    head_dim: int
    intermediate_dim: int
    act_fn: Callable
    max_seq_len: int
    layer_norm_epsilon: float
    use_bias: bool


class GPTLayer(nnx.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_attention_heads: int,
            head_dim: int,
            intermediate_dim: int,
            act_fn: Callable,
            layer_norm_epsilon: float,
            use_bias: bool,
            dtype: jnp.dtype,
            kernel_init: nnx.Initializer,
            bias_init: nnx.Initializer,
            proj_init: nnx.Initializer,
            rngs: nnx.Rngs,
    ):
        super().__init__()
        self.attention = Attention(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            head_dim=head_dim,
            rope_theta=None,
            qk_norm=False,
            use_bias=use_bias,
            dtype=dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            proj_init=proj_init,
            rngs=rngs,
        )
        self.ln_1 = nnx.LayerNorm(num_features=hidden_dim, use_bias=use_bias, epsilon=layer_norm_epsilon, dtype=jnp.float32, rngs=rngs)
        self.mlp = MLP(hidden_dim, intermediate_dim, act_fn, use_bias, dtype=dtype, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init, proj_init=proj_init)
        self.ln_2 = nnx.LayerNorm(num_features=hidden_dim, use_bias=use_bias, epsilon=layer_norm_epsilon, dtype=jnp.float32, rngs=rngs)

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

        std = 0.02
        resid_scale = 1.0 / jnp.sqrt(2 * config.num_layers)
        kernel_init = nnx.initializers.normal(stddev=std)
        proj_init   = nnx.initializers.normal(stddev=std * resid_scale)
        bias_init   = nnx.initializers.zeros_init()

        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_dim,
            dtype=config.dtype,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.pos_embed = nnx.Embed(
            num_embeddings=config.max_seq_len,
            features=config.hidden_dim,
            dtype=config.dtype,
            embedding_init=kernel_init,
            rngs=rngs,
        )
        self.layers = [
            GPTLayer(
                hidden_dim=config.hidden_dim,
                num_attention_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                intermediate_dim=config.intermediate_dim,
                act_fn=config.act_fn,
                layer_norm_epsilon=config.layer_norm_epsilon,
                use_bias=config.use_bias,
                dtype=config.dtype,
                kernel_init=kernel_init,
                bias_init=bias_init,
                proj_init=proj_init,
                rngs=rngs
            )
            for _ in range(config.num_layers)
        ]
        self.ln_f = nnx.LayerNorm(
            num_features=config.hidden_dim,
            use_bias=config.use_bias,
            epsilon=config.layer_norm_epsilon,
            dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        if attention_mask is None:
            attention_mask = nnx.make_causal_mask(input_ids)
        x = self.token_embedding(input_ids)
        x = x + self.pos_embed(jnp.arange(input_ids.shape[1]))
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.ln_f(x)
        return self.token_embedding.attend(x)
    
