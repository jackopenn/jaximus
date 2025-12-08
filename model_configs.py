from jax import numpy as jnp
from flax import nnx

from sws import Config

def get_gpt2_config():
    cfg = Config()
    cfg.vocab_size=50304
    cfg.hidden_dim=768
    cfg.num_layers=12
    cfg.num_attention_heads=12
    cfg.intermediate_dim=3072
    cfg.head_dim=64
    cfg.act_fn="gelu"
    cfg.max_seq_len=1024
    cfg.layer_norm_epsilon=1e-5
    cfg.use_bias=False
    cfg.dtype="bfloat16"
    return cfg