from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from utils.getters import ModelConfig, DataConfig, OptimConfig, TrainConfig, ExpConfig

from modelling.models.gpt import GPTConfig


model_config = GPTConfig(
    vocab_size=50257,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    intermediate_dim=3072,
    act_fn=nnx.gelu,
    max_seq_len=1024,
    layer_norm_epsilon=1e-5,
    use_bias=False,
    dtype=jnp.bfloat16,
)

data_config = DataConfig(
    source="hf",
    hf_name=["allenai/c4", "realnewslike"],
    tokenizer_name="gpt2",
    max_length=1024,
)

optim_config = OptimConfig(
    name="adamw",
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    batch_size=16,
    accum_steps=32,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=6e-4,
        warmup_steps=1_000,
        decay_steps=99_000,
        end_value=6e-5
    )
)

train_config = TrainConfig(
    num_steps=100_000,
    log_every=10,
    generate_every=100,
    eval_every=-1,
    save_every=1_000,
    save_dir="checkpoints",
)

exp_config = ExpConfig(
    name="mini-gpt2",
    seed=42,
    model=model_config,
    data=data_config,
    optim=optim_config,
    train=train_config,
)

from train import train

train(exp_config)