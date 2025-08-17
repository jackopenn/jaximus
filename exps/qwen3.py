from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from utils.configs import DataConfig, OptimConfig, TrainConfig, ExpConfig, ParallelConfig

from modelling.models.qwen3 import Qwen3Config


model_config = Qwen3Config(
    # vocab_size=151936,
    vocab_size=50257,
    hidden_dim=768,
    intermediate_dim=2048,
    num_layers=12,
    num_attention_heads=12,
    num_key_value_heads=6,
    head_dim=64,
    act_fn=nnx.silu,
    max_seq_len=1024,
    dtype=jnp.bfloat16,
)

data_config = DataConfig(
    source="hf",
    hf_name=["allenai/c4", "realnewslike"],
    # tokenizer_name="Qwen/Qwen3-0.6B",
    tokenizer_name="gpt2",
    max_length=1024,
)

optim_config = OptimConfig(
    name="adamw",
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    batch_size=32,
    accum_steps=16,
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

parallel_config = ParallelConfig(
    num_devices=4,
    data_parallel=True,
)

exp_config = ExpConfig(
    name="mini-qwen3",
    seed=42,
    model=model_config,
    data=data_config,
    optim=optim_config,
    parallel=parallel_config,
    train=train_config,
)

from train import train

train(exp_config)