import os 

from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from modelling.models.llama import LlamaConfig
from utils.configs import DataConfig, DummyDataConfig, HFDataConfig, OptimizerConfig, ExperimentConfig, ParallelConfig


model_config = LlamaConfig(
    vocab_size=128256,
    hidden_dim=2048,
    num_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_dim=8192,
    head_dim=64,
    act_fn=nnx.silu,
    max_seq_len=4096,
    use_attention_bias=False,
    use_mlp_bias=False,
    rms_norm_eps=1e-5,
    rope_theta=500000,
    dtype=jnp.bfloat16,
)

# train_data = DummyDataConfig(
train_data = HFDataConfig(
     source="hf",
     hf_name=["allenai/c4", "realnewslike"],
     tokenizer_name="gpt2",
    #  source="dummy",
    max_length=4096,
)

optim_config = OptimizerConfig(
    name="adamw",
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    batch_size=1,
    accum_steps=1,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=6e-4,
        warmup_steps=1_000,
        decay_steps=99_000,
        end_value=6e-5
    )
)


parallel_config = ParallelConfig(
    data_parallel=1,
)

exp_config = ExperimentConfig(
    name="llama",
    seed=42,
    model=model_config,
    optimizer=optim_config,
    parallel=parallel_config,
    train_data=train_data,
    val_data=None,
    steps=100,
    log_every=50,
    generate_every=10000,
    eval_every=-1,
    save_every=1_0000,
    save_dir="checkpoints",
    trace_dir="traces",
    start_trace_step=10,
    end_trace_step=20,
    gpu="H100",
)

from train import train

train(exp_config)