import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["JAX_COMPILER_ENABLE_REMAT_PASS"] = "true"

from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from utils.configs import OptimizerConfig, ExperimentConfig, ParallelConfig, HFDataConfig

from modelling.models.gpt import GPTConfig

sequence_length = 1024

max_steps = 20_000
warmup_steps = 700

model_config = GPTConfig(
    vocab_size=50304,
    hidden_dim=768,
    num_layers=12,
    num_attention_heads=12,
    intermediate_dim=3072,
    head_dim=64,
    act_fn=nnx.gelu,
    max_seq_len=sequence_length,
    layer_norm_epsilon=1e-5,
    use_bias=False,
    dtype=jnp.bfloat16,
)

train_data = HFDataConfig(
    source="hf",
    hf_name=["HuggingFaceFW/fineweb-edu"],
    tokenizer_name="gpt2",
    max_length=sequence_length,
)


optim_config = OptimizerConfig(
    name="adamw",
    weight_decay=0.1,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    batch_size=64*8,
    accum_steps=1,
    eps=1e-8,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=6e-4,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=6e-5
    )
)


parallel_config = ParallelConfig(
    data_parallel=8,
)

exp_config = ExperimentConfig(
    name="gpt2",
    seed=42,
    model=model_config,
    optimizer=optim_config,
    parallel=parallel_config,
    train_data=train_data,
    val_data=None,
    steps=max_steps,
    log_every=1,
    generate_every=500,
    eval_every=-1,
    save_every=2000,
    save_dir="checkpoints",
    trace_dir="traces",
    start_trace_micro_step=10,
    end_trace_micro_step=20,
    gpu="H100",
)

from train import train

train(exp_config)