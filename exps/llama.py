import os 

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_COMPILER_ENABLE_REMAT_PASS"] = "false"

# no difference
os.environ['XLA_FLAGS'] = "--xla_gpu_enable_latency_hiding_scheduler=true" # --xla_gpu_enable_command_buffer='' "


# os.environ["NCCL_DEBUG"] = "INFO"


from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from modelling.models.llama import LlamaConfig
from utils.configs import DataConfig, DummyDataConfig, HFDataConfig, OptimizerConfig, ExperimentConfig, ParallelConfig

sequence_length = 4096
n_gpu = 4
micro_batch_size = 8
accum_steps = 1
gpu_name = "v5p"

model_config = LlamaConfig(
    vocab_size=128256,
    hidden_dim=2048,
    num_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_dim=8192,
    head_dim=64,
    act_fn=nnx.silu,
    max_seq_len=sequence_length,
    use_attention_bias=False,
    use_mlp_bias=False,
    rms_norm_eps=1e-5,
    rope_theta=500000,
    dtype=jnp.bfloat16,
)

train_data = HFDataConfig(
    hf_name=["HuggingFaceFW/fineweb-edu", "sample-10BT"],
    tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
    max_length=sequence_length,
)

optim_config = OptimizerConfig(
    name="adamw",
    weight_decay=0.01,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    batch_size=micro_batch_size * n_gpu,
    accum_steps=accum_steps,
    eps=1e-8,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=6e-4,
        warmup_steps=1_000,
        decay_steps=99_000,
        end_value=6e-5
    )
)


parallel_config = ParallelConfig(
    data_parallel=n_gpu,
    zero_stage=3
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
    generate_every=10000,
    eval_every=-1,
    save_every=1_0000,
    save_dir="checkpoints",
    trace_dir="traces",
    start_trace_micro_step=10,
    end_trace_micro_step=20,
    gpu=gpu_name,
)

from train import train

train(exp_config)