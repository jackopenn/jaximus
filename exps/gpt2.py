import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["JAX_COMPILER_ENABLE_REMAT_PASS"] = "false"
# # os.environ["NCCL_DEBUG"] = "INFO"

# # Critical P2P memory reductions
# os.environ["NCCL_BUFFSIZE"] = "1048576"        # Reduce from 4MB to 1MB per channel
# os.environ["NCCL_NTHREADS"] = "2"              # Reduce NCCL worker threads  
# os.environ["NCCL_MAX_NCHANNELS"] = "2"         # Limit channels (was 24!)
# os.environ["NCCL_MIN_NCHANNELS"] = "2"         # Force minimal channels
# os.environ["NCCL_P2P_DISABLE"] = "0"           # Keep P2P but reduce memory
# os.environ["NCCL_SHM_DISABLE"] = "1"   

from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from utils.configs import OptimizerConfig, ExperimentConfig, ParallelConfig, HFDataConfig, ArrayRecordDataConfig

from modelling.models.gpt import GPTConfig

sequence_length = 4096

max_steps = 60_000
warmup_steps = 700

model_config = GPTConfig(
    vocab_size=50304,
    hidden_dim=2048,
    num_layers=24,
    num_attention_heads=16,
    intermediate_dim=8192,
    head_dim=128,
    act_fn=nnx.gelu,
    max_seq_len=sequence_length,
    layer_norm_epsilon=1e-5,
    use_bias=False,
    dtype=jnp.bfloat16,
)

train_data = HFDataConfig(
    hf_name=["HuggingFaceFW/fineweb-edu", "sample-10BT"],
    tokenizer_name="gpt2",
    max_length=sequence_length,
    streaming=True,
)

# train_data = ArrayRecordDataConfig(
#     path="/root/jaximus/notebooks/saved/HuggingFaceFW/fineweb-edu/sample-10BT",
#     max_length=sequence_length,
#     tokenizer_name="gpt2",
# )


optim_config = OptimizerConfig(
    name="adamw",
    weight_decay=0.1,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    batch_size=2,
    accum_steps=1,
    eps=1e-8,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=warmup_steps,
        decay_steps=max_steps - warmup_steps,
        end_value=6e-5
    )
)


parallel_config = ParallelConfig(
    data_parallel=1,
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
    log_every=20,
    generate_every=500,
    eval_every=-1,
    save_every=5000,
    save_dir="checkpoints",
    trace_dir="traces",
    start_trace_micro_step=10,
    end_trace_micro_step=20,
    gpu="H100",
)

from train import train

train(exp_config)