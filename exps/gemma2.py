import os 
from functools import partial
from functools import partial
from functools import partial

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_COMPILER_ENABLE_REMAT_PASS"] = "false"

# no difference
os.environ['XLA_FLAGS'] = "--xla_gpu_enable_latency_hiding_scheduler=true" # --xla_gpu_enable_command_buffer='' "


# os.environ["NCCL_DEBUG"] = "INFO"

import jax
from jax import numpy as jnp
from flax import nnx
import optax

from modelling.models.gemma2 import Gemma2Config
from utils.configs import HFDataConfig, OptimizerConfig, ExperimentConfig, ParallelConfig

sequence_length = 8192
n_gpu = 4
micro_batch_size = 8
accum_steps = 1
gpu_name = "v5p"

# ganna2-2b
model_config = Gemma2Config(
    vocab_size=256128,
    hidden_dim=2304,
    num_layers=26,
    num_attention_heads=8,
    num_key_value_heads=4,
    intermediate_dim=18432,
    head_dim=256,
    act_fn=partial(nnx.gelu, approximate=True),
    max_seq_len=sequence_length,
    use_attention_bias=False,
    use_mlp_bias=False,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    dtype=jnp.bfloat16,
    sliding_window=4096,
    attn_logit_softcapping=30.0,
    final_logit_softcapping=50.0,
)

train_data = HFDataConfig(
    hf_name=["HuggingFaceFW/fineweb-edu", "sample-10BT"],
    tokenizer_name="google/gemma-2-2b",
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
    name="gemma2",
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