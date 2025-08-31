from jax import numpy as jnp
from flax import nnx
import optax

from utils.configs import OptimizerConfig, ParallelConfig, DataConfig, ExperimentConfig
from modelling.models.qwen3 import Qwen3Config

model = Qwen3Config(
    vocab_size=50257,
    hidden_dim=768,
    intermediate_dim=2048,
    num_layers=12,
    num_attention_heads=12,
    num_key_value_heads=6,
    head_dim=64,
    act_fn=nnx.silu,
    max_seq_len=1024,
    rope_theta=1000000,
    dtype=jnp.bfloat16,
)

optimizer = OptimizerConfig(
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

parallel = ParallelConfig(
    data_parallel=1,
)

train_data = DataConfig(
    source="hf",
    hf_name=["allenai/c4", "realnewslike"],
    tokenizer_name="gpt2",
    max_length=1024,
)

experiment = ExperimentConfig(
    name="mini-qwen3",
    model=model,
    optimizer=optimizer,
    parallel=parallel,
    train_data=train_data,
    val_data=None,
    steps=1000,
    log_every=1,
    generate_every=10000,
    eval_every=-1,
    save_every=1_0000,
    save_dir="checkpoints",
    gpu="5090",
)


from train import train

train(experiment)