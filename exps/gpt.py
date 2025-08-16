from dataclasses import dataclass, field
from typing import Callable

import jax
from flax import nnx

from utils import ModelConfig, DataConfig, OptimConfig, TrainConfig, ExpConfig

from modelling.models.gpt import GPTConfig

# @dataclass
# class ModelConfig(GPTConfig):
#     vocab_size: int = 50257
#     num_layers: int = 12
#     num_heads: int = 12
#     hidden_dim: int = 768
#     intermediate_dim: int = 3072
#     act_fn: Callable = jax.nn.gelu
#     max_seq_len: int = 1024
#     layer_norm_epsilon: float = 1e-5


# @dataclass  
# class DataConfig:
#     source: str = "hf"
#     hf_name: str = "allenai/c4"
#     tokenizer_name: str = "gpt2"
#     max_length: int = 1024


# @dataclass
# class OptimConfig:
#     name: str = 'adamw'
#     lr: float = 1e-4
#     weight_decay: float = 0.0
#     betas: tuple[float, float] = (0.9, 0.95)


# @dataclass
# class TrainConfig:
#     num_steps: int = 1000
#     eval_every: int = 100
#     save_every: int = 100
#     save_dir: str = "checkpoints"


model_config = GPTConfig(
    vocab_size=50257,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    intermediate_dim=3072,
    act_fn=nnx.gelu,
    max_seq_len=1024,
    layer_norm_epsilon=1e-5,
)

data_config = DataConfig(
    source="hf",
    hf_name=["allenai/c4", "realnewslike"],
    tokenizer_name="gpt2",
    max_length=1024,
    batch_size=2,
)

optim_config = OptimConfig(
    name="adamw",
    lr=5e-4,
    weight_decay=0.0,
    betas=(0.9, 0.95),
    grad_clip=1.0,
    accum_steps=2,
)

train_config = TrainConfig(
    num_steps=1000,
    log_every=2,
    eval_every=10,
    save_every=100,
    save_dir="checkpoints",
)

exp_config = ExpConfig(
    name="gpt-test",
    seed=42,
    model=model_config,
    data=data_config,
    optim=optim_config,
    train=train_config,
)

from train import train

train(exp_config)