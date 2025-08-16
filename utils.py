from dataclasses import dataclass
from typing import Callable
from ds.hf import get_hf_dataset
from modelling.models.gpt import GPT, GPTConfig

import optax
import jax
from flax import nnx

@dataclass
class DataConfig:
    source: str
    hf_name: str
    tokenizer_name: str
    max_length: int
    batch_size: int


def get_dataset(config: DataConfig):
    if config.source == "hf":
        return get_hf_dataset(
            hf_name=config.hf_name,
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
        )
    else:
        raise ValueError(f"Dataset source {config.source} not found")
    

@dataclass
class ModelConfig:
    name: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    act_fn: str
    max_seq_len: int

def get_model(config: ModelConfig, seed: int):
    if isinstance(config, GPTConfig):
        return GPT(config, nnx.Rngs(jax.random.PRNGKey(seed)))
    else:
        raise ValueError(f"Model {config.name} not found")
    
@dataclass
class OptimConfig:
    name: str
    batch_size: int
    lr: float | Callable
    weight_decay: float
    betas: tuple[float, float]
    grad_clip: float
    accum_steps: int

def get_optimizer(model, config: OptimConfig):
    if config.name == "adamw":
        tx = optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(config.grad_clip),
                optax.adamw(
                    learning_rate=config.lr,
                    weight_decay=config.weight_decay,
                    b1=config.betas[0],
                    b2=config.betas[1],
                )
            ),
            every_k_schedule=config.accum_steps,
        )
    else:
        raise ValueError(f"Optimizer {config.name} not found")
    
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


@dataclass
class TrainConfig:
    num_steps: int
    log_every: int
    eval_every: int
    save_every: int
    save_dir: str


@dataclass
class ExpConfig:
    name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    train: TrainConfig

