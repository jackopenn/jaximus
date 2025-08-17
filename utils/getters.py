from dataclasses import dataclass
from typing import Callable
from data.hf import get_hf_dataset
from modelling.models.gpt import GPT, GPTConfig
from modelling.models.qwen3 import Qwen3, Qwen3Config
from utils.configs import DataConfig, ModelConfig, OptimConfig

import optax
import jax
from flax import nnx

def get_dataset(config: DataConfig):
    if config.source == "hf":
        return get_hf_dataset(
            hf_name=config.hf_name,
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
        )
    else:
        raise ValueError(f"Dataset source {config.source} not found")
    
def get_model(config: ModelConfig, seed: int):
    if isinstance(config, GPTConfig):
        return GPT(config, nnx.Rngs(jax.random.PRNGKey(seed)))
    elif isinstance(config, Qwen3Config):
        return Qwen3(config, nnx.Rngs(jax.random.PRNGKey(seed)))
    else:
        raise ValueError(f"Model {config.name} not found")
    


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
