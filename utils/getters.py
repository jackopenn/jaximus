from dataclasses import dataclass
from typing import Callable
from data.dummy import get_dummy_dataset
from data.hf import get_hf_dataset
from data.array_records import get_array_record_dataset
from modelling.models.gpt import GPT, GPTConfig
from modelling.models.qwen3 import Qwen3, Qwen3Config
from modelling.models.llama import Llama, LlamaConfig
from utils.configs import DataConfig, ModelConfig, OptimizerConfig, HFDataConfig, DummyDataConfig, ArrayRecordDataConfig, ParallelConfig

import optax
import jax
from flax import nnx

def get_dataset(config: DataConfig, batch_size: int):
    if isinstance(config, HFDataConfig):
        return get_hf_dataset(
            hf_name=config.hf_name,
            tokenizer_name=config.tokenizer_name,
            sequence_length=config.max_length,
            batch_size=batch_size
        )
    elif isinstance(config, DummyDataConfig):
        return get_dummy_dataset(
            max_length=config.max_length,
            batch_size=batch_size
        )
    elif isinstance(config, ArrayRecordDataConfig):
        return get_array_record_dataset(
            path=config.path,
            sequence_length=config.max_length,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Dataset {config} not found")
    

def get_model(config: ModelConfig, seed: int, parallel: ParallelConfig):
    shard_axis_name = "data" if parallel.data_parallel > 1 else None
    if isinstance(config, GPTConfig):
        return GPT(config, nnx.Rngs(jax.random.PRNGKey(seed)))
    elif isinstance(config, Qwen3Config):
        return Qwen3(config, nnx.Rngs(jax.random.PRNGKey(seed)))
    elif isinstance(config, LlamaConfig):
        return Llama(config, rngs=nnx.Rngs(jax.random.PRNGKey(seed)), shard_axis_name=shard_axis_name)
    else:
        raise ValueError(f"Model {config} not found")
    

def get_optimizer(model, config: OptimizerConfig):
    if config.name == "adamw":
        tx = optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(config.grad_clip),
                optax.adamw(
                    learning_rate=config.lr,
                    weight_decay=config.weight_decay,
                    mask=lambda params: jax.tree.map(lambda x: x.ndim != 1, params), # only wd for 2d tensors
                    b1=config.betas[0],
                    b2=config.betas[1],
                    eps=config.eps,
                )
            ),
            every_k_schedule=config.accum_steps,
        )
    else:
        raise ValueError(f"Optimizer {config} not found")
    
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
