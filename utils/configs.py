import chz
from typing import Callable, List, Optional, Tuple
from jax import numpy as jnp

@chz.chz
class ModelConfig:
    name: str
    dtype: jnp.dtype

    
@chz.chz
class OptimizerConfig:
    name: str
    weight_decay: float
    betas: Tuple[float, float]
    grad_clip: float
    batch_size: int
    accum_steps: int
    lr: float | Callable
    eps: float


@chz.chz
class ParallelConfig:
    data_parallel: int


@chz.chz
class DataConfig:
    source: str
    # hf_name: List[str]
    # tokenizer_name: str
    max_length: int

@chz.chz
class HFDataConfig(DataConfig):
    hf_name: List[str]
    tokenizer_name: str

@chz.chz
class DummyDataConfig(DataConfig):
    pass

@chz.chz
class ExperimentConfig:
    name: str
    model: ModelConfig
    optimizer: OptimizerConfig
    parallel: ParallelConfig
    train_data: DataConfig
    val_data: Optional[DataConfig]    
    steps: int
    log_every: int
    generate_every: int
    eval_every: int
    save_every: int
    save_dir: str
    trace_dir: str
    start_trace_micro_step: int
    end_trace_micro_step: int
    gpu: str
    seed: int = 42