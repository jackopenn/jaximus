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
    zero_stage: Optional[int]


@chz.chz
class DataConfig:
    max_length: int

@chz.chz
class HFDataConfig(DataConfig):
    hf_name: List[str]
    tokenizer_name: str
    streaming: bool = True
    num_proc: Optional[int] = None

@chz.chz
class DummyDataConfig(DataConfig):
    pass

@chz.chz
class ArrayRecordDataConfig(DataConfig):
    path: str
    tokenizer_name: str
    max_length: int

@chz.chz
class ExperimentConfig:
    name: str
    model: ModelConfig
    optimizer: OptimizerConfig
    parallel: ParallelConfig
    train_data: DataConfig
    val_data: Optional[DataConfig]    
    steps: int
    generate_every: int
    eval_every: int
    save_every: int
    save_dir: str
    trace_dir: str
    start_trace_micro_step: int
    end_trace_micro_step: int
    gpu: str
    seed: int = 42