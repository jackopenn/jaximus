from dataclasses import dataclass
from typing import Callable


@dataclass
class DataConfig:
    source: str
    hf_name: str
    tokenizer_name: str
    max_length: int


@dataclass
class ModelConfig:
    name: str


@dataclass
class OptimConfig:
    name: str
    batch_size: int
    lr: float | Callable
    weight_decay: float
    betas: tuple[float, float]
    grad_clip: float
    accum_steps: int


@dataclass
class ParallelConfig:
    data_parallel: int


@dataclass
class TrainConfig:
    num_steps: int
    log_every: int
    generate_every: int
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
    parallel: ParallelConfig
    train: TrainConfig
    gpu: str
