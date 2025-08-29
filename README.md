# Jaximus 🚀

**A barebones library for pretraining Large Language Models**

Built with JAX ecosystem: **Jax/Flax/Optax/Orbax/Grain**

## Overview

Jaximus is a minimalistic yet powerful framework designed for efficient LLM pretraining. It leverages the JAX ecosystem to provide high-performance, scalable training with clean, readable code that's easy to understand and modify.

### Key Features

- **🔥 High Performance**: Built on JAX for XLA compilation and efficient GPU/TPU utilization
- **📊 Model Support**: GPT-2 and Qwen3 architectures with easy extensibility
- **⚡ Efficient Training**: Multi-device data parallelism with gradient accumulation
- **🎯 MFU Tracking**: Model FLOPS Utilization monitoring for performance optimization
- **📈 W&B Integration**: Comprehensive logging and experiment tracking
- **🔄 Checkpointing**: Automatic model saving with Orbax
- **📦 Modern Stack**: Uses Flax NNX for clean, Pythonic neural network definitions

## Architecture

```
jaximus/
├── modelling/           # Model implementations
│   ├── models/         # GPT, Qwen3 architectures
│   └── layers/         # Core layers (MLP, GLU, GQA, RoPE)
├── data/               # Data loading (HuggingFace integration)
├── utils/              # Configuration, optimizers, metrics
├── exps/               # Experiment configurations
└── train.py           # Main training loop
```

## Supported Models

### GPT-2
- Standard transformer architecture with learned positional embeddings
- Multi-head attention with configurable bias settings
- Layer normalization with pre-norm or post-norm options

### Qwen3
- Modern architecture with RoPE (Rotary Position Embedding)
- Grouped Query Attention (GQA) for efficient inference
- GLU (Gated Linear Units) in feed-forward layers
- RMS normalization with QK normalization

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd jaximus

# Install dependencies (requires Python 3.11+)
pip install -e .
```

### Basic Usage

1. **Configure your experiment** (see `exps/` for examples):

```python
from utils.configs import *
from modelling.models.gpt import GPTConfig

# Model configuration
model_config = GPTConfig(
    vocab_size=50257,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    intermediate_dim=3072,
    max_seq_len=1024,
    dtype=jnp.bfloat16,
)

# Data configuration
data_config = DataConfig(
    source="hf",
    hf_name=["allenai/c4", "realnewslike"],
    tokenizer_name="gpt2",
    max_length=1024,
)

# Optimizer with warmup + cosine decay
optim_config = OptimConfig(
    name="adamw",
    batch_size=16,
    accum_steps=32,
    lr=optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=6e-4,
        warmup_steps=1_000,
        decay_steps=99_000,
    )
)
```

2. **Run training**:

```python
from train import train

exp_config = ExpConfig(
    name="my-experiment",
    model=model_config,
    data=data_config,
    optim=optim_config,
    # ... other configs
)

train(exp_config)
```

## Features in Detail

### Data Pipeline
- **HuggingFace Integration**: Load datasets directly from HF Hub
- **Grain Processing**: Efficient data loading with concatenation and chunking
- **Tokenization**: Automatic tokenization with configurable tokenizers

### Training Features
- **Multi-device**: Automatic data parallelism across available GPUs/TPUs
- **Gradient Accumulation**: Effective large batch training on limited hardware
- **Mixed Precision**: bfloat16 training for memory efficiency
- **Gradient Clipping**: Stable training with configurable gradient norms

### Monitoring & Logging
- **Real-time Metrics**: Loss, learning rate, tokens/second, MFU
- **Text Generation**: Periodic sample generation during training
- **Checkpointing**: Automatic model saving with configurable intervals
- **W&B Integration**: Comprehensive experiment tracking

### Performance Optimizations
- **JIT Compilation**: `@nnx.jit` for optimized training steps
- **Cached Partials**: Efficient function compilation with `nnx.cached_partial`
- **Sharding**: Automatic model and data sharding for multi-device setups

## Configuration System

Jaximus uses a clean dataclass-based configuration system:

```python
@dataclass
class ExpConfig:
    name: str
    seed: int
    model: ModelConfig      # Model architecture
    data: DataConfig        # Dataset configuration  
    optim: OptimConfig      # Optimizer settings
    parallel: ParallelConfig # Parallelism settings
    train: TrainConfig      # Training hyperparameters
    gpu: str               # GPU type for MFU calculation
```

## Example Experiments

### Mini GPT-2 (124M parameters)
```bash
python exps/gpt2.py
```

### Mini Qwen3 (Similar size)
```bash
python exps/qwen3.py
```

## Performance

Jaximus is designed for efficiency:

- **Model FLOPS Utilization (MFU)**: Track how efficiently you're using your hardware
- **Tokens/second**: Monitor training throughput
- **Memory Efficient**: bfloat16 precision and gradient accumulation
- **Scalable**: Data parallelism across multiple devices

## Dependencies

Core dependencies from `pyproject.toml`:

- **jax** - Core computation framework
- **flax** - Neural network library
- **optax** - Gradient-based optimization
- **orbax-checkpoint** - Model checkpointing
- **grain** - Data loading pipeline
- **transformers** - Tokenizers and utilities
- **datasets** - HuggingFace datasets
- **wandb** - Experiment tracking

## Contributing

Jaximus is designed to be simple and extensible. To add new models:

1. Create a new model class in `modelling/models/`
2. Add the config dataclass
3. Update `utils/getters.py` to include your model
4. Create an experiment config in `exps/`

## License

[Add your license here]

## Citation

If you use Jaximus in your research, please cite:

```bibtex
@software{jaximus,
  title={Jaximus: A Barebones Library for LLM Pretraining},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

---

**Built with ❤️ and JAX**

For questions, issues, or contributions, please visit our [GitHub repository](repository-url).