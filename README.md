# Jaximus

An opinionated, explicit, and functional JAX library for model experimentation.

## Setup

```bash
# Clone and install
git clone <repo> && cd jaximus
uv venv && source .venv/bin/activate
uv pip install -e ".[tpu]"  # or [gpu] or [cpu]
```

## Usage

```bash
python train.py --config exps/nanochat100_base.py
```

## Configuration

Experiments are Python files in `exps/` that return a `Config` object. Values can be constants or lambdas for derived values:

```python
from sws import Config

def get_config():
    cfg = Config()
    cfg.seed = 42

    # Model
    cfg.model.num_layers = 20
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: cfg.model.hidden_dim // 128

    # Data
    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.batch_size = 64

    # Optimizer (partitioned by parameter name)
    cfg.optimizer.embed.patterns = ["embed", "pos_embed"]
    cfg.optimizer.embed.type = "adamw"
    cfg.optimizer.embed.peak_lr = 0.01

    cfg.optimizer.other.patterns = "*"
    cfg.optimizer.other.type = "muon"
    cfg.optimizer.other.peak_lr = 0.02

    # Training
    cfg.max_steps = 10000
    cfg.parallel.strategy = "fsdp"  # or "dp"
    cfg.parallel.data = 16

    return cfg
```

Override any value from the command line:

```bash
python train.py --config exps/nanochat100_base.py --model.num_layers 12 --data.batch_size 32
```
