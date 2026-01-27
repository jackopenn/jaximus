# Jaximus

An opinionated, explicit, and functional JAX library for model experimentation.

## Setup

```bash
git clone <repo> && cd jaximus
uv venv && source .venv/bin/activate
uv pip install -e ".[tpu]"  # or [gpu] or [cpu]
```

## Usage

```bash
python train.py --config experiments/nanochat/config.py
```

## Creating Experiments

Each experiment is a self-contained directory under `experiments/` with 3 required files:

```
experiments/
└── my_experiment/
    ├── __init__.py
    ├── config.py      # get_config() -> Config
    ├── model.py       # init_model_weights(), model_forward()
    └── optimizer.py   # make_optimizer()
```

### config.py

```python
from sws import Config

def get_config():
    cfg = Config()
    cfg.seed = 42
    cfg.exp_name = "my-experiment"

    cfg.model.num_layers = 20
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64  # derived values

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.batch_size = 64

    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.embed.peak_lr = 0.01
    cfg.optimizer.other.peak_lr = 0.02

    cfg.max_steps = 10000
    cfg.parallel.strategy = "fsdp"  # or "dp"
    cfg.parallel.data = 16

    return cfg.finalize()
```

### model.py

```python
def init_model_weights(config, key):
    """Initialize model weights. Returns a pytree of arrays."""
    ...

def make_model_forward(config):
    """Factory that returns a forward function with precomputed values.

    Returns a partial function: forward(x, weights, mask=None) -> logits

    The factory precomputes rope embeddings and binds config via functools.partial.
    Called after mesh is set to enable proper sharding of precomputed values.
    """
    rope_cos, rope_sin = precompute_rope_embeddings(...)
    return partial(_model_forward, config=config, rope_cos=rope_cos, rope_sin=rope_sin)

def _model_forward(x, weights, config, rope_cos, rope_sin, mask=None):
    """Internal forward pass. Returns logits."""
    ...
```

### optimizer.py

```python
def make_optimizer(cfg):
    """Create optimizer. Returns (optax_tx, config_for_logging, schedule_fns_for_logging)."""
    tx = ...  # optax optimizer
    resolved_config = {...}  # dict of optimizer settings for logging
    schedule_fns = {  # pure Python schedule functions for logging
        "lr_embed": lambda step: ...,
        "lr_other": lambda step: ...,
        "momentum_other": lambda step: ...,
    }
    return tx, resolved_config, schedule_fns
```
