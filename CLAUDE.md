# CLAUDE.md

---
## Jack Suggestions

### Overview
- Jaximus is a jax codebase for experimenting with different LLM architectures and training methods
- The main approach is to create an experiment in `experiments/`. Which contains at minimum 3 files:
    - `config.py`: contains the config of the exp.
    - `model.py`: contains the model definition and forward pass. Implements at minimum `init_model_weights()` and `model_forward()` functions.
    - `optimizer.py`: contains `make_optimizer()` function that returns an optax optimizer.
- The rest of the code not in `experiments/` contains general training logic that should work with any implementation of the functions above (checkpointing, logging, sampling etc.)

### Style
- Don't use large comments. Max 1 line per function definition.
- Only comment if understanding the code is not trivial.
- Do not create variables if only used once, prefer inlining.
- Keep style consistent with the file.
- 120 character line limit
- Readability over speed

---
## Claude Suggestions
*Claude can update this section whenever useful*

### Commands
- `source .venv/bin/activate` before running
- `python train.py --config experiments/nanochat/config_debug.py` - quick validation
- `python train.py --config experiments/nanochat/config.py` - full training

### Key Modules
- `modelling/layers/` - Reusable building blocks (norm, attention, mlp, position)
- `parallel.py` - Sharding rules and `l2p()`
- `scheduler.py` - LR schedule utilities (`make_lr_schedule_fns`)
- `generate.py` - Text generation/sampling

### Patterns
- All weight dataclasses use `@jax.tree_util.register_dataclass`
- Sharding via `out_sharding=l2p(...)` on ops
- Forward passes are pure functional (no state mutation)
- Config values can be lambdas for derived values: `cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64`
