# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jaximus is a JAX transformer implementation with configurable sharding for training language models on TPUs and GPUs. It uses a functional programming style with explicit weight dataclasses.

## Commands

### Training
```bash
# Single-host training (uses sws-config for experiment configs)
python train.py --config exps/nanochat100_base.py

# Multi-host training (set JAX_MULTIHOST=1 to initialize jax.distributed)
JAX_MULTIHOST=1 python train.py --config exps/nanochat100_base.py
```

### Dependencies
```bash
# Install with uv (recommended)
uv pip install -e ".[tpu]"  # or [gpu] or [cpu]
```

## Architecture

### Weight System (Functional Style)
The codebase uses explicit dataclasses for weights:
- `ModelWeights` in `modelling/model.py` - top-level container with embed, layer_weights, unembed
- `LayerWeights` - per-layer attention + MLP weights and norms
- `AttentionWeights`, `MLPWeights`, `GLUWeights` in `modelling/layers/core.py`
- All weight dataclasses are registered with `@jax.tree_util.register_dataclass` for JAX compatibility

### Sharding System
- `parallel.py` defines logical-to-physical axis mapping via `SHARDING_RULES`
- Two strategies: `dp` (data parallel) and `fsdp` (fully sharded data parallel)
- `logical_to_physical()` converts logical axes (e.g., "batch", "model_embed") to PartitionSpec
- Sharding is applied inline in ops via `out_sharding` parameter on JAX operations

### Configuration (sws-config)
- Experiment configs in `exps/` define all hyperparameters
- Model configs in `configs/` define architecture presets (e.g., gpt2_small)
- Configs support lambdas for derived values (e.g., `hidden_dim = lambda: num_layers * 64`)
- Run with `sws.run(train)` which handles CLI argument parsing

### Optimizer System
- `optimizer.py` supports partitioned optimizers with different settings per parameter group
- Partitions defined by patterns matching weight names (e.g., "embed", "unembed", "*")
- Supports AdamW and Muon optimizers with separate hyperparameters per partition
- Muon optimizer in `muon.py` implements Newton-Schulz orthogonalization with layer sharding

### Data Pipeline
- Uses Grain for data loading with HuggingFace datasets
- `data/hf.py` handles streaming datasets with automatic sharding across hosts
- Tokenization and sequence packing via `ConcatThenSplitIterDataset`

### Key Model Options
- Norm types: `rms`, `layer` (pre or post position)
- MLP types: `mlp` (standard), `glu` (gated linear unit)
- Position embeddings: `learned`, `rope`, `none`
- Supports QK normalization, sliding window attention, logit softcap
- Init strategies: `default` (lecun), `nanochat` (uniform with zero residual projections)

## Important Patterns

- Forward pass in `modelling/model.py:forward()` is pure functional
- Training step in `train.py:make_train_step()` returns JIT-compiled function with explicit shardings
- Multi-host: all processes must call `generate()` together; results returned only on main process
- Checkpointing uses Orbax with GCS support (`gs://` paths)
- Profiling automatically captures steps 10-20 and uploads to wandb
