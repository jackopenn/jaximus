"""Optimizer creation and learning rate schedules."""
import jax
from jax import numpy as jnp
import optax

from muon import muon


def warmup_stable_decay_schedule(
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    max_steps: int,
    decay_type: str = "linear",
):
    """Learning rate schedule with warmup, stable, and decay phases.

    Args:
        peak_value: Peak learning rate
        warmup_steps: Steps to warm up from 0 to peak
        decay_steps: Steps to decay from peak to 0
        max_steps: Total training steps
        decay_type: "linear" or "cosine"
    """
    def schedule(step):
        # Warmup phase: 0 -> peak
        warmup_pct = step / jnp.maximum(warmup_steps, 1)
        warmup_value = peak_value * warmup_pct

        # Decay phase
        decay_start = max_steps - decay_steps
        decay_pct = (step - decay_start) / jnp.maximum(decay_steps, 1)

        if decay_type == "cosine":
            # Cosine decay: peak -> 0
            decay_value = peak_value * 0.5 * (1 + jnp.cos(jnp.pi * decay_pct))
        else:
            # Linear decay: peak -> 0
            decay_value = peak_value * (1 - decay_pct)

        return jnp.where(
            step < warmup_steps,
            warmup_value,
            jnp.where(step < decay_start, peak_value, decay_value)
        )
    return schedule


def muon_momentum_schedule(step):
    """Momentum schedule for Muon optimizer."""
    frac = jnp.minimum(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


# Reserved names for optimizer type defaults
OPTIMIZER_TYPES = {'adamw', 'muon'}

# Global config keys (not partitions or type defaults)
GLOBAL_KEYS = {'grad_clip_norm', 'accum_steps', 'peak_lr', 'warmup_steps', 'decay_steps', 'decay_type'}


def make_optimizer(cfg):
    """Create optimizer from config.

    Config structure:
        cfg.optimizer.grad_clip_norm - optional global grad clipping
        cfg.optimizer.accum_steps - gradient accumulation steps
        cfg.optimizer.adamw.* - defaults for adamw partitions
        cfg.optimizer.muon.* - defaults for muon partitions
        cfg.optimizer.<name>.patterns - partition definition (has 'patterns' = partition)

    All values must be explicitly set in config - no defaults.

    Returns:
        Tuple of (optax optimizer, resolved config dict for logging)
    """
    opt_cfg = cfg.optimizer
    opt_dict = opt_cfg.to_dict() if hasattr(opt_cfg, 'to_dict') else opt_cfg

    # Separate into: global settings, type defaults, and partitions
    global_settings = {}
    type_defaults = {}
    partitions = {}

    for key, value in opt_dict.items():
        if key in GLOBAL_KEYS:
            global_settings[key] = value
        elif key in OPTIMIZER_TYPES and isinstance(value, dict):
            type_defaults[key] = value
        elif isinstance(value, dict) and 'patterns' in value:
            partitions[key] = value

    # Default partition if none specified
    if not partitions:
        peak_lr = global_settings.get('peak_lr')
        if peak_lr is None:
            raise ValueError("No partitions defined and no cfg.optimizer.peak_lr set")
        partitions = {
            "default": {
                "patterns": "*",
                "type": "adamw",
                "peak_lr": peak_lr,
            }
        }

    def resolve(val):
        return val() if callable(val) else val

    resolved_config = {}
    exact_patterns = {}
    catch_all = None
    partition_optimizers = {}

    for name, part_cfg in partitions.items():
        patterns = part_cfg['patterns']
        if isinstance(patterns, str):
            patterns = [patterns]
        opt_type = part_cfg['type']
        defaults = type_defaults.get(opt_type, {})

        # Get values with fallback chain: partition -> type defaults -> global (no defaults)
        def get_val(key):
            if key in part_cfg:
                return resolve(part_cfg[key])
            if key in defaults:
                return resolve(defaults[key])
            if key in global_settings:
                return resolve(global_settings[key])
            return None

        def require_val(key):
            val = get_val(key)
            if val is None:
                raise ValueError(f"Partition '{name}' missing required '{key}'")
            return val

        peak_lr = require_val('peak_lr')

        # Schedule params are optional - if not set, use constant LR
        warmup_steps = get_val('warmup_steps')
        decay_steps = get_val('decay_steps')
        decay_type = get_val('decay_type') or 'linear'

        # Build resolved partition config
        resolved_partition = {
            "patterns": patterns,
            "type": opt_type,
            "peak_lr": peak_lr,
        }

        # Create schedule (optional warmup/decay)
        if warmup_steps is not None or decay_steps is not None:
            warmup_steps = warmup_steps or 0
            decay_steps = decay_steps or 0
            schedule_fn = warmup_stable_decay_schedule(
                peak_value=peak_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                max_steps=cfg.max_steps,
                decay_type=decay_type,
            )
            resolved_partition["warmup_steps"] = warmup_steps
            resolved_partition["decay_steps"] = decay_steps
            resolved_partition["decay_type"] = decay_type
        else:
            schedule_fn = peak_lr  # constant LR

        # Create optimizer based on type
        if opt_type == "adamw":
            weight_decay = require_val('weight_decay')
            eps = require_val('eps')
            b1 = require_val('b1')
            b2 = require_val('b2')

            partition_optimizers[name] = optax.adamw(
                learning_rate=schedule_fn,
                weight_decay=weight_decay,
                eps=eps,
                b1=b1,
                b2=b2,
            )
            resolved_partition.update({
                "weight_decay": weight_decay,
                "eps": eps,
                "b1": b1,
                "b2": b2,
            })

        elif opt_type == "muon":
            beta_raw = get_val('beta') or muon_momentum_schedule
            if callable(beta_raw):
                beta_schedule = beta_raw
                beta_val = "schedule"
            else:
                beta_schedule = lambda step, b=beta_raw: b
                beta_val = beta_raw
            nesterov = require_val('nesterov')
            layer_sharding = require_val('layer_sharding')

            partition_optimizers[name] = optax.inject_hyperparams(muon)(
                learning_rate=schedule_fn,
                nesterov=nesterov,
                beta=beta_schedule,
                layer_sharding=layer_sharding,
            )
            resolved_partition.update({
                "beta": beta_val,
                "nesterov": nesterov,
                "layer_sharding": layer_sharding,
            })

        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        resolved_config[name] = resolved_partition

        # Build pattern matching
        for pattern in patterns:
            if pattern == "*":
                if catch_all is not None:
                    raise ValueError(f"Multiple catch-all partitions: {catch_all} and {name}")
                catch_all = name
            else:
                if pattern in exact_patterns:
                    raise ValueError(f"Pattern '{pattern}' used by multiple partitions")
                exact_patterns[pattern] = name

    # Create router
    def router(state):
        def route_path(path, _):
            if len(path) == 0:
                return catch_all or list(partitions.keys())[0]
            param_name = path[0].name
            if param_name in exact_patterns:
                return exact_patterns[param_name]
            return catch_all or list(partitions.keys())[0]
        return jax.tree.map_with_path(route_path, state)

    # Build optimizer chain
    transforms = []

    # Optional grad clipping
    grad_clip_norm = get_val('grad_clip_norm')
    if grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(grad_clip_norm))

    transforms.append(optax.partition(partition_optimizers, router))

    tx = optax.chain(*transforms)

    # Optional gradient accumulation
    accum_steps = get_val('accum_steps')
    if accum_steps is not None and accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=accum_steps)

    return tx, resolved_config
