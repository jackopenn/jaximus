"""Optimizer creation and learning rate schedules."""
import jax
from jax import numpy as jnp
import optax

from muon import muon


def warmup_linear_decay_schedule(
    init_value: float,
    peak_value: float,
    end_value: float,
    warmup_steps: int,
    decay_steps: int,
    max_steps: int,
):
    """Learning rate schedule with warmup and linear decay."""
    def schedule(step):
        warmup_pct = step / jnp.maximum(warmup_steps, 1)
        warmup_value = init_value + (peak_value - init_value) * warmup_pct
        
        decay_start = max_steps - decay_steps
        decay_pct = (step - decay_start) / jnp.maximum(decay_steps, 1)
        decay_value = peak_value + (end_value - peak_value) * decay_pct
        
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


def make_optimizer(cfg):
    """Create optimizer and schedule functions based on config.
    
    Expects cfg.optim to have:
        - embed_lr: peak learning rate for embeddings
        - lm_head_lr: peak learning rate for lm_head
        - other_lr: peak learning rate for other params (muon)
        - warmup_ratio: fraction of max_steps for warmup
        - decay_ratio: fraction of max_steps for decay
        - grad_clip_norm: max gradient norm for clipping
        - adamw_weight_decay: weight decay for adamw
        - adamw_eps: epsilon for adamw
        - adamw_b1: beta1 for adamw
        - adamw_b2: beta2 for adamw
        - accum_steps: gradient accumulation steps
    """
    # Get optimizer params from config with defaults
    embed_lr = getattr(cfg.optim, 'embed_lr', 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5))
    lm_head_lr = getattr(cfg.optim, 'lm_head_lr', 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5))
    other_lr = getattr(cfg.optim, 'other_lr', 0.02)
    warmup_ratio = getattr(cfg.optim, 'warmup_ratio', 0.0)
    decay_ratio = getattr(cfg.optim, 'decay_ratio', 0.4)
    grad_clip_norm = getattr(cfg.optim, 'grad_clip_norm', 1.0)
    adamw_weight_decay = getattr(cfg.optim, 'adamw_weight_decay', 0.0)
    adamw_eps = getattr(cfg.optim, 'adamw_eps', 1e-10)
    adamw_b1 = getattr(cfg.optim, 'adamw_b1', 0.8)
    adamw_b2 = getattr(cfg.optim, 'adamw_b2', 0.95)
    
    adamw_params = dict(
        weight_decay=adamw_weight_decay,
        eps=adamw_eps,
        b1=adamw_b1,
        b2=adamw_b2,
    )
    
    # Schedule params shared between optimizer and logging
    schedule_params = dict(
        init_value=0.0,
        end_value=0.0,
        warmup_steps=warmup_ratio * cfg.max_steps,
        decay_steps=decay_ratio * cfg.max_steps,
        max_steps=cfg.max_steps,
    )
    
    schedule_fns = {
        "embed": warmup_linear_decay_schedule(peak_value=embed_lr, **schedule_params),
        "lm_head": warmup_linear_decay_schedule(peak_value=lm_head_lr, **schedule_params),
        "other": warmup_linear_decay_schedule(peak_value=other_lr, **schedule_params),
        "muon": muon_momentum_schedule,
    }
    
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.partition(
            {
                "embed": optax.adamw(
                    learning_rate=schedule_fns["embed"],
                    **adamw_params,
                ),
                "unembed": optax.adamw(
                    learning_rate=schedule_fns["lm_head"],
                    **adamw_params,
                ),
                "other": optax.inject_hyperparams(muon)(
                    learning_rate=schedule_fns["other"],
                    nesterov=True,
                    beta=muon_momentum_schedule,
                ),
            },
            lambda state: jax.tree.map_with_path(
                lambda path, _: path[0].name if path[0].name in ("embed", "unembed") else "other",
                state
            )
        )
    )
    tx = optax.MultiSteps(tx, every_k_schedule=cfg.optim.accum_steps)
    
    return tx, schedule_fns
