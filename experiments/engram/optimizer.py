import jax
import optax

from muon import muon
from scheduler import (
    warmup_stable_decay_schedule,
    warmup_stable_decay_schedule_py,
)


def make_optimizer(cfg):
    """Create partitioned optimizer. Engram weights use AdamW like embeddings."""
    opt = cfg.optimizer

    def make_schedule(peak_lr):
        return warmup_stable_decay_schedule(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    def router(state):
        def route_path(path, _):
            return "embed" if path[0].name in ("embed", "unembed") else "other"

        return jax.tree.map_with_path(route_path, state)

    learning_rate = make_schedule(opt.peak_lr)
    tx = optax.chain(
        optax.clip_by_global_norm(opt.clip_grad_norm),
        optax.partition(
            {
                "embed": optax.adamw(
                    learning_rate=learning_rate,
                    weight_decay=opt.weight_decay,
                    eps=1e-8,
                    b1=0.9, 
                    b2=0.95,
                ),
                "other": muon(
                    learning_rate=learning_rate,
                    weight_decay=opt.weight_decay,
                    nesterov=True,
                    layer_sharding=True,
                    eps=1e-8,
                    adjust_lr_fn="match_rms_adamw"
                ),
            },
            router,
        ),
    )
    if opt.accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=opt.accum_steps)

    resolved_config = {
        "embed": {
            "type": "adamw",
            "peak_lr": opt.peak_lr,
            "warmup_steps": opt.warmup_steps,
            "decay_steps": opt.decay_steps,
        },
        "other": {
            "type": "muon",
            "peak_lr": opt.peak_lr,
            "warmup_steps": opt.warmup_steps,
            "decay_steps": opt.decay_steps,
        },
    }

    def make_lr_schedule_py(peak_lr):
        return warmup_stable_decay_schedule_py(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    schedule_fns = {
        "lr_embed": make_lr_schedule_py(opt.peak_lr),
        "lr_other": make_lr_schedule_py(opt.peak_lr),
    }

    return tx, resolved_config, schedule_fns
