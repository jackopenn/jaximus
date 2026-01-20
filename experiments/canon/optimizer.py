import jax
import optax

from muon import muon
from scheduler import (
    muon_momentum_schedule,
    muon_momentum_schedule_py,
    warmup_stable_decay_schedule,
    warmup_stable_decay_schedule_py,
)


def make_optimizer(cfg):
    """Create partitioned optimizer. Returns (optimizer, config_for_logging, schedule_fns_for_logging)."""
    opt = cfg.optimizer

    def make_schedule(peak_lr):
        return warmup_stable_decay_schedule(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    def router(state):
        def route_path(path, _):
            if len(path) == 0:
                return "other"
            name = path[0].name
            return "embed" if name in ("embed", "pos_embed") else "unembed" if name == "unembed" else "other"

        return jax.tree.map_with_path(route_path, state)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.partition(
            {
                "embed": optax.adamw(
                    learning_rate=make_schedule(opt.embed.peak_lr), weight_decay=0.0, eps=1e-10, b1=0.8, b2=0.95
                ),
                "unembed": optax.adamw(
                    learning_rate=make_schedule(opt.unembed.peak_lr), weight_decay=0.0, eps=1e-10, b1=0.8, b2=0.95
                ),
                "other": optax.inject_hyperparams(muon)(
                    learning_rate=make_schedule(opt.other.peak_lr),
                    nesterov=True,
                    layer_sharding=True,
                    beta=muon_momentum_schedule(opt.momentum_start, opt.momentum_end, opt.momentum_warmup_steps),
                    adamw_b1=0.8,
                    adamw_b2=0.95,
                    adamw_weight_decay=0.0,
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
            "peak_lr": opt.embed.peak_lr,
            "warmup_steps": opt.warmup_steps,
            "decay_steps": opt.decay_steps,
        },
        "unembed": {
            "type": "adamw",
            "peak_lr": opt.unembed.peak_lr,
            "warmup_steps": opt.warmup_steps,
            "decay_steps": opt.decay_steps,
        },
        "other": {
            "type": "muon",
            "peak_lr": opt.other.peak_lr,
            "warmup_steps": opt.warmup_steps,
            "decay_steps": opt.decay_steps,
            "momentum_start": opt.momentum_start,
            "momentum_end": opt.momentum_end,
            "momentum_warmup_steps": opt.momentum_warmup_steps,
        },
    }

    def make_lr_schedule_py(peak_lr):
        return warmup_stable_decay_schedule_py(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    schedule_fns = {
        "lr_embed": make_lr_schedule_py(opt.embed.peak_lr),
        "lr_unembed": make_lr_schedule_py(opt.unembed.peak_lr),
        "lr_other": make_lr_schedule_py(opt.other.peak_lr),
        "momentum_other": muon_momentum_schedule_py(opt.momentum_start, opt.momentum_end, opt.momentum_warmup_steps),
    }

    return tx, resolved_config, schedule_fns
