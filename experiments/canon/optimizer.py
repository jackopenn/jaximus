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
            names = [getattr(k, "name", getattr(k, "key", None)) for k in path]
            if "embed" in names or "pos_embed" in names:
                return "embed"
            if "unembed" in names or "canon" in names:
                return "unembed"
            return "other"

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
                ),
            },
            router,
        ),
    )
    if opt.accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=opt.accum_steps)

    def make_lr_schedule_py(peak_lr):
        return warmup_stable_decay_schedule_py(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    schedule_fns = {
        "lr_embed": make_lr_schedule_py(opt.embed.peak_lr),
        "lr_unembed": make_lr_schedule_py(opt.unembed.peak_lr),
        "lr_other": make_lr_schedule_py(opt.other.peak_lr),
        "momentum_other": muon_momentum_schedule_py(opt.momentum_start, opt.momentum_end, opt.momentum_warmup_steps),
    }

    return tx, schedule_fns
