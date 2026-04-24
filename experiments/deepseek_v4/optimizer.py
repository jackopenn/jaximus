import jax
import optax

from muon import muon
from scheduler import warmup_stable_decay_schedule, warmup_stable_decay_schedule_py


def make_optimizer(cfg):
    opt = cfg.optimizer

    def make_schedule(peak_lr):
        return warmup_stable_decay_schedule(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    def router(state):
        def route_path(path, leaf):
            names = [getattr(k, "name", getattr(k, "key", None)) for k in path]
            if any(name in ("embed", "unembed") for name in names):
                return names[-1]
            if leaf.ndim < 2 or any(
                "bias" in str(name) or "alpha" in str(name) or "sink" in str(name) for name in names
            ):
                return "adam"
            return "muon"

        return jax.tree.map_with_path(route_path, state)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.partition(
            {
                "embed": optax.adamw(
                    learning_rate=make_schedule(opt.embed.peak_lr),
                    weight_decay=0.0,
                    eps=1e-20,
                    b1=0.9,
                    b2=0.95,
                ),
                "unembed": optax.adamw(
                    learning_rate=make_schedule(opt.unembed.peak_lr),
                    weight_decay=0.0,
                    eps=1e-20,
                    b1=0.9,
                    b2=0.95,
                ),
                "adam": optax.adamw(
                    learning_rate=make_schedule(opt.adam.peak_lr),
                    weight_decay=opt.weight_decay,
                    eps=1e-20,
                    b1=0.9,
                    b2=0.95,
                ),
                "muon": optax.inject_hyperparams(muon)(
                    learning_rate=make_schedule(opt.muon.peak_lr),
                    beta=opt.muon.momentum,
                    weight_decay=opt.weight_decay,
                    nesterov=True,
                    layer_sharding=True,
                    ns_coeffs=(3.4445, -4.7750, 2.0315),
                    ns_steps=8,
                    ns_coeffs_final=(2.0, -1.5, 0.5),
                    ns_steps_final=2,
                    adjust_lr_fn="match_rms_adamw",
                ),
            },
            router,
        ),
    )
    if opt.accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=opt.accum_steps)

    def make_lr_schedule_py(peak_lr):
        return warmup_stable_decay_schedule_py(peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps)

    return tx, {
        "lr_embed": make_lr_schedule_py(opt.embed.peak_lr),
        "lr_unembed": make_lr_schedule_py(opt.unembed.peak_lr),
        "lr_adam": make_lr_schedule_py(opt.adam.peak_lr),
        "lr_muon": make_lr_schedule_py(opt.muon.peak_lr),
    }
