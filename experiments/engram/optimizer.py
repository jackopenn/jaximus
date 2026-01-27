import jax
import optax
import muon

from scheduler import (
    warmup_stable_decay_schedule,
    warmup_stable_decay_schedule_py,
)


def make_optimizer(cfg):
    """Create partitioned optimizer with Muon for weights and AdamW for embeddings."""
    opt = cfg.optimizer

    tx = optax.chain(
        optax.clip_by_global_norm(opt.clip_grad_norm),
        optax.partition(
            {
                "embed": optax.adamw(
                    learning_rate=warmup_stable_decay_schedule(opt.peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps),
                    weight_decay=opt.weight_decay,
                    eps=1e-8,
                    b1=0.9,
                    b2=0.95,
                ),
                "other": optax.adamw(
                    learning_rate=warmup_stable_decay_schedule(opt.peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps),
                    weight_decay=opt.weight_decay,
                    # nesterov=True,
                    # layer_sharding=True,
                    eps=1e-8,
                    # adjust_lr_fn="match_rms_adamw",
                    b1=0.9,
                    b2=0.95,
                ),
            },
            lambda state: jax.tree.map_with_path(lambda path, _: "embed" if path[0].name in ("embed", "unembed") else "other", state)
        ),
    )
    if opt.accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=opt.accum_steps)

    schedule_fns = {
        "lr_embed": warmup_stable_decay_schedule_py(opt.peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps),
        "lr_other": warmup_stable_decay_schedule_py(opt.peak_lr, opt.warmup_steps, opt.decay_steps, cfg.max_steps),
    }

    return tx, schedule_fns
