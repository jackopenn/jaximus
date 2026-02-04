import jax
import optax

from muon import muon
from scheduler import warmup_stable_decay_schedule, warmup_stable_decay_schedule_py


def make_optimizer(cfg):
    """Create partitioned optimizer with Complete(d)P scaling. Returns (optimizer, schedule_fns_for_logging)."""
    m_N = cfg.model.hidden_dim / cfg.model.completedp.base_width
    m_L = cfg.model.num_layers / cfg.model.completedp.base_depth
    warmup_steps = int(cfg.optimizer.warmup_ratio * cfg.max_steps)
    decay_steps = int(cfg.optimizer.warmdown_ratio * cfg.max_steps)

    embed_lr = cfg.optimizer.embed.peak_lr
    unembed_lr = cfg.optimizer.unembed.peak_lr / m_N
    scalar_lr = cfg.optimizer.scalar.peak_lr * m_L ** (cfg.model.completedp.alpha - 1)
    matrix_lr = cfg.optimizer.matrix.peak_lr
    weight_decay = cfg.optimizer.weight_decay * m_N
    print(f"Complete(d)P optimizer: m_N={m_N:.4f}, m_L={m_L:.4f}, alpha={cfg.model.completedp.alpha}")
    print(f"  LRs: embed={embed_lr:.4f}, unembed={unembed_lr:.6f}, scalar={scalar_lr:.4f}, matrix={matrix_lr:.4f}")
    print(f"  weight_decay={weight_decay:.4f}")

    def make_schedule(peak_lr):
        return warmup_stable_decay_schedule(peak_lr, warmup_steps, decay_steps, cfg.max_steps)

    def router(state):
        def route_path(path, _):
            if len(path) == 0:
                return "matrix"
            name = path[0].name
            if name in ("embed", "value_embeds"):
                return "embed"
            if name == "unembed":
                return "unembed"
            if name in ("resid_lambdas", "x0_lambdas"):
                return "scalar"
            return "matrix"

        return jax.tree.map_with_path(route_path, state)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.partition(
            {
                "embed": optax.adamw(
                    learning_rate=make_schedule(embed_lr),
                    weight_decay=0.0,
                    eps=1e-10 / m_N,
                    b1=cfg.optimizer.adam_beta1,
                    b2=cfg.optimizer.adam_beta2,
                ),
                "unembed": optax.adamw(
                    learning_rate=make_schedule(unembed_lr),
                    weight_decay=0.0,
                    eps=1e-10 / m_N,
                    b1=cfg.optimizer.adam_beta1,
                    b2=cfg.optimizer.adam_beta2,
                ),
                "scalar": optax.adamw(
                    learning_rate=make_schedule(scalar_lr),
                    weight_decay=0.0,
                    eps=1e-10 / m_N * m_L ** (-cfg.model.completedp.alpha),
                    b1=cfg.optimizer.adam_beta1,
                    b2=cfg.optimizer.adam_beta2,
                ),
                "matrix": optax.inject_hyperparams(muon)(
                    learning_rate=make_schedule(matrix_lr),
                    nesterov=True,
                    layer_sharding=True,
                    beta=cfg.optimizer.momentum,
                    weight_decay=weight_decay,
                ),
            },
            router,
        ),
    )
    if cfg.optimizer.accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=cfg.optimizer.accum_steps)

    def make_lr_schedule_py(peak_lr):
        return warmup_stable_decay_schedule_py(peak_lr, warmup_steps, decay_steps, cfg.max_steps)

    schedule_fns = {
        "lr_embed": make_lr_schedule_py(embed_lr),
        "lr_unembed": make_lr_schedule_py(unembed_lr),
        "lr_scalar": make_lr_schedule_py(scalar_lr),
        "lr_matrix": make_lr_schedule_py(matrix_lr),
    }

    return tx, schedule_fns
