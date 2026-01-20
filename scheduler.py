from jax import numpy as jnp


def warmup_stable_decay_schedule(peak_value, warmup_steps, decay_steps, max_steps):
    """LR schedule: warmup -> stable -> linear decay."""

    def schedule(step):
        warmup_value = peak_value * step / jnp.maximum(warmup_steps, 1)
        decay_start = max_steps - decay_steps
        decay_pct = (step - decay_start) / jnp.maximum(decay_steps, 1)
        decay_value = peak_value * (1 - decay_pct)
        return jnp.where(step < warmup_steps, warmup_value, jnp.where(step < decay_start, peak_value, decay_value))

    return schedule


def muon_momentum_schedule(start, end, warmup_steps):
    def schedule(step):
        frac = jnp.minimum(step / warmup_steps, 1.0)
        return (1 - frac) * start + frac * end

    return schedule


def warmup_stable_decay_schedule_py(peak_value, warmup_steps, decay_steps, max_steps):
    """Pure Python LR schedule for logging."""
    decay_start = max_steps - decay_steps

    def schedule(step):
        if warmup_steps and step < warmup_steps:
            return peak_value * step / max(warmup_steps, 1)
        if step < decay_start:
            return peak_value
        decay_pct = (step - decay_start) / max(decay_steps, 1)
        return peak_value * (1 - decay_pct)

    return schedule


def muon_momentum_schedule_py(start, end, warmup_steps):
    """Pure Python momentum schedule for logging."""

    def schedule(step):
        frac = min(step / warmup_steps, 1.0)
        return (1 - frac) * start + frac * end

    return schedule
