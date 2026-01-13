from functools import partial
import jax
from jax import numpy as jnp
import optax
from sws import Config

from muon import muon


def warmup_linear_decay_schedule(
    init_value,
    peak_value,
    end_value,
    warmup_steps,
    decay_steps,
    max_steps
):
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
    frac = jnp.minimum(step / 300, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95

def make_optimizer(cfg):
    """Create optimizer and logging helpers based on config."""
    adamw_params = dict(weight_decay=0.0, eps=1e-10, b1=0.8, b2=0.95)
    te_peak_value = 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)
    lm_head_peak_value = 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    other_peak_value = 0.02
    
    # Schedule params shared between optimizer and logging
    schedule_params = dict(
        init_value=0.0, end_value=0.0,
        warmup_steps=0.0 * cfg.max_steps,
        decay_steps=0.4 * cfg.max_steps,
        max_steps=cfg.max_steps,
    )
    
    schedule_fns = {
        "embed": warmup_linear_decay_schedule(peak_value=te_peak_value, **schedule_params),
        "lm_head": warmup_linear_decay_schedule(peak_value=lm_head_peak_value, **schedule_params),
        "other": warmup_linear_decay_schedule(peak_value=other_peak_value, **schedule_params),
        "muon": muon_momentum_schedule,
    }
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
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


def get_config():
    cfg = Config()
    cfg.seed = 42
    cfg.exp_name = "nanochat100-base"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 1
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.max_seq_len = 256
    cfg.model.norm_type = "rms"
    cfg.model.norm_position = "pre"
    cfg.model.norm_epsilon = 1e-6
    cfg.model.norm_scale = False
    cfg.model.norm_bias = False
    cfg.model.mlp_type = "mlp"
    cfg.model.act_fn = "relu_squared"
    cfg.model.attn_use_bias = False
    cfg.model.mlp_use_bias = False
    cfg.model.lm_head_use_bias = False
    cfg.model.qk_norm = False
    cfg.model.qk_norm_type = "rms"
    cfg.model.qk_norm_epsilon = 1e-6
    cfg.model.sliding_window = None
    cfg.model.position_embedding_type = 'none'
    cfg.model.rope_theta = 10000.0
    cfg.model.tie_word_embeddings = False
    cfg.model.init_strategy = "nanochat"
    cfg.model.softcap = None
    cfg.model.dtype = "bfloat16"
    cfg.model.post_embed_norm = False
    cfg.model.pre_lm_head_norm = False

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len

    # target batch size of 524288 tokens (2048 seq_len)
    cfg.data.batch_size = 4
    cfg.optim.accum_steps = 1

    max_steps = 25_000
    cfg.max_steps = max_steps
    cfg.generate_every = 10
    cfg.eval_every = -1
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "checkpoints"
    cfg.xpu = "v4"
    cfg.wandb = False

    cfg.parallel.strategy = "dp"
    cfg.parallel.data = 1

    cfg.optim.make_optimizer = lambda: make_optimizer

    return cfg
