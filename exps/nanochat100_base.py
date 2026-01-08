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


def make_muon_momentum_schedule(start, end, warmup_steps):
    """Factory for momentum schedule for Muon optimizer"""
    def schedule(step):
        frac = jnp.minimum(step / warmup_steps, 1.0)
        return (1 - frac) * start + frac * end
    return schedule


def get_config():
    cfg = Config()
    cfg.seed = 42
    cfg.exp_name = "nanochat100-base"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 20
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.max_seq_len = 2048
    cfg.model.norm_type = "rms"
    cfg.model.norm_position = "pre"
    cfg.model.norm_epsilon = 1e-6
    cfg.model.mlp_type = "mlp"
    cfg.model.act_fn = "relu_squared"
    cfg.model.attn_use_bias = False
    cfg.model.mlp_use_bias = False
    cfg.model.lm_head_use_bias = False
    cfg.model.qk_norm = True
    cfg.model.qk_norm_type = "rms"
    cfg.model.qk_norm_epsilon = 1e-6
    cfg.model.sliding_window = None
    cfg.model.position_embedding_type = "rope"
    cfg.model.rope_theta = 10000.0
    cfg.model.tie_word_embeddings = False
    cfg.model.init_strategy = "nanochat"
    cfg.model.softcap = 15.0
    cfg.model.dtype = "bfloat16"
    cfg.model.post_embed_norm = True
    cfg.model.pre_lm_head_norm = True

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len

    # target batch size of 524288 tokens (2048 seq_len)
    cfg.data.batch_size = 32
    cfg.optim.accum_steps = 8

    max_steps = 3.8e9 // 52488
    cfg.max_steps = max_steps
    cfg.generate_every = 500
    cfg.eval_every = -1
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/nanochat100-base-v2"
    cfg.xpu = "v4"
    cfg.wandb = False

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    # ---------- optimizer config ----------
    adamw_params = dict(weight_decay=0.0, eps=1e-10, b1=0.8, b2=0.95)
    
    cfg.optim.warmup_pct = 0.0
    cfg.optim.decay_pct = 0.4
    
    schedule_params = dict(
        init_value=0.0,
        end_value=0.0,
        warmup_steps=cfg.optim.warmup_pct * max_steps,
        decay_steps=cfg.optim.decay_pct * max_steps,
        max_steps=max_steps,
    )
    
    cfg.optim.te_peak_value = lambda: 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optim.lm_head_peak_value = lambda: 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optim.other_peak_value = 0.02
    
    cfg.optim.muon_momentum_start = 0.85
    cfg.optim.muon_momentum_end = 0.95
    cfg.optim.muon_momentum_warmup_steps = 300
    
    lr_schedule_te = warmup_linear_decay_schedule(peak_value=cfg.optim.te_peak_value, **schedule_params)
    lr_schedule_lm_head = warmup_linear_decay_schedule(peak_value=cfg.optim.lm_head_peak_value, **schedule_params)
    lr_schedule_other = warmup_linear_decay_schedule(peak_value=cfg.optim.other_peak_value, **schedule_params)
    
    cfg.optim.tx = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.partition(
            {
                "token_embedding": optax.adamw(
                    learning_rate=lr_schedule_te,
                    **adamw_params,
                ),
                "lm_head": optax.adamw(
                    learning_rate=lr_schedule_lm_head,
                    **adamw_params,
                ),
                "other": optax.inject_hyperparams(muon)(
                    learning_rate=lr_schedule_other,
                    nesterov=True,
                    beta=make_muon_momentum_schedule(
                        cfg.optim.muon_momentum_start,
                        cfg.optim.muon_momentum_end,
                        cfg.optim.muon_momentum_warmup_steps,
                    ),
                ),
            },
            lambda state: jax.tree.map_with_path(lambda path, _: path[0].key if path[0].key in ("token_embedding", "lm_head") else "other", state)
        )
    )

    return cfg
