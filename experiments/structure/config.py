# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    """Config for sliding window structure experiment. Vary sliding_window and max_flops per run."""
    cfg = Config()
    cfg.experiment = "experiments.structure"
    cfg.seed = 42

    def _make_exp_name():
        w = cfg.model.sliding_window or "full"
        return f"d{cfg.model.num_layers}-w{w}-{cfg.max_flops:.0e}"

    cfg.exp_name = _make_exp_name

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 20
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: ((4 * cfg.model.hidden_dim + 63) // 64) * 64
    cfg.model.max_seq_len = 2048
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6
    cfg.model.sliding_window = None  # None=full, or 256/1024

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len

    tokens_per_batch = 524288
    cfg.data.batch_size = 64
    cfg.optimizer.accum_steps = lambda: int(tokens_per_batch // (cfg.data.batch_size * cfg.data.max_length))
    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.momentum_start = 0.85
    cfg.optimizer.momentum_end = 0.95
    cfg.optimizer.momentum_warmup_steps = 300
    cfg.optimizer.embed.peak_lr = lambda: 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optimizer.unembed.peak_lr = lambda: 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optimizer.other.peak_lr = 0.02

    cfg.max_flops = 1e18

    def _compute_max_steps():
        D, L, N, K, H, I = (
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            cfg.model.num_attention_heads,
            cfg.model.num_key_value_heads,
            cfg.model.head_dim,
            cfg.model.intermediate_dim,
        )
        S = cfg.model.max_seq_len
        W = cfg.model.sliding_window if cfg.model.sliding_window else S

        # Core params (per layer: qkvo projections + up/down MLP)
        params_per_layer = (D * N * H) + 2 * (D * K * H) + (N * H * D) + 2 * (D * I)
        num_core_params = L * params_per_layer

        # FLOPs per token: 6*params (fwd+bwd matmuls) + attention cost
        effective_seq = min(W, S)
        num_flops_per_token = 6 * num_core_params + L * 12 * N * H * effective_seq

        tokens_per_step = cfg.data.batch_size * cfg.optimizer.accum_steps * cfg.data.max_length
        flops_per_step = num_flops_per_token * tokens_per_step
        return max(1, int(cfg.max_flops / flops_per_step))

    cfg.max_steps = _compute_max_steps

    cfg.generate_every = -1
    cfg.eval_every = -1
    cfg.eval_max_per_task = 500
    cfg.eval_batch_size = 128
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = -1
    cfg.checkpoint_dir = lambda: f"gs://trm-jax-123/jaximus/checkpoints/structure/{cfg.exp_name}"
    cfg.xpu = "v4"
    cfg.wandb = True
    cfg.wandb_project = "structure"

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
