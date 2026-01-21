# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    cfg = Config()
    cfg.experiment = "experiments.canon"
    cfg.seed = 42
    cfg.exp_name = "nanochat-canon"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 20
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.max_seq_len = 2048
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6

    # canon
    cfg.model.canon_a = True
    cfg.model.canon_b = True
    cfg.model.canon_c = True
    cfg.model.canon_d = True
    cfg.model.canon_depth = 4
    cfg.model.canon_init = "uniform"

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 128

    cfg.optimizer.accum_steps = 2
    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.momentum_start = 0.85
    cfg.optimizer.momentum_end = 0.95
    cfg.optimizer.momentum_warmup_steps = 300
    cfg.optimizer.embed.peak_lr = lambda: 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optimizer.unembed.peak_lr = lambda: 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optimizer.other.peak_lr = 0.02

    cfg.max_steps = int(3.82e9 // 524288)
    cfg.generate_every = 500
    cfg.eval_every = -1
    cfg.eval_max_per_task = -1
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/canon"
    cfg.xpu = "v4"
    cfg.wandb = True
    cfg.wandb_project = "canon"

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
