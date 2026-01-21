# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config as Config


def get_config():
    """Debug config for local testing."""
    cfg = Config()
    cfg.experiment = "experiments.canon"
    cfg.seed = 42
    cfg.exp_name = "canon100-debug"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 1
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.max_seq_len = 256
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6
    cfg.model.canon_a = True
    cfg.model.canon_b = True
    cfg.model.canon_c = True
    cfg.model.canon_d = True
    cfg.model.canon_depth = 4
    cfg.model.canon_init = "zeros"

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 4

    cfg.optimizer.accum_steps = 1
    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.momentum_start = 0.85
    cfg.optimizer.momentum_end = 0.95
    cfg.optimizer.momentum_warmup_steps = 300
    cfg.optimizer.embed.peak_lr = lambda: 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optimizer.unembed.peak_lr = lambda: 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optimizer.other.peak_lr = 0.02

    cfg.max_steps = 5
    cfg.generate_every = 10
    cfg.eval_every = -1
    cfg.eval_max_per_task = -1
    cfg.eval_batch_size = 256
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = -1
    cfg.checkpoint_dir = "checkpoints"
    cfg.xpu = "v4"
    cfg.wandb = False
    cfg.wandb_project = "canon"

    cfg.parallel.strategy = "dp"
    cfg.parallel.data = 1

    return cfg
