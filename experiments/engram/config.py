# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    """Base config for baseline transformer."""
    cfg = Config()
    cfg.experiment = "experiments.engram"
    cfg.seed = 42
    cfg.exp_name = "baseline"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 12
    cfg.model.hidden_dim = 1536
    cfg.model.num_attention_heads = 12
    cfg.model.num_key_value_heads = 12
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: int(8 / 3 * cfg.model.hidden_dim)
    cfg.model.max_seq_len = 2048
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6

    cfg.data.hf_name = ["karpathy/fineweb-edu-100b-shuffle", "default"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len

    tokens_per_batch = 96 * 8192
    cfg.data.batch_size = 96
    cfg.optimizer.accum_steps = lambda: tokens_per_batch // (cfg.data.batch_size * cfg.model.max_seq_len)

    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad_norm = 1.0
    cfg.optimizer.warmup_steps = lambda: int(0.02 * cfg.max_steps)
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.peak_lr = 9.503e-4

    cfg.max_steps = int(8.92e9 // tokens_per_batch)
    cfg.generate_every = 500
    cfg.eval_every = 1000
    cfg.eval_max_per_task = 500
    cfg.eval_batch_size = 128
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/engram"
    cfg.xpu = "v4"
    cfg.wandb = True
    cfg.wandb_project = "engram"

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
