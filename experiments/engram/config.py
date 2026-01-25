# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    """Base config with engram memory injection."""
    cfg = Config()
    cfg.experiment = "experiments.engram"
    cfg.seed = 42
    cfg.exp_name = "engram-base"

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

    # Engram config
    cfg.model.engram.enabled = True
    cfg.model.engram.vocab_size_per_ngram = [251459, 251459]  # ~vocab*5, prime
    cfg.model.engram.max_ngram_size = 3  # bigrams + trigrams
    cfg.model.engram.n_embed_per_ngram = 512
    cfg.model.engram.n_head_per_ngram = 8
    cfg.model.engram.layer_ids = [1, 15]
    cfg.model.engram.kernel_size = 4
    cfg.model.engram.seed = 0
    cfg.model.engram.mode = "consecutive"  # "consecutive" or "attention"
    cfg.model.engram.attn_lag = 0  # 0=prev layer, 1=layer-2, etc.
    cfg.model.engram.attn_exclude_self = True  # mask diagonal when finding top_k

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 64

    cfg.optimizer.accum_steps = 4
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
    cfg.eval_every = 1000
    cfg.eval_max_per_task = 500
    cfg.eval_batch_size = 128
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/engram-base"
    cfg.xpu = "v4"
    cfg.wandb = True
    cfg.wandb_project = "engram"

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
