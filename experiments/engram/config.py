# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    """Base config with engram memory injection."""
    cfg = Config()
    cfg.experiment = "experiments.engram"
    cfg.seed = 42
    cfg.exp_name = "engram-base"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 12
    cfg.model.hidden_dim = 1536
    cfg.model.num_attention_heads = 12
    cfg.model.num_key_value_heads = 12
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: int(8/3 * cfg.model.hidden_dim)
    cfg.model.max_seq_len = 2048
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6

    # Engram config
    cfg.model.engram.enabled = True
    cfg.model.engram.vocab_size_per_ngram = [251459, 251459]  # ~vocab*5, prime
    cfg.model.engram.ngram_sizes = [2, 3]  # bigrams + trigrams
    cfg.model.engram.n_embed_per_ngram = 256
    cfg.model.engram.n_head_per_ngram = 8
    cfg.model.engram.layer_ids = [2, 10]
    cfg.model.engram.kernel_size = 4
    cfg.model.engram.seed = 0

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    
    global_batch_size = 96 * 8192
    cfg.data.batch_size = 96
    cfg.optimizer.accum_steps = lambda: global_batch_size // cfg.data.batch_size

    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad_norm = 0.0
    cfg.optimizer.warmup_steps = lambda: int(0.1 * cfg.max_steps)
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.peak_lr = 9.503e-4

    cfg.max_steps = int(8.92e9 // global_batch_size)
    cfg.generate_every = -1
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
