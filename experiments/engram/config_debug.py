# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    """Debug config for local testing."""
    cfg = Config()
    cfg.experiment = "experiments.engram"
    cfg.seed = 42
    cfg.exp_name = "baseline-debug"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 4
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * 64
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.max_seq_len = 256
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6

    cfg.model.engram.enabled = True
    cfg.model.engram.vocab_size = 1000
    cfg.model.engram.ngrams = [2, 3]
    cfg.model.engram.n_heads = 2
    cfg.model.engram.embed_dim = 8
    cfg.model.engram.layer_ids = [1, 3]
    cfg.model.engram.kernel_size = 4
    cfg.model.engram.lr_multiplier = 5.0
    cfg.model.engram.weight_decay = 0.0

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 4

    cfg.optimizer.accum_steps = 1
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad_norm = 1.0
    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.peak_lr = 9.503e-4
    cfg.optimizer.engram_lr_multiplier = lambda: cfg.model.engram.lr_multiplier
    cfg.optimizer.engram_weight_decay = lambda: cfg.model.engram.weight_decay

    cfg.max_steps = 3
    cfg.generate_every = -1
    cfg.eval_every = -1
    cfg.eval_max_per_task = 10
    cfg.eval_batch_size = 4
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = -1
    cfg.checkpoint_dir = "checkpoints"
    cfg.xpu = "v4"
    cfg.wandb = False
    cfg.wandb_project = "baseline"

    cfg.parallel.strategy = "dp"
    cfg.parallel.data = 1

    return cfg
