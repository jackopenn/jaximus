# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    """Base config. Hardcodes: no bias, MLP, RMSNorm pre-norm, RoPE, QK norm, softcap=15, relu_squared, bf16."""
    cfg = Config()
    cfg.experiment = "experiments.nanochat"
    cfg.seed = 42
    cfg.exp_name = "nanochat100-base"

    # Model architecture (nanochat defaults)
    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 20
    cfg.model.aspect_ratio = 64
    cfg.model.hidden_dim = lambda: cfg.model.num_layers * cfg.model.aspect_ratio
    cfg.model.num_attention_heads = lambda: max(1, (cfg.model.hidden_dim + 127) // 128)
    cfg.model.num_key_value_heads = lambda: cfg.model.num_attention_heads
    cfg.model.head_dim = 128
    cfg.model.intermediate_dim = lambda: 4 * cfg.model.hidden_dim
    cfg.model.max_seq_len = 2048
    cfg.model.rope_theta = 10000.0
    cfg.model.norm_epsilon = 1e-6

    # Sliding window attention pattern (tiled across layers, final layer always L)
    # L=long (full context), S=short (half context)
    cfg.model.window_pattern = "SSSL"

    # Complete(d)P parameterization for HP transfer across width/depth/batch/tokens
    cfg.model.completedp.base_width = 768
    cfg.model.completedp.base_depth = 12
    cfg.model.completedp.alpha = 0.5  # depth scaling exponent (0.5=feature diversity, 1.0=CompleteP)
    cfg.model.completedp.base_std = 1.0  # base init std for matrix weights
    cfg.model.completedp.base_embed_std = 1.0  # base init std for embedding
    cfg.model.completedp.base_unembed_std = 0.001  # base init std for unembedding

    # Data
    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 32  # device batch size

    # Optimizer (nanochat defaults, base LRs at base_width/base_depth)
    cfg.optimizer.total_batch_size = 524288
    cfg.optimizer.accum_steps = lambda: (
        cfg.optimizer.total_batch_size // (cfg.data.batch_size * cfg.model.max_seq_len)
    )
    cfg.optimizer.warmup_ratio = 0.0
    cfg.optimizer.warmdown_ratio = 0.5
    cfg.optimizer.momentum = 0.95
    # Base LRs (Complete(d)P scaling applied in optimizer)
    cfg.optimizer.embed.peak_lr = 0.3
    cfg.optimizer.unembed.peak_lr = 0.004
    cfg.optimizer.matrix.peak_lr = 0.02  # Muon - constant across scales
    cfg.optimizer.scalar.peak_lr = 0.5
    cfg.optimizer.adam_beta1 = 0.8
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.weight_decay = 0.2

    # Training (target 10.5x data:param ratio like nanochat)
    cfg.target_param_data_ratio = 10.5
    cfg.max_steps = lambda: int(
        cfg.target_param_data_ratio * cfg.model.num_layers * cfg.model.hidden_dim ** 2 * 12 // cfg.optimizer.total_batch_size
    )  # rough estimate: params ≈ 12 * L * D²
    cfg.generate_every = 500
    cfg.val_every = 250
    cfg.val_batches = 50
    cfg.eval_every = 2000
    cfg.eval_max_per_task = 500
    cfg.eval_batch_size = 128
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/nanochat-base-core"
    cfg.xpu = "v4"
    cfg.wandb = True
    cfg.wandb_project = "nanochat"

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
