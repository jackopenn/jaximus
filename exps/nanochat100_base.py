from sws import Config


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

    max_steps = 3.8e9 // 524288
    cfg.max_steps = max_steps
    cfg.generate_every = 500
    cfg.eval_every = -1
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/nanochat100-base-v2"
    cfg.xpu = "v4"
    cfg.wandb = False

    cfg.parallel.strategy = "dp"
    cfg.parallel.data = 16
    cfg.parallel.zero_stage = 2

    # ---------- optimizer config ----------
    cfg.optim.warmup_pct = 0.0
    cfg.optim.decay_pct = 0.4
    cfg.optim.te_peak_value = lambda: 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optim.lm_head_peak_value = lambda: 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    cfg.optim.other_peak_value = 0.02
    cfg.optim.muon_momentum_start = 0.85
    cfg.optim.muon_momentum_end = 0.95
    cfg.optim.muon_momentum_warmup_steps = 300

    return cfg
