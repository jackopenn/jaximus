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
    cfg.model.norm_scale = False
    cfg.model.norm_bias = False
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

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-100BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len

    # target batch size of 524288 tokens (2048 seq_len)
    cfg.data.batch_size = 64

    # optimizer settings
    cfg.optimizer.accum_steps = 4
    cfg.optimizer.grad_clip_norm = 1.0
    cfg.optimizer.decay_type = "linear"
    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)

    # adamw defaults
    cfg.optimizer.adamw.weight_decay = 0.0
    cfg.optimizer.adamw.eps = 1e-10
    cfg.optimizer.adamw.b1 = 0.8
    cfg.optimizer.adamw.b2 = 0.95

    # muon defaults
    cfg.optimizer.muon.nesterov = True
    cfg.optimizer.muon.layer_sharding = True

    # partitions
    cfg.optimizer.embed.patterns = ["embed", "pos_embed"]
    cfg.optimizer.embed.type = "adamw"
    cfg.optimizer.embed.peak_lr = lambda: 0.3 * ((cfg.model.hidden_dim / 768) ** -0.5)

    cfg.optimizer.unembed.patterns = ["unembed"]
    cfg.optimizer.unembed.type = "adamw"
    cfg.optimizer.unembed.peak_lr = lambda: 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)

    cfg.optimizer.other.patterns = "*"
    cfg.optimizer.other.type = "muon"
    cfg.optimizer.other.peak_lr = 0.02


    cfg.max_steps = int(10 * 3.82e9 // 524288)
    cfg.generate_every = 500
    cfg.eval_every = -1
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/nanochat100-we-go-again"
    cfg.xpu = "v4"
    cfg.wandb = True

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
