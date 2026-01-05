from sws import Config

def gpt2_small():
    cfg = Config()
    cfg.vocab_size = 50304
    cfg.hidden_dim = 768
    cfg.num_layers = 12
    cfg.num_attention_heads = 12
    cfg.num_key_value_heads = 12
    cfg.head_dim = 64
    cfg.intermediate_dim = 3072
    cfg.max_seq_len = 1024
    cfg.norm_type = "layer"
    cfg.norm_position = "pre"
    cfg.norm_epsilon = 1e-5
    cfg.mlp_type = "mlp"
    cfg.act_fn = "gelu"
    cfg.attn_use_bias = False
    cfg.mlp_use_bias = False
    cfg.lm_head_use_bias = False
    cfg.qk_norm = False
    cfg.qk_norm_type = None
    cfg.qk_norm_epsilon = None
    cfg.sliding_window = None
    cfg.position_embedding_type = "learned"
    cfg.rope_theta = None
    cfg.tie_word_embeddings = True
    cfg.init_strategy = "mup"
    cfg.dtype = "bfloat16"
    return cfg
