# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    cfg = Config()
    cfg.experiment = "experiments.deepseek_v4"
    cfg.seed = 42
    cfg.exp_name = "deepseek-v4-scaled-flash"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 12
    cfg.model.hidden_dim = 1024
    cfg.model.num_attention_heads = 16
    cfg.model.head_dim = 128
    cfg.model.query_compression_dim = 256
    cfg.model.indexer_num_heads = 16
    cfg.model.indexer_head_dim = 64
    cfg.model.output_groups = 4
    cfg.model.output_group_dim = 256
    cfg.model.max_seq_len = 2048
    cfg.model.rope_theta = 10000.0
    cfg.model.partial_rope_dim = 64
    cfg.model.norm_epsilon = 1e-6
    cfg.model.attention_schedule = "flash"
    cfg.model.csa_compression_rate = 4
    cfg.model.csa_top_k = 128
    cfg.model.hca_compression_rate = 128
    cfg.model.sliding_window = 128

    cfg.model.num_shared_experts = 1
    cfg.model.num_routed_experts = 32
    cfg.model.num_experts_per_tok = 4
    cfg.model.expert_intermediate_dim = 512
    cfg.model.hash_routing_layers = 3
    cfg.model.moe_up_clip = 10.0

    cfg.model.mhc_width = 4
    cfg.model.mhc_sinkhorn_steps = 20
    cfg.model.mhc_dynamic_std = 0.001
    cfg.model.mhc_residual_diag_init = 2.0

    cfg.model.mtp_depth = 0
    cfg.model.logit_softcap = 15.0
    cfg.model.dtype = "bfloat16"

    cfg.data.hf_name = ["karpathy/fineweb-edu-100b-shuffle", "default"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 8

    cfg.optimizer.total_batch_size = 524288
    cfg.optimizer.accum_steps = lambda: cfg.optimizer.total_batch_size // (cfg.data.batch_size * cfg.model.max_seq_len)
    cfg.optimizer.warmup_steps = 2000
    cfg.optimizer.decay_steps = lambda: int(0.1 * cfg.max_steps)
    cfg.optimizer.embed.peak_lr = 2.7e-4
    cfg.optimizer.unembed.peak_lr = 2.7e-4
    cfg.optimizer.adam.peak_lr = 2.7e-4
    cfg.optimizer.muon.peak_lr = 2.7e-4
    cfg.optimizer.muon.momentum = 0.95
    cfg.optimizer.weight_decay = 0.1

    cfg.max_steps = int(8.1e9 // 524288)
    cfg.generate_every = 500
    cfg.val_every = 99999
    cfg.val_batches = 50
    cfg.eval_every = 1000
    cfg.eval_max_per_task = 500
    cfg.eval_batch_size = 64
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/deepseek-v4-scaled-flash"
    cfg.xpu = "v4"
    cfg.wandb = True
    cfg.wandb_project = "deepseek-v4"

    cfg.parallel.strategy = "fsdp"
    cfg.parallel.data = 16

    return cfg
