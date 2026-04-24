# pyright: reportAttributeAccessIssue=false, reportOperatorIssue=false
from sws import Config


def get_config():
    cfg = Config()
    cfg.experiment = "experiments.deepseek_v4"
    cfg.seed = 42
    cfg.exp_name = "deepseek-v4-debug"

    cfg.model.vocab_size = 50304
    cfg.model.num_layers = 4
    cfg.model.hidden_dim = 128
    cfg.model.num_attention_heads = 4
    cfg.model.head_dim = 32
    cfg.model.query_compression_dim = 64
    cfg.model.indexer_num_heads = 4
    cfg.model.indexer_head_dim = 32
    cfg.model.output_groups = 2
    cfg.model.output_group_dim = 64
    cfg.model.max_seq_len = 64
    cfg.model.rope_theta = 10000.0
    cfg.model.partial_rope_dim = 32
    cfg.model.norm_epsilon = 1e-6
    cfg.model.attention_schedule = "flash"
    cfg.model.csa_compression_rate = 4
    cfg.model.csa_top_k = 2
    cfg.model.hca_compression_rate = 8
    cfg.model.sliding_window = 16

    cfg.model.num_shared_experts = 1
    cfg.model.num_routed_experts = 4
    cfg.model.num_experts_per_tok = 2
    cfg.model.expert_intermediate_dim = 64
    cfg.model.hash_routing_layers = 1
    cfg.model.moe_up_clip = 10.0

    cfg.model.mhc_width = 2
    cfg.model.mhc_sinkhorn_steps = 6
    cfg.model.mhc_dynamic_std = 0.001
    cfg.model.mhc_residual_diag_init = 2.0

    cfg.model.mtp_depth = 0
    cfg.model.logit_softcap = 15.0
    cfg.model.dtype = "bfloat16"

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len
    cfg.data.batch_size = 2

    cfg.optimizer.accum_steps = 1
    cfg.optimizer.warmup_steps = 0
    cfg.optimizer.decay_steps = lambda: int(0.4 * cfg.max_steps)
    cfg.optimizer.embed.peak_lr = 0.003
    cfg.optimizer.unembed.peak_lr = 0.003
    cfg.optimizer.adam.peak_lr = 0.003
    cfg.optimizer.muon.peak_lr = 0.02
    cfg.optimizer.muon.momentum = 0.95
    cfg.optimizer.weight_decay = 0.1

    cfg.max_steps = 5
    cfg.generate_every = 0
    cfg.val_every = 0
    cfg.val_batches = 0
    cfg.eval_every = 0
    cfg.eval_max_per_task = 10
    cfg.eval_batch_size = 2
    cfg.eval_data_path = "cache"
    cfg.checkpoint_every = -1
    cfg.checkpoint_dir = "checkpoints/deepseek-v4-debug"
    cfg.xpu = "v4"
    cfg.wandb = False
    cfg.wandb_project = "deepseek-v4"

    cfg.parallel.strategy = "dp"
    cfg.parallel.data = 1

    return cfg
