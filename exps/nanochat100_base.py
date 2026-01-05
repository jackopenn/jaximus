
from functools import partial
import jax
from jax import numpy as jnp
import optax
from sws import Config

# def warmup_linear_decay_schedule(
#     init_value,
#     peak_value,
#     end_value,
#     warmup_steps,
#     decay_steps,
#     max_steps
# ):
#     def schedule(step):
#         warmup_pct = step / jnp.maximum(warmup_steps, 1)
#         warmup_value = init_value + (peak_value - init_value) * warmup_pct
        
#         decay_start = max_steps - decay_steps
#         decay_pct = (step - decay_start) / jnp.maximum(decay_steps, 1)
#         decay_value = peak_value + (end_value - peak_value) * decay_pct
        
#         return jnp.where(
#             step < warmup_steps,
#             warmup_value,
#             jnp.where(step < decay_start, peak_value, decay_value)
#         )
#     return schedule


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

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = lambda: cfg.model.max_seq_len

    # target batch size of 524288 tokens (2048 seq_len)
    cfg.data.batch_size = 32
    cfg.optim.accum_steps = 8

    max_steps = 25_000
    cfg.max_steps = max_steps
    cfg.generate_every = 500
    cfg.eval_every = -1
    cfg.checkpoint_every = 5000
    cfg.checkpoint_dir = "gs://trm-jax-123/jaximus/checkpoints/nanochat100-base"
    cfg.xpu = "v4"
    cfg.wandb = False

    cfg.parallel.multihost = True
    cfg.parallel.strategy = "dp"
    cfg.parallel.data = 16


    # ---------- optimizer config ----------
    # learning_rate_schedule = partial(warmup_linear_decay_schedule,
    #     init_value=0.0,
    #     end_value=0.0,
    #     warmup_steps= 0.0 * max_steps,
    #     decay_steps= 0.2 * max_steps,
    #     max_steps=max_steps,
    # )
    # adamw_params = dict(weight_decay=0.0, eps=1e-10, b1=0.8, b2=0.95)
    # cfg.optim.tx = lambda: optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.partition(
    #         {
    #             "token_embedding": optax.adamw(
    #                 learning_rate=learning_rate_schedule(peak_value=0.2 * ((cfg.model.hidden_dim / 768) ** -0.5)),
    #                 **adamw_params,
    #             ),
    #             "lm_head": optax.adamw(
    #                 learning_rate=learning_rate_schedule(peak_value=0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)),
    #                 **adamw_params,
    #             ),
    #             "other": optax.contrib.muon(
    #                 learning_rate=learning_rate_schedule(peak_value=0.02),
    #                 nesterov=True,
    #                 beta=0.95,
    #             ),
    #         },
    #         lambda state: jax.tree.map_with_path(lambda path, _: path[0].key if path[0].key in ("token_embedding", "lm_head") else "other", state)
    #     )
    # )

    return cfg
