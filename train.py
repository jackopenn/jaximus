import os

# Must initialize distributed JAX BEFORE any other JAX imports
# Check env var to determine if multihost mode is needed
if os.environ.get("JAX_MULTIHOST", "0") == "1":
    import jax
    jax.distributed.initialize()

from functools import partial
import time
from datetime import datetime
import warnings

import jax
from jax.sharding import AxisType, PartitionSpec as P
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax import numpy as jnp
from sws import run as sws_run
from transformers import AutoTokenizer

import wandb
from data.hf import get_hf_dataset
from generate import generate
from modelling.model import Model
from parallel import logical_to_physical, set_sharding_strategy
from utils import DummyWandb, get_num_params_and_flops, pretty_print_samples, MetricLogger


def validate_config(cfg):
    """Validate config consistency before model creation."""
    m = cfg.model
    
    if m.position_embedding_type == "rope" and m.rope_theta is None:
        raise ValueError("rope_theta required when using RoPE")
    
    if m.qk_norm and m.qk_norm_type is None:
        raise ValueError("qk_norm_type required when qk_norm=True")
    
    if m.qk_norm and m.qk_norm_epsilon is None:
        raise ValueError("qk_norm_epsilon required when qk_norm=True")
    
    if m.num_attention_heads % m.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    
    if m.mlp_type not in ("mlp", "glu"):
        raise ValueError(f"mlp_type must be 'mlp' or 'glu', got '{m.mlp_type}'")
    
    if m.norm_type not in ("rms", "layer"):
        raise ValueError(f"norm_type must be 'rms' or 'layer', got '{m.norm_type}'")
    
    if m.position_embedding_type not in ("learned", "rope", "none"):
        raise ValueError(f"position_embedding_type must be 'learned', 'rope', or 'none', got '{m.position_embedding_type}'")

    if m.sliding_window is not None and m.sliding_window < 0:
        raise ValueError(f"sliding_window must be positive or None, got {m.sliding_window}")
    
    if m.init_strategy not in ("nanochat", "default"):
        raise ValueError(f"init_strategy must be 'nanochat' or 'default', got '{m.init_strategy}'")
    
    # Conflicting parameters - warn and override
    if m.position_embedding_type != "rope" and m.rope_theta is not None:
        warnings.warn(f"rope_theta={m.rope_theta} ignored because position_embedding_type='{m.position_embedding_type}'")
        m.rope_theta = None
    
    if not m.qk_norm and m.qk_norm_type is not None:
        warnings.warn(f"qk_norm_type='{m.qk_norm_type}' ignored because qk_norm=False")
        m.qk_norm_type = None
    
    if not m.qk_norm and m.qk_norm_epsilon is not None:
        warnings.warn(f"qk_norm_epsilon={m.qk_norm_epsilon} ignored because qk_norm=False")
        m.qk_norm_epsilon = None
    
    if m.tie_word_embeddings and m.lm_head_use_bias:
        warnings.warn("lm_head_use_bias=True ignored because tie_word_embeddings=True")
        m.lm_head_use_bias = False


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]).astype(jnp.float32),
            y.reshape(-1)
        ).mean()
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    grad_norm = optax.global_norm(grads)
    return loss, grad_norm


def train(cfg):

    # init mesh
    mesh = jax.make_mesh((cfg.parallel.data, ), ("data", ), (AxisType.Explicit))
    jax.set_mesh(mesh)
    main_process = jax.process_index() == 0
    if main_process:
        print(f"{mesh=}")

    # validate configV
    validate_config(cfg)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)

    # init dataset
    dataset = get_hf_dataset(
        hf_name=cfg.data.hf_name,
        sequence_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        tokenizer_name=cfg.data.tokenizer_name,
        streaming=True,
        num_proc=None,
    )

    # set sharding strategy and init model 
    set_sharding_strategy(cfg.parallel.strategy)
    model = Model(rngs=nnx.Rngs(cfg.seed), **cfg.model.to_dict())
    
    # init optimizer
    def warmup_linear_decay_schedule(
        init_value,
        peak_value,
        end_value,
        warmup_steps,
        decay_steps,
        max_steps
    ):
        def schedule(step):
            warmup_pct = step / jnp.maximum(warmup_steps, 1)
            warmup_value = init_value + (peak_value - init_value) * warmup_pct
            
            decay_start = max_steps - decay_steps
            decay_pct = (step - decay_start) / jnp.maximum(decay_steps, 1)
            decay_value = peak_value + (end_value - peak_value) * decay_pct
            
            return jnp.where(
                step < warmup_steps,
                warmup_value,
                jnp.where(step < decay_start, peak_value, decay_value)
            )
        return schedule
    
    adamw_params = dict(weight_decay=0.0, eps=1e-10, b1=0.8, b2=0.95)
    te_peak_value = 0.2 * ((cfg.model.hidden_dim / 768) ** -0.5)
    lm_head_peak_value = 0.004 * ((cfg.model.hidden_dim / 768) ** -0.5)
    other_peak_value = 0.02
    print(f"{te_peak_value=}")
    print(f"{lm_head_peak_value=}")
    print(f"{other_peak_value=}")
    
    # Schedule params shared between optimizer and logging
    schedule_params = dict(
        init_value=0.0, end_value=0.0,
        warmup_steps=0.0 * cfg.max_steps, decay_steps=0.2 * cfg.max_steps, max_steps=cfg.max_steps,
    )
    
    # JAX schedules for optimizer (also used for logging on CPU)
    lr_schedule_te = warmup_linear_decay_schedule(peak_value=te_peak_value, **schedule_params)
    lr_schedule_lm_head = warmup_linear_decay_schedule(peak_value=lm_head_peak_value, **schedule_params)
    lr_schedule_other = warmup_linear_decay_schedule(peak_value=other_peak_value, **schedule_params)
    
    # Momentum schedule for Muon optimizer (0.85 -> 0.95 over first 300 steps)
    def muon_momentum_schedule(step):
        frac = jnp.minimum(step / 300, 1.0)
        return (1 - frac) * 0.85 + frac * 0.95
    
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.partition(
            {
                "token_embedding": optax.adamw(
                    learning_rate=lr_schedule_te,
                    **adamw_params,
                ),
                "lm_head": optax.adamw(
                    learning_rate=lr_schedule_lm_head,
                    **adamw_params,
                ),
                "other": optax.inject_hyperparams(optax.contrib.muon)(
                    learning_rate=lr_schedule_other,
                    nesterov=True,
                    beta=muon_momentum_schedule,
                ),
            },
            lambda state: jax.tree.map_with_path(lambda path, _: path[0].key if path[0].key in ("token_embedding", "lm_head") else "other", state)
        )
    )
    tx = optax.MultiSteps(tx, every_k_schedule=cfg.optim.accum_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    if main_process:
        # print model stats
        num_params, num_flops_per_token = get_num_params_and_flops(model)
        print(f"{num_params=}")
        print(f"{num_flops_per_token=}")

        # init logging
        wandb_run = wandb.init(project="transformers", config=cfg.to_dict()) if cfg.wandb else DummyWandb()
        train_logger = MetricLogger(
            batch_size=cfg.data.batch_size,
            accum_steps=cfg.optim.accum_steps,
            sequence_length=cfg.data.max_length,
            num_flops_per_token=num_flops_per_token,
            xpu_name=cfg.xpu,
            wandb_run=wandb_run,
        )

        # init profiler
        os.makedirs("profiles", exist_ok=True)
        profile_dir = f"profiles/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.host_tracer_level = 3
        if jax.default_backend() == "gpu":
            profiler_options.gpu_enable_nvtx_tracking = True
            profiler_options.gpu_enable_cupti_activity_graph_trace = True
            profiler_options.gpu_dump_graph_node_mapping = True
    
    # init checkpoint manager
    checkpoint_dir = cfg.checkpoint_dir if cfg.checkpoint_dir.startswith("gs://") else os.path.join(os.getcwd(), cfg.checkpoint_dir)
    if jax.process_index() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=1,cleanup_tmp_directories=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=checkpoint_options)

    # https://flax.readthedocs.io/en/stable/guides/performance.html#caching-graph-node-traversals
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)

    train_iter = iter(dataset)
    step = 1
    micro_step = 0
    t0 = time.time()
    while step <= cfg.max_steps:
        batch = next(train_iter)
        batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(logical_to_physical(("batch", "seq")), x), batch)

        # train step (profile steps 10-20)
        if main_process and micro_step == 10: 
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            loss, grad_norm = cached_train_step(batch)
        if main_process and micro_step == 20:
            jax.profiler.stop_trace()
            wandb_run.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"{wandb_run.id}_profile", type="profile")

        micro_step += 1

        if micro_step % cfg.optim.accum_steps == 0:
            step_time = time.time() - t0
            t0 = time.time()

            # log metrics
            if main_process:
                # with jax.default_device(jax.devices("cpu")[0]):
                lr_te = lr_schedule_te(step)
                lr_lm_head = lr_schedule_lm_head(step)
                lr_other = lr_schedule_other(step)
                muon_momentum = muon_momentum_schedule(step)
                train_logger.log({
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "step_time": step_time,
                    "step": step,
                    "muon_momentum": muon_momentum,
                    "lr/token_embedding": lr_te,
                    "lr/lm_head": lr_lm_head,
                    "lr/other": lr_other,
                })

            # generate samples
            if step % cfg.generate_every == 0:
                samples = generate(model, tokenizer)
                if main_process:
                    pretty_print_samples(samples)

            # checkpoint
            if step % cfg.checkpoint_every == 0:
                checkpoint_manager.save(step, args=ocp.args.StandardSave(nnx.state(model)))
                checkpoint_manager.wait_until_finished() # must wait before logging to wandb
                if main_process:
                    wandb_run.log_artifact(f"{checkpoint_dir}/{step}", name=f"{wandb_run.id}_model", type="model", aliases=[f"step_{step}"])

            step += 1
    
    # final checkpoint (skip if last step was already checkpointed)
    if cfg.max_steps % cfg.checkpoint_every != 0:
        checkpoint_manager.save(cfg.max_steps, args=ocp.args.StandardSave(nnx.state(model)))
        checkpoint_manager.wait_until_finished() # must wait before logging to wandb
        if main_process:
            wandb_run.log_artifact(f"{checkpoint_dir}/{cfg.max_steps}", name=f"{wandb_run.id}_model", type="model", aliases=[f"step_{cfg.max_steps}"])
    
    if main_process:
        train_logger.flush()
        wandb_run.finish()

if __name__ == "__main__":
    sws_run(train)
