import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

# XLA performance flags for TPU
# os.environ["XLA_FLAGS"] = " ".join([
#     "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
# #   "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false",
# #   "--xla_enable_async_all_gather=true",
# ])

# os.environ['XLA_FLAGS'] = (
#     # '--xla_gpu_triton_gemm_any=True '
#     # '--xla_gpu_enable_latency_hiding_scheduler=true '
#     # '--xla_tpu_enable_data_parallel_all_reduce_opt=true '
#     # '--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false '
#     # '--xla_enable_async_all_gather=true '
#     # "--xla_tpu_megacore_fusion_allow_ags=true "
# )

# Must initialize distributed JAX BEFORE any other JAX imports
# Check env var to determine if multihost mode is needed
import jax
if os.environ.get("JAX_MULTIHOST", "0") == "1":
    jax.distributed.initialize()

import time
from datetime import datetime

from jax.sharding import AxisType, PartitionSpec as P, reshard
import optax
import orbax.checkpoint as ocp
from jax import numpy as jnp
from jax.sharding import reshard
from sws import run as sws_run
from transformers import AutoTokenizer

import wandb
from data.hf import get_hf_dataset
from generate import generate
from modelling.model import forward, init_model_weights, make_config
from modelling.layers.position import precompute_rope_embeddings
from optimizer import make_optimizer
from parallel import logical_to_physical, set_sharding_strategy
from utils import DummyWandb, pretty_print_samples, MetricLogger


def make_train_step(optimizer, model_config, model_weights, opt_weights):
    model_weights_sharding = jax.tree.map(lambda x: x.sharding, model_weights)
    opt_weights_sharding = jax.tree.map(lambda x: x.sharding, opt_weights)
    input_sharding = logical_to_physical(("batch", "seq"))
    
    rope_cos, rope_sin = None, None
    if model_config.position_embedding_type == "rope":
        rope_cos, rope_sin = precompute_rope_embeddings(
            model_config.max_seq_len, model_config.head_dim, model_config.rope_theta, model_config.dtype
        )
        rope_cos = reshard(rope_cos, P())
        rope_sin = reshard(rope_sin, P())
    
    def loss_fn(model_weights, x, y):
        logits = forward(x, model_weights, model_config, rope_cos=rope_cos, rope_sin=rope_sin)
        label_logits = jnp.take_along_axis(logits, y[..., jnp.newaxis], axis=-1)
        log_normalizers = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        loss = jnp.mean(log_normalizers - label_logits)
        return loss
    
    @jax.jit(
        in_shardings=(model_weights_sharding, opt_weights_sharding, (input_sharding, input_sharding)),
        out_shardings=(model_weights_sharding, opt_weights_sharding, None, None)
    )
    def train_step(model_weights, opt_weights, batch):
        x, y = batch
        loss, grads = jax.value_and_grad(loss_fn)(model_weights, x, y)
        grad_norm = optax.global_norm(grads)
        updates, opt_weights = optimizer.update(grads, opt_weights, model_weights)
        model_weights = optax.apply_updates(model_weights, updates)
        return model_weights, opt_weights, loss, grad_norm
    
    return train_step, input_sharding

def train(cfg):
    # init mesh
    mesh = jax.make_mesh((cfg.parallel.data, ), ("data", ), (AxisType.Explicit,))
    jax.set_mesh(mesh)
    main_process = jax.process_index() == 0
    if main_process:
        print(f"{mesh=}")
        print()

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
    model_config = make_config(cfg.model.to_dict())
    key = jax.random.PRNGKey(cfg.seed)
    model_weights = init_model_weights(model_config, key)

    # init optimizer
    tx, optimizer_config = make_optimizer(cfg)
    opt_weights = tx.init(model_weights)
    
    print("model_weights sharding:")
    print(f"embed: {jax.typeof(model_weights.embed)}")
    print(f"layer_weights:")
    print(f"    attention_weights:")
    print(f"        q_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.q_proj)}")
    print(f"        k_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.k_proj)}")
    print(f"        v_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.v_proj)}")
    print(f"        o_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.o_proj)}")
    print(f"    mlp_weights:")
    print(f"        up_proj: {jax.typeof(model_weights.layer_weights[0].mlp_weights.up_proj)}")
    print(f"        down_proj: {jax.typeof(model_weights.layer_weights[0].mlp_weights.down_proj)}")
    print(f"unembed: {jax.typeof(model_weights.unembed)}")
    print()
    
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model_weights))
    num_embed_params = model_weights.embed.size + (model_weights.pos_embed.size if model_weights.pos_embed is not None else 0)
    num_flops_per_token = (
        6 * (num_params - num_embed_params) 
        + model_config.num_layers * 12 * model_config.num_attention_heads * model_config.head_dim * model_config.max_seq_len
    )
    if main_process:
        print(f"{num_params=}")
        print(f"{num_flops_per_token=}")
        print()
        
        # init logging
        cfg_dict = cfg.to_dict()
        # Add resolved optimizer config
        cfg_dict["optimizers_resolved"] = optimizer_config
        wandb_run = wandb.init(project="transformers", config=cfg_dict) if cfg.wandb else DummyWandb()
        accum_steps = _get_value(getattr(cfg.optimizer, 'accum_steps', None), 1)
        
        train_logger = MetricLogger(
            batch_size=cfg.data.batch_size,
            accum_steps=accum_steps,
            sequence_length=cfg.data.max_length,
            num_flops_per_token=num_flops_per_token,
            xpu_name=cfg.xpu,
            max_steps=cfg.max_steps,
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
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=1, cleanup_tmp_directories=True)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=checkpoint_options)

    train_step, input_sharding = make_train_step(tx, model_config, model_weights, opt_weights)

    train_iter = iter(dataset)
    step = 1
    micro_step = 0
    t0 = time.time()
    while step <= cfg.max_steps:
        batch = next(train_iter)
        batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(input_sharding, x), batch)

        # train step (profile steps 10-20)
        if main_process and micro_step == 10: 
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            model_weights, opt_weights, loss, grad_norm = train_step(model_weights, opt_weights, batch)
        if main_process and micro_step == 20:
            jax.profiler.stop_trace()
            wandb_run.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"{wandb_run.id}_profile", type="profile")

        micro_step += 1

        accum_steps_val = cfg.optimizer.accum_steps() if callable(cfg.optimizer.accum_steps) else cfg.optimizer.accum_steps

        if micro_step % accum_steps_val == 0:
            step_time = time.time() - t0
            t0 = time.time()

            # log metrics
            if main_process:
                # Pure Python LR computation (no JAX, doesn't block TPU)
                import math
                lrs = {}
                for partition_name, part_cfg in optimizer_config.items():
                    peak_lr = part_cfg['peak_lr']
                    warmup_steps = part_cfg.get('warmup_steps', 0)
                    decay_steps = part_cfg.get('decay_steps', 0)
                    decay_type = part_cfg.get('decay_type', 'linear')
                    decay_start = cfg.max_steps - decay_steps
                    opt_type = part_cfg['type']

                    if warmup_steps or decay_steps:
                        if step < warmup_steps:
                            lr = peak_lr * step / max(warmup_steps, 1)
                        elif step < decay_start:
                            lr = peak_lr
                        else:
                            decay_pct = (step - decay_start) / max(decay_steps, 1)
                            if decay_type == "cosine":
                                lr = peak_lr * 0.5 * (1 + math.cos(math.pi * decay_pct))
                            else:
                                lr = peak_lr * (1 - decay_pct)
                    else:
                        lr = peak_lr

                    lrs[f"{partition_name}_lr"] = lr

                    if opt_type == "muon":
                        lrs[f"{partition_name}_muon_lr"] = 0.85 + min(step / 300, 1.0) * 0.10
                
                train_logger.log({
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "step_time": step_time,
                    "step": step,
                    **lrs,
                })

            # generate samples
            if step % cfg.generate_every == 0:
                samples = generate(model_weights, model_config, tokenizer)
                if main_process:
                    pretty_print_samples(samples)

            # checkpoint
            if step % cfg.checkpoint_every == 0:
                checkpoint_manager.save(step, args=ocp.args.StandardSave(model_weights))
                checkpoint_manager.wait_until_finished() # must wait before logging to wandb
                if main_process:
                    wandb_run.log_artifact(f"{checkpoint_dir}/{step}", name=f"{wandb_run.id}_model", type="model", aliases=[f"step_{step}"])

            step += 1
    
    # final checkpoint (skip if last step was already checkpointed)
    if cfg.max_steps % cfg.checkpoint_every != 0:
        checkpoint_manager.save(int(cfg.max_steps), args=ocp.args.StandardSave(model_weights))
        checkpoint_manager.wait_until_finished() # must wait before logging to wandb
        if main_process:
            wandb_run.log_artifact(f"{checkpoint_dir}/{cfg.max_steps}", name=f"{wandb_run.id}_model", type="model", aliases=[f"step_{cfg.max_steps}"])
    
    if main_process:
        train_logger.flush()
        wandb_run.finish()

if __name__ == "__main__":
    sws_run(train)
