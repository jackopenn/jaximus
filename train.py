import os
import re
import time
from datetime import datetime
from functools import partial

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
from sws import run
from transformers import AutoTokenizer

import wandb
from data.hf import get_hf_dataset
from generate import generate
from modelling.model import Model
from utils.common import get_nparams_and_flops, pretty_print_samples
from utils.metric_logger import MetricLogger
from utils.parallel import init_model_and_optimizer_with_sharding,make_and_set_mesh


def train(cfg):
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

    # init model
    partial_model = partial(Model, config=cfg.model, rngs=nnx.Rngs(cfg.seed))

    # init optimizer
    tx = optax.MultiSteps(
            optax.chain(
                optax.clip_by_global_norm(cfg.optim.grad_clip),
                optax.adamw(
                    learning_rate=optax.warmup_cosine_decay_schedule(
                        init_value=cfg.optim.schedule.init_value,
                        peak_value=cfg.optim.schedule.peak_value,
                        warmup_steps=cfg.optim.schedule.warmup_steps,
                        decay_steps=cfg.optim.schedule.decay_steps,
                        end_value=cfg.optim.schedule.end_value,
                    ),
                    weight_decay=cfg.optim.weight_decay,
                    b1=cfg.optim.betas[0],
                    b2=cfg.optim.betas[1],
                    eps=cfg.optim.eps,
                    mask=lambda params: jax.tree.map(lambda x: x.ndim != 1, params), # only wd for 2d tensors
                )
            ),
            every_k_schedule=cfg.optim.accum_steps,
        )

    if cfg.parallel:
        mesh = make_and_set_mesh(cfg.parallel)
        model, optimizer = init_model_and_optimizer_with_sharding(partial_model, tx, cfg.parallel)
        data_sharding = NamedSharding(mesh, PartitionSpec("data", None))
        shard_batch = lambda batch: jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)
        train_iter = (shard_batch(batch) for batch in dataset)
    else:
        model = partial_model()
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        train_iter = iter(dataset)


    def zero_shard(zero_stage, model, optimizer, grads):
        # in stage 1 | 2 force repl of model + grads | model
        repl_sharding = NamedSharding(mesh, PartitionSpec())
        if zero_stage in {1, 2}:
            model = jax.lax.with_sharding_constraint(model, repl_sharding)
            if zero_stage == 1:
                grads = jax.lax.with_sharding_constraint(grads, repl_sharding)
        return model, optimizer, grads


    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]).astype(jnp.float32),
            y.reshape(-1)
        ).mean()
        return loss
    

    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = jax.value_and_grad(loss_fn)(model, batch)
        model, optimizer, grads = zero_shard(cfg.parallel.zero_stage, model, optimizer, grads)
        optimizer.update(model, grads)
        grad_norm = optax.global_norm(grads)
        return loss, grad_norm


    nparams, nflops_per_token = get_nparams_and_flops(model)
    print(f"Number of trainable parameters: {nparams:,}")
    print(f"Number of FLOPS per token: {nflops_per_token:,}")


    # init checkpoint manager
    ckpt_dir = ocp.test_utils.erase_and_create_empty(f'{os.getcwd()}/{cfg.save_dir}/')
    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=5,
        best_fn = lambda x: x,
        best_mode="min",
        cleanup_tmp_directories=True,
        enable_async_checkpointing=False # otherwise wandb logging fails
    )
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)


    # init logging
    wandb.init(project="transformers", config=cfg.to_dict())
    train_logger = MetricLogger(
        batch_size=cfg.optim.batch_size,
        accum_steps=cfg.optim.accum_steps,
        sequence_length=cfg.data.max_length,
        n_flops_per_token=nflops_per_token,
        gpu_name=cfg.gpu,
        optimizer_scheduler=cfg.optim.schedule,
        wandb=wandb,
    )


    # init profiler
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = 3
    profiler_options.gpu_enable_nvtx_tracking = True
    profiler_options.gpu_enable_cupti_activity_graph_trace = True
    profiler_options.gpu_dump_graph_node_mapping = True
    profile_dir = f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


    # https://flax.readthedocs.io/en/latest/guides/performance.html#performance-considerations
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)

    step = 1
    micro_step = 0
    t0 = time.time()
    while step <= cfg.max_steps:
        batch = next(train_iter)

        if micro_step == 10:
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            loss, grad_norm = cached_train_step(batch)

        if micro_step == 20:
            jax.profiler.stop_trace()
            wandb.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"run_{wandb.run.id}_profile", type="profile")

        micro_step += 1

        if micro_step % cfg.optim.accum_steps == 0:
            step_time = time.time() - t0
            t0 = time.time()
            train_logger.log({"loss": loss, "grad_norm": grad_norm, "step_time": step_time, "step": step})

            if step > 1 and step % cfg.generate_every == 0:
                samples = generate(model, tokenizer)
                pretty_print_samples(samples)

            if step > 1 and step % cfg.save_every == 0:
                _, state = nnx.split(model)
                ckpt_mngr.save(step, metrics=loss, args=ocp.args.StandardSave(state))
                wandb.log_artifact(f"{ckpt_dir}/{step}", name=f"run_{wandb.run.id}_model", type="model", aliases=[f"step_{step}"])

            step += 1


if __name__ == "__main__":
    run(train)