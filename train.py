import time
import os
import chz
from functools import partial
from utils.common import get_gpu_peak_flops, get_nparams_and_flops
from utils.getters import get_dataset, get_model, get_optimizer
from utils.configs import ExperimentConfig
from utils.metric_logger import MetricLogger
from generate import generate
from transformers import AutoTokenizer
import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from flax import nnx
import optax

import wandb

import orbax.checkpoint as ocp
import re

from utils.parallel import shard_model_and_optimizer


def pretty_print_samples(samples):
    for prompt, samples_list in samples.items():
        print(f"prompt: {prompt}")
        for i, sample in enumerate(samples_list):
            clean = re.sub(r'^(?:<\|endoftext\|>)+', '', sample)
            print(f"sample {i}: {clean}")
        print()

def train(cfg: ExperimentConfig):
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = 3

    tokenizer = AutoTokenizer.from_pretrained(cfg.train_data.tokenizer_name)
    dataset = get_dataset(cfg.train_data, cfg.optimizer.batch_size)
    model = get_model(cfg.model, cfg.seed, cfg.parallel)
    optimizer = get_optimizer(model, cfg.optimizer)

    # shard_batch = lambda batch: batch
    # if cfg.parallel.data_parallel > 1:
    #     num_devices = jax.device_count()

    #     assert 0 < cfg.parallel.data_parallel <= num_devices, (
    #         f"data_parallel must be less than or equal to the number of devices: {num_devices} and greater than 0"
    #     )

    #     mesh = jax.make_mesh((cfg.parallel.data_parallel,), ("data",))
    #     data_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    #     replicated_sharding = NamedSharding(mesh, PartitionSpec())

    #     _, model_state = nnx.split(model)
    #     sharded_model_state = jax.lax.with_sharding_constraint(model_state, replicated_sharding)
    #     nnx.update(model, sharded_model_state)

    #     _, optim_state = nnx.split(optimizer)
    #     sharded_optim_state = jax.lax.with_sharding_constraint(optim_state, replicated_sharding)
    #     nnx.update(optimizer, sharded_optim_state)
        
    #     shard_batch = lambda batch: jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)

    print(model)
    print(jax.device_count(), jax.devices())

    if cfg.parallel:
        num_devices = jax.device_count()
        
        assert 0 < cfg.parallel.data_parallel <= num_devices, (
            f"data_parallel must be less than or equal to the number of devices: {num_devices} and greater than 0"
        )

        mesh = jax.make_mesh((cfg.parallel.data_parallel,), ("data",))
        print(mesh)
        model, optimizer = shard_model_and_optimizer(model, optimizer, cfg.parallel, mesh)
        print(model)
        data_sharding = NamedSharding(mesh, PartitionSpec("data", None))
        shard_batch = lambda batch: jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)
    print("visualizing")
    print("token_embedding")
    jax.debug.visualize_array_sharding(model.token_embedding.embedding.value)
    print("q_proj")
    jax.debug.visualize_array_sharding(model.layers[0].attention.q_proj.kernel.value)
    print("k_proj")
    jax.debug.visualize_array_sharding(model.layers[0].attention.k_proj.kernel.value)
    print("v_proj")
    jax.debug.visualize_array_sharding(model.layers[0].attention.v_proj.kernel.value)
    print("o_proj")
    jax.debug.visualize_array_sharding(model.layers[0].attention.o_proj.kernel.value)
    # jax.debug.visualize_array_sharding(model.layers[0].norm_1.scla)
    # jax.debug.visualize_array_sharding(model.layers[0].norm_2.kernel)
    print("up_proj")
    jax.debug.visualize_array_sharding(model.layers[0].mlp.up_proj.kernel.value)
    print("down_proj")
    jax.debug.visualize_array_sharding(model.layers[0].mlp.down_proj.kernel.value)
    # jax.debug.visualize_array_sharding(model.lm_norm.value)
    
    print("sharding")
    print("token_embedding")
    print(model.token_embedding.embedding.value.sharding)
    print("q_proj")
    print(model.layers[0].attention.q_proj.kernel.value.sharding)
    print("k_proj")
    print(model.layers[0].attention.k_proj.kernel.value.sharding)
    print("v_proj")
    print(model.layers[0].attention.v_proj.kernel.value.sharding)
    print("o_proj")
    print(model.layers[0].attention.o_proj.kernel.value.sharding)
    # jax.debug.visualize_array_sharding(model.layers[0].norm_1.scla)
    # jax.debug.visualize_array_sharding(model.layers[0].norm_2.kernel)
    print("up_proj")
    print(model.layers[0].mlp.up_proj.kernel.value.sharding)
    print("down_proj")
    print(model.layers[0].mlp.down_proj.kernel.value.sharding)
    # jax.debug.visualize_array_sharding(model.lm_norm.value)

    def loss_fn(model, batch):
        x, y = batch
        logits = print(jax.make_jaxpr(model)(x))
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]).astype(jnp.float32),
            y.reshape(-1)
        ).mean()
        return loss
    

    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = jax.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)
        # grad_norm = jnp.sqrt(sum(jnp.sum(jnp.abs(x)**2) for x in jax.tree.leaves(grads)))
        grad_norm = optax.global_norm(grads)
        return loss, grad_norm


    nparams, nflops_per_token = get_nparams_and_flops(model)
    print(f"Number of trainable parameters: {nparams:,}")
    print(f"Number of FLOPS per token: {nflops_per_token:,}")

    ckpt_dir = ocp.test_utils.erase_and_create_empty(f'{os.getcwd()}/{cfg.save_dir}/')
    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=5,
        best_fn = lambda x: x,
        best_mode="min",
        cleanup_tmp_directories=True,
        enable_async_checkpointing=False # otherwise wandb logging fails
    )
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)


    # wandb.init(
    #     project="transformers",
    #     config=cfg,
    # )

    train_logger = MetricLogger(
        batch_size=cfg.optimizer.batch_size,
        accum_steps=cfg.optimizer.accum_steps,
        sequence_length=cfg.train_data.max_length,
        n_flops_per_token=nflops_per_token,
        gpu_name=cfg.gpu,
        optimizer_scheduler=cfg.optimizer.lr,
        # wandb=wandb,
        wandb=None,
    )

    # https://flax.readthedocs.io/en/latest/guides/performance.html#performance-considerations
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)

    if cfg.parallel:
        train_iter = (shard_batch(batch) for batch in dataset)
    else:
        train_iter = iter(dataset)

    step = 1
    micro_step = 0
    t0 = time.time()

    while step <= cfg.steps:

        if micro_step == cfg.start_trace_micro_step:
            jax.profiler.start_trace(cfg.trace_dir, profiler_options=profiler_options)
        
        batch = next(train_iter)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            loss, grad_norm = cached_train_step(batch)

        if micro_step == cfg.end_trace_micro_step:
            jax.profiler.stop_trace()

        micro_step += 1

        if micro_step % cfg.optimizer.accum_steps == 0:
            # loss.block_until_ready()
            step_time = time.time() - t0
            t0 = time.time()

            train_logger.log({
                "loss": loss,
                "grad_norm": grad_norm,
                "step_time": step_time,
            })

            if step > 1:
                if step % cfg.generate_every == 0:
                    samples = generate(model, tokenizer)
                    pretty_print_samples(samples)
            
                if step % cfg.save_every == 0:
                    _, state = nnx.split(model)
                    ckpt_mngr.save(step, metrics=loss, args=ocp.args.StandardSave(state))
                    wandb.log_artifact(f"{ckpt_dir}/{step}", name=f"run_{wandb.run.id}_model", type="model", aliases=[f"step_{step}"])

            step += 1
            

if __name__ == "__main__":
    chz.nested_entrypoint(train)