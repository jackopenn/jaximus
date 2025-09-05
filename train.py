import time
import os
import chz
from utils.common import get_gpu_peak_flops, get_nparams_and_flops, pretty_log
from utils.getters import get_dataset, get_model, get_optimizer
from utils.configs import ExperimentConfig

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from flax import nnx
import optax

import wandb

import orbax.checkpoint as ocp


def generate(model, tokenizer, prompt, max_length):
    x = tokenizer.encode(prompt, return_tensors="np")
    x = x.reshape(1, -1)
    for _ in range(max_length):
        logits = model(x)
        next_token = jnp.argmax(logits[0,-1, :], axis=-1)
        x = jnp.concatenate([x, next_token.reshape(1, 1)], axis=1)
    return tokenizer.decode(x[0])


def train(cfg: ExperimentConfig):
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    jax.config.update("jax_compiler_enable_remat_pass", False)
    profiler_options = jax.profiler.ProfileOptions()
    profiler_options.host_tracer_level = 3


    assert cfg.generate_every % cfg.log_every == 0, (
        f"generate_every must be a multiple of log_every for accurate timing :)"
    )
    # assert cfg.eval_every % cfg.log_every == 0, (
    #     f"eval_every must be a multiple of log_every"
    # )
    assert cfg.save_every % cfg.log_every == 0, (
        f"save_every must be a multiple of log_every"
    )

    tokenizer, dataset = get_dataset(cfg.train_data)
    model = get_model(cfg.model, cfg.seed)
    optimizer = get_optimizer(model,cfg.optimizer)

    dataset = dataset.batch(cfg.optimizer.batch_size)
    
    print(cfg.optimizer.batch_size, next(iter(dataset))[0].shape)



    shard_batch = lambda batch: batch
    if cfg.parallel.data_parallel > 1:
        num_devices = jax.device_count()

        assert 0 < cfg.parallel.data_parallel <= num_devices, (
            f"data_parallel must be less than or equal to the number of devices: {num_devices} and greater than 0"
        )

        mesh = jax.make_mesh((cfg.parallel.data_parallel,), ("data",))
        data_sharding = NamedSharding(mesh, PartitionSpec("data", None))
        replicated_sharding = NamedSharding(mesh, PartitionSpec())

        _, model_state = nnx.split(model)
        sharded_model_state = jax.lax.with_sharding_constraint(model_state, replicated_sharding)
        nnx.update(model, sharded_model_state)

        _, optim_state = nnx.split(optimizer)
        sharded_optim_state = jax.lax.with_sharding_constraint(optim_state, replicated_sharding)
        nnx.update(optimizer, sharded_optim_state)
        
        shard_batch = lambda batch: jax.tree_util.tree_map(lambda x: jax.device_put(x, data_sharding), batch)


    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        ).mean()
        return loss
    
    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = jax.value_and_grad(loss_fn)(model, batch)
        optimizer.update(model, grads)
        return loss


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

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss")
    )

    wandb.init(
        project="transformers",
        config=cfg,
    )

    train_iter = iter(dataset)

    tokens_per_batch = cfg.optimizer.batch_size * cfg.train_data.max_length
    step = 0
    micro_step = 0
    tokens_consumed = 0
    gpus_peak_flops = get_gpu_peak_flops(cfg.gpu) * cfg.parallel.data_parallel

    # https://flax.readthedocs.io/en/latest/guides/performance.html#performance-considerations
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)
    
    # warmup
    t0 = time.time()
    batch = shard_batch(next(train_iter))
    loss = cached_train_step(batch)
    loss.block_until_ready()
    t1 = time.time()
    print(f"warmup time: {t1 - t0}")
    tokens_consumed += tokens_per_batch
    micro_step += 1
    step = micro_step // cfg.optimizer.accum_steps
    

    t0 = time.time()
    while step < cfg.steps:
        
        if step == cfg.start_trace_step:
            jax.profiler.start_trace(cfg.trace_dir, profiler_options=profiler_options)
        
        batch = next(train_iter)
        batch = shard_batch(batch)

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            loss = cached_train_step(batch)
        metrics.update(loss=loss)

        if step == cfg.end_trace_step:
            jax.profiler.stop_trace()

        tokens_consumed += tokens_per_batch
        micro_step += 1
        

        if step % cfg.log_every == 0 and micro_step % cfg.optimizer.accum_steps == 0:
            loss.block_until_ready()
            t1 = time.time()
            
            step_time = (t1 - t0) / (cfg.log_every * cfg.optimizer.accum_steps if step > 0 else 1)

            log_stats = metrics.compute()
            log_stats["step_time"] = step_time
            log_stats["tokens_consumed"] = tokens_consumed
            log_stats["tokens_per_second"] = tokens_per_batch / step_time
            log_stats["tokens_per_second_per_device"] = log_stats["tokens_per_second"] / cfg.parallel.data_parallel
            log_stats["mfu"] = ((nflops_per_token * log_stats["tokens_per_second"]) / gpus_peak_flops) * 100
            log_stats["lr"] = cfg.optimizer.lr if isinstance(cfg.optimizer.lr, float) else cfg.optimizer.lr(step)

            pretty_log(step, log_stats)
            wandb.log(log_stats, step=step)
            
            metrics.reset()

            if step % cfg.generate_every == 0:
                sample = generate(model, tokenizer, "What is the meaning of life?", 16)
                print(f"step: {step}, sample: {sample}")
                wandb.log({"sample": sample}, step=step)
            

            if step > 0 and step % cfg.save_every == 0:
                _, state = nnx.split(model)
                ckpt_mngr.save(step, metrics=log_stats['loss'].item(), args=ocp.args.StandardSave(state))
                wandb.log_artifact(f"{ckpt_dir}/{step}", name=f"run_{wandb.run.id}_model", type="model", aliases=[f"step_{step}"])
            
            t0 = time.time()

        if micro_step % cfg.optimizer.accum_steps == 0:
            step += 1
            

if __name__ == "__main__":
    chz.nested_entrypoint(train)