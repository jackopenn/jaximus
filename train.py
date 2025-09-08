import time
import os
import chz
from functools import partial
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

@partial(jax.jit, static_argnames=["max_length", "top_k", "temperature"])
def sample(model, tokens, max_length=64, top_k=50, temperature=1.0):
    for _ in range(max_length):
        logits = model(tokens)
        logits = logits[:, -1, :] / temperature
        values, indices = jax.lax.top_k(logits, k=top_k)
        logits = jnp.where(logits < values[:, -1][:, jnp.newaxis], -jnp.inf, logits)
        next_token = jax.random.categorical(jax.random.PRNGKey(0), logits)
        tokens = jnp.concatenate([tokens, next_token.reshape(-1, 1)], axis=1)
    return tokens


def generate(model, tokenizer, prompt, max_length, n_samples=1, top_k=50, temperature=1.0):
    x = tokenizer.encode(prompt, return_tensors="np")[0]
    x = jnp.stack(jnp.concatenate([jnp.array([tokenizer.bos_token_id]), x]))
    x = jnp.stack([x for _ in range(n_samples)])
    
    x = sample(model, x, max_length, top_k, temperature)

    return tokenizer.batch_decode(x[:, 1:])


def train(cfg: ExperimentConfig):
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

    tokenizer, dataset = get_dataset(cfg.train_data, cfg.optimizer.batch_size)
    model = get_model(cfg.model, cfg.seed)
    optimizer = get_optimizer(model,cfg.optimizer)

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
            logits.reshape(-1, logits.shape[-1]).astype(jnp.float32),
            y.reshape(-1)
        ).mean()
        return loss
    
    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = jax.value_and_grad(loss_fn)(model, batch)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.abs(x)**2) for x in jax.tree.leaves(grads)))
        optimizer.update(model, grads)
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

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm")
    )

    wandb.init(
        project="transformers",
        config=cfg,
    )

    # https://flax.readthedocs.io/en/latest/guides/performance.html#performance-considerations
    cached_train_step = nnx.cached_partial(train_step, model, optimizer)

    train_iter = (shard_batch(batch) for batch in dataset)

    tokens_per_batch = cfg.optimizer.batch_size * cfg.train_data.max_length
    step = 1
    micro_step = 0
    tokens_consumed = 0
    gpus_peak_flops = get_gpu_peak_flops(cfg.gpu) * cfg.parallel.data_parallel

    t0 = time.time()
    while step <= cfg.steps:
        
        if micro_step == cfg.start_trace_micro_step:
            jax.profiler.start_trace(cfg.trace_dir, profiler_options=profiler_options)
        
        batch = next(train_iter)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            loss, grad_norm = cached_train_step(batch)
        metrics.update(loss=loss, grad_norm=grad_norm)

        if micro_step == cfg.end_trace_micro_step:
            jax.profiler.stop_trace()

        tokens_consumed += tokens_per_batch
        micro_step += 1

        if (step == 1 or step % cfg.log_every == 0) and micro_step % cfg.optimizer.accum_steps == 0:
            loss.block_until_ready()
            t1 = time.time()
            
            step_time = (t1 - t0) / (cfg.log_every * cfg.optimizer.accum_steps if step > 1 else cfg.optimizer.accum_steps)

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

            if step == 1 or step % cfg.generate_every == 0:
                for prompt in [
                    "What is the meaning of life?",
                    "Hello, I'm a language model",
                    "5+7=",
                    "What is the capital of France?",
                ]:
                    samples = generate(model, tokenizer, prompt, max_length=8, n_samples=5, top_k=50, temperature=1.0)
                    print(f"prompt: {prompt}")
                    for i, sample in enumerate(samples):
                        print(f"sample {i}: {sample}")
                    wandb.log({"prompt": prompt, "sample": samples}, step=step)
            
            if step > 0 and step % cfg.save_every == 0:
                _, state = nnx.split(model)
                ckpt_mngr.save(step, metrics=log_stats['loss'].item(), args=ocp.args.StandardSave(state))
                wandb.log_artifact(f"{ckpt_dir}/{step}", name=f"run_{wandb.run.id}_model", type="model", aliases=[f"step_{step}"])
            
            t0 = time.time()

        if micro_step % cfg.optimizer.accum_steps == 0:
            step += 1
            

if __name__ == "__main__":
    chz.nested_entrypoint(train)