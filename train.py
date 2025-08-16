import time
import os
from utils import ExpConfig, get_dataset, get_model, get_optimizer

import jax
from jax import numpy as jnp

from flax import nnx
import optax

import wandb

# import orbax.checkpoint as ocp

def pretty_print(step, metrics):
    print(f"step: {step}", end=", ")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}", end=", ")
    print()


def generate(model, tokenizer, prompt, max_length):
    x = tokenizer.encode(prompt, return_tensors="np")
    x = x.reshape(1, -1)
    for _ in range(max_length):
        logits = model(x)
        next_token = jnp.argmax(logits[0,-1, :], axis=-1)
        x = jnp.concatenate([x, next_token.reshape(1, 1)], axis=1)
    return tokenizer.decode(x[0])

def train(config: ExpConfig):

    tokenizer, dataset = get_dataset(config.data)
    model = get_model(config.model, config.seed)
    optimizer = get_optimizer(model,config.optim)

    dataset = dataset.batch(config.optim.batch_size)

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


    num_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"Number of trainable parameters: {num_params:,}")

    # ckpt_dir = ocp.test_utils.erase_and_create_empty(f'{os.getcwd()}/checkpoints/')
    # ckpt_options = ocp.CheckpointManagerOptions(
    #     max_to_keep=1,
    #     best_fn = lambda x: x,
    #     best_mode="min",
    #     cleanup_tmp_directories=True,
    #     enable_async_checkpointing=False
    # )
    # ckpt_mngr = ocp.CheckpointManager(ckpt_dir, options=ckpt_options)

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        step_time=nnx.metrics.Average("step_time"),
    )

    wandb.init(
        project="transformers",
        config=config,
    )

    train_iter = iter(dataset)

    step = 0
    accum_steps = 0


    while step < config.train.num_steps:
        batch = next(train_iter)

        # train step
        t0 = time.time()
        loss = train_step(model, optimizer, batch)
        t1 = time.time()
    
        metrics.update(loss=loss, step_time=(t1 - t0))
        
        if step % config.train.log_every == 0 and accum_steps == 0:
            log_stats = metrics.compute()
            log_stats["toks_s"] = (config.optim.batch_size * config.data.max_length) / log_stats["step_time"]
            log_stats["lr"] = config.optim.lr if isinstance(config.optim.lr, float) else config.optim.lr(step)
            pretty_print(step, log_stats)
            wandb.log(log_stats)
            metrics.reset()

        if step % config.train.eval_every == 0 and accum_steps == 0:
            sample = generate(model, tokenizer, "My favourite food is ", 16)
            print(f"step: {step}, sample: {sample}")
            wandb.log({"sample": sample})

        # if step % config.train.save_every == 0:
        #     nnx.save(model, f"checkpoints/model_{step}.flax")

        accum_steps += 1
        if accum_steps == config.optim.accum_steps:
            accum_steps = 0
            step += 1