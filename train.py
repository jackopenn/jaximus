# pyright: reportArgumentType=false, reportOperatorIssue=false
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax

if os.environ.get("JAX_MULTIHOST", "0") == "1":
    jax.distributed.initialize()

import argparse
import time
from datetime import datetime

import optax
import orbax.checkpoint as ocp
from jax import numpy as jnp
from jax.sharding import AxisType, reshard
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer

import wandb
from data.hf import get_hf_dataset
from generate import generate
from modelling.layers.position import precompute_rope_embeddings
from parallel import l2p, set_sharding_strategy
from utils import DummyLogger, DummyWandb, MetricLogger, pretty_print_samples


def make_train_step(optimizer, model_config, model_weights, opt_weights, forward_fn):
    model_weights_sharding = jax.tree.map(lambda x: x.sharding, model_weights)
    opt_weights_sharding = jax.tree.map(lambda x: x.sharding, opt_weights)
    input_sharding = l2p(("batch", "seq"))

    rope_cos, rope_sin = None, None
    if hasattr(model_config, "rope_theta") and model_config.rope_theta is not None:
        rope_cos, rope_sin = precompute_rope_embeddings(
            model_config.max_seq_len, model_config.head_dim, model_config.rope_theta, "bfloat16"
        )
        rope_cos, rope_sin = reshard(rope_cos, P()), reshard(rope_sin, P())

    def loss_fn(model_weights, x, y):
        logits = forward_fn(x, model_weights, model_config, rope_cos=rope_cos, rope_sin=rope_sin)
        return jnp.mean(
            jax.nn.logsumexp(logits, axis=-1, keepdims=True) - jnp.take_along_axis(logits, y[..., jnp.newaxis], axis=-1)
        )

    @jax.jit(
        in_shardings=(model_weights_sharding, opt_weights_sharding, (input_sharding, input_sharding)),
        out_shardings=(model_weights_sharding, opt_weights_sharding, None, None),
    )
    def train_step(model_weights, opt_weights, batch):
        x, y = batch
        loss, grads = jax.value_and_grad(loss_fn)(model_weights, x, y)
        updates, opt_weights = optimizer.update(grads, opt_weights, model_weights)
        return optax.apply_updates(model_weights, updates), opt_weights, loss, optax.global_norm(grads)

    return train_step, input_sharding


def train(cfg, init_model_weights, model_forward, make_optimizer):
    mesh = jax.make_mesh((cfg.parallel.data,), ("data",), (AxisType.Explicit,))
    jax.set_mesh(mesh)
    main_process = jax.process_index() == 0
    if main_process:
        print(f"{mesh=}\n")

    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    dataset = get_hf_dataset(
        hf_name=cfg.data.hf_name,
        sequence_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        tokenizer_name=cfg.data.tokenizer_name,
        streaming=True,
        num_proc=None,
    )

    set_sharding_strategy(cfg.parallel.strategy)
    model_config = cfg.model
    model_weights = init_model_weights(model_config, jax.random.PRNGKey(cfg.seed))

    tx, optimizer_config, schedule_fns = make_optimizer(cfg)
    opt_weights = tx.init(model_weights)

    print("model_weights sharding:")
    print(f"embed: {jax.typeof(model_weights.embed)}")
    print("layer_weights:\n    attention_weights:")
    print(f"        q_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.q_proj)}")
    print(f"        k_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.k_proj)}")
    print(f"        v_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.v_proj)}")
    print(f"        o_proj: {jax.typeof(model_weights.layer_weights[0].attention_weights.o_proj)}")
    print("    mlp_weights:")
    print(f"        up_proj: {jax.typeof(model_weights.layer_weights[0].mlp_weights.up_proj)}")
    print(f"        down_proj: {jax.typeof(model_weights.layer_weights[0].mlp_weights.down_proj)}")
    print(f"unembed: {jax.typeof(model_weights.unembed)}\n")

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model_weights))
    pos_embed = getattr(model_weights, "pos_embed", None)
    num_embed_params = model_weights.embed.size + (pos_embed.size if pos_embed is not None else 0)
    num_flops_per_token = (
        6 * (num_params - num_embed_params)
        + model_config.num_layers
        * 12
        * model_config.num_attention_heads
        * model_config.head_dim
        * model_config.max_seq_len
    )
    if main_process:
        print(f"{num_params=}\n{num_flops_per_token=}\n")

    cfg_dict = cfg.to_dict()
    cfg_dict["optimizers_resolved"] = optimizer_config
    accum_steps_raw = getattr(cfg.optimizer, "accum_steps", 1)
    accum_steps = accum_steps_raw() if callable(accum_steps_raw) else accum_steps_raw

    wandb_run = (
        (wandb.init(project="transformers", config=cfg_dict) if cfg.wandb else DummyWandb())
        if main_process
        else DummyWandb()
    )
    train_logger = (
        MetricLogger(
            batch_size=cfg.data.batch_size,
            accum_steps=accum_steps,
            sequence_length=cfg.data.max_length,
            num_flops_per_token=num_flops_per_token,
            xpu_name=cfg.xpu,
            max_steps=cfg.max_steps,
            wandb_run=wandb_run,
        )
        if main_process
        else DummyLogger()
    )

    if main_process:
        os.makedirs("profiles", exist_ok=True)
    profile_dir = f"profiles/{datetime.now().strftime('%Y%m%d_%H%M%S')}" if main_process else None
    profiler_options = None
    if main_process:
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.host_tracer_level = 3
        if jax.default_backend() == "gpu":
            profiler_options.gpu_enable_nvtx_tracking = True
            profiler_options.gpu_enable_cupti_activity_graph_trace = True
            profiler_options.gpu_dump_graph_node_mapping = True

    checkpoint_dir = (
        cfg.checkpoint_dir if cfg.checkpoint_dir.startswith("gs://") else os.path.join(os.getcwd(), cfg.checkpoint_dir)
    )
    if jax.process_index() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, options=ocp.CheckpointManagerOptions(max_to_keep=1, cleanup_tmp_directories=True)
    )

    train_step, input_sharding = make_train_step(tx, model_config, model_weights, opt_weights, model_forward)

    train_iter = iter(dataset)
    step, micro_step, t0 = 1, 0, time.time()
    while step <= cfg.max_steps:
        batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(input_sharding, x), next(train_iter))

        if main_process and micro_step == 10 and profile_dir and profiler_options:
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            model_weights, opt_weights, loss, grad_norm = train_step(model_weights, opt_weights, batch)
        if main_process and micro_step == 20 and profile_dir:
            jax.profiler.stop_trace()
            wandb_run.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"{wandb_run.id}_profile", type="profile")

        micro_step += 1
        accum_steps_val = int(
            cfg.optimizer.accum_steps() if callable(cfg.optimizer.accum_steps) else cfg.optimizer.accum_steps
        )

        if micro_step % accum_steps_val == 0:
            step_time, t0 = time.time() - t0, time.time()

            if main_process:
                train_logger.log(
                    {
                        "loss": loss,
                        "grad_norm": grad_norm,
                        "step_time": step_time,
                        "step": step,
                        **{name: fn(step) for name, fn in schedule_fns.items()},
                    }
                )

            if step % cfg.generate_every == 0:
                samples = generate(model_weights, model_config, tokenizer, model_forward)
                if main_process:
                    pretty_print_samples(samples)

            if cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0:
                checkpoint_manager.save(step, args=ocp.args.StandardSave(model_weights))
                checkpoint_manager.wait_until_finished()
                if main_process:
                    wandb_run.log_artifact(
                        f"{checkpoint_dir}/{step}", name=f"{wandb_run.id}_model", type="model", aliases=[f"step_{step}"]
                    )

            step += 1

    if cfg.checkpoint_every > 0 and cfg.max_steps % cfg.checkpoint_every != 0:
        checkpoint_manager.save(int(cfg.max_steps), args=ocp.args.StandardSave(model_weights))
        checkpoint_manager.wait_until_finished()
        if main_process:
            wandb_run.log_artifact(
                f"{checkpoint_dir}/{cfg.max_steps}",
                name=f"{wandb_run.id}_model",
                type="model",
                aliases=[f"step_{cfg.max_steps}"],
            )

    if main_process:
        train_logger.flush()
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/nanochat/config_debug.py")
    args = parser.parse_args()

    # Convert path to module: experiments/nanochat/config.py -> experiments.nanochat.config
    import importlib

    exp_dir = os.path.dirname(args.config).replace("/", ".")
    config_module = args.config.replace("/", ".").removesuffix(".py")

    cfg = importlib.import_module(config_module).get_config()
    model_module = importlib.import_module(f"{exp_dir}.model")
    optimizer_module = importlib.import_module(f"{exp_dir}.optimizer")

    train(cfg, model_module.init_model_weights, model_module.model_forward, optimizer_module.make_optimizer)

    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
