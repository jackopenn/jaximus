# pyright: reportPossiblyUnboundVariable=false
import importlib
import os
import time
from datetime import datetime

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax

if os.environ.get("JAX_MULTIHOST", "0") == "1":
    jax.distributed.initialize()

import optax
import orbax.checkpoint as ocp
from jax import numpy as jnp
from jax.sharding import AxisType, reshard
from jax.sharding import PartitionSpec as P
from sws import run
from transformers import AutoTokenizer

import wandb
from data.hf import get_hf_dataset
from eval import evaluate_model
from generate import generate
from modelling.layers.position import precompute_rope_embeddings
from parallel import l2p, set_sharding_strategy
from utils import DummyWandb, MetricLogger, pretty_print_model, pretty_print_samples


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
    # init mesh 
    mesh = jax.make_mesh((cfg.parallel.data,), ("data",), (AxisType.Explicit,))
    jax.set_mesh(mesh)
    main_process = jax.process_index() == 0
    if main_process:
        print(f"{mesh=}\n")


    # init tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    dataset = get_hf_dataset(
        hf_name=cfg.data.hf_name,
        sequence_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        tokenizer_name=cfg.data.tokenizer_name,
        streaming=True,
        num_proc=None,
    )

    
    # init sharding strategy, model and optimizer
    set_sharding_strategy(cfg.parallel.strategy)
    model_config = cfg.model
    model_weights = init_model_weights(model_config, jax.random.PRNGKey(cfg.seed))
    tx, optimizer_config, schedule_fns = make_optimizer(cfg)
    opt_weights = tx.init(model_weights)
    accum_steps = cfg.optimizer.accum_steps
    if main_process:
        pretty_print_model(model_weights)


    # calculate num_flops_per_token for mfu
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(model_weights))
    pos_embed = getattr(model_weights, "pos_embed", None)
    num_embed_params = model_weights.embed.size + (pos_embed.size if pos_embed is not None else 0)
    L, N, H, S = (
        model_config.num_layers,
        model_config.num_attention_heads,
        model_config.head_dim,
        model_config.max_seq_len,
    )
    num_flops_per_token = 6 * (num_params - num_embed_params) + L * 12 * N * H * S
    if main_process:
        print(f"{num_params=}\n{num_flops_per_token=}\n")


    # init wandb and logger 
    cfg_dict = cfg.to_dict()
    cfg_dict["optimizers_resolved"] = optimizer_config
    if main_process:
        project_name = cfg.wandb_project if cfg.wandb_project else "transformers"
        wandb_run = wandb.init(project=project_name, config=cfg_dict) if cfg.wandb else DummyWandb()
        train_logger = MetricLogger(
            cfg.data.batch_size,
            accum_steps,
            cfg.data.max_length,
            num_flops_per_token,
            cfg.xpu,
            cfg.max_steps,
            wandb_run,
        )
    

    # init checkpoint manager
    checkpoint_dir = cfg.checkpoint_dir if cfg.checkpoint_dir.startswith("gs://") else os.path.join(os.getcwd(), cfg.checkpoint_dir)
    if main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, cleanup_tmp_directories=True)
    )
    

    # init profiler
    if main_process:
        os.makedirs("profiles", exist_ok=True)
        profile_dir = f"profiles/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.host_tracer_level = 3
        if jax.default_backend() == "gpu":
            profiler_options.gpu_enable_nvtx_tracking = True
            profiler_options.gpu_enable_cupti_activity_graph_trace = True
            profiler_options.gpu_dump_graph_node_mapping = True



    # make jit train step
    train_step, input_sharding = make_train_step(tx, model_config, model_weights, opt_weights, model_forward)

    train_iter = iter(dataset)
    step, micro_step, t0 = 1, 0, time.time()
    while step <= cfg.max_steps:
        batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(input_sharding, x), next(train_iter))

        if main_process and micro_step == 10:
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            model_weights, opt_weights, loss, grad_norm = train_step(model_weights, opt_weights, batch)
        if main_process and micro_step == 20:
            jax.profiler.stop_trace()
            wandb_run.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"{wandb_run.id}_profile", type="profile")

        micro_step += 1
        if micro_step % accum_steps == 0:
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

            if cfg.generate_every > 0 and step % cfg.generate_every == 0:
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

            if cfg.eval_every > 0 and step % cfg.eval_every == 0:
                eval_results = evaluate_model(
                    model_weights, model_config, model_forward, tokenizer, cfg.eval_data_path, cfg.eval_max_per_task, cfg.eval_batch_size
                )
                if main_process:
                    wandb_run.log({f"eval/{k}": v for k, v in eval_results["results"].items()}, step=step)
                    wandb_run.log({f"eval/centered_{k}": v for k, v in eval_results["centered_results"].items()}, step=step)
                    wandb_run.log({"eval/core_metric": eval_results["core_metric"]}, step=step)

            step += 1

    # checkpoint at max steps (ignore if we just did on last step)
    if cfg.checkpoint_every > 0 and cfg.max_steps % cfg.checkpoint_every != 0:
        checkpoint_manager.save(cfg.max_steps, args=ocp.args.StandardSave(model_weights))
        checkpoint_manager.wait_until_finished()
        if main_process:
            wandb_run.log_artifact(
                f"{checkpoint_dir}/{cfg.max_steps}",
                name=f"{wandb_run.id}_model",
                type="model",
                aliases=[f"step_{cfg.max_steps}"],
            )

    # full eval at end (max_per_task=-1)
    if cfg.eval_every > 0:
        eval_results = evaluate_model(
            model_weights, model_config, model_forward, tokenizer, cfg.eval_data_path, -1, cfg.eval_batch_size
        )
        if main_process:
            wandb_run.log({f"eval_final/{k}": v for k, v in eval_results["results"].items()}, step=cfg.max_steps)
            wandb_run.log({f"eval_final/centered_{k}": v for k, v in eval_results["centered_results"].items()}, step=cfg.max_steps)
            wandb_run.log({"eval_final/core_metric": eval_results["core_metric"]}, step=cfg.max_steps)

    if main_process:
        train_logger.flush()
        wandb_run.finish()


def main(cfg):
    model_module = importlib.import_module(f"{cfg.experiment}.model")
    optimizer_module = importlib.import_module(f"{cfg.experiment}.optimizer")
    train(cfg, model_module.init_model_weights, model_module.model_forward, optimizer_module.make_optimizer)

    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    run(main)
