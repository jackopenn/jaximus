import argparse
import importlib
import shutil

import jax
import orbax.checkpoint as ocp
import wandb
from jax import numpy as jnp
from jax.sharding import AxisType
from sws import Config
from transformers import AutoTokenizer

from data.hf import get_hf_dataset
from parallel import set_sharding_strategy
from train import make_val_step


def make_config_from_wandb(wandb_config):
    """Convert flat wandb config dict to nested config object."""
    cfg = Config()
    for key, value in wandb_config.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg


def evaluate_run(run, val_batches=50):
    """Download model artifact and run validation."""
    config = run.config
    cfg = make_config_from_wandb(config)

    artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    if not artifacts:
        print(f"No model artifact for run {run.name}")
        return None
    artifact = artifacts[-1]
    artifact_dir = artifact.download()

    mesh = jax.make_mesh((1,), ("data",), (AxisType.Explicit,))
    jax.set_mesh(mesh)
    set_sharding_strategy(cfg.parallel.strategy)

    model_module = importlib.import_module("experiments.structure.model")
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    model_forward = model_module.make_model_forward(cfg.model, tokenizer)
    abstract_weights = model_module.init_model_weights(cfg.model, jax.random.PRNGKey(0))

    ckpt_mngr = ocp.CheckpointManager(artifact_dir)
    model_weights = ckpt_mngr.restore(ckpt_mngr.latest_step(), args=ocp.args.StandardRestore(abstract_weights))

    val_dataset = get_hf_dataset(
        hf_name=[cfg.data.hf_name[0], cfg.data.hf_name[1]] if isinstance(cfg.data.hf_name, list) else cfg.data.hf_name,
        sequence_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        tokenizer_name=cfg.data.tokenizer_name,
        streaming=True,
        split="train",
        data_files="shard_01822.parquet",
    )
    val_iter = iter(val_dataset)

    val_step, val_input_sharding = make_val_step(model_weights, model_forward)
    val_loss_sum = jnp.zeros(())
    for _ in range(val_batches):
        val_batch = jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(val_input_sharding, x), next(val_iter)
        )
        val_loss_sum += val_step(model_weights, val_batch)

    shutil.rmtree(artifact_dir)
    return float(val_loss_sum) / val_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="structure")
    parser.add_argument("--val-batches", type=int, default=50)
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(args.project)

    results = []
    for run in runs:
        if run.state != "finished":
            print(f"Skipping {run.name} (state: {run.state})")
            continue
        print(f"Evaluating {run.name}...")
        val_loss = evaluate_run(run, args.val_batches)
        if val_loss is not None:
            run.summary["eval_val_loss"] = val_loss
            run.summary.update()
            results.append({"name": run.name, "val_loss": val_loss})
            print(f"  val_loss: {val_loss:.4f}")

    print("\n=== Results ===")
    for r in sorted(results, key=lambda x: x["val_loss"]):
        print(f"{r['name']}: {r['val_loss']:.4f}")


if __name__ == "__main__":
    main()
