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
if os.environ.get("JAX_MULTIHOST", "0") == "1":
    import jax
    jax.distributed.initialize()

from datetime import datetime

import jax
from jax.sharding import AxisType
import optax
from jax import numpy as jnp
from sws import run as sws_run

import wandb
from modelling.model import forward, init_model_weights, make_config
from parallel import logical_to_physical, set_sharding_strategy
from utils import DummyWandb


def make_train_step(optimizer, model_config):
    """Create a JIT-compiled train step function."""
    
    def loss_fn(weights, batch):
        x, y = batch
        logits = forward(x, weights, model_config)
        with jax.named_scope("loss"):
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]).astype(jnp.float32),
                y.reshape(-1)
            ).mean()
        return loss
    
    @jax.jit
    def train_step(weights, opt_state, batch):
        with jax.named_scope("value_and_grad"):
            loss, grads = jax.value_and_grad(loss_fn)(weights, batch)
        with jax.named_scope("update"):
            updates, opt_state = optimizer.update(grads, opt_state, weights)
        with jax.named_scope("apply_updates"):
            weights = optax.apply_updates(weights, updates)
        return weights, opt_state, loss
    
    return train_step


def train(cfg):

    # init mesh
    mesh = jax.make_mesh((cfg.parallel.data, ), ("data", ), (AxisType.Explicit))
    jax.set_mesh(mesh)
    main_process = jax.process_index() == 0
    if main_process:
        print(f"{mesh=}")

    # set sharding strategy and init model 
    set_sharding_strategy(cfg.parallel.strategy)
    
    # Create model config and initialize weights
    model_config = make_config(cfg.model.to_dict())
    key = jax.random.PRNGKey(cfg.seed)
    weights = init_model_weights(model_config, key, init_strategy=cfg.model.init_strategy)
  
    tx = optax.adamw(learning_rate=0.01)
    opt_state = tx.init(weights)

    if main_process:
        # init logging
        wandb_run = wandb.init(project="transformers", config=cfg.to_dict()) if cfg.wandb else DummyWandb()

        # init profiler
        os.makedirs("profiles", exist_ok=True)
        profile_dir = f"profiles/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.host_tracer_level = 3

    # Create JIT-compiled train step
    train_step = make_train_step(tx, model_config)
    
    key = jax.random.key(cfg.seed)
    xkey, ykey = jax.random.split(key)
    batch = (
        jax.random.randint(xkey, (cfg.data.batch_size, cfg.data.max_length), minval=0, maxval=cfg.model.vocab_size, dtype=jnp.int32, out_sharding=logical_to_physical(("batch", "seq"))),
        jax.random.randint(ykey, (cfg.data.batch_size, cfg.data.max_length), minval=0, maxval=cfg.model.vocab_size, dtype=jnp.int32, out_sharding=logical_to_physical(("batch", "seq")))
    )
    micro_step = 0
    while True:

        # train step (profile steps 10-20)
        if main_process and micro_step == 10: 
            jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
        with jax.profiler.StepTraceAnnotation("train", step_num=micro_step):
            weights, opt_state, loss = train_step(weights, opt_state, batch)
        if main_process and micro_step == 30:
            jax.profiler.stop_trace()
            wandb_run.log_artifact(f"{os.getcwd()}/{profile_dir}/", name=f"{wandb_run.id}_profile", type="profile")
            print("done profiling")
            exit()

        micro_step += 1

if __name__ == "__main__":
    sws_run(train)
