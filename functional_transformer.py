from dataclasses import dataclass, field
from typing import List
from functools import partial

import jax
from jax import numpy as jnp
import optax

from sws import Config
import wandb
import os
from datetime import datetime
from tqdm import tqdm

from jax.sharding import AxisType, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map


mesh = jax.make_mesh((1,), ("data",), (AxisType.Explicit,))
jax.set_mesh(mesh)
print(f"{mesh=}")



SHARDING_RULES = {
    "dp": {
        "batch": "data",
        "act_seq": None,
        "act_vocab": None,
        "act_embed": None,
        "act_intermediate": None,
        "act_q": None,
        "act_kv": None,
        "model_seq": None,
        "model_vocab": None,
        "model_embed": None,
        "model_intermediate": None,
        "model_q": None,
        "model_kv": None,
    },
    "fsdp": {
        "batch": "data",
        "act_seq": None,
        "act_vocab": None,
        "act_embed": None,
        "act_intermediate": None,
        "act_q": None,
        "act_kv": None,
        "model_seq": None,
        "model_vocab": "data",
        "model_embed": None,
        "model_intermediate": "data",
        "model_q": None,
        "model_kv": None,
        "model_head": "data",
    },

}

_current_strategy = "fsdp"


def logical_to_physical(logical_axes):
    rules = SHARDING_RULES[_current_strategy]
    return P(*[rules.get(axis, None) for axis in logical_axes])


@jax.tree_util.register_dataclass
@dataclass
class AttentionWeights:
  q_proj: jax.Array
  k_proj: jax.Array
  v_proj: jax.Array
  o_proj: jax.Array

@jax.tree_util.register_dataclass
@dataclass
class MLPWeights:
  up_proj: jax.Array
  down_proj: jax.Array

@jax.tree_util.register_dataclass
@dataclass
class GLUWeights:
  gate_proj: jax.Array
  up_proj: jax.Array
  down_proj: jax.Array

@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
  attention_weights: AttentionWeights
  mlp_weights: MLPWeights

@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
  embed: jax.Array
  layer_weights: List[LayerWeights]
  unembed: jax.Array

def rms_norm(x, eps = 1e-6):
  return (x * jax.lax.rsqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def precompute_rope_embeddings(seq_len, head_dim, base):
  channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
  inv_freq = 1.0 / (base ** (channel_range / head_dim))
  t = jnp.arange(seq_len, dtype=jnp.float32)
  freqs = jnp.outer(t, inv_freq)
  cos, sin = jnp.cos(freqs), jnp.sin(freqs)
  cos, sin = cos.astype(jnp.bfloat16), sin.astype(jnp.bfloat16)
  cos, sin = cos[None, :, None, :], sin[None, :, None, :]
  return cos, sin


def apply_rope(x, cos, sin):
    H = x.shape[-1] // 2
    x1, x2 = x[..., :H], x[..., H:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concat([y1, y2], axis=-1)
  

def attention(x, w: AttentionWeights, cos, sin):
  # B = batch size
  # D = embedding dimension
  # S = length of the key/value (source)
  # T = length of the query (target)
  # N = number of attention heads
  # H = dimensions of each attention head
  # K = number of key/value heads
  # G = number of groups, which equals to N // K
  
  T = x.shape[1]
  H = w.q_proj.shape[2]
  G = w.q_proj.shape[1] // w.k_proj.shape[1]
  
  q = jnp.einsum(
    "BTD, DNH -> BTNH", x, w.q_proj.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_q", "act_head"))
    )
  k = jnp.einsum(
    "BSD, DKH -> BSKH", x, w.k_proj.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_kv", "act_head"))
  )
  v = jnp.einsum(
    "BSD, DKH -> BSKH", x, w.v_proj.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_kv", "act_head"))
  )

  q = apply_rope(q, cos, sin)
  k = apply_rope(k, cos, sin)

  q = rms_norm(q)
  k = rms_norm(k)

  k = jnp.repeat(
    k, G, axis=2,
    out_sharding=logical_to_physical(("batch", "act_seq", "act_q", "act_head"))
  )
  v = jnp.repeat(
    v, G, axis=2,
    out_sharding=logical_to_physical(("batch", "act_seq", "act_q", "act_head"))
  )
  
  logits = jnp.einsum(
    "BTNH, BSNH -> BNTS", q, k,
    out_sharding=logical_to_physical(("batch", "act_q", "act_seq", "act_seq"))
  )
  logits *= jax.lax.rsqrt(jnp.array(H, dtype=jnp.bfloat16))
  causal_mask = jnp.tril(jnp.ones((T, T,), dtype=jnp.bfloat16))
  masked_logits = jnp.where(causal_mask, logits, jnp.array(float("-inf"), dtype=jnp.bfloat16))
  probs = jax.nn.softmax(masked_logits.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
  encoded = jnp.einsum(
    "BNTS, BSNH -> BTNH", probs, v,
    out_sharding=logical_to_physical(("batch", "act_seq", "act_q", "act_head"))
  )
  out = jnp.einsum(
    "BTNH, NHD -> BTD", encoded, w.o_proj.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_embed"))
  )

  return out

def mlp(x, w: MLPWeights):
  intermediate = jnp.matmul(
    x, w.up_proj.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_intermediate"))
  )
  return jnp.matmul(
    jax.nn.silu(intermediate), w.down_proj.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_embed"))
  )


def layer(x, w: LayerWeights, cos, sin):
  x = x + attention(rms_norm(x), w.attention_weights, cos, sin)
  x = x + mlp(rms_norm(x), w.mlp_weights)
  return x

@jax.jit
def forward(x, w: ModelWeights, cos, sin):
  x = w.embed.at[x].get(out_sharding=logical_to_physical(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
  for layer_weights in w.layer_weights:
    x = layer(x, layer_weights, cos, sin)
  logits = jnp.matmul(
    x, w.unembed.astype(jnp.bfloat16),
    out_sharding=logical_to_physical(("batch", "act_seq", "act_vocab"))
  )
  return logits


c = Config()

c.model.seq_len = 2048
c.model.vocab_size = 50304
c.model.num_layers = 1
c.model.hidden_dim = 256
c.model.intermediate_dim = lambda: 4 * c.model.hidden_dim
c.model.num_attention_heads = 2
c.model.num_key_value_heads = 2
c.model.head_dim = lambda: c.model.hidden_dim // c.model.num_attention_heads
c.model.rope_base = 10000

c.optimizer.learning_rate = 0.0001
c.optimizer.weight_decay = 0.01
c.optimizer.beta1 = 0.9
c.optimizer.beta2 = 0.999
c.optimizer.eps = 1e-8

c = c.finalize()

def init_model_weights(
    vocab_size,
    num_layers,
    hidden_dim,
    intermediate_dim,
    num_attention_heads,
    num_key_value_heads,
    head_dim
):
    num_weight_arrays = 1 + (num_layers * 6) + 1
    key = jax.random.key(69420)
    key_iter = iter(jax.random.split(key, num_weight_arrays))
    
    init_fn = jax.nn.initializers.lecun_normal()
    
    embed = init_fn(
        next(key_iter), (vocab_size, hidden_dim), dtype=jnp.float32,
        out_sharding=logical_to_physical(("model_vocab", "model_embed"))
    )
    layer_weights = [
        LayerWeights(
            attention_weights=AttentionWeights(
                q_proj=init_fn(
                    next(key_iter), (hidden_dim, num_attention_heads, head_dim), dtype=jnp.float32,
                    out_sharding=logical_to_physical(("model_embed", "model_q", "model_head"))
                ),
                k_proj=init_fn(
                    next(key_iter), (hidden_dim, num_key_value_heads, head_dim), dtype=jnp.float32,
                    out_sharding=logical_to_physical(("model_embed", "model_kv", "model_head"))
                ),
                v_proj=init_fn(
                    next(key_iter), (hidden_dim, num_key_value_heads, head_dim), dtype=jnp.float32,
                    out_sharding=logical_to_physical(("model_embed", "model_kv", "model_head"))
                ),
                o_proj=init_fn(
                    next(key_iter), (num_attention_heads, head_dim, hidden_dim), dtype=jnp.float32,
                    out_sharding=logical_to_physical(("model_q", "model_head", "model_embed"))
                )
            ),
            mlp_weights = MLPWeights(
                up_proj=init_fn(
                    next(key_iter), (hidden_dim, intermediate_dim), dtype=jnp.float32,
                    out_sharding=logical_to_physical(("model_embed", "model_intermediate"))
                ),
                down_proj=init_fn(
                    next(key_iter), (intermediate_dim, hidden_dim), dtype=jnp.float32,
                    out_sharding=logical_to_physical(("model_intermediate", "model_embed"))
                )
            )
        )
        for _ in range(num_layers)
    ]
    unembed = init_fn(
        next(key_iter), (hidden_dim, vocab_size), dtype=jnp.float32,
        out_sharding=logical_to_physical(("model_embed", "model_vocab"))
    )
    model_weights = ModelWeights(embed=embed, layer_weights=layer_weights, unembed=unembed)

    return model_weights



model_weights = init_model_weights(
    vocab_size=c.model.vocab_size,
    num_layers=c.model.num_layers,
    hidden_dim=c.model.hidden_dim,
    intermediate_dim=c.model.intermediate_dim,
    num_attention_heads=c.model.num_attention_heads,
    num_key_value_heads=c.model.num_key_value_heads,
    head_dim=c.model.head_dim
)
optimizer = optax.adamw(
    learning_rate=c.optimizer.learning_rate,
    weight_decay=c.optimizer.weight_decay,
    b1=c.optimizer.beta1,
    b2=c.optimizer.beta2,
    eps=c.optimizer.eps,
)
optimizer_state = optimizer.init(model_weights)

# Reshard optimizer state to data axis
# optimizer_state = (
#     jax.tree.map(lambda x: jax.sharding.reshard(x, P("data",)) if x.ndim > 1 else x, optimizer_state[0]),
#     optimizer_state[1],
#     optimizer_state[2],
# )

if jax.process_index() == 0:
  print("model_weights sharding:")
  print(f"{model_weights.layer_weights[0].attention_weights.k_proj.sharding=}")
  print(f"{model_weights.layer_weights[0].attention_weights.v_proj.sharding=}")
  print(f"{model_weights.layer_weights[0].attention_weights.o_proj.sharding=}")
  print(f"{model_weights.layer_weights[0].mlp_weights.up_proj.sharding=}")
  print(f"{model_weights.layer_weights[0].mlp_weights.down_proj.sharding=}")
  print("optimizer_state sharding:")
  print(f"{optimizer_state[0].mu.layer_weights[0].attention_weights.k_proj.sharding=}")
  print(f"{optimizer_state[0].mu.layer_weights[0].attention_weights.v_proj.sharding=}")
  print(f"{optimizer_state[0].mu.layer_weights[0].attention_weights.o_proj.sharding=}")
  print(f"{optimizer_state[0].mu.layer_weights[0].mlp_weights.up_proj.sharding=}")
  print(f"{optimizer_state[0].mu.layer_weights[0].mlp_weights.down_proj.sharding=}")

def loss_fn(w, x, y, cos, sin):
    logits = forward(x, w, cos, sin)
    label_logits = jnp.take_along_axis(logits, y[..., jnp.newaxis], axis=-1)
    log_normalizers = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return jnp.mean(log_normalizers - label_logits)


# def make_zero_optimizer_update(optimizer, model_weights, optimizer_state):
#     """Create a ZeRO-1/2 style update function with proper specs for the given structures."""
    
#     # Build partition specs matching the pytree structures
#     # Grads and model weights are replicated (P())
#     # Optimizer state is sharded on first axis for arrays with ndim > 1
#     grads_spec = jax.tree.map(lambda _: P(), model_weights)
#     model_spec = jax.tree.map(lambda _: P(), model_weights)
    
#     # For optimizer state: shard arrays with ndim > 1 on first axis, replicate scalars/1D
#     def opt_state_to_spec(x):
#         if hasattr(x, 'ndim') and x.ndim > 1:
#             return P("data",)
#         return P()
    
#     opt_state_spec = jax.tree.map(opt_state_to_spec, optimizer_state)
    
#     # Output specs: updates are replicated, optimizer state stays sharded
#     updates_spec = jax.tree.map(lambda _: P(), model_weights)
    
#     @partial(shard_map, mesh=mesh,
#              in_specs=(grads_spec, opt_state_spec, model_spec),
#              out_specs=(updates_spec, opt_state_spec),
#              check_rep=False)
#     def _zero_update(grads, opt_state, model):
#         # Get axis info for slicing
#         axis_size = jax.lax.psum(1, "data")
#         axis_index = jax.lax.axis_index("data")
        
#         # Reduce-scatter gradients: each device gets 1/N of fully-reduced grads
#         def reduce_scatter_grad(g):
#             if g.ndim > 1:
#                 return jax.lax.psum_scatter(g, "data", scatter_dimension=0, tiled=True)
#             else:
#                 # For 1D arrays (scalars after vmap, biases, etc.), just reduce
#                 return jax.lax.pmean(g, "data")
        
#         grads_sharded = jax.tree.map(reduce_scatter_grad, grads)
        
#         # Slice model weights to match sharded gradients (for weight decay)
#         def slice_to_shard(arr):
#             if arr.ndim > 1:
#                 shard_size = arr.shape[0] // axis_size
#                 return jax.lax.dynamic_slice_in_dim(arr, axis_index * shard_size, shard_size, axis=0)
#             return arr
        
#         model_sharded = jax.tree.map(slice_to_shard, model)
        
#         # Local optimizer update with sharded grads, sharded state, and sliced model
#         updates_sharded, new_opt_state = optimizer.update(grads_sharded, opt_state, model_sharded)
        
#         # All-gather updates to replicate them
#         def all_gather_update(u):
#             if u.ndim > 1:
#                 return jax.lax.all_gather(u, "data", axis=0, tiled=True)
#             else:
#                 return u
        
#         updates_full = jax.tree.map(all_gather_update, updates_sharded)
        
#         return updates_full, new_opt_state
    
#     return _zero_update

# # Create the ZeRO update function with the actual structures
# zero_update = make_zero_optimizer_update(optimizer, model_weights, optimizer_state)

model_sharding = jax.tree.map(lambda x: x.sharding, model_weights)
optimizer_sharding = jax.tree.map(lambda x: x.sharding, optimizer_state)

@jax.jit(
  in_shardings=(model_sharding, optimizer_sharding, P("data",), P("data",), P(), P()),
  out_shardings=(model_sharding, optimizer_sharding, None)
)
def train_step(model_weights, optimizer_state, x, y, cos, sin):
    with jax.named_scope("value_and_grad"):
        loss, grads = jax.value_and_grad(loss_fn)(model_weights, x, y, cos, sin)
    with jax.named_scope("update"):
        # updates, optimizer_state = zerop_update(grads, optimizer_state, model_weights)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, model_weights)
    with jax.named_scope("apply_updates"):
        model_weights = optax.apply_updates(model_weights, updates)
    return model_weights, optimizer_state, loss

if jax.process_index() == 0:
  wandb.init(project="functional-transformer", config=c.to_dict())

  # init profiler
  os.makedirs("profiles", exist_ok=True)
  profile_dir = f"profiles/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  profiler_options = jax.profiler.ProfileOptions()
  profiler_options.host_tracer_level = 3

x = jnp.ones((32, c.model.seq_len), dtype=jnp.int32, out_sharding=logical_to_physical(("batch", "seq")))
y = jnp.ones((32, c.model.seq_len), dtype=jnp.int32, out_sharding=logical_to_physical(("batch", "seq")))

cos, sin = precompute_rope_embeddings(c.model.seq_len, c.model.head_dim, c.model.rope_base)
cos = jax.sharding.reshard(cos, P())
sin = jax.sharding.reshard(sin, P())

step = 0
while True:
    if jax.process_index() == 0 and step == 10:
        jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
        model_weights, optimizer_state, loss = train_step(model_weights, optimizer_state, x, y, cos, sin)
    if jax.process_index() == 0 and step == 30:
        jax.profiler.stop_trace()
        wandb.log_artifact(f"{profile_dir}/", name=f"{wandb.run.id}_profile", type="profile")
        print("done profiling")
        wandb.finish()
        exit()
    step += 1