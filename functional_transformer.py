from dataclasses import dataclass, field
from typing import List

import jax
from jax import numpy as jnp
import optax

from sws import Config
import wandb
import os
from datetime import datetime
from tqdm import tqdm


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

def rms_norm(x, eps=1e-6):
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

    with jax.named_scope("q_proj"):
        q = jnp.einsum("BTD, DNH -> BTNH", x, w.q_proj.astype(jnp.bfloat16))
    with jax.named_scope("k_proj"):
        k = jnp.einsum("BSD, DKH -> BSKH", x, w.k_proj.astype(jnp.bfloat16))
    with jax.named_scope("v_proj"):
        v = jnp.einsum("BSD, DKH -> BSKH", x, w.v_proj.astype(jnp.bfloat16))

    with jax.named_scope("rope"):
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

    with jax.named_scope("qk_norm"):
        q = rms_norm(q)
        k = rms_norm(k)

    with jax.named_scope("repeat_kv"):
        k = jnp.repeat(k, G, axis=2)
        v = jnp.repeat(v, G, axis=2)

    with jax.named_scope("attention_scores"):
        logits = jnp.einsum("BTNH, BSNH -> BNTS", q, k)
        logits *= jax.lax.rsqrt(jnp.array(H, dtype=jnp.bfloat16))
        causal_mask = jnp.tril(jnp.ones((T, T,), dtype=jnp.bfloat16))
        masked_logits = jnp.where(causal_mask, logits, jnp.array(float("-inf"), dtype=jnp.bfloat16))

    with jax.named_scope("softmax"):
        probs = jax.nn.softmax(masked_logits.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)

    with jax.named_scope("attention_output"):
        encoded = jnp.einsum("BNTS, BSNH -> BTNH", probs, v)

    with jax.named_scope("o_proj"):
        out = jnp.einsum("BTNH, NHD -> BTD", encoded, w.o_proj.astype(jnp.bfloat16))

    return out

def mlp(x, w: MLPWeights):
    with jax.named_scope("up_proj"):
        h = x @ w.up_proj.astype(jnp.bfloat16)
    with jax.named_scope("act"):
        h = jax.nn.silu(h)
    with jax.named_scope("down_proj"):
        out = h @ w.down_proj.astype(jnp.bfloat16)
    return out


def glu(x, w: GLUWeights):
    with jax.named_scope("gate_proj"):
        gate = x @ w.gate_proj.astype(jnp.bfloat16)
    with jax.named_scope("up_proj"):
        up = x @ w.up_proj.astype(jnp.bfloat16)
    with jax.named_scope("act"):
        h = jax.nn.silu(gate) * up
    with jax.named_scope("down_proj"):
        out = h @ w.down_proj.astype(jnp.bfloat16)
    return out


def layer(x, w: LayerWeights, cos, sin):
    residual = x
    with jax.named_scope("att_pre_norm"):
        x = rms_norm(x)
    with jax.named_scope("attention"):
        x = attention(x, w.attention_weights, cos, sin)
    with jax.named_scope("add_residual"):
        x = x + residual

    residual = x
    with jax.named_scope("mlp_pre_norm"):
        x = rms_norm(x)
    with jax.named_scope("mlp"):
        x = mlp(x, w.mlp_weights)
    with jax.named_scope("add_residual"):
        x = x + residual
    return x


@jax.jit
def forward(x, w: ModelWeights, cos, sin):
    with jax.named_scope("embed"):
        x = w.embed[x].astype(jnp.bfloat16)
    for i, layer_weights in enumerate(w.layer_weights):
        with jax.named_scope(f"layer_{i}"):
            x = layer(x, layer_weights, cos, sin)
    with jax.named_scope("unembed"):
        logits = x @ w.unembed.astype(jnp.bfloat16)
    return logits


c = Config()

c.model.seq_len = 2048
c.model.vocab_size = 50304
c.model.num_layers = 1
c.model.hidden_dim = 768
c.model.intermediate_dim = lambda: 4 * c.model.hidden_dim
c.model.num_attention_heads = 6
c.model.num_key_value_heads = 3
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
    
    embed = init_fn(next(key_iter), (vocab_size, hidden_dim), dtype=jnp.float32)
    layer_weights = [
        LayerWeights(
            attention_weights=AttentionWeights(
                q_proj=init_fn(next(key_iter), (hidden_dim, num_attention_heads, head_dim), dtype=jnp.float32),
                k_proj=init_fn(next(key_iter), (hidden_dim, num_key_value_heads, head_dim), dtype=jnp.float32),
                v_proj=init_fn(next(key_iter), (hidden_dim, num_key_value_heads, head_dim), dtype=jnp.float32),
                o_proj=init_fn(next(key_iter), (num_attention_heads, head_dim, hidden_dim), dtype=jnp.float32)
            ),
            mlp_weights = MLPWeights(
                up_proj=init_fn(next(key_iter), (hidden_dim, intermediate_dim), dtype=jnp.float32),
                down_proj=init_fn(next(key_iter), (intermediate_dim, hidden_dim), dtype=jnp.float32)
            )
        )
        for _ in range(num_layers)
    ]
    unembed = init_fn(next(key_iter), (hidden_dim, vocab_size), dtype=jnp.float32)
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


def loss_fn(w, x, y, cos, sin):
    with jax.named_scope("forward"):
        logits = forward(x, w, cos, sin)
    with jax.named_scope("loss"):
        logits = logits.reshape(-1, logits.shape[-1])
        labels = y.reshape(-1, 1)
        label_logits = jnp.take_along_axis(logits, labels, axis=-1)
        log_normalizers = jax.nn.logsumexp(logits, axis=-1)
        loss = jnp.mean(log_normalizers - label_logits)
    return loss
    
@jax.jit
def train_step(model_weights, optimizer_state, x, y, cos, sin):
    with jax.named_scope("value_and_grad"):
        loss, grads = jax.value_and_grad(loss_fn)(model_weights, x, y, cos, sin)
    with jax.named_scope("update"):
        updates, optimizer_state = optimizer.update(grads, optimizer_state, model_weights)
    with jax.named_scope("apply_updates"):
        model_weights = optax.apply_updates(model_weights, updates)
    return model_weights, optimizer_state, loss

wandb.init(project="functional-transformer", config=c.to_dict())
# init profiler
os.makedirs("profiles", exist_ok=True)
profile_dir = f"profiles/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
profiler_options = jax.profiler.ProfileOptions()
profiler_options.host_tracer_level = 3

x = jnp.ones((16, c.model.seq_len), dtype=jnp.int32)
y = jnp.ones((16, c.model.seq_len), dtype=jnp.int32)
cos, sin = precompute_rope_embeddings(c.model.seq_len, c.model.head_dim, c.model.rope_base)
step = 0
while True:
    if step == 10:
        jax.profiler.start_trace(profile_dir, profiler_options=profiler_options)
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
        model_weights, optimizer_state, loss = train_step(model_weights, optimizer_state, x, y, cos, sin)
    if step == 20:
        jax.profiler.stop_trace()
        wandb.log_artifact(f"{profile_dir}/", name=f"{wandb.run.id}_profile", type="profile")
        print("done profiling")
        exit()
    step += 1