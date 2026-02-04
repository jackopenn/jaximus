from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights
from modelling.layers.mlp import MLPWeights, mlp
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope, precompute_rope_embeddings
from parallel import l2p


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_window_sizes(window_pattern, num_layers, max_seq_len):
    """Compute per-layer window sizes. L=full context, S=half context. Final layer always L."""
    pattern = window_pattern.upper()
    char_to_window = {"L": max_seq_len, "S": max_seq_len // 2}
    window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(num_layers)]
    window_sizes[-1] = max_seq_len
    return window_sizes


def attention(x, weights, rope_cos, rope_sin, eps, num_heads, num_kv_heads, ve, ve_gate, sliding_window, mask=None):
    """Attention with RoPE, QK norm, optional value embeddings, and sliding window."""
    dtype = jnp.bfloat16
    B, S, _ = x.shape
    H = weights.q_proj.shape[1] // num_heads

    q = jnp.matmul(x, weights.q_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q")))
    q = q.reshape(B, S, num_heads, H, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))

    k = jnp.matmul(x, weights.k_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
    k = k.reshape(B, S, num_kv_heads, H, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    v = jnp.matmul(x, weights.v_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
    v = v.reshape(B, S, num_kv_heads, H, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    if ve is not None:
        ve = ve.reshape(B, S, num_kv_heads, H)
        gate = 2.0 * jax.nn.sigmoid(jnp.matmul(x[..., :32].astype(jnp.float32), ve_gate))  # (B, S, K)
        v = v + gate.astype(dtype)[..., None] * ve

    cos, sin = rope_cos[:, :S, :, :], rope_sin[:, :S, :, :]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q, k = rms_norm(q, None, eps).astype(dtype), rms_norm(k, None, eps).astype(dtype)

    if mask is not None:
        mask = (mask[:, None, None, :] & mask[:, None, :, None]).astype(jnp.bool_)

    att = jax.nn.dot_product_attention(
        query=q,
        key=k,
        value=v,
        is_causal=True,
        implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
        mask=mask,
        local_window_size=(sliding_window, 0),
    )
    att = att.reshape(B, S, num_heads * H, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    return jnp.matmul(att, weights.o_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_embed")))


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    mlp_weights: MLPWeights
    ve_gate: Optional[jax.Array] = None  # (32, K) for layers with value embeddings


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array
    resid_lambdas: jax.Array  # (L,) per-layer residual scaling
    x0_lambdas: jax.Array  # (L,) per-layer skip to initial embedding
    value_embeds: List[Optional[jax.Array]]  # (V, K*H) per layer with VE, None otherwise


def init_model_weights(cfg, key):
    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    V, D, N, K, H, I, L = (
        cfg.model.vocab_size,
        cfg.model.hidden_dim,
        cfg.model.num_attention_heads,
        cfg.model.num_key_value_heads,
        cfg.model.head_dim,
        cfg.model.intermediate_dim,
        cfg.model.num_layers,
    )
    keys = iter(jax.random.split(key, 4 + L * 7 + sum(1 for i in range(L) if has_ve(i, L))))

    # Complete(d)P init stds
    m_N = D / cfg.completedp.base_width
    embed_std = cfg.completedp.base_embed_std
    hidden_std = cfg.completedp.base_std / (cfg.completedp.base_width**0.5) * (m_N**-0.5)
    unembed_std = cfg.completedp.base_unembed_std * (m_N**-1)

    def trunc(std):
        return jax.nn.initializers.truncated_normal(stddev=std, lower=-2 * std, upper=2 * std)

    zeros = jax.nn.initializers.zeros

    layer_weights, value_embeds = [], []
    for i in range(L):
        layer_weights.append(
            LayerWeights(
                attention_weights=AttentionWeights(
                    q_proj=w(next(keys), trunc(hidden_std), (D, N * H), ("model_embed", "model_q")),
                    k_proj=w(next(keys), trunc(hidden_std), (D, K * H), ("model_embed", "model_kv")),
                    v_proj=w(next(keys), trunc(hidden_std), (D, K * H), ("model_embed", "model_kv")),
                    o_proj=w(next(keys), zeros, (N * H, D), ("model_q", "model_embed")),
                ),
                mlp_weights=MLPWeights(
                    up_proj=w(next(keys), trunc(hidden_std), (D, I), ("model_embed", "model_intermediate")),
                    down_proj=w(next(keys), zeros, (I, D), ("model_intermediate", "model_embed")),
                ),
                ve_gate=w(next(keys), zeros, (32, K), (None, "model_kv")) if has_ve(i, L) else None,
            )
        )
        value_embeds.append(w(next(keys), trunc(embed_std), (V, K * H), ("model_vocab", "model_kv")) if has_ve(i, L) else None)

    return ModelWeights(
        embed=w(next(keys), trunc(embed_std), (V, D), ("model_vocab", "model_embed")),
        layer_weights=layer_weights,
        unembed=w(next(keys), trunc(unembed_std), (D, V), ("model_embed", "model_vocab")),
        resid_lambdas=w(next(keys), jax.nn.initializers.ones, (L,), (None,)),
        x0_lambdas=w(next(keys), jax.nn.initializers.constant(0.1), (L,), (None,)),
        value_embeds=value_embeds,
    )


def make_model_forward(cfg, tokenizer=None):
    rope_cos, rope_sin = precompute_rope_embeddings(
        cfg.model.max_seq_len, cfg.model.head_dim, cfg.model.rope_theta, "bfloat16", sharding=l2p(())
    )
    window_pattern = getattr(cfg.model, "window_pattern", "L")
    window_sizes = compute_window_sizes(window_pattern, cfg.model.num_layers, cfg.model.max_seq_len)
    m_L = cfg.model.num_layers / cfg.completedp.base_depth
    resid_scale = m_L ** (-cfg.completedp.alpha)
    return partial(
        model_forward,
        config=cfg.model,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        resid_scale=resid_scale,
        window_sizes=window_sizes,
    )


def model_forward(x, weights, config, rope_cos, rope_sin, resid_scale, window_sizes, mask=None):
    """Forward pass: pre-norm RMSNorm, RoPE, QK norm, relu_squared MLP, softcap=15.0, bfloat16, value embeds, resid lambdas, Complete(d)P, sliding window."""
    eps = config.norm_epsilon
    N, K, H = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    tokens = x  # save for value embedding lookup
    B, S = x.shape

    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
    x = rms_norm(x, None, eps)
    x0 = x  # save initial embedding for x0_lambdas

    mlp_fn = partial(mlp, act_fn="relu_squared", dtype="bfloat16")

    for i, layer_weights in enumerate(weights.layer_weights):
        # apply resid and x0 lambdas at start of block (like nanochat)
        resid_lambda = weights.resid_lambdas[i].astype(jnp.bfloat16)
        x0_lambda = weights.x0_lambdas[i].astype(jnp.bfloat16)
        x = resid_lambda * x + x0_lambda * x0

        # value embedding lookup (None for layers without VE)
        ve = weights.value_embeds[i]
        ve = ve.at[tokens].get(out_sharding=l2p(("batch", "act_seq", "act_kv"))).astype(jnp.bfloat16) if ve is not None else None

        # attention with Complete(d)P residual scaling and sliding window
        residual = x
        x = rms_norm(x, None, eps)
        x = attention(
            x, layer_weights.attention_weights, rope_cos, rope_sin, eps, N, K, ve, layer_weights.ve_gate, window_sizes[i], mask
        )
        x = residual + resid_scale * x

        # MLP with Complete(d)P residual scaling
        residual = x
        x = rms_norm(x, None, eps)
        x = mlp_fn(x, layer_weights.mlp_weights)
        x = residual + resid_scale * x

    x = rms_norm(x, None, eps)
    logits = jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)
