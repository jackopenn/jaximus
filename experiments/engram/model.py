from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import jax
from jax import numpy as jnp
from jax.sharding import reshard

from modelling.layers.attention import AttentionWeights, attention
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import precompute_rope_embeddings
from parallel import l2p


@dataclass
class EngramConfig:
    """Precomputed engram configuration (not part of weights tree)."""

    vocab_sizes: jax.Array
    head_offsets: jax.Array
    multipliers: jax.Array
    ngram_size: int
    n_heads: int
    pad_id: int
    kernel_size: int
    layer_ids: Tuple[int, ...]


@jax.tree_util.register_dataclass
@dataclass
class EngramWeights:
    embeddings: jax.Array
    key_proj: jax.Array
    value_proj: jax.Array
    conv_weight: jax.Array
    conv_norm_scale: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    glu_weights: GLUWeights


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array
    engram: Optional[List[EngramWeights]] = None


def _init_engram_weights(config, key):
    """Initialize engram weights for one layer."""
    cfg_e = config.engram
    n_total_heads = cfg_e.n_heads * (cfg_e.ngram_size - 1)
    flat_embed_dim = n_total_heads * cfg_e.embed_dim
    total_vocab = cfg_e.vocab_size * n_total_heads
    D = config.hidden_dim

    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    keys = jax.random.split(key, 5)
    normal = jax.nn.initializers.normal(0.02)
    zeros = jax.nn.initializers.zeros

    return EngramWeights(
        embeddings=w(keys[0], normal, (total_vocab, cfg_e.embed_dim), ("model_engram_vocab", "model_engram_embed")),
        key_proj=w(keys[1], normal, (flat_embed_dim, D), ("model_engram_embed", "model_engram_hidden")),
        value_proj=w(keys[2], zeros, (flat_embed_dim, D), ("model_engram_embed", "model_engram_hidden")),
        conv_weight=w(keys[3], zeros, (D, cfg_e.kernel_size), ("model_engram_hidden",)),
        conv_norm_scale=jnp.ones((D,)),
    )


def init_model_weights(config, key):
    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    num_engram_layers = len(config.engram.layer_ids) if getattr(config, "engram", None) and config.engram.enabled else 0
    keys = iter(jax.random.split(key, 3 + config.num_layers * 7 + num_engram_layers))

    V, D, N, K, H, I = (
        config.vocab_size,
        config.hidden_dim,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_dim,
    )
    zeros, normal = jax.nn.initializers.zeros, jax.nn.initializers.lecun_normal()

    layer_weights = []
    for _ in range(config.num_layers):
        layer_weights.append(
            LayerWeights(
                attention_weights=AttentionWeights(
                    q_proj=w(next(keys), normal, (D, N * H), ("model_embed", "model_q")),
                    k_proj=w(next(keys), normal, (D, K * H), ("model_embed", "model_kv")),
                    v_proj=w(next(keys), normal, (D, K * H), ("model_embed", "model_kv")),
                    o_proj=w(next(keys), zeros, (N * H, D), ("model_q", "model_embed")),
                ),
                glu_weights=GLUWeights(
                    gate_proj=w(next(keys), normal, (D, I), ("model_embed", "model_intermediate")),
                    up_proj=w(next(keys), normal, (D, I), ("model_embed", "model_intermediate")),
                    down_proj=w(next(keys), zeros, (I, D), ("model_intermediate", "model_embed")),
                ),
            )
        )

    engram_weights = None
    if getattr(config, "engram", None) and config.engram.enabled:
        engram_keys = jax.random.split(next(keys), len(config.engram.layer_ids))
        engram_weights = [_init_engram_weights(config, k) for k in engram_keys]

    return ModelWeights(
        embed=w(next(keys), normal, (V, D), ("model_vocab", "model_embed")),
        layer_weights=layer_weights,
        unembed=w(next(keys), normal, (D, V), ("model_embed", "model_vocab")),
        engram=engram_weights,
    )


def hash_ngrams_multihead(x, engram_cfg):
    """Multi-head n-gram hashing."""
    B, S = x.shape
    all_hashes = []

    # Get sharding for traced arrays
    x_sharding = jax.typeof(x).sharding

    for n in range(2, engram_cfg.ngram_size + 1):
        mix = x * engram_cfg.multipliers[0]
        for k in range(1, n):
            # Shift x to the right by k positions, padding with pad_id on the left
            pad = jnp.full((B, k), engram_cfg.pad_id, dtype=x.dtype)
            sliced = x[:, :-k] if k < S else jnp.zeros((B, 0), dtype=x.dtype)
            # Reshard to match x's sharding for concatenation
            pad = reshard(pad, x_sharding)
            sliced = reshard(sliced, x_sharding)
            shifted = jnp.concatenate([pad, sliced], axis=1)
            mix = jnp.bitwise_xor(mix, shifted * engram_cfg.multipliers[k])

        for h in range(engram_cfg.n_heads):
            head_idx = (n - 2) * engram_cfg.n_heads + h
            all_hashes.append(mix % engram_cfg.vocab_sizes[head_idx])

    return jnp.stack(all_hashes, axis=-1)


def engram_forward(x, input_ids, engram_weights, engram_cfg, hidden_dim):
    """Full engram with multi-head hash, gates, and ShortConv."""
    B, S, D = x.shape
    kernel_size, dilation = engram_cfg.kernel_size, 3

    # Hash and embed lookup
    hash_indices = hash_ngrams_multihead(input_ids, engram_cfg)
    shifted_indices = hash_indices + engram_cfg.head_offsets
    embeds = engram_weights.embeddings.at[shifted_indices].get(out_sharding=l2p(("batch", "act_seq", None, None)))
    flat_embeds = embeds.reshape(B, S, -1).astype(jnp.bfloat16)

    # Project to key/value
    key = jnp.matmul(flat_embeds, engram_weights.key_proj.astype(jnp.bfloat16))
    value = jnp.matmul(flat_embeds, engram_weights.value_proj.astype(jnp.bfloat16))

    # Gate: sigmoid of RMS-normed key-query alignment
    key_norm = rms_norm(key, None, 1e-6)
    query_norm = rms_norm(x, None, 1e-6)
    gate = (key_norm * query_norm).sum(axis=-1, keepdims=True) / jnp.sqrt(hidden_dim)
    gate = jnp.sign(gate) * jnp.sqrt(jnp.abs(gate).clip(min=1e-6))
    gate = jax.nn.sigmoid(gate).astype(jnp.bfloat16)
    gated_value = gate * value

    # Depthwise dilated conv with RMSNorm and SiLU
    gv_norm = rms_norm(gated_value, None, 1e-5) * engram_weights.conv_norm_scale

    # Causal padding for dilated conv
    pad_len = (kernel_size - 1) * dilation
    gv_padded = jnp.pad(gv_norm, ((0, 0), (pad_len, 0), (0, 0)), mode="constant")

    # Depthwise 1D dilated conv: (B, S+pad, D) -> (B, S, D)
    # Reshape for conv: (B, S+pad, D) -> (B, 1, S+pad, D) for NHWC-like layout
    gv_conv = gv_padded[:, None, :, :]  # (B, 1, S+pad, D)
    # Kernel: (D, kernel_size) -> (1, kernel_size, 1, D) for depthwise
    kernel = engram_weights.conv_weight.astype(gv_conv.dtype).T[None, :, None, :]  # (1, kernel_size, 1, D)
    conv_out = jax.lax.conv_general_dilated(
        gv_conv,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        rhs_dilation=(1, dilation),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=D,
    )[:, 0, :, :]  # (B, S, D)
    conv_out = jax.nn.silu(conv_out)

    return (x + gated_value + conv_out).astype(jnp.bfloat16)


def _make_hash_multipliers(ngram_size, seed):
    """Generate deterministic odd multipliers for XOR hashing."""
    key = jax.random.PRNGKey(seed)
    return jax.random.randint(key, (ngram_size,), 1, 1000) * 2 + 1


def make_model_forward(config):
    """Factory that returns forward function with precomputed rope and engram config."""
    rope_cos, rope_sin = precompute_rope_embeddings(
        config.max_seq_len, config.head_dim, config.rope_theta, "bfloat16", sharding=l2p(())
    )

    if getattr(config, "engram", None) and config.engram.enabled:
        cfg_e = config.engram
        n_total_heads = cfg_e.n_heads * (cfg_e.ngram_size - 1)
        engram_cfg = EngramConfig(
            vocab_sizes=jnp.array([cfg_e.vocab_size] * n_total_heads),
            head_offsets=jnp.arange(n_total_heads) * cfg_e.vocab_size,
            multipliers=_make_hash_multipliers(cfg_e.ngram_size, getattr(config, "seed", 42)),
            ngram_size=cfg_e.ngram_size,
            n_heads=cfg_e.n_heads,
            pad_id=cfg_e.pad_id,
            kernel_size=cfg_e.kernel_size,
            layer_ids=tuple(cfg_e.layer_ids),
        )
        return partial(
            _model_forward_with_engram, config=config, rope_cos=rope_cos, rope_sin=rope_sin, engram_cfg=engram_cfg
        )

    return partial(_model_forward, config=config, rope_cos=rope_cos, rope_sin=rope_sin)


def _model_forward(x, weights, config, rope_cos, rope_sin, mask=None):
    """Base forward pass."""
    norm_fn = partial(rms_norm, weights=None, eps=config.norm_epsilon)
    attention_fn = partial(
        attention,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm=True,
        qk_norm_type="rms",
        qk_norm_epsilon=config.norm_epsilon,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        dtype="bfloat16",
        sliding_window=None,
    )
    mlp_fn = partial(glu, act_fn="silu", dtype="bfloat16")

    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
    x = norm_fn(x)

    for layer_weights in weights.layer_weights:
        residual = x
        x = norm_fn(x)
        x = attention_fn(x, layer_weights.attention_weights, mask=mask)
        x = x + residual

        residual = x
        x = norm_fn(x)
        x = mlp_fn(x, layer_weights.glu_weights)
        x = x + residual

    x = norm_fn(x)
    return jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))


def _model_forward_with_engram(x, weights, config, rope_cos, rope_sin, engram_cfg, mask=None):
    """Forward pass with engram at specified layers."""
    norm_fn = partial(rms_norm, weights=None, eps=config.norm_epsilon)
    attention_fn = partial(
        attention,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm=True,
        qk_norm_type="rms",
        qk_norm_epsilon=config.norm_epsilon,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        dtype="bfloat16",
        sliding_window=None,
    )
    mlp_fn = partial(glu, act_fn="silu", dtype="bfloat16")

    input_ids = x
    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
    x = norm_fn(x)

    engram_layer_set = set(engram_cfg.layer_ids)
    engram_idx = 0

    for layer_idx, layer_weights in enumerate(weights.layer_weights):
        if layer_idx in engram_layer_set:
            x = engram_forward(x, input_ids, weights.engram[engram_idx], engram_cfg, config.hidden_dim)
            engram_idx += 1

        residual = x
        x = norm_fn(x)
        x = attention_fn(x, layer_weights.attention_weights, mask=mask)
        x = x + residual

        residual = x
        x = norm_fn(x)
        x = mlp_fn(x, layer_weights.glu_weights)
        x = x + residual

    x = norm_fn(x)
    return jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
