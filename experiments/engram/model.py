from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, attention
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import precompute_rope_embeddings
from parallel import l2p


@dataclass
class EngramConfig:
    """Precomputed engram configuration (not part of weights tree)."""

    vocab_size: int              # total vocab size
    head_offsets: jax.Array      # [n_total_heads] offsets for per-head vocab
    multipliers: jax.Array       # [max_ngram] XOR hash multipliers
    ngrams: Tuple[int, ...]      # which n-grams to use, e.g. (2, 3)
    n_heads: int                 # heads per n-gram type
    per_head_vocab: int          # vocab_size // n_total_heads
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
    n_total_heads = cfg_e.n_heads * len(cfg_e.ngrams)
    per_head_embed = cfg_e.embed_dim // n_total_heads
    D = config.hidden_dim

    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    keys = jax.random.split(key, 4)
    normal = jax.nn.initializers.normal(0.02)
    zeros = jax.nn.initializers.zeros

    return EngramWeights(
        embeddings=w(keys[0], normal, (cfg_e.vocab_size, per_head_embed), ("model_engram_vocab", "model_engram_embed")),
        key_proj=w(keys[1], normal, (cfg_e.embed_dim, D), ("model_engram_embed", "model_engram_hidden")),
        value_proj=w(keys[2], zeros, (cfg_e.embed_dim, D), ("model_engram_embed", "model_engram_hidden")),
        conv_weight=w(keys[3], zeros, (D, cfg_e.kernel_size), ("model_engram_hidden",)),
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
    all_hashes = []

    for ngram_idx, n in enumerate(engram_cfg.ngrams):
        mix = x * engram_cfg.multipliers[0]
        for k in range(1, n):
            shifted = jnp.pad(x[:, :-k], ((0, 0), (k, 0)), constant_values=engram_cfg.pad_id)
            mix = jnp.bitwise_xor(mix, shifted * engram_cfg.multipliers[k])

        for h in range(engram_cfg.n_heads):
            head_idx = ngram_idx * engram_cfg.n_heads + h
            all_hashes.append(mix % engram_cfg.per_head_vocab + engram_cfg.head_offsets[head_idx])

    return jnp.stack(all_hashes, axis=-1)


def engram_forward(x, input_ids, engram_weights, engram_cfg, hidden_dim):
    """Full engram with multi-head hash, gates, and ShortConv."""
    B, S, D = x.shape
    kernel_size, dilation = engram_cfg.kernel_size, max(engram_cfg.ngrams)

    # Hash and embed lookup (indices already include head offsets)
    hash_indices = hash_ngrams_multihead(input_ids, engram_cfg)
    embeds = engram_weights.embeddings.at[hash_indices].get(out_sharding=l2p(("batch", "act_seq", None, None)))
    flat_embeds = embeds.reshape(B, S, -1).astype(jnp.bfloat16)

    # Project to key/value
    key = jnp.matmul(
        flat_embeds, engram_weights.key_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_embed"))
    )
    value = jnp.matmul(
        flat_embeds, engram_weights.value_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_embed"))
    )

    # Gate: sigmoid of RMS-normed key-query alignment
    key_norm = rms_norm(key, None, 1e-6)
    query_norm = rms_norm(x, None, 1e-6)
    gate = (key_norm * query_norm).sum(axis=-1, keepdims=True) / jnp.sqrt(hidden_dim)
    gate = jnp.sign(gate) * jnp.sqrt(jnp.abs(gate).clip(min=1e-6))
    gate = jax.nn.sigmoid(gate).astype(jnp.bfloat16)
    gated_value = gate * value

    # Depthwise dilated conv with RMSNorm and SiLU
    gv_norm = rms_norm(gated_value, None, 1e-5)[:, None, :, :]
    kernel = engram_weights.conv_weight.astype(gv_norm.dtype).T[None, :, None, :]
    pad_len = (kernel_size - 1) * dilation
    conv_out = jax.lax.conv_general_dilated(
        gv_norm,
        kernel,
        window_strides=(1, 1),
        padding=((0, 0), (pad_len, 0)),
        rhs_dilation=(1, dilation),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=D,
    )[:, 0, :, :]
    conv_out = jax.nn.silu(conv_out)

    return (x + gated_value + conv_out).astype(jnp.bfloat16)


def _make_hash_multipliers(max_ngram, seed):
    """Generate deterministic odd multipliers for XOR hashing."""
    key = jax.random.PRNGKey(seed)
    return jax.random.randint(key, (max_ngram,), 1, 1000) * 2 + 1


def make_model_forward(config):
    """Factory that returns forward function with precomputed rope and engram config."""
    rope_cos, rope_sin = precompute_rope_embeddings(
        config.max_seq_len, config.head_dim, config.rope_theta, "bfloat16", sharding=l2p(())
    )

    if getattr(config, "engram", None) and config.engram.enabled:
        cfg_e = config.engram
        ngrams = tuple(cfg_e.ngrams)
        n_total_heads = cfg_e.n_heads * len(ngrams)
        per_head_vocab = cfg_e.vocab_size // n_total_heads
        engram_cfg = EngramConfig(
            vocab_size=cfg_e.vocab_size,
            head_offsets=jnp.arange(n_total_heads) * per_head_vocab,
            multipliers=_make_hash_multipliers(max(ngrams), getattr(config, "seed", 42)),
            ngrams=ngrams,
            n_heads=cfg_e.n_heads,
            per_head_vocab=per_head_vocab,
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
