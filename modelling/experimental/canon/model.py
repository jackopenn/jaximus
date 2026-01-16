from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Union
import warnings

import jax
from jax import numpy as jnp

from modelling.layers.core import (
    RMSNormWeights,
    LayerNormWeights,
    NormWeights,
    MLPWeights,
    GLUWeights,
    AttentionWeights,
    norm,
    rms_norm,
    layer_norm,
    resolve_act_fn,
    make_attention_mask,
    validate_enum,
)
from modelling.layers.position import precompute_rope_embeddings, apply_rope
from modelling.layers.init import get_initializers
from modelling.model import (
    ModelConfig,
    VALID_NORM_TYPES,
    VALID_INIT_STRATEGIES,
    VALID_MLP_TYPES,
    VALID_POSITION_EMBEDDING_TYPES,
    _init_weight,
    _init_norm_weights,
    _init_attention_weights,
    _init_mlp_weights,
)
from parallel import logical_to_physical

from modelling.experimental.canon.canon import CanonWeights, canon_layer, init_canon_weights


VALID_CANON_INIT_STRATEGIES = ("identity", "ones", "normal")


@jax.tree_util.register_dataclass
@dataclass
class CanonBlockWeights:
    """Canon weights for all positions within a transformer layer."""
    canon_a: Optional[CanonWeights]  # After att_norm, before attention (hidden_dim)
    canon_b_q: Optional[CanonWeights]  # After Q projection (num_heads * head_dim)
    canon_b_k: Optional[CanonWeights]  # After K projection (num_kv_heads * head_dim)
    canon_b_v: Optional[CanonWeights]  # After V projection (num_kv_heads * head_dim)
    canon_c: Optional[CanonWeights]  # After mlp_norm, before MLP (hidden_dim)
    canon_d: Optional[CanonWeights]  # After up_proj, before activation (intermediate_dim)


@jax.tree_util.register_dataclass
@dataclass
class CanonLayerWeights:
    """Weights for a single transformer layer with optional Canon layers."""
    attention_weights: AttentionWeights
    mlp_weights: Union[MLPWeights, GLUWeights]
    att_norm: Optional[NormWeights]
    mlp_norm: Optional[NormWeights]
    canon: Optional[CanonBlockWeights]


@jax.tree_util.register_dataclass
@dataclass
class CanonModelWeights:
    """Model weights with Canon layer support."""
    embed: jax.Array
    layer_weights: List[CanonLayerWeights]
    unembed: Optional[jax.Array]
    pos_embed: Optional[jax.Array] = None
    embed_norm: Optional[NormWeights] = None
    lm_head_norm: Optional[NormWeights] = None
    lm_head_bias: Optional[jax.Array] = None


@dataclass
class CanonModelConfig:
    """Model configuration with Canon layer options."""
    # Base model config fields
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_dim: int
    max_seq_len: int
    norm_type: str
    norm_position: str
    norm_epsilon: float
    norm_scale: bool
    norm_bias: bool
    mlp_type: str
    act_fn: str
    attn_use_bias: bool
    mlp_use_bias: bool
    lm_head_use_bias: bool
    qk_norm: bool
    qk_norm_type: Optional[str]
    qk_norm_epsilon: Optional[float]
    sliding_window: Optional[int]
    position_embedding_type: str
    rope_theta: Optional[float]
    tie_word_embeddings: bool
    softcap: Optional[float]
    dtype: str
    post_embed_norm: bool
    pre_lm_head_norm: bool
    init_strategy: str
    # Canon-specific config fields
    canon_enabled: bool  # Master switch for Canon layers
    canon_a: bool  # Enable Canon-A (after att_norm, before attention)
    canon_b: bool  # Enable Canon-B (after Q/K/V projections)
    canon_c: bool  # Enable Canon-C (after mlp_norm, before MLP)
    canon_d: bool  # Enable Canon-D (after up_proj, before activation)
    canon_init: str  # "identity" | "ones" | "normal"


def make_config(cfg_dict: dict) -> CanonModelConfig:
    """Create a CanonModelConfig from a config dictionary."""
    # Valid value checks
    validate_enum(cfg_dict["norm_type"], "norm_type", VALID_NORM_TYPES)
    validate_enum(cfg_dict["init_strategy"], "init_strategy", VALID_INIT_STRATEGIES)
    validate_enum(cfg_dict["mlp_type"], "mlp_type", VALID_MLP_TYPES)
    validate_enum(cfg_dict["position_embedding_type"], "position_embedding_type", VALID_POSITION_EMBEDDING_TYPES)
    validate_enum(cfg_dict["qk_norm_type"], "qk_norm_type", VALID_NORM_TYPES, allow_none=True)
    validate_enum(cfg_dict.get("canon_init", "identity"), "canon_init", VALID_CANON_INIT_STRATEGIES)

    # Required parameter checks
    if cfg_dict["position_embedding_type"] == "rope" and cfg_dict["rope_theta"] is None:
        raise ValueError("rope_theta required when position_embedding_type='rope'")
    if cfg_dict["qk_norm"] and cfg_dict["qk_norm_type"] is None:
        raise ValueError("qk_norm_type required when qk_norm=True")
    if cfg_dict["qk_norm"] and cfg_dict["qk_norm_epsilon"] is None:
        raise ValueError("qk_norm_epsilon required when qk_norm=True")

    # Consistency checks
    if cfg_dict["num_attention_heads"] % cfg_dict["num_key_value_heads"] != 0:
        raise ValueError(f"num_attention_heads ({cfg_dict['num_attention_heads']}) must be divisible by num_key_value_heads ({cfg_dict['num_key_value_heads']})")
    if cfg_dict["sliding_window"] is not None and cfg_dict["sliding_window"] < 0:
        raise ValueError(f"sliding_window must be positive or None, got: {cfg_dict['sliding_window']}")

    # Handle conflicting parameters (warn and override)
    rope_theta = cfg_dict["rope_theta"]
    qk_norm_type = cfg_dict["qk_norm_type"]
    qk_norm_epsilon = cfg_dict["qk_norm_epsilon"]
    lm_head_use_bias = cfg_dict["lm_head_use_bias"]

    if cfg_dict["position_embedding_type"] != "rope" and rope_theta is not None:
        warnings.warn(f"rope_theta={rope_theta} ignored because position_embedding_type='{cfg_dict['position_embedding_type']}'")
        rope_theta = None
    if not cfg_dict["qk_norm"] and qk_norm_type is not None:
        warnings.warn(f"qk_norm_type='{qk_norm_type}' ignored because qk_norm=False")
        qk_norm_type = None
    if not cfg_dict["qk_norm"] and qk_norm_epsilon is not None:
        warnings.warn(f"qk_norm_epsilon={qk_norm_epsilon} ignored because qk_norm=False")
        qk_norm_epsilon = None
    if cfg_dict["tie_word_embeddings"] and lm_head_use_bias:
        warnings.warn("lm_head_use_bias=True ignored because tie_word_embeddings=True")
        lm_head_use_bias = False

    return CanonModelConfig(
        vocab_size=cfg_dict["vocab_size"],
        hidden_dim=cfg_dict["hidden_dim"],
        num_layers=cfg_dict["num_layers"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        head_dim=cfg_dict["head_dim"],
        intermediate_dim=cfg_dict["intermediate_dim"],
        max_seq_len=cfg_dict["max_seq_len"],
        norm_type=cfg_dict["norm_type"],
        norm_position=cfg_dict["norm_position"],
        norm_epsilon=cfg_dict["norm_epsilon"],
        norm_scale=cfg_dict["norm_scale"],
        norm_bias=cfg_dict["norm_bias"],
        mlp_type=cfg_dict["mlp_type"],
        act_fn=cfg_dict["act_fn"],
        attn_use_bias=cfg_dict["attn_use_bias"],
        mlp_use_bias=cfg_dict["mlp_use_bias"],
        lm_head_use_bias=lm_head_use_bias,
        qk_norm=cfg_dict["qk_norm"],
        qk_norm_type=qk_norm_type,
        qk_norm_epsilon=qk_norm_epsilon,
        sliding_window=cfg_dict["sliding_window"],
        position_embedding_type=cfg_dict["position_embedding_type"],
        rope_theta=rope_theta,
        tie_word_embeddings=cfg_dict["tie_word_embeddings"],
        softcap=cfg_dict["softcap"],
        dtype=cfg_dict["dtype"],
        post_embed_norm=cfg_dict["post_embed_norm"],
        pre_lm_head_norm=cfg_dict["pre_lm_head_norm"],
        init_strategy=cfg_dict["init_strategy"],
        # Canon fields with defaults
        canon_enabled=cfg_dict.get("canon_enabled", False),
        canon_a=cfg_dict.get("canon_a", False),
        canon_b=cfg_dict.get("canon_b", False),
        canon_c=cfg_dict.get("canon_c", False),
        canon_d=cfg_dict.get("canon_d", False),
        canon_init=cfg_dict.get("canon_init", "identity"),
    )


def attention_with_canon(
    x: jax.Array,
    weights: AttentionWeights,
    canon_b_q: Optional[CanonWeights],
    canon_b_k: Optional[CanonWeights],
    canon_b_v: Optional[CanonWeights],
    rope_cos: Optional[jax.Array],
    rope_sin: Optional[jax.Array],
    qk_norm: bool,
    qk_norm_type: Optional[str],
    qk_norm_epsilon: Optional[float],
    sliding_window: Optional[int],
    dtype: str,
    num_heads: int,
    num_kv_heads: int,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Attention with optional Canon-B layers applied after Q/K/V projections."""
    dtype = getattr(jnp, dtype)
    batch, seq_len, _ = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads

    with jax.named_scope("q_proj"):
        q = jnp.matmul(x, weights.q_proj.astype(dtype), out_sharding=logical_to_physical(("batch", "seq", "act_q")))
        if weights.q_bias is not None:
            q = q + weights.q_bias.astype(dtype)
        # Apply Canon-B to Q before reshape
        if canon_b_q is not None:
            with jax.named_scope("canon_b_q"):
                q = canon_layer(q, canon_b_q)
        q = q.reshape(batch, seq_len, num_heads, head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_q", "act_head")))

    with jax.named_scope("k_proj"):
        k = jnp.matmul(x, weights.k_proj.astype(dtype), out_sharding=logical_to_physical(("batch", "seq", "act_kv")))
        if weights.k_bias is not None:
            k = k + weights.k_bias.astype(dtype)
        # Apply Canon-B to K before reshape
        if canon_b_k is not None:
            with jax.named_scope("canon_b_k"):
                k = canon_layer(k, canon_b_k)
        k = k.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_kv", "act_head")))

    with jax.named_scope("v_proj"):
        v = jnp.matmul(x, weights.v_proj.astype(dtype), out_sharding=logical_to_physical(("batch", "seq", "act_kv")))
        if weights.v_bias is not None:
            v = v + weights.v_bias.astype(dtype)
        # Apply Canon-B to V before reshape
        if canon_b_v is not None:
            with jax.named_scope("canon_b_v"):
                v = canon_layer(v, canon_b_v)
        v = v.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_kv", "act_head")))

    if rope_cos is not None and rope_sin is not None:
        with jax.named_scope("apply_rope"):
            cos = rope_cos[:, :seq_len, :, :]
            sin = rope_sin[:, :seq_len, :, :]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

    if qk_norm:
        norm_factory = partial(
            rms_norm if qk_norm_type == "rms" else layer_norm,
            eps=qk_norm_epsilon,
        )
        with jax.named_scope("q_norm"):
            q = norm_factory(q, weights.q_norm).astype(dtype)
        with jax.named_scope("k_norm"):
            k = norm_factory(k, weights.k_norm).astype(dtype)

    if mask is not None:
        mask = make_attention_mask(mask)

    with jax.named_scope("dot_product_attention"):
        att = jax.nn.dot_product_attention(
            query=q, key=k, value=v,
            is_causal=True,
            implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
            mask=mask,
            local_window_size=(sliding_window, 0) if sliding_window else None
        )

    with jax.named_scope("o_proj"):
        att = att.reshape(batch, seq_len, num_heads * head_dim, out_sharding=logical_to_physical(("batch", "seq", "act_q", "act_head")))
        out = jnp.matmul(
            att, weights.o_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_embed"))
        )
        if weights.o_bias is not None:
            out = out + weights.o_bias.astype(dtype)

    return out


def mlp_with_canon(
    x: jax.Array,
    weights: MLPWeights,
    canon_d: Optional[CanonWeights],
    act_fn: str,
    dtype: str,
) -> jax.Array:
    """MLP with optional Canon-D layer applied after up_proj, before activation."""
    dtype = getattr(jnp, dtype)
    with jax.named_scope("up_proj"):
        h = jnp.matmul(
            x, weights.up_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_intermediate"))
        )
        if weights.up_bias is not None:
            h = h + weights.up_bias.astype(dtype)

    # Apply Canon-D after up_proj, before activation
    if canon_d is not None:
        with jax.named_scope("canon_d"):
            h = canon_layer(h, canon_d)

    with jax.named_scope("act_fn"):
        h = resolve_act_fn(act_fn)(h)

    with jax.named_scope("down_proj"):
        out = jnp.matmul(
            h, weights.down_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_embed"))
        )
        if weights.down_bias is not None:
            out = out + weights.down_bias.astype(dtype)

    return out


def glu_with_canon(
    x: jax.Array,
    weights: GLUWeights,
    canon_d: Optional[CanonWeights],
    act_fn: str,
    dtype: str,
) -> jax.Array:
    """GLU with optional Canon-D layer applied after up_proj, before activation."""
    dtype = getattr(jnp, dtype)
    with jax.named_scope("up_proj"):
        up = jnp.matmul(
            x, weights.up_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_intermediate"))
        )
        if weights.up_bias is not None:
            up = up + weights.up_bias.astype(dtype)

    # Apply Canon-D to up projection, before the gate multiplication
    if canon_d is not None:
        with jax.named_scope("canon_d"):
            up = canon_layer(up, canon_d)

    with jax.named_scope("gate_proj"):
        gate = jnp.matmul(
            x, weights.gate_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_intermediate"))
        )
        if weights.gate_bias is not None:
            gate = gate + weights.gate_bias.astype(dtype)

    with jax.named_scope("down_proj"):
        out = jnp.matmul(
            resolve_act_fn(act_fn)(gate) * up, weights.down_proj.astype(dtype),
            out_sharding=logical_to_physical(("batch", "seq", "act_embed"))
        )
        if weights.down_bias is not None:
            out = out + weights.down_bias.astype(dtype)

    return out


def forward(
    x: jax.Array,
    weights: CanonModelWeights,
    config: CanonModelConfig,
    rope_cos: Optional[jax.Array] = None,
    rope_sin: Optional[jax.Array] = None,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Forward pass with Canon layers."""
    dtype = getattr(jnp, config.dtype)
    eps = config.norm_epsilon

    with jax.named_scope("token_embedding"):
        x = weights.embed.at[x].get(
            out_sharding=logical_to_physical(("batch", "act_seq", "act_embed"))
        ).astype(dtype)

    if config.position_embedding_type == "learned" and weights.pos_embed is not None:
        with jax.named_scope("pos_embedding"):
            seq_len = x.shape[1]
            pos_emb = weights.pos_embed[:seq_len].astype(dtype)
            x = x + pos_emb[None, :, :]

    if config.post_embed_norm:
        with jax.named_scope("embed_norm"):
            x = norm(x, weights.embed_norm, eps)

    # Create attention factory with Canon-B support
    def attention_factory(x, attn_weights, canon_block, mask=None):
        canon_b_q = canon_block.canon_b_q if (canon_block and config.canon_enabled and config.canon_b) else None
        canon_b_k = canon_block.canon_b_k if (canon_block and config.canon_enabled and config.canon_b) else None
        canon_b_v = canon_block.canon_b_v if (canon_block and config.canon_enabled and config.canon_b) else None
        return attention_with_canon(
            x, attn_weights,
            canon_b_q=canon_b_q,
            canon_b_k=canon_b_k,
            canon_b_v=canon_b_v,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            qk_norm=config.qk_norm,
            qk_norm_type=config.qk_norm_type,
            qk_norm_epsilon=config.qk_norm_epsilon,
            sliding_window=config.sliding_window,
            dtype=config.dtype,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            mask=mask,
        )

    # Create MLP factory with Canon-D support
    def mlp_factory(x, mlp_weights, canon_block):
        canon_d = canon_block.canon_d if (canon_block and config.canon_enabled and config.canon_d) else None
        if config.mlp_type == "mlp":
            return mlp_with_canon(x, mlp_weights, canon_d, config.act_fn, config.dtype)
        else:
            return glu_with_canon(x, mlp_weights, canon_d, config.act_fn, config.dtype)

    for idx, layer_weights in enumerate(weights.layer_weights):
        with jax.named_scope(f"layer_{idx}"):
            canon_block = layer_weights.canon

            if config.norm_position == "pre":
                residual = x
                with jax.named_scope("att_pre_norm"):
                    x = norm(x, layer_weights.att_norm, eps)
                # Apply Canon-A after att_norm, before attention
                if canon_block and config.canon_enabled and config.canon_a and canon_block.canon_a:
                    with jax.named_scope("canon_a"):
                        x = canon_layer(x, canon_block.canon_a)
                with jax.named_scope("att"):
                    x = attention_factory(x, layer_weights.attention_weights, canon_block, mask=mask)
                with jax.named_scope("add_residual"):
                    x = x + residual
                residual = x
                with jax.named_scope("mlp_pre_norm"):
                    x = norm(x, layer_weights.mlp_norm, eps)
                # Apply Canon-C after mlp_norm, before MLP
                if canon_block and config.canon_enabled and config.canon_c and canon_block.canon_c:
                    with jax.named_scope("canon_c"):
                        x = canon_layer(x, canon_block.canon_c)
                with jax.named_scope("mlp"):
                    x = mlp_factory(x, layer_weights.mlp_weights, canon_block)
                with jax.named_scope("add_residual"):
                    x = x + residual
            else:
                residual = x
                # Canon-A would be applied before attention in post-norm, but there's no norm before attention
                with jax.named_scope("att"):
                    x = attention_factory(x, layer_weights.attention_weights, canon_block, mask=mask)
                with jax.named_scope("att_post_norm"):
                    x = norm(x, layer_weights.att_norm, eps)
                with jax.named_scope("add_residual"):
                    x = x + residual
                residual = x
                with jax.named_scope("mlp"):
                    x = mlp_factory(x, layer_weights.mlp_weights, canon_block)
                with jax.named_scope("mlp_post_norm"):
                    x = norm(x, layer_weights.mlp_norm, eps)
                with jax.named_scope("add_residual"):
                    x = x + residual

    if config.pre_lm_head_norm:
        with jax.named_scope("lm_head_norm"):
            x = norm(x, weights.lm_head_norm, eps)

    with jax.named_scope("lm_head"):
        if config.tie_word_embeddings:
            logits = jnp.matmul(
                x, weights.embed.T.astype(dtype),
                out_sharding=logical_to_physical(("batch", "act_seq", "act_vocab"))
            )
        else:
            logits = jnp.matmul(
                x, weights.unembed.astype(dtype),
                out_sharding=logical_to_physical(("batch", "act_seq", "act_vocab"))
            )
            if weights.lm_head_bias is not None:
                logits = logits + weights.lm_head_bias.astype(dtype)

    if config.softcap is not None:
        with jax.named_scope("softcap"):
            logits = config.softcap * jnp.tanh(logits.astype(jnp.float32) / config.softcap)

    return logits


def _init_canon_block_weights(
    config: CanonModelConfig,
    key: jax.random.PRNGKey,
) -> Optional[CanonBlockWeights]:
    """Initialize Canon weights for a single layer."""
    if not config.canon_enabled:
        return None

    keys = iter(jax.random.split(key, 6))
    D = config.hidden_dim
    N, K, H = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    I = config.intermediate_dim

    canon_a = None
    if config.canon_a:
        canon_a = init_canon_weights(D, config.canon_init, ("model_embed",), next(keys))

    canon_b_q = canon_b_k = canon_b_v = None
    if config.canon_b:
        canon_b_q = init_canon_weights(N * H, config.canon_init, ("model_q",), next(keys))
        canon_b_k = init_canon_weights(K * H, config.canon_init, ("model_kv",), next(keys))
        canon_b_v = init_canon_weights(K * H, config.canon_init, ("model_kv",), next(keys))

    canon_c = None
    if config.canon_c:
        canon_c = init_canon_weights(D, config.canon_init, ("model_embed",), next(keys))

    canon_d = None
    if config.canon_d:
        canon_d = init_canon_weights(I, config.canon_init, ("model_intermediate",), next(keys))

    return CanonBlockWeights(
        canon_a=canon_a,
        canon_b_q=canon_b_q,
        canon_b_k=canon_b_k,
        canon_b_v=canon_b_v,
        canon_c=canon_c,
        canon_d=canon_d,
    )


def _init_canon_layer_weights(
    config: CanonModelConfig,
    inits,
    keys,
    canon_key: jax.random.PRNGKey,
) -> CanonLayerWeights:
    """Initialize all weights for a single transformer layer with Canon support."""
    return CanonLayerWeights(
        attention_weights=_init_attention_weights(config, inits, keys),
        mlp_weights=_init_mlp_weights(config, inits, keys),
        att_norm=_init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",)),
        mlp_norm=_init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",)),
        canon=_init_canon_block_weights(config, canon_key),
    )


def init_model_weights(config: CanonModelConfig, key: jax.random.PRNGKey) -> CanonModelWeights:
    """Initialize all model weights including Canon layers."""
    inits = get_initializers(config.init_strategy, config.hidden_dim)
    # Split keys: base weights + canon weights per layer
    num_base_keys = 4 + config.num_layers * 14
    keys = jax.random.split(key, num_base_keys + config.num_layers + 1)
    base_keys = iter(keys[:num_base_keys])
    canon_keys = keys[num_base_keys:num_base_keys + config.num_layers]

    embed = _init_weight(inits, next(base_keys), "embed", (config.vocab_size, config.hidden_dim), ("model_vocab", "model_embed"))

    pos_embed = None
    if config.position_embedding_type == "learned":
        pos_embed = _init_weight(inits, next(base_keys), "embed", (config.max_seq_len, config.hidden_dim), ("model_seq", "model_embed"))

    embed_norm = None
    if config.post_embed_norm:
        embed_norm = _init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",))

    layer_weights_list = [
        _init_canon_layer_weights(config, inits, base_keys, canon_keys[i])
        for i in range(config.num_layers)
    ]

    lm_head_norm = None
    if config.pre_lm_head_norm:
        lm_head_norm = _init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",))

    unembed = None
    lm_head_bias = None
    if not config.tie_word_embeddings:
        unembed = _init_weight(inits, next(base_keys), "lm_head", (config.hidden_dim, config.vocab_size), ("model_embed", "model_vocab"))
        if config.lm_head_use_bias:
            lm_head_bias = _init_weight(inits, next(base_keys), "bias", (config.vocab_size,), ("model_vocab",))

    return CanonModelWeights(
        embed=embed,
        layer_weights=layer_weights_list,
        unembed=unembed,
        pos_embed=pos_embed,
        embed_norm=embed_norm,
        lm_head_norm=lm_head_norm,
        lm_head_bias=lm_head_bias,
    )
