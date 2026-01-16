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
    attention,
    mlp,
    glu,
    norm,
    validate_enum,
)
from modelling.layers.position import precompute_rope_embeddings
from modelling.layers.init import get_initializers
from parallel import logical_to_physical


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    mlp_weights: Union[MLPWeights, GLUWeights]
    att_norm: Optional[NormWeights]
    mlp_norm: Optional[NormWeights]


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: Optional[jax.Array]
    pos_embed: Optional[jax.Array] = None
    embed_norm: Optional[NormWeights] = None
    lm_head_norm: Optional[NormWeights] = None
    lm_head_bias: Optional[jax.Array] = None


@dataclass
class ModelConfig:
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
    norm_scale: bool  # For RMSNorm: optional. For LayerNorm: always True (ignored)
    norm_bias: bool   # For RMSNorm: always False (ignored). For LayerNorm: optional
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


VALID_NORM_TYPES = ("rms", "layer")
VALID_INIT_STRATEGIES = ("default", "nanochat")
VALID_MLP_TYPES = ("mlp", "glu")
VALID_POSITION_EMBEDDING_TYPES = ("learned", "rope", "none")


def make_config(cfg_dict: dict) -> ModelConfig:

    # Valid value checks
    validate_enum(cfg_dict["norm_type"], "norm_type", VALID_NORM_TYPES)
    validate_enum(cfg_dict["init_strategy"], "init_strategy", VALID_INIT_STRATEGIES)
    validate_enum(cfg_dict["mlp_type"], "mlp_type", VALID_MLP_TYPES)
    validate_enum(cfg_dict["position_embedding_type"], "position_embedding_type", VALID_POSITION_EMBEDDING_TYPES)
    validate_enum(cfg_dict["qk_norm_type"], "qk_norm_type", VALID_NORM_TYPES, allow_none=True)
    
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
    
    return ModelConfig(
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
    )


def forward(
    x: jax.Array,
    weights: ModelWeights,
    config: ModelConfig,
    rope_cos: Optional[jax.Array] = None,
    rope_sin: Optional[jax.Array] = None,
    mask: Optional[jax.Array] = None,
) -> jax.Array:
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

    attention_factory = partial(attention,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm=config.qk_norm,
        qk_norm_type=config.qk_norm_type,
        qk_norm_epsilon=config.qk_norm_epsilon,
        sliding_window=config.sliding_window,
        dtype=config.dtype,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads)
    
    mlp_factory = partial(
        mlp if config.mlp_type == "mlp" else glu,
        act_fn=config.act_fn,
        dtype=config.dtype,
    )
    
    for idx, layer_weights in enumerate(weights.layer_weights):
        with jax.named_scope(f"layer_{idx}"):
            if config.norm_position == "pre":
                residual = x
                with jax.named_scope("att_pre_norm"):
                    x = norm(x, layer_weights.att_norm, eps)
                with jax.named_scope("att"):
                    x = attention_factory(
                        x, layer_weights.attention_weights,
                        mask=mask,
                    )
                with jax.named_scope("add_residual"):
                    x = x + residual
                residual = x
                with jax.named_scope("mlp_pre_norm"):
                    x = norm(x, layer_weights.mlp_norm, eps)
                with jax.named_scope("mlp"):
                    x = mlp_factory(x, layer_weights.mlp_weights)
                with jax.named_scope("add_residual"):
                    x = x + residual
            else:
                residual = x
                with jax.named_scope("att"):
                    x = attention_factory(
                        x, layer_weights.attention_weights,
                        mask=mask,
                    )
                with jax.named_scope("att_post_norm"):
                    x = norm(x, layer_weights.att_norm, eps)
                with jax.named_scope("add_residual"):
                    x = x + residual
                residual = x
                with jax.named_scope("mlp"):
                    x = mlp_factory(x, layer_weights.mlp_weights)
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


def _init_weight(inits, key, init_name: str, shape: tuple, sharding: tuple):
    """Initialize a single weight array."""
    return inits[init_name](key, shape, dtype=jnp.float32, out_sharding=logical_to_physical(sharding))


def _init_norm_weights(norm_type: str, num_features: int, use_scale: bool, use_bias: bool, sharding: tuple):
    """Initialize norm weights based on norm type."""
    # RMSNorm: scale is optional (use_scale), never has bias
    # LayerNorm: always has scale, bias is optional (use_bias)
    if norm_type == "rms":
        scale = jnp.ones((num_features,), dtype=jnp.float32, out_sharding=logical_to_physical(sharding)) if use_scale else None
        return RMSNormWeights(scale=scale)
    else:
        scale = jnp.ones((num_features,), dtype=jnp.float32, out_sharding=logical_to_physical(sharding))
        bias = jnp.zeros((num_features,), dtype=jnp.float32, out_sharding=logical_to_physical(sharding)) if use_bias else None
        return LayerNormWeights(scale=scale, bias=bias)


def _init_attention_weights(config: ModelConfig, inits, keys) -> AttentionWeights:
    """Initialize attention weights for a single layer."""
    D, N, K, H = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim

    q_proj = _init_weight(inits, next(keys), "qkv", (D, N * H), ("model_embed", "model_q"))
    k_proj = _init_weight(inits, next(keys), "qkv", (D, K * H), ("model_embed", "model_kv"))
    v_proj = _init_weight(inits, next(keys), "qkv", (D, K * H), ("model_embed", "model_kv"))
    o_proj = _init_weight(inits, next(keys), "o_proj", (N * H, D), ("model_q", "model_embed"))

    q_bias = k_bias = v_bias = o_bias = None
    if config.attn_use_bias:
        q_bias = _init_weight(inits, next(keys), "bias", (N * H,), ("model_q",))
        k_bias = _init_weight(inits, next(keys), "bias", (K * H,), ("model_kv",))
        v_bias = _init_weight(inits, next(keys), "bias", (K * H,), ("model_kv",))
        o_bias = _init_weight(inits, next(keys), "bias", (D,), ("model_embed",))

    q_norm = k_norm = None
    if config.qk_norm:
        q_norm = _init_norm_weights(config.qk_norm_type, H, config.norm_scale, config.norm_bias, ("head_embed",))
        k_norm = _init_norm_weights(config.qk_norm_type, H, config.norm_scale, config.norm_bias, ("head_embed",))

    return AttentionWeights(
        q_proj=q_proj, k_proj=k_proj, v_proj=v_proj, o_proj=o_proj,
        q_bias=q_bias, k_bias=k_bias, v_bias=v_bias, o_bias=o_bias,
        q_norm=q_norm, k_norm=k_norm,
    )


def _init_mlp_weights(config: ModelConfig, inits, keys) -> Union[MLPWeights, GLUWeights]:
    """Initialize MLP weights for a single layer."""
    D, I = config.hidden_dim, config.intermediate_dim

    if config.mlp_type == "mlp":
        up_proj = _init_weight(inits, next(keys), "mlp_up", (D, I), ("model_embed", "model_intermediate"))
        down_proj = _init_weight(inits, next(keys), "mlp_down", (I, D), ("model_intermediate", "model_embed"))
        up_bias = down_bias = None
        if config.mlp_use_bias:
            up_bias = _init_weight(inits, next(keys), "bias", (I,), ("model_intermediate",))
            down_bias = _init_weight(inits, next(keys), "bias", (D,), ("model_embed",))
        return MLPWeights(up_proj=up_proj, down_proj=down_proj, up_bias=up_bias, down_bias=down_bias)
    else:
        gate_proj = _init_weight(inits, next(keys), "mlp_up", (D, I), ("model_embed", "model_intermediate"))
        up_proj = _init_weight(inits, next(keys), "mlp_up", (D, I), ("model_embed", "model_intermediate"))
        down_proj = _init_weight(inits, next(keys), "mlp_down", (I, D), ("model_intermediate", "model_embed"))
        gate_bias = up_bias = down_bias = None
        if config.mlp_use_bias:
            gate_bias = _init_weight(inits, next(keys), "bias", (I,), ("model_intermediate",))
            up_bias = _init_weight(inits, next(keys), "bias", (I,), ("model_intermediate",))
            down_bias = _init_weight(inits, next(keys), "bias", (D,), ("model_embed",))
        return GLUWeights(
            gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj,
            gate_bias=gate_bias, up_bias=up_bias, down_bias=down_bias,
        )


def _init_layer_weights(config: ModelConfig, inits, keys) -> LayerWeights:
    """Initialize all weights for a single transformer layer."""
    return LayerWeights(
        attention_weights=_init_attention_weights(config, inits, keys),
        mlp_weights=_init_mlp_weights(config, inits, keys),
        att_norm=_init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",)),
        mlp_norm=_init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",)),
    )


def init_model_weights(config: ModelConfig, key: jax.random.PRNGKey) -> ModelWeights:
    inits = get_initializers(config.init_strategy, config.hidden_dim)
    # Worst case: embed + pos_embed + unembed + lm_head_bias + num_layers * (4 qkvo + 4 bias + 3 glu + 3 glu_bias)
    num_keys = 4 + config.num_layers * 14
    keys = iter(jax.random.split(key, num_keys))

    embed = _init_weight(inits, next(keys), "embed", (config.vocab_size, config.hidden_dim), ("model_vocab", "model_embed"))

    pos_embed = None
    if config.position_embedding_type == "learned":
        pos_embed = _init_weight(inits, next(keys), "embed", (config.max_seq_len, config.hidden_dim), ("model_seq", "model_embed"))

    embed_norm = None
    if config.post_embed_norm:
        embed_norm = _init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",))

    layer_weights_list = [_init_layer_weights(config, inits, keys) for _ in range(config.num_layers)]

    lm_head_norm = None
    if config.pre_lm_head_norm:
        lm_head_norm = _init_norm_weights(config.norm_type, config.hidden_dim, config.norm_scale, config.norm_bias, ("model_embed",))

    unembed = None
    lm_head_bias = None
    if not config.tie_word_embeddings:
        unembed = _init_weight(inits, next(keys), "lm_head", (config.hidden_dim, config.vocab_size), ("model_embed", "model_vocab"))
        if config.lm_head_use_bias:
            lm_head_bias = _init_weight(inits, next(keys), "bias", (config.vocab_size,), ("model_vocab",))

    return ModelWeights(
        embed=embed,
        layer_weights=layer_weights_list,
        unembed=unembed,
        pos_embed=pos_embed,
        embed_norm=embed_norm,
        lm_head_norm=lm_head_norm,
        lm_head_bias=lm_head_bias,
    )
