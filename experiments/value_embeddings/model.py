from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights, make_attention_mask
from modelling.layers.mlp import MLPWeights, mlp
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class ValueResidualWeights:
    lambda_v: jax.Array  # Scalar per layer, init to 0.5


@jax.tree_util.register_dataclass
@dataclass
class ValueEmbeddingGateWeights:
    gate_proj: jax.Array  # (gate_channels, num_kv_heads), zero-init


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    mlp_weights: MLPWeights
    value_residual_weights: Optional[ValueResidualWeights] = None
    value_embedding_gate: Optional[ValueEmbeddingGateWeights] = None
    value_embedding_lambda: Optional[jax.Array] = None  # (num_kv_heads,), init 0.5


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array
    value_embeddings: Optional[jax.Array] = None  # (vocab_size, K*H)


def _init_weight(key, init_fn, shape, sharding):
    return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))


def _should_use_value_residuals(config, layer_idx):
    if config.value_residuals is None:
        return False
    if layer_idx == 0:
        return False  # First layer defines v1
    if layer_idx == config.num_layers - 1:
        return True  # Last layer always uses it
    return (layer_idx % config.value_residuals_every_layer) == 0


def _should_use_value_embeddings(config, layer_idx):
    if not config.value_embeddings:
        return False
    if layer_idx == config.num_layers - 1:
        return True  # Last layer always uses it
    return (layer_idx % config.value_embeddings_every_layer) == 0


def _init_attention_weights(config, keys):
    D, N, K, H = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim
    bound = (3**0.5) * (D**-0.5)
    qkv_init, zero_init = jax.nn.initializers.uniform(scale=bound), jax.nn.initializers.zeros
    return AttentionWeights(
        q_proj=_init_weight(next(keys), qkv_init, (D, N * H), ("model_embed", "model_q")),
        k_proj=_init_weight(next(keys), qkv_init, (D, K * H), ("model_embed", "model_kv")),
        v_proj=_init_weight(next(keys), qkv_init, (D, K * H), ("model_embed", "model_kv")),
        o_proj=_init_weight(next(keys), zero_init, (N * H, D), ("model_q", "model_embed")),
    )


def _init_mlp_weights(config, keys):
    D, I = config.hidden_dim, config.intermediate_dim
    bound = (3**0.5) * (D**-0.5)
    return MLPWeights(
        up_proj=_init_weight(
            next(keys), jax.nn.initializers.uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")
        ),
        down_proj=_init_weight(next(keys), jax.nn.initializers.zeros, (I, D), ("model_intermediate", "model_embed")),
    )


def _init_value_residual_weights(config, layer_idx):
    if not _should_use_value_residuals(config, layer_idx):
        return None
    if config.value_residuals != "learnt":
        return None
    return ValueResidualWeights(lambda_v=jnp.array(0.5, dtype=jnp.float32))


def _init_value_embedding_gate(config, layer_idx):
    if not _should_use_value_embeddings(config, layer_idx):
        return None
    if not config.value_embeddings_use_gate:
        return None
    return ValueEmbeddingGateWeights(
        gate_proj=jnp.zeros((config.value_embeddings_gate_channels, config.num_key_value_heads), dtype=jnp.float32)
    )


def _init_value_embedding_lambda(config, layer_idx):
    if not _should_use_value_embeddings(config, layer_idx):
        return None
    if config.value_embeddings_use_gate:
        return None
    return jnp.full((config.num_key_value_heads,), 0.5, dtype=jnp.float32)


def _init_layer_weights(config, layer_idx, keys):
    return LayerWeights(
        attention_weights=_init_attention_weights(config, keys),
        mlp_weights=_init_mlp_weights(config, keys),
        value_residual_weights=_init_value_residual_weights(config, layer_idx),
        value_embedding_gate=_init_value_embedding_gate(config, layer_idx),
        value_embedding_lambda=_init_value_embedding_lambda(config, layer_idx),
    )


def _init_value_embeddings(config, key):
    if not config.value_embeddings:
        return None
    K, H = config.num_key_value_heads, config.head_dim
    return _init_weight(
        key, jax.nn.initializers.normal(stddev=1.0), (config.vocab_size, K * H), ("model_vocab", "model_kv")
    )


def init_model_weights(config, key):
    keys = iter(jax.random.split(key, 3 + config.num_layers * 6))
    return ModelWeights(
        embed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=1.0),
            (config.vocab_size, config.hidden_dim),
            ("model_vocab", "model_embed"),
        ),
        layer_weights=[_init_layer_weights(config, i, keys) for i in range(config.num_layers)],
        unembed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=0.001),
            (config.hidden_dim, config.vocab_size),
            ("model_embed", "model_vocab"),
        ),
        value_embeddings=_init_value_embeddings(config, next(keys)),
    )


def _attention_with_v(
    x: jax.Array,
    weights: AttentionWeights,
    rope_cos: jax.Array,
    rope_sin: jax.Array,
    qk_norm_epsilon: float,
    num_heads: int,
    num_kv_heads: int,
    v1: Optional[jax.Array],
    value_residual_weights: Optional[ValueResidualWeights],
    value_residuals_mode: Optional[str],
    input_ids: Optional[jax.Array],
    value_embeddings: Optional[jax.Array],
    value_embedding_gate: Optional[ValueEmbeddingGateWeights],
    value_embedding_lambda: Optional[jax.Array],
    mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Attention that returns both output and v (for v1 capture). Supports value residual and value embedding mixing."""
    dtype = jnp.bfloat16
    batch, seq_len, _ = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads

    with jax.named_scope("q_proj"):
        q = jnp.matmul(x, weights.q_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q")))
        q = q.reshape(batch, seq_len, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))

    with jax.named_scope("k_proj"):
        k = jnp.matmul(x, weights.k_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
        k = k.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    with jax.named_scope("v_proj"):
        v = jnp.matmul(x, weights.v_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_kv")))
        v = v.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    v_out = v  # Capture v before mixing for v1

    if v1 is not None and value_residuals_mode is not None:
        with jax.named_scope("value_residual"):
            if value_residuals_mode == "fixed":
                v = 0.5 * v + 0.5 * v1
            elif value_residuals_mode == "learnt":
                lam = value_residual_weights.lambda_v.astype(dtype)
                v = lam * v + (1.0 - lam) * v1

    if value_embeddings is not None and input_ids is not None:
        with jax.named_scope("value_embedding"):
            ve = value_embeddings.at[input_ids].get(out_sharding=l2p(("batch", "act_seq", "act_kv")))
            ve = ve.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))
            ve = ve.astype(dtype)
            if value_embedding_gate is not None:
                gate_input = x[:, :, :value_embedding_gate.gate_proj.shape[0]]
                gate = jnp.matmul(gate_input, value_embedding_gate.gate_proj.astype(dtype))  # (B, T, K)
                gate = 2.0 * jax.nn.sigmoid(gate)  # Range (0, 2), init at 1.0
                v = v + gate[:, :, :, None] * ve
            else:
                lam = value_embedding_lambda.astype(dtype)  # (K,)
                v = (1.0 - lam)[None, None, :, None] * v + lam[None, None, :, None] * ve

    if rope_cos is not None and rope_sin is not None:
        with jax.named_scope("apply_rope"):
            cos = rope_cos[:, :seq_len, :, :]
            sin = rope_sin[:, :seq_len, :, :]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

    with jax.named_scope("q_norm"):
        q = rms_norm(q, None, qk_norm_epsilon).astype(dtype)
    with jax.named_scope("k_norm"):
        k = rms_norm(k, None, qk_norm_epsilon).astype(dtype)

    if mask is not None:
        mask = make_attention_mask(mask)

    with jax.named_scope("dot_product_attention"):
        att = jax.nn.dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=True,
            implementation="cudnn" if jax.default_backend() == "gpu" else "xla",
            mask=mask,
        )

    with jax.named_scope("o_proj"):
        att = att.reshape(batch, seq_len, num_heads * head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
        out = jnp.matmul(att, weights.o_proj.astype(dtype), out_sharding=l2p(("batch", "seq", "act_embed")))

    return out, v_out


def model_forward(input_ids, weights, config, rope_cos=None, rope_sin=None, mask=None):
    """Forward pass: pre-norm RMSNorm, RoPE, QK norm, relu_squared MLP, softcap=15.0, bfloat16."""
    eps = config.norm_epsilon

    x = weights.embed.at[input_ids].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)
    x = rms_norm(x, None, eps)

    mlp_fn = partial(mlp, act_fn="relu_squared", dtype="bfloat16")

    v1 = None
    for layer_idx, layer_weights in enumerate(weights.layer_weights):
        use_value_res = _should_use_value_residuals(config, layer_idx)
        value_res_mode = config.value_residuals if use_value_res else None
        use_value_emb = _should_use_value_embeddings(config, layer_idx)

        residual = x
        x = rms_norm(x, None, eps)
        x, v = _attention_with_v(
            x,
            layer_weights.attention_weights,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            qk_norm_epsilon=eps,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            v1=v1,
            value_residual_weights=layer_weights.value_residual_weights,
            value_residuals_mode=value_res_mode,
            input_ids=input_ids if use_value_emb else None,
            value_embeddings=weights.value_embeddings if use_value_emb else None,
            value_embedding_gate=layer_weights.value_embedding_gate,
            value_embedding_lambda=layer_weights.value_embedding_lambda,
            mask=mask,
        )
        x = x + residual

        if layer_idx == 0:
            v1 = v

        residual = x
        x = rms_norm(x, None, eps)
        x = mlp_fn(x, layer_weights.mlp_weights)
        x = x + residual

    x = rms_norm(x, None, eps)
    logits = jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)
