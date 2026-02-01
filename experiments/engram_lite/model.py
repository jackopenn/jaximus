from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import jax
import numpy as np
from jax import numpy as jnp
from tokenizers import Regex, normalizers

from modelling.layers.attention import AttentionWeights, attention, make_attention_mask
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope, precompute_rope_embeddings
from parallel import l2p


class CompressedTokenizer:
    """Builds lookup table mapping raw token IDs to normalized canonical IDs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        SENTINEL = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )
        self.lookup_table, self.num_new_tokens = self._build_lookup_table()

    def _build_lookup_table(self):
        old2new, key2new, new_tokens = {}, {}, []
        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)


@jax.tree_util.register_dataclass
@dataclass
class EngramWeights:
    embeddings: jax.Array  # (vocab_size * table_multiplier, embed_dim)
    lambdas: jax.Array  # (num_layers,) - per-layer scalar scaling


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
    engram: Optional[EngramWeights] = None


def _init_engram_weights(config, key):
    """Initialize engram-lite weights."""
    table_size = config.vocab_size * config.engram.table_multiplier
    L = config.num_layers
    injection = getattr(config.engram, "injection", "residual")
    embed_dim = config.head_dim if injection == "attention" else config.hidden_dim

    def w(init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    zeros = jax.nn.initializers.zeros
    lambda_init = jax.nn.initializers.constant(config.engram.lambda_init)

    return EngramWeights(
        embeddings=w(zeros, (table_size, embed_dim), ("model_engram_vocab", "model_embed")),
        lambdas=w(lambda_init, (L,), ()),
    )


def init_model_weights(config, key):
    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    keys = iter(jax.random.split(key, 3 + config.num_layers * 7 + 1))

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
        engram_weights = _init_engram_weights(config, next(keys))

    return ModelWeights(
        embed=w(next(keys), normal, (V, D), ("model_vocab", "model_embed")),
        layer_weights=layer_weights,
        unembed=w(next(keys), normal, (D, V), ("model_embed", "model_vocab")),
        engram=engram_weights,
    )


def bigram_hash(input_ids, table_size, compression_map=None):
    """Simple bigram hash: (36313*curr) ^ (27191*prev) % (table_size - 1)."""
    if compression_map is not None:
        input_ids = compression_map.at[input_ids].get(out_sharding=l2p(("batch", "seq")))
    curr = input_ids[:, 1:]
    prev = input_ids[:, :-1]
    h = (36313 * curr) ^ (27191 * prev)
    return h % (table_size - 1)


def make_model_forward(config, tokenizer=None):
    """Factory that returns forward function with precomputed rope embeddings."""
    rope_cos, rope_sin = precompute_rope_embeddings(
        config.max_seq_len, config.head_dim, config.rope_theta, "bfloat16", sharding=l2p(())
    )

    if getattr(config, "engram", None) and config.engram.enabled:
        table_size = config.vocab_size * config.engram.table_multiplier
        compression_map = None
        if tokenizer and getattr(config.engram, "token_compression", False):
            compression_map = jnp.array(CompressedTokenizer(tokenizer).lookup_table)
        injection = getattr(config.engram, "injection", "residual")
        if injection == "attention":
            return partial(
                _model_forward_with_engram_attention,
                config=config,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                table_size=table_size,
                compression_map=compression_map,
            )
        return partial(
            _model_forward_with_engram_lite,
            config=config,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            table_size=table_size,
            compression_map=compression_map,
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


def _model_forward_with_engram_lite(
    x, weights, config, rope_cos, rope_sin, table_size, compression_map=None, mask=None
):
    """Forward pass with engram-lite: simple bigram hash + per-layer lambdas."""
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

    # Compute bigram embeddings once (pad first position with zeros)
    hash_indices = bigram_hash(input_ids, table_size, compression_map)
    hash_indices = jnp.pad(hash_indices, ((0, 0), (1, 0)), constant_values=0)
    bigram_embed = (
        weights.engram.embeddings.at[hash_indices]
        .get(out_sharding=l2p(("batch", "act_seq", "act_embed")))
        .astype(jnp.bfloat16)
    )

    x = norm_fn(x)

    for layer_idx, layer_weights in enumerate(weights.layer_weights):
        lam = weights.engram.lambdas[layer_idx].astype(jnp.bfloat16)
        x = x + lam * bigram_embed

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


def _attention_with_engram(
    x, weights, engram_embed, rope_cos, rope_sin, qk_norm_epsilon, num_heads, num_kv_heads, mask=None
):
    """Attention with engram injected into values, gated by per-head prev-token attention."""
    batch, seq_len, _ = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads
    assert engram_embed.shape[-1] == head_dim, f"engram dim {engram_embed.shape[-1]} != head_dim {head_dim}"

    q = jnp.matmul(x, weights.q_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_q")))
    q = q.reshape(batch, seq_len, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))

    k = jnp.matmul(x, weights.k_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    k = k.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    v = jnp.matmul(x, weights.v_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    v = v.reshape(batch, seq_len, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))

    cos, sin = rope_cos[:, :seq_len, :, :], rope_sin[:, :seq_len, :, :]
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    q = rms_norm(q, weights=None, eps=qk_norm_epsilon)
    k = rms_norm(k, weights=None, eps=qk_norm_epsilon)

    if num_kv_heads != num_heads:
        k = jnp.repeat(k, num_heads // num_kv_heads, axis=2)
        v = jnp.repeat(v, num_heads // num_kv_heads, axis=2)

    scores = jnp.einsum("bshd,bthd->bhst", q, k) / jnp.sqrt(head_dim).astype(jnp.bfloat16)
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    scores = jnp.where(causal_mask, scores, -1e9)
    if mask is not None:
        scores = jnp.where(make_attention_mask(mask), scores, -1e9)
    attn_weights = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)

    # gate[b, h, i] = attn_weights[b, h, i, i-1] (attention to previous token)
    seq_idx = jnp.arange(seq_len)
    prev_attn = attn_weights[:, :, seq_idx[1:], seq_idx[:-1]]  # (batch, heads, seq-1)
    prev_attn = jnp.pad(prev_attn, ((0, 0), (0, 0), (1, 0)))  # (batch, heads, seq)
    gate = prev_attn.transpose(0, 2, 1)[:, :, :, None]  # (batch, seq, heads, 1)

    # engram_embed: (batch, seq, head_dim) -> broadcast to (batch, seq, heads, head_dim)
    v = v + gate * engram_embed[:, :, None, :]

    att = jnp.einsum("bhst,bthd->bshd", attn_weights, v)
    att = att.reshape(batch, seq_len, num_heads * head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    return jnp.matmul(att, weights.o_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_embed")))


def _model_forward_with_engram_attention(
    x, weights, config, rope_cos, rope_sin, table_size, compression_map=None, mask=None
):
    """Forward pass with engram injected into attention values, gated by per-head prev-token attention."""
    norm_fn = partial(rms_norm, weights=None, eps=config.norm_epsilon)
    attention_engram_fn = partial(
        _attention_with_engram,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        qk_norm_epsilon=config.norm_epsilon,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )
    mlp_fn = partial(glu, act_fn="silu", dtype="bfloat16")

    input_ids = x
    x = weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16)

    # Compute bigram embeddings once (pad first position with zeros)
    hash_indices = bigram_hash(input_ids, table_size, compression_map)
    hash_indices = jnp.pad(hash_indices, ((0, 0), (1, 0)), constant_values=0)
    bigram_embed = (
        weights.engram.embeddings.at[hash_indices]
        .get(out_sharding=l2p(("batch", "act_seq", "act_embed")))
        .astype(jnp.bfloat16)
    )

    x = norm_fn(x)

    for layer_weights in weights.layer_weights:
        residual = x
        x = norm_fn(x)
        x = attention_engram_fn(x, layer_weights.attention_weights, engram_embed=bigram_embed, mask=mask)
        x = x + residual

        residual = x
        x = norm_fn(x)
        x = mlp_fn(x, layer_weights.glu_weights)
        x = x + residual

    x = norm_fn(x)
    return jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
