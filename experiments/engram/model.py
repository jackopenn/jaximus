from dataclasses import dataclass
from typing import List, Optional

import jax
from jax import numpy as jnp

from modelling.layers.attention import AttentionWeights
from modelling.layers.mlp import GLUWeights, glu
from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope
from parallel import l2p


def attention(x, weights, rope_cos, rope_sin, eps, num_heads, num_kv_heads):
    # this only works when num_heads = num_kv_heads
    B, L, D = x.shape
    head_dim = weights.q_proj.shape[1] // num_heads
    q = jnp.matmul(x, weights.q_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_q")))
    q = q.reshape(B, L, num_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    k = jnp.matmul(x, weights.k_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    k = k.reshape(B, L, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))
    v = jnp.matmul(x, weights.v_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_kv")))
    v = v.reshape(B, L, num_kv_heads, head_dim, out_sharding=l2p(("batch", "seq", "act_kv", "act_head")))
    q, k = apply_rope(q, rope_cos[:, :L], rope_sin[:, :L]), apply_rope(k, rope_cos[:, :L], rope_sin[:, :L])
    q, k = rms_norm(q, None, eps).astype(jnp.bfloat16), rms_norm(k, None, eps).astype(jnp.bfloat16)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim**-0.5)
    logits = jnp.where(jnp.tril(jnp.ones((L, L), dtype=jnp.bool_)), logits, -1e9)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
    encoded = jnp.einsum("bhqk,bkhd->bqhd", probs, v)
    encoded = jnp.reshape(encoded, (B, L, D), out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    out = jnp.matmul(encoded, weights.o_proj.astype(jnp.bfloat16), out_sharding=l2p(("batch", "seq", "act_embed")))
    return out


@jax.tree_util.register_dataclass
@dataclass
class EngramWeights:
    embedding: jax.Array
    key_proj: jax.Array
    value_proj: jax.Array
    conv_weight: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    attention_weights: AttentionWeights
    glu_weights: GLUWeights
    engram_weights: Optional[EngramWeights] = None


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array


_HASH_CONFIG_CACHE = {}


def _next_prime(n):
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    while not is_prime(n):
        n += 1
    return n


def _generate_distinct_primes(base_size, count, start_offset=0):
    primes, candidate = [], base_size + start_offset
    while len(primes) < count:
        candidate = _next_prime(candidate)
        primes.append(candidate)
        candidate += 1
    return primes


# =================================================================================================
# N-GRAM HASHING EXPLANATION
# =================================================================================================
# Goal: Map n-gram token sequences to embedding indices. Each position looks up embeddings based on
# the current token AND previous tokens (bigrams, trigrams, etc).
#
# Example config: ngram_sizes=[2,3], n_head_per_ngram=2, vocab_size_per_ngram=[10007, 10007]
# This gives 4 heads total: 2 for bigrams + 2 for trigrams
#
# _make_hash_config() returns 3 dicts keyed by layer_id:
#
#   multipliers[layer_id] = [M0, M1, M2]  # random odd ints, one per position in largest n-gram
#       Example: [738492817, 192847561, 847261539]
#
#   prime_vocab_sizes[layer_id] = [[p0, p1], [p2, p3]]  # primes for [bigram_heads, trigram_heads]
#       Example: [[10007, 10009], [10037, 10039]]  # 2 primes per n-gram type
#
#   offsets[layer_id] = [0, 10007, 20016, 30053]  # cumsum of primes, one per head
#       Head 0 (bigram):  indices 0 to 10006
#       Head 1 (bigram):  indices 10007 to 20015
#       Head 2 (trigram): indices 20016 to 30052
#       Head 3 (trigram): indices 30053 to 40091
#
# compute_hash_ids() example with input_ids = [101, 42, 7, 999] at position 3 (token 999):
#
#   For bigram (ngram_size=2):
#       mix = 999 * M[0]  XOR  7 * M[1]
#       hash_head0 = |mix| % 10007  -> e.g. 4521
#       hash_head1 = |mix| % 10009  -> e.g. 4523  (different prime = different index)
#
#   For trigram (ngram_size=3):
#       mix = 999 * M[0]  XOR  7 * M[1]  XOR  42 * M[2]
#       hash_head2 = |mix| % 10037  -> e.g. 8842
#       hash_head3 = |mix| % 10039  -> e.g. 8844
#
#   Output: hash_ids[pos=3] = [4521, 4523, 8842, 8844]  # shape [B, L, 4]
#
# engram_forward() then adds offsets to get final table indices:
#   final_indices = [4521+0, 4523+10007, 8842+20016, 8844+30053] = [4521, 14530, 28858, 38897]
#   embeddings = table[final_indices]  # shape [B, L, 4, embed_dim]
# =================================================================================================
def _make_hash_config(engram_cfg):
    rng = jax.random.PRNGKey(engram_cfg.seed)
    n_ngram_types = len(engram_cfg.ngram_sizes)
    total_heads = n_ngram_types * engram_cfg.n_head_per_ngram
    multipliers, prime_vocab_sizes, offsets = {}, {}, {}

    for layer_idx, layer_id in enumerate(engram_cfg.layer_ids):
        rng, key = jax.random.split(rng)
        r = jax.random.randint(key, (max(engram_cfg.ngram_sizes),), 0, 2**29, dtype=jnp.int32)
        multipliers[layer_id] = r * 2 + 1
        layer_primes = [
            _generate_distinct_primes(
                engram_cfg.vocab_size_per_ngram[i], engram_cfg.n_head_per_ngram,
                start_offset=layer_idx * total_heads * 100 + i * engram_cfg.n_head_per_ngram * 10
            )
            for i in range(n_ngram_types)
        ]
        prime_vocab_sizes[layer_id] = layer_primes
        all_primes = [p for primes in layer_primes for p in primes]
        offsets[layer_id] = jnp.array([0] + list(jnp.cumsum(jnp.array(all_primes[:-1]))), dtype=jnp.int32)

    return multipliers, prime_vocab_sizes, offsets


def compute_hash_ids(input_ids, layer_id, multipliers, prime_vocab_sizes, ngram_sizes):
    def shift_k(k):
        return jnp.pad(input_ids[:, :-k], ((0, 0), (k, 0)), constant_values=0)

    all_hash_ids = []
    for ngram_idx, ngram_size in enumerate(ngram_sizes):
        mix = input_ids * multipliers[layer_id][0]
        for k in range(1, ngram_size):
            mix = mix ^ (shift_k(k) * multipliers[layer_id][k])
        for prime in prime_vocab_sizes[layer_id][ngram_idx]:
            all_hash_ids.append(jnp.abs(mix) % prime)
    return jnp.stack(all_hash_ids, axis=-1)


def engram_forward(hidden_states, input_ids, engram_weights, layer_id, hash_config, engram_cfg, eps):
    B, L, D = hidden_states.shape
    multipliers, prime_vocab_sizes, offsets = hash_config

    hash_ids = compute_hash_ids(input_ids, layer_id, multipliers, prime_vocab_sizes, engram_cfg.ngram_sizes)
    hash_ids = hash_ids + offsets[layer_id][None, None, :]
    embeddings = engram_weights.embedding.at[hash_ids].get(out_sharding=l2p(("batch", "act_seq", None, None)))
    embeddings = embeddings.reshape(B, L, -1).astype(jnp.bfloat16)

    key = jnp.matmul(embeddings, engram_weights.key_proj.astype(jnp.bfloat16))
    gate = jnp.sum(rms_norm(key, None, eps) * rms_norm(hidden_states, None, eps), axis=-1, keepdims=True) / jnp.sqrt(D)
    gate = jax.nn.sigmoid(jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate))

    gated_value = gate * jnp.matmul(embeddings, engram_weights.value_proj.astype(jnp.bfloat16))

    normed_value = rms_norm(gated_value, None, eps).astype(jnp.bfloat16)
    max_ngram = max(engram_cfg.ngram_sizes)
    conv_out = jax.lax.conv_general_dilated(
        normed_value.transpose(0, 2, 1),
        engram_weights.conv_weight.astype(jnp.bfloat16).T[:, None, :],
        window_strides=(1,),
        padding=[((engram_cfg.kernel_size - 1) * max_ngram, 0)],
        rhs_dilation=(max_ngram,),
        feature_group_count=D,
        dimension_numbers=("NCH", "OIH", "NCH")
    ).transpose(0, 2, 1)

    return gated_value + jax.nn.silu(conv_out)


def init_model_weights(config, key):
    def w(key, init_fn, shape, sharding):
        return init_fn(key, shape, dtype=jnp.float32, out_sharding=l2p(sharding))

    if config.engram.enabled:
        _HASH_CONFIG_CACHE["config"] = _make_hash_config(config.engram)
        prime_vocab_sizes = _HASH_CONFIG_CACHE["config"][1]
    else:
        prime_vocab_sizes = None
    num_engram_layers = len(config.engram.layer_ids) if config.engram.enabled else 0
    keys = iter(jax.random.split(key, 2 + config.num_layers * 7 + num_engram_layers * 4))

    D, N, K, H, I = config.hidden_dim, config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.intermediate_dim
    bound = (3**0.5) * (D**-0.5)
    uniform, zeros, normal = jax.nn.initializers.uniform, jax.nn.initializers.zeros, jax.nn.initializers.normal

    layer_weights = []
    for layer_idx in range(config.num_layers):
        engram_weights = None
        if config.engram.enabled and layer_idx in config.engram.layer_ids:
            cfg = config.engram
            engram_hidden = len(cfg.ngram_sizes) * cfg.n_embed_per_ngram
            total_vocab = ((sum(sum(primes) for primes in prime_vocab_sizes[layer_idx]) + 63) // 64) * 64
            engram_weights = EngramWeights(
                embedding=w(next(keys), normal(stddev=0.02), (total_vocab, cfg.n_embed_per_ngram // cfg.n_head_per_ngram), ("model_engram_vocab", "model_engram_embed")),
                key_proj=w(next(keys), uniform(scale=(3**0.5) * (engram_hidden**-0.5)), (engram_hidden, D), ("model_engram_hidden", "model_embed")),
                value_proj=w(next(keys), zeros, (engram_hidden, D), ("model_engram_hidden", "model_embed")),
                conv_weight=w(next(keys), normal(stddev=0.01), (cfg.kernel_size, D), (None, "model_embed")),
            )
        layer_weights.append(LayerWeights(
            attention_weights=AttentionWeights(
                q_proj=w(next(keys), uniform(scale=bound), (D, N * H), ("model_embed", "model_q")),
                k_proj=w(next(keys), uniform(scale=bound), (D, K * H), ("model_embed", "model_kv")),
                v_proj=w(next(keys), uniform(scale=bound), (D, K * H), ("model_embed", "model_kv")),
                o_proj=w(next(keys), zeros, (N * H, D), ("model_q", "model_embed")),
            ),
            glu_weights=GLUWeights(
                gate_proj=w(next(keys), uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")),
                up_proj=w(next(keys), uniform(scale=bound), (D, I), ("model_embed", "model_intermediate")),
                down_proj=w(next(keys), zeros, (I, D), ("model_intermediate", "model_embed")),
            ),
            engram_weights=engram_weights,
        ))

    return ModelWeights(
        embed=w(next(keys), normal(stddev=1.0), (config.vocab_size, D), ("model_vocab", "model_embed")),
        layer_weights=layer_weights,
        unembed=w(next(keys), normal(stddev=0.001), (D, config.vocab_size), ("model_embed", "model_vocab")),
    )


def model_forward(x, weights, config, rope_cos=None, rope_sin=None, mask=None):
    eps = config.norm_epsilon
    engram_enabled = hasattr(config, "engram") and config.engram.enabled
    hash_config = _HASH_CONFIG_CACHE.get("config") if engram_enabled else None

    input_ids = x
    x = rms_norm(weights.embed.at[x].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(jnp.bfloat16), None, eps)

    for layer_idx, layer_weights in enumerate(weights.layer_weights):
        if engram_enabled and layer_weights.engram_weights is not None:
            x = x + engram_forward(x, input_ids, layer_weights.engram_weights, layer_idx, hash_config, config.engram, eps)
        x = x + attention(rms_norm(x, None, eps), layer_weights.attention_weights, rope_cos, rope_sin, eps, config.num_attention_heads, config.num_key_value_heads)
        x = x + glu(rms_norm(x, None, eps), layer_weights.glu_weights, act_fn="silu", dtype="bfloat16")
    
    x = rms_norm(x, None, eps)
    logits = jnp.matmul(x, weights.unembed.astype(jnp.bfloat16), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return logits
