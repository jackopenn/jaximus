import math
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import jax
from jax import numpy as jnp
from jax.sharding import reshard

from modelling.layers.norm import rms_norm
from modelling.layers.position import apply_rope, precompute_rope_embeddings
from parallel import l2p


@jax.tree_util.register_dataclass
@dataclass
class MHCWeights:
    w_pre: jax.Array
    w_res: jax.Array
    w_post: jax.Array
    s_pre: jax.Array
    s_res: jax.Array
    s_post: jax.Array
    alpha_pre: jax.Array
    alpha_res: jax.Array
    alpha_post: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class HybridAttentionWeights:
    q_down: jax.Array
    q_up: jax.Array
    hca_kv: jax.Array
    hca_z: jax.Array
    hca_bias: jax.Array
    csa_kv_a: jax.Array
    csa_kv_b: jax.Array
    csa_z_a: jax.Array
    csa_z_b: jax.Array
    csa_bias_a: jax.Array
    csa_bias_b: jax.Array
    csa_idx_a: jax.Array
    csa_idx_b: jax.Array
    csa_idx_z_a: jax.Array
    csa_idx_z_b: jax.Array
    csa_idx_bias_a: jax.Array
    csa_idx_bias_b: jax.Array
    idx_q_up: jax.Array
    idx_w: jax.Array
    local_kv: jax.Array
    group_o: jax.Array
    o_proj: jax.Array
    sink: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class MoEWeights:
    router: jax.Array
    router_bias: jax.Array
    shared_gate: jax.Array
    shared_up: jax.Array
    shared_down: jax.Array
    expert_gate: jax.Array
    expert_up: jax.Array
    expert_down: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class LayerWeights:
    mhc: MHCWeights
    attention: HybridAttentionWeights
    moe: MoEWeights


@jax.tree_util.register_dataclass
@dataclass
class ModelWeights:
    embed: jax.Array
    layer_weights: List[LayerWeights]
    unembed: jax.Array


def _init_weight(key, init_fn, shape, sharding):
    return init_fn(key, shape, dtype=jnp.float32, out_sharding=None if sharding is None else l2p(sharding))


def _constant(shape, value):
    return jnp.full(shape, value, dtype=jnp.float32)


def _logit(x):
    return math.log(x / (1.0 - x))


def _init_mhc_weights(config, keys):
    D, M = config.hidden_dim, config.mhc_width
    flat = D * M
    dyn = jax.nn.initializers.normal(config.mhc_dynamic_std)
    return MHCWeights(
        w_pre=_init_weight(next(keys), dyn, (flat, M), ("model_embed", None)),
        w_res=_init_weight(next(keys), dyn, (flat, M * M), ("model_embed", None)),
        w_post=_init_weight(next(keys), dyn, (flat, M), ("model_embed", None)),
        s_pre=_constant((M,), _logit(1.0 / M)),
        s_res=config.mhc_residual_diag_init * jnp.eye(M, dtype=jnp.float32),
        s_post=jnp.zeros((M,), dtype=jnp.float32),
        alpha_pre=jnp.zeros((), dtype=jnp.float32),
        alpha_res=jnp.zeros((), dtype=jnp.float32),
        alpha_post=jnp.zeros((), dtype=jnp.float32),
    )


def _init_attention_weights(config, keys):
    D, N, H, Q, G, DG = (
        config.hidden_dim,
        config.num_attention_heads,
        config.head_dim,
        config.query_compression_dim,
        config.output_groups,
        config.output_group_dim,
    )
    I_N, I_H = config.indexer_num_heads, config.indexer_head_dim
    csa_m, hca_m = config.csa_compression_rate, config.hca_compression_rate
    init = jax.nn.initializers.uniform(scale=(3**0.5) * (D**-0.5))
    zeros = jax.nn.initializers.zeros
    return HybridAttentionWeights(
        q_down=_init_weight(next(keys), init, (D, Q), ("model_embed", "model_intermediate")),
        q_up=_init_weight(next(keys), init, (Q, N * H), ("model_intermediate", "model_q")),
        hca_kv=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        hca_z=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        hca_bias=_init_weight(next(keys), zeros, (hca_m, H), None),
        csa_kv_a=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        csa_kv_b=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        csa_z_a=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        csa_z_b=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        csa_bias_a=_init_weight(next(keys), zeros, (csa_m, H), None),
        csa_bias_b=_init_weight(next(keys), zeros, (csa_m, H), None),
        csa_idx_a=_init_weight(next(keys), init, (D, I_H), ("model_embed", "model_head")),
        csa_idx_b=_init_weight(next(keys), init, (D, I_H), ("model_embed", "model_head")),
        csa_idx_z_a=_init_weight(next(keys), init, (D, I_H), ("model_embed", "model_head")),
        csa_idx_z_b=_init_weight(next(keys), init, (D, I_H), ("model_embed", "model_head")),
        csa_idx_bias_a=_init_weight(next(keys), zeros, (csa_m, I_H), None),
        csa_idx_bias_b=_init_weight(next(keys), zeros, (csa_m, I_H), None),
        idx_q_up=_init_weight(next(keys), init, (Q, I_N * I_H), ("model_intermediate", "model_q")),
        idx_w=_init_weight(next(keys), init, (D, I_N), ("model_embed", "model_q")),
        local_kv=_init_weight(next(keys), init, (D, H), ("model_embed", "model_head")),
        group_o=_init_weight(next(keys), init, (G, (N // G) * H, DG), None),
        o_proj=_init_weight(next(keys), zeros, (G * DG, D), ("model_intermediate", "model_embed")),
        sink=jnp.zeros((N,), dtype=jnp.float32),
    )


def _init_moe_weights(config, keys):
    D, E, I = config.hidden_dim, config.num_routed_experts, config.expert_intermediate_dim
    init = jax.nn.initializers.uniform(scale=(3**0.5) * (D**-0.5))
    zeros = jax.nn.initializers.zeros
    return MoEWeights(
        router=_init_weight(next(keys), init, (D, E), ("model_embed", "model_expert")),
        router_bias=jnp.zeros((E,), dtype=jnp.float32),
        shared_gate=_init_weight(next(keys), init, (D, I), ("model_embed", "model_intermediate")),
        shared_up=_init_weight(next(keys), init, (D, I), ("model_embed", "model_intermediate")),
        shared_down=_init_weight(next(keys), zeros, (I, D), ("model_intermediate", "model_embed")),
        expert_gate=_init_weight(next(keys), init, (E, D, I), ("model_expert", "model_embed", "model_intermediate")),
        expert_up=_init_weight(next(keys), init, (E, D, I), ("model_expert", "model_embed", "model_intermediate")),
        expert_down=_init_weight(next(keys), zeros, (E, I, D), ("model_expert", "model_intermediate", "model_embed")),
    )


def _init_layer_weights(config, key):
    keys = iter(jax.random.split(key, 64))
    return LayerWeights(
        mhc=_init_mhc_weights(config, keys),
        attention=_init_attention_weights(config, keys),
        moe=_init_moe_weights(config, keys),
    )


def init_model_weights(config, key):
    keys = iter(jax.random.split(key, config.num_layers + 2))
    return ModelWeights(
        embed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=1.0),
            (config.vocab_size, config.hidden_dim),
            ("model_vocab", "model_embed"),
        ),
        layer_weights=[_init_layer_weights(config, next(keys)) for _ in range(config.num_layers)],
        unembed=_init_weight(
            next(keys),
            jax.nn.initializers.normal(stddev=0.001),
            (config.hidden_dim, config.vocab_size),
            ("model_embed", "model_vocab"),
        ),
    )


def _attention_modes(config) -> Tuple[str, ...]:
    modes = []
    for i in range(config.num_layers):
        if config.attention_schedule == "flash" and i < 2:
            modes.append("sliding")
        elif config.attention_schedule == "pro" and i < 2:
            modes.append("hca")
        else:
            modes.append("csa" if (i - 2) % 2 == 0 else "hca")
    return tuple(modes)


def make_model_forward(config, tokenizer=None):
    del tokenizer
    rope_cos, rope_sin = precompute_rope_embeddings(
        config.max_seq_len, config.partial_rope_dim, config.rope_theta, "bfloat16", sharding=l2p(())
    )
    return partial(model_forward, config=config, rope_cos=rope_cos, rope_sin=rope_sin, modes=_attention_modes(config))


def _pad_to_multiple(x, multiple, value):
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x
    return jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2), constant_values=value)


def _valid_mask(mask, S, multiple):
    valid = jnp.ones((1, S), dtype=jnp.bool_) if mask is None else mask.astype(jnp.bool_)
    return _pad_to_multiple(valid, multiple, False)


def _compress_hca(values, logits, bias, rate, mask):
    B, S, H = values.shape
    values = _pad_to_multiple(values, rate, 0.0)
    valid = _valid_mask(mask, S, rate)
    logits = jnp.where(valid[..., None], _pad_to_multiple(logits, rate, -1e30), -1e30)
    chunks = values.reshape(B, values.shape[1] // rate, rate, H)
    weights = jax.nn.softmax(logits.reshape(B, values.shape[1] // rate, rate, H) + bias[None, None, :, :], axis=2)
    return jnp.sum(weights * chunks, axis=2), valid.reshape(valid.shape[0], values.shape[1] // rate, rate).any(axis=2)


def _compress_csa(values_a, values_b, logits_a, logits_b, bias_a, bias_b, rate, mask):
    B, S, H = values_a.shape
    values_a, values_b = _pad_to_multiple(values_a, rate, 0.0), _pad_to_multiple(values_b, rate, 0.0)
    valid = _valid_mask(mask, S, rate)
    logits_a = jnp.where(valid[..., None], _pad_to_multiple(logits_a, rate, -1e30), -1e30)
    logits_b = jnp.where(valid[..., None], _pad_to_multiple(logits_b, rate, -1e30), -1e30)
    blocks = values_a.shape[1] // rate
    a = values_a.reshape(B, blocks, rate, H)
    b = values_b.reshape(B, blocks, rate, H)
    za = logits_a.reshape(B, blocks, rate, H) + bias_a[None, None, :, :]
    zb = logits_b.reshape(B, blocks, rate, H) + bias_b[None, None, :, :]
    prev_b = jnp.concatenate([jnp.zeros_like(b[:, :1]), b[:, :-1]], axis=1)
    prev_zb = jnp.concatenate([jnp.full_like(zb[:, :1], -1e30), zb[:, :-1]], axis=1)
    weights = jax.nn.softmax(jnp.concatenate([za, prev_zb], axis=2), axis=2)
    wa, wb = weights[:, :, :rate], weights[:, :, rate:]
    block_valid = valid.reshape(valid.shape[0], blocks, rate).any(axis=2)
    block_valid = block_valid | jnp.concatenate([jnp.zeros_like(block_valid[:, :1]), block_valid[:, :-1]], axis=1)
    return jnp.sum(wa * a, axis=2) + jnp.sum(wb * prev_b, axis=2), block_valid


def _partial_rope(x, rope_cos, rope_sin, positions=None, inverse=False):
    R = rope_cos.shape[-1] * 2
    if R == 0:
        return x
    prefix, tail = x[..., :-R], x[..., -R:]
    if positions is None:
        cos, sin = rope_cos[:, : x.shape[1], :, :], rope_sin[:, : x.shape[1], :, :]
    else:
        cos, sin = rope_cos[:, positions, :, :], rope_sin[:, positions, :, :]
    if x.ndim == 3:
        cos, sin = cos[:, :, 0, :], sin[:, :, 0, :]
    if inverse:
        sin = -sin
    return jnp.concatenate([prefix, apply_rope(tail, cos, sin)], axis=-1)


def _masked_sink_attention(q, kv, sink, key_mask):
    H = q.shape[-1]
    scores = jnp.einsum("bsnh,bslh->bsnl", q.astype(jnp.float32), kv.astype(jnp.float32)) / math.sqrt(H)
    scores = jnp.where(key_mask[:, :, None, :], scores, -1e30)
    sink = sink.astype(jnp.float32)[None, None, :, None]
    max_score = jnp.maximum(jnp.max(scores, axis=-1, keepdims=True), sink)
    exp_scores = jnp.where(key_mask[:, :, None, :], jnp.exp(scores - max_score), 0.0)
    probs = exp_scores / (jnp.sum(exp_scores, axis=-1, keepdims=True) + jnp.exp(sink - max_score))
    return jnp.einsum("bsnl,bslh->bsnh", probs.astype(kv.dtype), kv)


def _finish_attention(out, weights, config, rope_cos, rope_sin):
    B, S, N, H = out.shape
    G, DG = config.output_groups, config.output_group_dim
    out = _partial_rope(out, rope_cos, rope_sin, inverse=True)
    out = out.reshape(B, S, G, (N // G) * H, out_sharding=l2p(("batch", "seq", None, None)))
    out = jnp.einsum("bsgm,gmd->bsgd", out, weights.group_o.astype(out.dtype))
    out = out.reshape(B, S, G * DG, out_sharding=l2p(("batch", "seq", "act_intermediate")))
    return jnp.matmul(out, weights.o_proj.astype(out.dtype), out_sharding=l2p(("batch", "seq", "act_embed")))


def _local_mask(S, window):
    pos = jnp.arange(S)
    return (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - window)


def _attention(x, weights, config, mode, rope_cos, rope_sin, mask=None):
    dtype = getattr(jnp, config.dtype)
    B, S, _ = x.shape
    N, H = config.num_attention_heads, config.head_dim
    q_latent = jnp.matmul(x, weights.q_down.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
    q = jnp.matmul(q_latent, weights.q_up.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q")))
    q = q.reshape(B, S, N, H, out_sharding=l2p(("batch", "seq", "act_q", "act_head")))
    q = rms_norm(_partial_rope(q, rope_cos, rope_sin), None, config.norm_epsilon).astype(dtype)

    local = jnp.matmul(x, weights.local_kv.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head")))
    local = rms_norm(_partial_rope(local, rope_cos, rope_sin), None, config.norm_epsilon).astype(dtype)
    local_kv = jnp.broadcast_to(local[:, None, :, :], (B, S, S, H))
    local_kv = reshard(local_kv, l2p(("batch", "seq", None, "act_head")))
    local_key_mask = jnp.broadcast_to(_local_mask(S, config.sliding_window)[None, :, :], (B, S, S))
    if mask is not None:
        local_key_mask = local_key_mask & mask[:, None, :].astype(jnp.bool_)
    local_key_mask = reshard(local_key_mask, l2p(("batch", "seq", None)))

    if mode == "sliding":
        return _finish_attention(
            _masked_sink_attention(q, local_kv, weights.sink, local_key_mask), weights, config, rope_cos, rope_sin
        )

    if mode == "hca":
        rate = config.hca_compression_rate
        comp, comp_valid = _compress_hca(
            jnp.matmul(x, weights.hca_kv.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
            jnp.matmul(x, weights.hca_z.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
            weights.hca_bias.astype(dtype),
            rate,
            mask,
        )
        comp_pos = jnp.minimum(jnp.arange(comp.shape[1]) * rate + rate - 1, config.max_seq_len - 1)
        comp = rms_norm(_partial_rope(comp, rope_cos, rope_sin, comp_pos), None, config.norm_epsilon).astype(dtype)
        comp_kv = jnp.broadcast_to(comp[:, None, :, :], (B, S, comp.shape[1], H))
        comp_kv = reshard(comp_kv, l2p(("batch", "seq", None, "act_head")))
        comp_key_mask = jnp.arange(comp.shape[1])[None, :] < (jnp.arange(S)[:, None] // rate)
        comp_key_mask = jnp.broadcast_to(comp_key_mask[None, :, :], (B, S, comp.shape[1])) & comp_valid[:, None, :]
        comp_key_mask = reshard(comp_key_mask, l2p(("batch", "seq", None)))
        kv = jnp.concatenate([comp_kv, local_kv], axis=2)
        key_mask = jnp.concatenate([comp_key_mask, local_key_mask], axis=2)
        return _finish_attention(
            _masked_sink_attention(q, kv, weights.sink, key_mask), weights, config, rope_cos, rope_sin
        )

    rate = config.csa_compression_rate
    comp, comp_valid = _compress_csa(
        jnp.matmul(x, weights.csa_kv_a.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        jnp.matmul(x, weights.csa_kv_b.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        jnp.matmul(x, weights.csa_z_a.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        jnp.matmul(x, weights.csa_z_b.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        weights.csa_bias_a.astype(dtype),
        weights.csa_bias_b.astype(dtype),
        rate,
        mask,
    )
    idx_k, idx_valid = _compress_csa(
        jnp.matmul(x, weights.csa_idx_a.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        jnp.matmul(x, weights.csa_idx_b.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        jnp.matmul(x, weights.csa_idx_z_a.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        jnp.matmul(x, weights.csa_idx_z_b.astype(dtype), out_sharding=l2p(("batch", "seq", "act_head"))),
        weights.csa_idx_bias_a.astype(dtype),
        weights.csa_idx_bias_b.astype(dtype),
        rate,
        mask,
    )
    comp_pos = jnp.minimum(jnp.arange(comp.shape[1]) * rate + rate - 1, config.max_seq_len - 1)
    comp = rms_norm(_partial_rope(comp, rope_cos, rope_sin, comp_pos), None, config.norm_epsilon).astype(dtype)
    idx_k = rms_norm(idx_k, None, config.norm_epsilon).astype(dtype)
    idx_q = jnp.matmul(q_latent, weights.idx_q_up.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q")))
    idx_q = idx_q.reshape(B, S, config.indexer_num_heads, config.indexer_head_dim)
    idx_q = rms_norm(idx_q, None, config.norm_epsilon).astype(dtype)
    idx_w = jnp.matmul(x, weights.idx_w.astype(dtype), out_sharding=l2p(("batch", "seq", "act_q"))).astype(jnp.float32)
    dots = jnp.einsum("bsnh,bkh->bsnk", idx_q.astype(jnp.float32), idx_k.astype(jnp.float32))
    idx_scores = jnp.sum(idx_w[:, :, :, None] * jax.nn.relu(dots), axis=2)
    visible = jnp.arange(comp.shape[1])[None, :] < (jnp.arange(S)[:, None] // rate)
    idx_scores = jnp.where(visible[None, :, :] & comp_valid[:, None, :] & idx_valid[:, None, :], idx_scores, -1e30)
    top_vals, top_idx = jax.lax.top_k(idx_scores, min(config.csa_top_k, comp.shape[1]))
    comp_for_gather = jnp.broadcast_to(comp[:, None, :, :], (B, S, comp.shape[1], H))
    comp_for_gather = reshard(comp_for_gather, l2p(("batch", "seq", None, "act_head")))
    selected = jnp.take_along_axis(comp_for_gather, top_idx[..., None], axis=2)
    selected = reshard(selected, l2p(("batch", "seq", None, "act_head")))
    kv = jnp.concatenate([selected, local_kv], axis=2)
    top_mask = reshard(top_vals > -1e20, l2p(("batch", "seq", None)))
    key_mask = jnp.concatenate([top_mask, local_key_mask], axis=2)
    return _finish_attention(_masked_sink_attention(q, kv, weights.sink, key_mask), weights, config, rope_cos, rope_sin)


def _hash_router(input_ids, layer_idx, num_experts, top_k):
    offsets = jnp.arange(top_k, dtype=jnp.int32) * 8191 + layer_idx * 131071
    return (input_ids.astype(jnp.int32)[..., None] * 65537 + offsets) % num_experts


def _moe(x, input_ids, weights, config, layer_idx):
    dtype = getattr(jnp, config.dtype)
    B, S, D = x.shape
    E, K = config.num_routed_experts, config.num_experts_per_tok
    gate = jnp.matmul(x, weights.shared_gate.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
    up = jnp.matmul(x, weights.shared_up.astype(dtype), out_sharding=l2p(("batch", "seq", "act_intermediate")))
    if config.moe_up_clip:
        up = jnp.clip(up, -config.moe_up_clip, config.moe_up_clip)
    shared = jnp.matmul(
        (jax.nn.silu(gate) * up).astype(dtype),
        weights.shared_down.astype(dtype),
        out_sharding=l2p(("batch", "seq", "act_embed")),
    )

    if layer_idx < config.hash_routing_layers:
        top_idx = _hash_router(input_ids, layer_idx, E, K)
        top_weight = jnp.full((B, S, K), 1.0 / K, dtype=jnp.float32)
    else:
        scores = jnp.sqrt(
            jax.nn.softplus(jnp.matmul(x, weights.router.astype(dtype)).astype(jnp.float32) + weights.router_bias)
        )
        top_score, top_idx = jax.lax.top_k(scores, K)
        top_weight = top_score / jnp.maximum(jnp.sum(top_score, axis=-1, keepdims=True), 1e-9)

    expert_gate = (
        weights.expert_gate.astype(dtype)
        .at[top_idx]
        .get(out_sharding=l2p(("batch", "seq", None, "act_embed", "act_intermediate")))
    )
    expert_up = (
        weights.expert_up.astype(dtype)
        .at[top_idx]
        .get(out_sharding=l2p(("batch", "seq", None, "act_embed", "act_intermediate")))
    )
    expert_down = (
        weights.expert_down.astype(dtype)
        .at[top_idx]
        .get(out_sharding=l2p(("batch", "seq", None, "act_intermediate", "act_embed")))
    )
    gate = jnp.einsum("bsd,bskdi->bski", x, expert_gate)
    up = jnp.einsum("bsd,bskdi->bski", x, expert_up)
    if config.moe_up_clip:
        up = jnp.clip(up, -config.moe_up_clip, config.moe_up_clip)
    routed = jnp.einsum("bski,bskid->bskd", (jax.nn.silu(gate) * up).astype(dtype), expert_down)
    routed = jnp.sum(top_weight[..., None].astype(dtype) * routed, axis=2)
    return shared + routed


def _sinkhorn(logits, steps):
    x = jnp.exp(logits - jnp.max(logits, axis=(-2, -1), keepdims=True))
    for _ in range(steps):
        x = x / jnp.maximum(jnp.sum(x, axis=-2, keepdims=True), 1e-9)
        x = x / jnp.maximum(jnp.sum(x, axis=-1, keepdims=True), 1e-9)
    return x


def _mhc_input(state, weights, config):
    B, S, M, D = state.shape
    flat = rms_norm(state.reshape(B, S, M * D), None, config.norm_epsilon)
    a = jax.nn.sigmoid(weights.s_pre + weights.alpha_pre * jnp.matmul(flat, weights.w_pre))
    b_raw = weights.s_res + weights.alpha_res * jnp.matmul(flat, weights.w_res).reshape(B, S, M, M)
    c = 2.0 * jax.nn.sigmoid(weights.s_post + weights.alpha_post * jnp.matmul(flat, weights.w_post))
    return jnp.einsum("bsm,bsmd->bsd", a, state), _sinkhorn(b_raw, config.mhc_sinkhorn_steps), c


def _transformer_block(x, input_ids, layer_weights, config, mode, rope_cos, rope_sin, layer_idx, mask):
    residual = x
    x = rms_norm(x, None, config.norm_epsilon)
    x = _attention(x, layer_weights.attention, config, mode, rope_cos, rope_sin, mask)
    x = x + residual

    residual = x
    x = rms_norm(x, None, config.norm_epsilon)
    x = _moe(x, input_ids, layer_weights.moe, config, layer_idx)
    return x + residual


def model_forward(input_ids, weights, config, rope_cos, rope_sin, modes, mask=None):
    dtype = getattr(jnp, config.dtype)
    x = weights.embed.at[input_ids].get(out_sharding=l2p(("batch", "act_seq", "act_embed"))).astype(dtype)
    x = rms_norm(x, None, config.norm_epsilon)
    state = jnp.broadcast_to(x[:, :, None, :], (x.shape[0], x.shape[1], config.mhc_width, config.hidden_dim))

    for i, layer_weights in enumerate(weights.layer_weights):
        x, b, c = _mhc_input(state, layer_weights.mhc, config)
        x = _transformer_block(x, input_ids, layer_weights, config, modes[i], rope_cos, rope_sin, i, mask)
        state = jnp.einsum("bsmn,bsnd->bsmd", b, state) + c[:, :, :, None].astype(dtype) * x[:, :, None, :]

    x = rms_norm(jnp.mean(state, axis=2), None, config.norm_epsilon)
    logits = jnp.matmul(x, weights.unembed.astype(dtype), out_sharding=l2p(("batch", "act_seq", "act_vocab")))
    return config.logit_softcap * jnp.tanh(logits.astype(jnp.float32) / config.logit_softcap)
