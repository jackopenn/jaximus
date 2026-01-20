# pyright: reportAttributeAccessIssue=false
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from modelling.layers.position import precompute_rope_embeddings
from parallel import l2p


def _make_sample_batch(model_config, rope_cos, rope_sin, forward_fn):
    @partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def _sample_batch(weights, tokens, prompt_len, gen_len, top_k, temperature, key):
        def body_fn(i, carry):
            tokens, pos, key = carry
            key, subkey = jax.random.split(key)

            logits = forward_fn(tokens, weights, model_config, rope_cos=rope_cos, rope_sin=rope_sin)
            logits = logits[:, pos, :].astype(jnp.float32) / temperature

            # Top-k filtering
            top_values, _ = jax.lax.top_k(logits, k=top_k)
            threshold = top_values[:, -1:]
            logits = jnp.where(logits < threshold, -jnp.inf, logits)

            # Sample
            next_token = jax.random.categorical(subkey, logits)

            # Update
            pos = pos + 1
            tokens = tokens.at[:, pos].set(next_token)

            return tokens, pos, key

        pos = prompt_len - 1
        tokens, _, _ = jax.lax.fori_loop(0, gen_len, body_fn, (tokens, pos, key))
        return tokens

    return _sample_batch


def generate(
    weights,
    model_config,
    tokenizer,
    forward_fn,
    prompts=None,
    max_length=64,
    n_samples=5,
    top_k=10,
    temperature=1.0,
    seed=0,
):
    """Generate text samples. For multi-host: ALL processes must call this, returns results only on main."""
    if prompts is None:
        prompts = [
            "The meaning of life is",
            "Hello, I'm a language model,",
            "5+7=",
            "five plus seven is",
            "The capital of France is",
            "The answer to the ultimate question of life, the universe, and everything is",
            "Once upon a time, there was a",
        ]

    n_devices = jax.device_count()
    main_process = jax.process_index() == 0

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    prompt_token_lists = [[tokenizer.bos_token_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_prompt_len = max(len(p) for p in prompt_token_lists)
    gen_len = max_length - max_prompt_len
    if gen_len <= 0:
        raise ValueError(f"max_length ({max_length}) must be > longest prompt ({max_prompt_len})")

    real_batch_size = len(prompts) * n_samples
    pad_samples = (n_devices - real_batch_size % n_devices) % n_devices
    global_batch_size = real_batch_size + pad_samples
    local_batch_size = global_batch_size // jax.process_count()

    # left-pad prompts to max_prompt_len, add generation space, repeat n_samples times
    all_tokens = []
    for prompt_ids in prompt_token_lists:
        padded = (
            [tokenizer.pad_token_id] * (max_prompt_len - len(prompt_ids))
            + prompt_ids
            + [tokenizer.pad_token_id] * gen_len
        )
        all_tokens.extend([padded] * n_samples)
    all_tokens.extend([all_tokens[-1]] * pad_samples)
    all_tokens = np.array(all_tokens, dtype=np.int32)

    start_idx = jax.process_index() * local_batch_size
    local_tokens = all_tokens[start_idx : start_idx + local_batch_size]
    local_indices = np.arange(start_idx, start_idx + local_batch_size, dtype=np.int32)

    tokens = jax.make_array_from_process_local_data(l2p(("batch", "seq")), local_tokens)
    indices = jax.make_array_from_process_local_data(l2p(("batch",)), local_indices)

    rope_cos, rope_sin = None, None
    if getattr(model_config, "position_embedding_type", None) == "rope" or hasattr(model_config, "rope_theta"):
        rope_cos, rope_sin = precompute_rope_embeddings(
            model_config.max_seq_len,
            model_config.head_dim,
            model_config.rope_theta,
            getattr(model_config, "dtype", "bfloat16"),
        )
        rope_cos, rope_sin = reshard(rope_cos, P()), reshard(rope_sin, P())

    sample_fn = _make_sample_batch(model_config, rope_cos, rope_sin, forward_fn)
    generated = sample_fn(weights, tokens, max_prompt_len, gen_len, top_k, temperature, jax.random.PRNGKey(seed))

    all_generated = jax.experimental.multihost_utils.process_allgather(generated, tiled=True)
    all_indices = jax.experimental.multihost_utils.process_allgather(indices, tiled=True)
    if not main_process:
        return None

    all_generated = all_generated[np.argsort(all_indices)][:real_batch_size]
    decoded = tokenizer.batch_decode(all_generated, skip_special_tokens=False)
    return {prompt: decoded[i * n_samples : (i + 1) * n_samples] for i, prompt in enumerate(prompts)}


if __name__ == "__main__":
    print("Usage: import generate and call generate(weights, model_config, tokenizer)")
