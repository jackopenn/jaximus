from functools import partial
from typing import Callable
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P, reshard
import numpy as np

from parallel import logical_to_physical
from modelling.layers.position import precompute_rope_embeddings


def _make_sample_batch(model_config, rope_cos, rope_sin, forward_fn: Callable):
    """Create a JIT-compiled sampling function for the given config."""
    
    @partial(jax.jit, static_argnums=(2, 3, 4, 5))
    def _sample_batch(weights, tokens, prompt_len, gen_len, top_k, temperature, key):
        """
        Autoregressive sampling for a batch. All prompts must have same length.
        
        Args:
            weights: ModelWeights
            tokens: [batch, seq_len] token ids
            prompt_len: static int, length of all prompts
            gen_len: static int, tokens to generate
            top_k: static int
            temperature: static float
            key: PRNGKey
        
        Returns:
            tokens: [batch, seq_len] with generated tokens
        """
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
    forward_fn: Callable,
    prompts: list[str] = None,
    max_length: int = 64,
    n_samples: int = 5,
    top_k: int = 10,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, list[str]] | None:
    """
    Generate text samples. Batches all prompts and shards across devices like training.

    For multi-host: ALL processes must call this. Returns results only on main process.

    Note: All prompts are padded to same length for efficient batched generation.

    Args:
        weights: Model weights
        model_config: Model config
        tokenizer: HuggingFace tokenizer
        forward_fn: Forward function for the model
        prompts: List of prompt strings (uses defaults if None)
        max_length: Maximum total sequence length
        n_samples: Number of samples per prompt
        top_k: Top-k sampling parameter
        temperature: Sampling temperature
        seed: Random seed
    
    Returns:
        Dict mapping prompts to lists of generated samples (None on non-main processes)
    """
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
    
    n_devices = jax.device_count()  # total devices across all hosts
    process_idx = jax.process_index()
    main_process = process_idx == 0
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    
    # Tokenize all prompts
    prompt_token_lists = []
    for prompt in prompts:
        ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_lists.append(ids)
    
    # Find max prompt length, pad all to same length
    max_prompt_len = max(len(p) for p in prompt_token_lists)
    gen_len = max_length - max_prompt_len
    
    if gen_len <= 0:
        raise ValueError(f"max_length ({max_length}) must be > longest prompt ({max_prompt_len})")
    
    # Build global batch: [n_prompts * n_samples, max_length]
    # Each prompt repeated n_samples times
    real_batch_size = len(prompts) * n_samples
    
    # Pad batch to be divisible by total device count (for sharding)
    remainder = real_batch_size % n_devices
    pad_samples = (n_devices - remainder) % n_devices
    global_batch_size = real_batch_size + pad_samples
    
    local_batch_size = global_batch_size // jax.process_count()
    
    # Build full token array (all processes build the same thing, then slice)
    # Left-pad prompts so they all end at max_prompt_len, then add space for generation
    all_tokens = []
    for prompt_ids in prompt_token_lists:
        left_pad = [pad_id] * (max_prompt_len - len(prompt_ids))
        gen_space = [pad_id] * gen_len
        padded = left_pad + prompt_ids + gen_space
        for _ in range(n_samples):
            all_tokens.append(padded)
    
    # Pad batch with copies of last sample if needed
    if pad_samples > 0:
        for _ in range(pad_samples):
            all_tokens.append(all_tokens[-1])
    
    # Use numpy array (not jax) so it stays local to this process
    all_tokens = np.array(all_tokens, dtype=np.int32)  # [global_batch_size, max_length]
    
    # Each process takes its slice
    start_idx = process_idx * local_batch_size
    end_idx = start_idx + local_batch_size
    local_tokens = all_tokens[start_idx:end_idx]
    local_indices = np.arange(start_idx, end_idx, dtype=np.int32)
    
    # Create sharded JAX arrays from local numpy data
    tokens = jax.make_array_from_process_local_data(
        logical_to_physical(("batch", "seq")),
        local_tokens
    )
    indices = jax.make_array_from_process_local_data(
        logical_to_physical(("batch",)),
        local_indices
    )
    
    rope_cos, rope_sin = None, None
    if model_config.position_embedding_type == "rope":
        rope_cos, rope_sin = precompute_rope_embeddings(
            model_config.max_seq_len, model_config.head_dim, model_config.rope_theta, model_config.dtype
        )
        rope_cos = reshard(rope_cos, P())
        rope_sin = reshard(rope_sin, P())

    # Create and call sampling function
    key = jax.random.PRNGKey(seed)
    sample_fn = _make_sample_batch(model_config, rope_cos, rope_sin, forward_fn)
    generated = sample_fn(weights, tokens, max_prompt_len, gen_len, top_k, temperature, key)
    
    # Gather from all hosts
    all_generated = jax.experimental.multihost_utils.process_allgather(generated, tiled=True)
    all_indices = jax.experimental.multihost_utils.process_allgather(indices, tiled=True)
    
    if not main_process:
        return None
    
    # Reorder by original global indices and slice off padding
    sort_order = np.argsort(all_indices)
    all_generated = all_generated[sort_order][:real_batch_size]
    
    decoded = tokenizer.batch_decode(all_generated, skip_special_tokens=False)
    
    # Organize into dict: prompt -> [samples]
    results = {}
    for i, prompt in enumerate(prompts):
        start = i * n_samples
        end = start + n_samples
        results[prompt] = decoded[start:end]
    
    return results


if __name__ == "__main__":
    print("Usage: import generate and call generate(weights, model_config, tokenizer)")
