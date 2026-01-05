from functools import partial
import jax
from jax import numpy as jnp
from flax import nnx

from parallel import logical_to_physical


@nnx.jit(static_argnums=(2, 3, 4, 5))
def _sample_batch(model, tokens, prompt_len, gen_len, top_k, temperature, key):
    """
    Autoregressive sampling for a batch. All prompts must have same length.
    
    Args:
        model: nnx model
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
        
        logits = model(tokens)
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


def generate(
    model,
    tokenizer,
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
    """
    if prompts is None:
        prompts = [
            "The meaning of life is",
            "Hello, I'm a language model",
            "The capital of France is",
        ]
    
    n_processes = jax.process_count()
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
    
    # Pad batch to be divisible by process count
    remainder = real_batch_size % n_processes
    pad_samples = (n_processes - remainder) % n_processes
    global_batch_size = real_batch_size + pad_samples
    
    local_batch_size = global_batch_size // n_processes
    
    # Build full token array (all processes build the same thing, then slice)
    all_tokens = []
    for prompt_ids in prompt_token_lists:
        # Pad prompt to max_prompt_len, then pad rest with pad tokens
        padded = prompt_ids + [pad_id] * (max_length - len(prompt_ids))
        for _ in range(n_samples):
            all_tokens.append(padded)
    
    # Pad batch with copies of last sample if needed
    if pad_samples > 0:
        for _ in range(pad_samples):
            all_tokens.append(all_tokens[-1])
    
    all_tokens = jnp.array(all_tokens)  # [global_batch_size, max_length]
    
    # Each process takes its slice
    start_idx = process_idx * local_batch_size
    end_idx = start_idx + local_batch_size
    local_tokens = all_tokens[start_idx:end_idx]
    
    # Create sharded array from local data
    tokens = jax.make_array_from_process_local_data(
        logical_to_physical(("batch", "seq")),
        local_tokens
    )
    
    # Generate with cached model
    key = jax.random.PRNGKey(seed)
    sample_fn = nnx.cached_partial(_sample_batch, model)
    generated = sample_fn(tokens, max_prompt_len, gen_len, top_k, temperature, key)
    
    # Gather all results to host 0
    # Convert sharded array to global array on all hosts
    all_generated = jnp.asarray(generated)
    
    if not main_process:
        return None
    
    # Decode on main process (only real samples, not padding)
    decoded = tokenizer.batch_decode(all_generated[:real_batch_size], skip_special_tokens=False)
    
    # Organize into dict: prompt -> [samples]
    results = {}
    for i, prompt in enumerate(prompts):
        start = i * n_samples
        end = start + n_samples
        results[prompt] = decoded[start:end]
    
    return results


if __name__ == "__main__":
    print("Usage: import generate and call generate(model, tokenizer)")
