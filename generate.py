from typing import List
from functools import partial
import jax
from jax import numpy as jnp
from flax import nnx
import chz
from utils.configs import ModelConfig
# from modelling.models.gpt import GPT, GPTConfig
# from transformers import AutoTokenizer
import wandb
import orbax.checkpoint as ocp

@chz.chz
class GenerateConfig:
    model: ModelConfig
    weights_path: str
    prompts: List[str]
    max_length: int
    n_samples: int
    top_k: int
    temperature: float
    tokenizer_name: str

@partial(
    jax.jit,
    static_argnames=("sample_length", "top_k", "temperature"),
    donate_argnums=(1,)
)
def sample(model, batch, mask, sample_length, top_k, temperature, key=jax.random.PRNGKey(0)):

    def body_fn(i, carry):
        batch, mask, key = carry
        key, subkey = jax.random.split(key)
    
        logits = model(batch, mask)
        logits = logits[:, -1, :].astype(jnp.float32) / temperature
        values, _ = jax.lax.top_k(logits, k=top_k)
        kth_value = values[:, -1]
        logits = jnp.where(logits < kth_value[:, jnp.newaxis], -jnp.inf, logits)
        next_token = jax.random.categorical(subkey, logits).reshape(-1, 1)
        batch = jnp.concatenate([batch[:, 1:], next_token], axis=-1)
        mask = jnp.concatenate([mask[:, 1:], jnp.ones_like(next_token, dtype=jnp.bool_)], axis=-1)
        return batch, mask, key
    
    batch, _, _ = jax.lax.fori_loop(0, sample_length, body_fn, (batch, mask, key))
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # for _ in range(sample_length):
    #     print(batch)
    #     print(mask)
    #     key, subkey = jax.random.split(key)
    #     logits = model(batch, mask)
    #     logits = logits[:, -1, :].astype(jnp.float32) / temperature
    #     values, indices = jax.lax.top_k(logits, k=top_k)
    #     print(values, indices)
    #     print(tokenizer.batch_decode(indices.reshape(-1, 1)))
    #     kth_value = values[:, -1]
    #     logits = jnp.where(logits < kth_value[:, jnp.newaxis], -jnp.inf, logits)
    #     next_token = jax.random.categorical(subkey, logits).reshape(-1, 1)
    #     print(next_token)

    #     batch = jnp.concatenate([batch[:, 1:], next_token], axis=-1)
    #     print(tokenizer.batch_decode(batch))
    #     mask = jnp.concatenate([mask[:, 1:], jnp.ones_like(next_token, dtype=jnp.bool_)], axis=-1)
    #     print()

    return batch


def generate(model, tokenizer, prompts, max_length, n_samples, top_k, temperature):
    # tokenizer = AutoTokenizer.from_pretrained(generate_config.tokenizer_name)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.bos_token

    # model = GPT(generate_config.model, nnx.Rngs(jax.random.PRNGKey(0)))
    # ckpt = ocp.StandardCheckpointer()
    # abstract_model = nnx.eval_shape(lambda: GPT(generate_config.model, nnx.Rngs(jax.random.PRNGKey(0))))
    # graphdef, abstract_state = nnx.split(abstract_model)

    # sharding = jax.sharding.NamedSharding(
    #     jax.sharding.Mesh(jax.devices(), ('data',)),
    #     jax.sharding.PartitionSpec(),
    # )
    # sharded_abstract_state = jax.tree_util.tree_map(lambda x: x.update(sharding=sharding), abstract_state)

    # state_restored = ckpt.restore(generate_config.weights_path + f"/default", sharded_abstract_state)
    # model = nnx.merge(graphdef, state_restored)

    # create a batch of size (n_samples * len(prompts), max_length) left padded with pad_token_id
    prompts_tokens = tokenizer.batch_encode_plus(prompts)['input_ids']
    # prompts_tokens = [jnp.concatenate([jnp.array([tokenizer.bos_token_id]), jnp.array(x)]) for x in prompts_tokens]
    prompts_tokens = [jnp.array(x) for x in prompts_tokens]

    # only sample max_length - longest_prompt_length tokens
    prompt_token_lengths = [x.shape[0] for x in prompts_tokens]
    sample_length = max_length - max(prompt_token_lengths)

    prompts_tokens = [jnp.pad(x, (max_length - x.shape[0], 0), mode='constant', constant_values=tokenizer.pad_token_id) for x in prompts_tokens]
    prompts_tokens = [jnp.repeat(x[jnp.newaxis, :], n_samples, axis=0) for x in prompts_tokens]
    prompts_tokens = jnp.concatenate(prompts_tokens, axis=0)
    # print(prompts_tokens)
    mask = prompts_tokens != tokenizer.pad_token_id
    generated_tokens = sample(model, prompts_tokens, mask, sample_length, top_k, temperature)

    # print(generated_tokens)
    decoded_tokens = tokenizer.batch_decode(generated_tokens)
    # print(decoded_tokens)
    
    samples = {}
    for i in range(0, n_samples * len(prompts), n_samples):
        prompt_idx = i // n_samples
        # print(f"prompt: {generate_config.prompts[prompt_idx]}, (prompt token length: {prompt_token_lengths[prompt_idx]})")
        for j in range(n_samples):
            # print(f"sample {j}: {decoded_tokens[i + j]}")
            samples[prompts[prompt_idx]] = samples.get(prompts[prompt_idx], []) + [decoded_tokens[i + j]]
        # print()
    
    return samples

    

# if __name__ == "__main__":

    # model_config = GPTConfig(
    #     vocab_size=50304,
    #     hidden_dim=768,
    #     num_layers=12,
    #     num_attention_heads=12,
    #     intermediate_dim=3072,
    #     head_dim=64,
    #     act_fn=nnx.gelu,
    #     max_seq_len=1024,
    #     layer_norm_epsilon=1e-5,
    #     use_bias=False,
    #     dtype=jnp.bfloat16,
    # )


    # sample_config = GenerateConfig(
    #     model=model_config,
    #     weights_path="jackpenn/transformers/run_nvneez36_model:v9",
    #     # weights_path="/Users/jack/projects/jaximus/artifacts/run_nvneez36_model:v9",
    #     prompts=[
    #         # "What is the meaning of life?",
    #         # "Hello, I'm a language model",
    #         # "5+7=",
    #         "The capital of France is",
    #     ],
    #     max_length=10,
    #     n_samples=1,
    #     top_k=10,
    #     temperature=1.0,
    #     tokenizer_name="gpt2",
    # )
    

    # run = wandb.init(project="transformers")
    # artifact = run.use_artifact(sample_config.weights_path, type='model')
    # artifact_dir = artifact.download()
    # sample_config = chz.replace(sample_config, weights_path=artifact_dir)

    # generate(sample_config)


    # # x = jnp.array([[1, 6, 3, 4, 5]], dtype=jnp.float32)
    # # counter = {}
    # # key = jax.random.PRNGKey(0)
    # # for i in range(10000):
    # #     key, subkey = jax.random.split(key)
    # #     next_token = jax.random.categorical(subkey, x)
    # #     counter[next_token.item()] = counter.get(next_token.item(), 0) + 1
    # # print(counter)