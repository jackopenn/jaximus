from typing import List
from functools import partial
import jax
from jax import numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
import wandb
import orbax.checkpoint as ocp


@partial(
    jax.jit,
    static_argnames=("prompt_length", "sample_length", "top_k", "temperature"),
    donate_argnums=(1,)
)
def sample(model, batch, mask, prompt_length, sample_length, top_k, temperature, key=jax.random.PRNGKey(0)):

    def body_fn(i, carry):
        batch, mask, sample_idx, key = carry
        key, subkey = jax.random.split(key)
    
        logits = model(batch, mask)
        logits = logits[:, sample_idx, :].astype(jnp.float32) / temperature
        values, _ = jax.lax.top_k(logits, k=top_k)
        kth_value = values[:, -1]
        logits = jnp.where(logits < kth_value[:, jnp.newaxis], -jnp.inf, logits)
        next_token = jax.random.categorical(subkey, logits)
        sample_idx += 1
        batch = batch.at[:, sample_idx].set(next_token)
        mask = mask.at[:, sample_idx].set(jnp.ones_like(next_token, dtype=jnp.bool_))
        return batch, mask, sample_idx, key
    
    sample_idx = prompt_length - 1
    batch, _, _, _ = jax.lax.fori_loop(0, sample_length, body_fn, (batch, mask, sample_idx, key))
    return batch


def generate(
    model,
    tokenizer,
    prompts=[
        "The meaning of life is",
        "Hello, I'm a language model",
        "5+7=",
        "five plus seven is",
        "The capital of France is",
        "The answer to the ultimate question of life, the universe, and everything is",
        "Once upon a time, there was a",
    ],
    max_length=64,
    n_samples=5,
    top_k=10,
    temperature=1.0
):

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.bos_token


    # create a batch of size (n_samples * len(prompts), max_length) left padded with pad_token_id
    prompts_tokens = tokenizer.batch_encode_plus(prompts)['input_ids']
    prompts_tokens = [jnp.concatenate([jnp.array([tokenizer.bos_token_id]), jnp.array(x)]) for x in prompts_tokens]

    prompt_token_lengths = [x.shape[0] for x in prompts_tokens]

    prompts_tokens = [jnp.pad(x, (0, max_length - x.shape[0]), mode='constant', constant_values=tokenizer.pad_token_id) for x in prompts_tokens]
    prompts_tokens = [jnp.repeat(x[jnp.newaxis, :], n_samples, axis=0) for x in prompts_tokens]
    masks = [x != tokenizer.pad_token_id for x in prompts_tokens]
    masks = [mask.at[:, 0].set(True) for mask in masks]

    generated_tokens = []
    for prompt_tokens, mask, prompt_token_length in zip(prompts_tokens, masks, prompt_token_lengths):
        sample_length = max_length - prompt_token_length
        # print(prompt_tokens)
        generated_tokens.append(sample(model, prompt_tokens, mask, prompt_token_length, sample_length, top_k, temperature))

    decoded_tokens = tokenizer.batch_decode(jnp.concatenate(generated_tokens, axis=0))
    
    samples = {}
    for i in range(0, n_samples * len(prompts), n_samples):
        prompt_idx = i // n_samples
        for j in range(n_samples):
            samples[prompts[prompt_idx]] = samples.get(prompts[prompt_idx], []) + [decoded_tokens[i + j]]
    
    return samples


if __name__ == "__main__":
    print("TODO: add generation")
    
    # # weight_path = "jackpenn/transformers/run_vckuuoa9_model:v11"
    
    # # run = wandb.init(project="transformers")
    # # artifact = run.use_artifact(weight_path, type='model')
    # # weight_path = artifact.download()
    
    # weight_path = "/Users/jack/projects/jaximus/artifacts/run_vckuuoa9_model:v11"

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

    # ckpt = ocp.StandardCheckpointer()
    # abstract_model = nnx.eval_shape(lambda: GPT(model_config, nnx.Rngs(jax.random.PRNGKey(0))))
    # graphdef, abstract_state = nnx.split(abstract_model)

    # sharding = jax.sharding.NamedSharding(
    #     jax.sharding.Mesh(jax.devices(), ('data',)),
    #     jax.sharding.PartitionSpec(),
    # )
    # sharded_abstract_state = jax.tree_util.tree_map(lambda x: x.update(sharding=sharding), abstract_state)

    # state_restored = ckpt.restore(weight_path + f"/default", sharded_abstract_state)
    # model = nnx.merge(graphdef, state_restored)

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # questions = [
    #     "Who wrote the book the origin of species?",
    #     "Who is the founder of the ubuntu project?",
    #     "Who is the quarterback for the green bay packers?",
    #     "Panda is a national animal of which country?",
    #     "Who came up with the theory of relativity?",
    #     "When was the first star wars film released?",
    #     "What is the most common blood type in sweden?",
    #     "Who is regarded as the founder of psychoanalysis?",
    #     "Who took the first steps on the moon in 1969?",
    #     "Who is the largest supermarket chain in the uk?",
    #     "What is the meaning of shalom in english?",
    #     "Who was the author of the art of war?",
    #     "Largest state in the us by land mass?",
    #     "Green algae is an example of which type of reproduction?",
    #     "Vikram samvat calender is official in which country?",
    #     "Who is mostly responsible for writing the declaration of independence?",
    #     "What us state forms the western boundary of montana?",
    #     "Who plays ser davos in game of thrones?",
    #     "Who appoints the chair of the federal reserve system?",
    #     "State the process that divides one nucleus into two genetically identical nuclei?",
    #     "Who won the most mvp awards in the nba?",
    #     "What river is associated with the city of rome?",
    #     "Who is the first president to be impeached?",
    #     "Who is the head of the department of homeland security 2017?",
    #     "What is the name given to the common currency to the european union?",
    #     "What was the emperor name in star wars?",
    #     "Do you have to have a gun permit to shoot at a range?",
    #     "Who proposed evolution in 1859 as the basis of biological development?",
    #     "Nuclear power plant that blew up in russia?",
    #     "Who played john connor in the original terminator?"
    # ]

    # maths = [
    #     "5+7=",
    #     "1+3=",
    #     "2*3=",
    #     "6/2=",
    #     "10-3=",
    #     "10+3=",
    #     "1+1=",
    #     "2+2=",
    #     "3+3=",
     
    # ]

    # prompts = [
    #     "The meaning of life is",
    #     "Hello, I'm a language model,",
    #     "5+7=",
    #     "five plus seven is",
    #     "The capital of France is",
    #     "The answer to the ultimate question of life, the universe, and everything is",
    #     "Once upon a time, there was a",
    # ]

    # samples = generate(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompts=prompts,
    #     max_length=64,
    #     n_samples=2,
    #     top_k=10,
    #     temperature=1.0,
    # )
    
    # for prompt, sample_list in samples.items():
    #     print(f"prompt: {prompt}")
    #     for i, sample in enumerate(sample_list):
    #         print(f"sample {i}: {sample}")
    #     print()