"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794
"""
import csv
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import jax
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import optax
import yaml
from jax.sharding import PartitionSpec as P, reshard
from jinja2 import Template

from modelling.layers.position import precompute_rope_embeddings

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def download_eval_bundle(eval_data_path):
    """Download and extract eval bundle if not already cached. Each host downloads independently."""
    cache_dir = Path(eval_data_path)
    bundle_path = cache_dir / "eval_bundle"

    if not bundle_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "eval_bundle.zip"

        if jax.process_index() == 0:
            print(f"Downloading eval bundle from {EVAL_BUNDLE_URL}...")
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)

        if jax.process_index() == 0:
            print("Extracting eval bundle...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

        zip_path.unlink()

    if jax.process_count() > 1:
        jax.experimental.multihost_utils.sync_global_devices("eval_bundle_download")

    return bundle_path


def load_core_config(bundle_path):
    """Load core.yaml task configurations."""
    with open(bundle_path / "core.yaml") as f:
        return yaml.safe_load(f)["icl_tasks"]


def load_task_data(bundle_path, dataset_uri):
    """Load task data from jsonl file."""
    data_path = bundle_path / "eval_data" / dataset_uri
    with open(data_path) as f:
        return [json.loads(line) for line in f]


def load_random_baselines(bundle_path):
    """Load random baselines from eval_meta_data.csv."""
    baselines = {}
    with open(bundle_path / "eval_meta_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            baseline_pct = float(row["Random baseline"]) if row["Random baseline"] else 0.0
            baselines[task_name] = baseline_pct / 100.0
    return baselines


MC_TEMPLATE = Template("""
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip())


SCHEMA_TEMPLATE = Template("""
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip())


LM_TEMPLATE = Template("""
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip())


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question."""
    fewshot_examples = fewshot_examples or []
    context = {"fewshot_examples": fewshot_examples, "continuation_delimiter": continuation_delimiter, "item": item}
    return [MC_TEMPLATE.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question."""
    fewshot_examples = fewshot_examples or []
    context = {"fewshot_examples": fewshot_examples, "continuation_delimiter": continuation_delimiter, "item": item}
    return [SCHEMA_TEMPLATE.render(context=ctx, **context) for ctx in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a language modeling task."""
    fewshot_examples = fewshot_examples or []
    context = {"fewshot_examples": fewshot_examples, "continuation_delimiter": continuation_delimiter, "item": item}
    prompt_without = LM_TEMPLATE.render(include_continuation=False, **context).strip()
    prompt_with = LM_TEMPLATE.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """Find the length of the common prefix or suffix across token sequences."""
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right."""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = np.full((bsz, seq_len), pad_token_id, dtype=np.int32)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = x
    return jnp.array(input_ids)


def batch_sequences_mc(tokenizer, prompts, bos_token_id):
    """Batch sequences for multiple choice (common prefix)."""
    tokens = [[bos_token_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts, bos_token_id):
    """Batch sequences for schema tasks (common suffix)."""
    tokens = [[bos_token_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts, bos_token_id):
    """Batch sequences for LM tasks (prompt without/with continuation)."""
    tokens = [[bos_token_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    return [tokens_with], [start_idx], [end_idx]


def get_fewshot_examples(data, idx, num_fewshot):
    """Get fewshot examples for a given data index."""
    if num_fewshot <= 0:
        return []
    rng = random.Random(1234 + idx)
    available_indices = [i for i in range(len(data)) if i != idx]
    fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
    return [data[i] for i in fewshot_indices]


def crop_sequences(tokens, start_idxs, end_idxs, max_seq_len):
    """Crop sequences to max_seq_len from the left, adjusting indices accordingly."""
    if max_seq_len is None:
        return tokens, start_idxs, end_idxs
    new_tokens, new_start_idxs, new_end_idxs = [], [], []
    for t, s, e in zip(tokens, start_idxs, end_idxs):
        if len(t) > max_seq_len:
            num_to_crop = len(t) - max_seq_len
            new_tokens.append(t[-max_seq_len:])
            new_start_idxs.append(max(1, s - num_to_crop))
            new_end_idxs.append(e - num_to_crop)
        else:
            new_tokens.append(t)
            new_start_idxs.append(s)
            new_end_idxs.append(e)
    return new_tokens, new_start_idxs, new_end_idxs


def prepare_batch(data, indices, tokenizer, task_meta, bos_token_id, max_seq_len):
    """Prepare a batch of examples for evaluation."""
    all_tokens = []
    example_ids = []
    start_idxs = []
    end_idxs = []
    gold_labels = []
    num_choices_list = []

    task_type = task_meta["task_type"]
    continuation_delimiter = task_meta["continuation_delimiter"]
    num_fewshot = task_meta["num_fewshot"]

    for batch_idx, data_idx in enumerate(indices):
        item = data[data_idx]
        fewshot_examples = get_fewshot_examples(data, data_idx, num_fewshot)

        if task_type == "multiple_choice":
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
            tokens, starts, ends = batch_sequences_mc(tokenizer, prompts, bos_token_id)
        elif task_type == "schema":
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
            tokens, starts, ends = batch_sequences_schema(tokenizer, prompts, bos_token_id)
        else:  # language_modeling
            prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
            tokens, starts, ends = batch_sequences_lm(tokenizer, prompts, bos_token_id)

        tokens, starts, ends = crop_sequences(tokens, starts, ends, max_seq_len)

        num_choices = len(tokens)
        for choice_idx in range(num_choices):
            all_tokens.append(tokens[choice_idx])
            example_ids.append(batch_idx)
            start_idxs.append(starts[choice_idx])
            end_idxs.append(ends[choice_idx])

        gold_labels.append(item.get("gold", 0))
        num_choices_list.append(num_choices)

    return {
        "tokens": all_tokens,
        "example_ids": example_ids,
        "start_idxs": start_idxs,
        "end_idxs": end_idxs,
        "gold_labels": gold_labels,
        "num_choices": num_choices_list,
        "task_type": task_type,
    }


def evaluate_batch(batch, eval_step_fn, pad_token_id, max_seq_len, eval_batch_size, timing=False):
    """Evaluate a batch and return list of is_correct per example."""
    tokens = batch["tokens"]
    start_idxs = batch["start_idxs"]
    end_idxs = batch["end_idxs"]
    gold_labels = batch["gold_labels"]
    num_choices = batch["num_choices"]
    task_type = batch["task_type"]

    if not tokens:
        return []

    t0 = time.perf_counter() if timing else 0
    input_ids = stack_sequences(tokens, pad_token_id)
    t1 = time.perf_counter() if timing else 0

    losses, predictions = forward_model(eval_step_fn, input_ids, max_seq_len, eval_batch_size)
    jax.block_until_ready((losses, predictions))
    t2 = time.perf_counter() if timing else 0

    # Transfer to CPU once, then do all slicing on numpy
    losses_np = np.asarray(losses)
    predictions_np = np.asarray(predictions)
    input_ids_np = np.asarray(input_ids)
    t3 = time.perf_counter() if timing else 0

    results = []
    row_offset = 0

    for ex_idx, n_choices in enumerate(num_choices):
        if task_type == "language_modeling":
            si, ei = start_idxs[row_offset], end_idxs[row_offset]
            pred_tokens = predictions_np[row_offset, si - 1 : ei - 1]
            actual_tokens = input_ids_np[row_offset, si:ei]
            is_correct = np.all(pred_tokens == actual_tokens)
        else:
            choice_losses = []
            for c in range(n_choices):
                row = row_offset + c
                si, ei = start_idxs[row], end_idxs[row]
                mean_loss = losses_np[row, si - 1 : ei - 1].mean()
                choice_losses.append(mean_loss)
            pred_idx = choice_losses.index(min(choice_losses))
            is_correct = pred_idx == gold_labels[ex_idx]

        results.append(is_correct)
        row_offset += n_choices

    t4 = time.perf_counter() if timing else 0

    if timing:
        print(f"stack: {t1-t0:.3f}s, forward: {t2-t1:.3f}s, transfer: {t3-t2:.3f}s, extract: {t4-t3:.3f}s")

    return results


def make_eval_step(forward_fn, weights, config, rope_cos, rope_sin):
    """Create a JIT-compiled eval step function with weights bound."""
    weights_sharding = jax.tree.map(lambda x: x.sharding, weights)

    def eval_step(input_ids, weights):
        logits = forward_fn(input_ids, weights, config, rope_cos=rope_cos, rope_sin=rope_sin)
        target_ids = jnp.roll(input_ids, -1, axis=-1)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
        predictions = jnp.argmax(logits, axis=-1)
        # Reshard outputs to replicated for easy slicing
        losses = reshard(losses, P(None, None))
        predictions = reshard(predictions, P(None, None))
        return losses, predictions

    jitted_step = jax.jit(eval_step, in_shardings=(P(None, None), weights_sharding), out_shardings=(P(), P()))
    return lambda input_ids: jitted_step(input_ids, weights)


def forward_model(eval_step_fn, input_ids, max_seq_len, eval_batch_size):
    """Run JIT-compiled eval step with fixed shape padding."""
    batch_size, seq_len = input_ids.shape

    # Pad to fixed shape (eval_batch_size, max_seq_len) for JIT stability
    batch_pad = eval_batch_size - batch_size
    seq_pad = max_seq_len - seq_len
    if batch_pad > 0 or seq_pad > 0:
        input_ids = jnp.pad(input_ids, [(0, batch_pad), (0, seq_pad)], constant_values=0)

    losses, predictions = eval_step_fn(input_ids)

    # Slice back to original size
    return losses[:batch_size, :seq_len], predictions[:batch_size, :seq_len]


def evaluate_task(task_config, bundle_path, tokenizer, eval_step_fn, max_seq_len, max_per_task=-1, eval_batch_size=256):
    """Evaluate all examples for a single task with batching."""
    dataset_uri = task_config["dataset_uri"]
    task_type = task_config["icl_task_type"]
    num_fewshot = task_config["num_fewshot"][0]
    continuation_delimiter = task_config.get("continuation_delimiter", "")
    label = task_config["label"]

    data = load_task_data(bundle_path, dataset_uri)
    if max_per_task > 0:
        data = data[:max_per_task]

    task_meta = {"task_type": task_type, "num_fewshot": num_fewshot, "continuation_delimiter": continuation_delimiter}
    main_process = jax.process_index() == 0
    bos_token_id = tokenizer.bos_token_id or 0
    pad_token_id = bos_token_id

    correct = []
    pending_indices = []
    pending_rows = 0
    iterator = tqdm(range(len(data)), desc=label, disable=not main_process, leave=False)

    for idx in iterator:
        item = data[idx]
        if task_type == "multiple_choice":
            num_choices = len(item["choices"])
        elif task_type == "schema":
            num_choices = len(item["context_options"])
        else:  # language_modeling
            num_choices = 1

        # If adding this example would exceed batch size, process current batch first
        if pending_rows + num_choices > eval_batch_size and pending_rows > 0:
            batch = prepare_batch(data, pending_indices, tokenizer, task_meta, bos_token_id, max_seq_len)
            results = evaluate_batch(batch, eval_step_fn, pad_token_id, max_seq_len, eval_batch_size)
            correct.extend([float(r) for r in results])
            pending_indices = []
            pending_rows = 0
            if main_process:
                iterator.set_postfix(acc=f"{sum(correct)/len(correct):.2%}")

        pending_indices.append(idx)
        pending_rows += num_choices

    # Process final batch
    if pending_indices:
        batch = prepare_batch(data, pending_indices, tokenizer, task_meta, bos_token_id, max_seq_len)
        results = evaluate_batch(batch, eval_step_fn, pad_token_id, max_seq_len, eval_batch_size)
        correct.extend([float(r) for r in results])

    return sum(correct) / len(correct) if correct else 0.0, task_type


def evaluate_model(weights, config, forward_fn, tokenizer, eval_data_path, max_per_task=-1, eval_batch_size=256):
    """
    Evaluate model on CORE benchmark tasks.

    Args:
        weights: ModelWeights dataclass (JAX pytree)
        config: Model config with hidden_dim, max_seq_len, rope_theta, etc.
        forward_fn: model_forward(x, weights, config, rope_cos, rope_sin, mask) -> logits
        tokenizer: HuggingFace tokenizer
        eval_data_path: Path to eval data cache (local or gs://)
        max_per_task: Limit examples per task for debugging (-1 = all)
        eval_batch_size: Number of examples to batch together (default 256)

    Returns:
        dict with results, centered_results, core_metric
    """
    bundle_path = download_eval_bundle(eval_data_path)
    main_process = jax.process_index() == 0
    max_seq_len = getattr(config, "max_seq_len", 2048)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rope_cos, rope_sin = None, None
    if hasattr(config, "rope_theta") and config.rope_theta is not None:
        rope_cos, rope_sin = precompute_rope_embeddings(
            max_seq_len, config.head_dim, config.rope_theta, getattr(config, "dtype", "bfloat16")
        )
        rope_cos, rope_sin = reshard(rope_cos, P()), reshard(rope_sin, P())

    # Create JIT-compiled eval step
    eval_step_fn = make_eval_step(forward_fn, weights, config, rope_cos, rope_sin)

    task_configs = load_core_config(bundle_path)
    random_baselines = load_random_baselines(bundle_path)

    results = {}
    task_types = {}
    for task_config in task_configs:
        label = task_config["label"]
        score, task_type = evaluate_task(
            task_config, bundle_path, tokenizer, eval_step_fn, max_seq_len, max_per_task, eval_batch_size
        )
        results[label] = score
        task_types[label] = task_type

        if main_process:
            print(f"{label}: {score:.4f}")

    centered_results = {}
    for label, score in results.items():
        task_type = task_types[label]
        if task_type == "language_modeling":
            centered_results[label] = score
        else:
            baseline = random_baselines.get(label, 0.0)
            centered_results[label] = (score - baseline) / (1.0 - baseline) if baseline < 1.0 else score

    accuracy_tasks = [k for k, v in task_types.items() if v != "language_modeling"]
    if accuracy_tasks:
        core_metric = sum(centered_results[k] for k in accuracy_tasks) / len(accuracy_tasks)
    else:
        core_metric = 0.0

    if main_process:
        print(f"\nCORE metric: {core_metric:.4f}")

    return {"results": results, "centered_results": centered_results, "core_metric": core_metric}


if __name__ == "__main__":
    print("Usage: from eval import evaluate_model")
    print("       results = evaluate_model(weights, config, model_forward, tokenizer)")
