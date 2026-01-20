import argparse
import multiprocessing as mp
import os
import pickle

from array_record.python.array_record_module import ArrayRecordWriter
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def process_shard(pool_args):
    shard_id, shard_dataset, tokenizer, output_path = pool_args
    shard_path = f"{output_path}/shard_{shard_id}.array_record"
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)
    writer = ArrayRecordWriter(shard_path, "group_size:1")
    for i, example in tqdm(enumerate(shard_dataset)):
        tokens = tokenizer.encode(example["text"])
        record = {"tokens": tokens}
        writer.write(pickle.dumps(record))
    writer.close()
    return shard_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to array records")
    parser.add_argument(
        "hf_name",
        nargs="+",
        help="HuggingFace dataset name (can be multiple parts like ['user/dataset'] or ['dataset'])",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to use (default: train)")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path (default: gpt2)")
    parser.add_argument("--num_shards", type=int, default=100, help="Number of shards to create (default: 100)")
    parser.add_argument(
        "--num_procs", type=int, default=None, help="Number of processes to use (default: cpu_count - 1)"
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output path for array records (default: None)")

    args = parser.parse_args()

    print(args.hf_name)
    # Set default num_procs if not provided
    if args.num_procs is None:
        args.num_procs = os.cpu_count() - 1

    if args.output_path is None:
        args.output_path = f"saved/{'_'.join(args.hf_name)}/{args.split}"

    dataset = load_dataset(*args.hf_name, split=args.split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    num_shards = args.num_shards
    num_procs = args.num_procs
    output_path = args.output_path

    res = []

    pool_args = [
        (shard_id, dataset.shard(num_shards=num_shards, index=shard_id), tokenizer, output_path)
        for shard_id in range(num_shards)
    ]

    with mp.Pool(num_procs) as pool:
        for r in pool.imap(process_shard, pool_args):
            res.append(r)

    print(res)
