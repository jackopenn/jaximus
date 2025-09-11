from transformers import AutoTokenizer
import pickle
from array_record.python.array_record_module import ArrayRecordWriter
import multiprocessing as mp
from tqdm import tqdm
import os
import argparse
import pyarrow.parquet as pq


GLOBAL_TOKENIZER = None


def _init_worker(tokenizer_name: str):
    global GLOBAL_TOKENIZER
    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def process_file(pool_args):
    file_index, file_path, output_dir, text_column, batch_size = pool_args
    global GLOBAL_TOKENIZER
    tokenizer = GLOBAL_TOKENIZER

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = f"{output_dir}/{base}.array_record"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer = ArrayRecordWriter(out_path, "group_size:1")
    try:
        parquet = pq.ParquetFile(file_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[text_column]):
            # batch is a pyarrow.RecordBatch
            column = batch.column(0)
            texts = column.to_pylist()
            for text in texts:
                if text is None:
                    continue
                tokens = tokenizer.encode(text)
                record = {"tokens": tokens}
                writer.write(pickle.dumps(record))
    finally:
        writer.close()

    return file_index


def list_parquet_files(input_dir: str):
    parquet_files = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith(".parquet"):
                parquet_files.append(os.path.join(root, name))
    parquet_files.sort()
    return parquet_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a directory of Parquet files to ArrayRecords")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column (default: text)")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path (default: gpt2)")
    parser.add_argument("--num_procs", type=int, default=None, help="Number of processes to use (default: cpu_count - 1)")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory for ArrayRecords")
    parser.add_argument("--batch_size", type=int, default=8192, help="Parquet read batch size (default: 8192)")

    args = parser.parse_args()

    if args.num_procs is None:
        args.num_procs = max(1, (os.cpu_count() or 2) - 1)

    if args.output_path is None:
        dir_name = os.path.basename(os.path.normpath(args.input_dir)) or "parquet"
        args.output_path = f"saved/{dir_name}"

    files = list_parquet_files(args.input_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"No .parquet files found in {args.input_dir}")

    pool_args = [
        (idx, f, args.output_path, args.text_column, args.batch_size)
        for idx, f in enumerate(files)
    ]

    results = []
    with mp.Pool(args.num_procs, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for r in tqdm(pool.imap(process_file, pool_args), total=len(pool_args), desc="files"):
            results.append(r)

    print(f"Finished processing {len(results)} files → {args.output_path}")