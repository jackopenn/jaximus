import grain
import jax
from datasets import load_dataset
from transformers import AutoTokenizer

from data.grain_transforms import Shift, Tokenize


class HFStreamingDataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, iterable_ds):
        self._ds = iterable_ds
        self._it = None

    def __len__(self) -> int:
        return 10_000_000_000

    def __getitem__(self, record_key: int):
        if self._it is None:
            self._it = iter(self._ds)
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._ds)
            return next(self._it)


def get_hf_dataset(
    hf_name,
    sequence_length,
    batch_size,
    tokenizer_name=None,
    streaming=True,
    num_proc=None,
    num_shards=None,
    shard_index=None,
    split="train",
    data_files=None,
):
    if streaming:
        num_proc = None

    num_shards = num_shards if num_shards is not None else jax.process_count()
    shard_index = shard_index if shard_index is not None else jax.process_index()
    hf_ds = load_dataset(*hf_name, split=split, data_files=data_files, streaming=streaming, num_proc=num_proc)
    if num_shards and num_shards > 1:
        hf_ds = hf_ds.shard(num_shards=num_shards, index=shard_index)

    source = HFStreamingDataSource(hf_ds) if streaming else hf_ds
    ds = grain.MapDataset.source(source).to_iter_dataset(
        read_options=grain.ReadOptions(num_threads=1 if streaming else 16)
    )

    # if no tokenizer, assume already tokenized
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        ds = ds.map(Tokenize(tokenizer, prepend_bos=True))

    ds = grain.experimental.ConcatThenSplitIterDataset(
        parent=ds,
        length_struct={"tokens": sequence_length + 1},
    )

    assert batch_size % jax.process_count() == 0, "batch_size must be divisible by process_count"

    local_batch_size = batch_size // jax.process_count()
    ds = ds.map(Shift()).batch(local_batch_size, drop_remainder=True)

    return ds
