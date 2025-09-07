import jax
from datasets import load_dataset
from transformers import AutoTokenizer
import grain
from typing import List, Callable, Iterator, Any
import numpy as np
import os
from torch.utils.data import DataLoader, default_collate
import tensorflow as tf
tf.config.set_visible_devices([], device_type='GPU')

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


class Tokenize(grain.transforms.Map):
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def map(self, x):
    return {"input_ids": self.tokenizer.encode(x['text'], return_tensors="np")[0]}

class GetInputAndTarget(grain.transforms.Map):
  def map(self, x):
    return x['input_ids'][:-1], x['input_ids'][1:]

# def get_hf_dataset(
#     hf_name: List[str],
#     tokenizer_name: str,
#     sequence_length: int,
#     batch_size: int,
#     split: str = "train",
# ):
    
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     hf_ds = load_dataset(*hf_name, split=split, streaming=True)
#     source = HFStreamingDataSource(hf_ds)

#     ds = (
#         grain.MapDataset.source(source)
#         .to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1024))
#         .map(Tokenize(tokenizer))
#     )
    
#     ds = (
#         grain.experimental.ConcatThenSplitIterDataset(
#             parent=ds,
#             length_struct={"input_ids": sequence_length+1},
#             bos_token_id=tokenizer.bos_token_id,
#             bos_handling=grain.experimental.BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS,
#             bos_features={"input_ids"}
#         )
#         .map(GetInputAndTarget())
#         .batch(batch_size)
#     )

#     return tokenizer, ds



# )
def get_hf_dataset(
    hf_name: List[str],
    tokenizer_name: str,
    sequence_length: int,
    batch_size: int,
    split: str = "train",
):
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_bos_token=True)

    # https://github.com/huggingface/nanotron/blob/7bc9923285a03069ebffe994379a311aceaea546/src/nanotron/data/processing.py#L47
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def tokenize_and_group_texts(texts):
        tokenized_batch = tokenizer(texts, return_attention_mask=False, return_token_type_ids=False, return_tensors="np")
        grouped_batch = group_texts(tokenized_batch)
        return grouped_batch


    hf_ds = load_dataset(*hf_name, split=split, streaming=True)
    hf_ds = hf_ds.map(
        tokenize_and_group_texts,
        input_columns=["text"],
        remove_columns=hf_ds.column_names,
        batched=True,
        # batch_size=batch_size,
    )

#     source = HFStreamingDataSource(hf_ds)

#     sampler = grain.samplers.IndexSampler(
#         num_records=len(source),
#         shuffle=False,
#         seed=0,
#     )


#     operations = []
#     operations.append(GetInputAndTarget())
#     operations.append(grain.transforms.Batch(batch_size))

#     data_loader = grain.DataLoader(
#         data_source=source,
#         sampler=sampler,
#         operations=operations,
#         # worker_count=1,
#         worker_buffer_size=1,
#         read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1024),
#         enable_profiling=True,
#     )
#     # cpu_buffer_size = 8
#     # data_loader = (
#     #     grain.MapDataset.source(source)
#     #     .to_iter_dataset(read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1024))
#     #     .map(GetInputAndTarget())
#     #     .batch(batch_size)
#     # )
#     # data_loader = (
#     #     grain.experimental.ThreadPrefetchIterDataset(
#     #         parent=data_loader,
#     #         prefetch_buffer_size=cpu_buffer_size,
#     #     )
#     # )
    def numpy_collate(batch):
        batch = jax.tree_util.tree_map(np.asarray, default_collate(batch))
        return (batch['input_ids'][:,:-1], batch['input_ids'][:,1:])

    
    
    data_loader = DataLoader(hf_ds, batch_size=batch_size, collate_fn=numpy_collate, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # data_loader = (
    #     tf.data.Dataset.from_generator(
    #         lambda: hf_ds,
    #         output_signature={"input_ids": tf.TensorSpec(shape=(sequence_length+1,), dtype=tf.int32)},
    #     )
    #     .batch(batch_size)
    #     .map(lambda x: (x['input_ids'][:, :-1], x['input_ids'][:, 1:]), num_parallel_calls=tf.data.AUTOTUNE)
    #     .prefetch(tf.data.AUTOTUNE)
    #     .as_numpy_iterator()
    # )
    return tokenizer, data_loader



# def get_hf_dataset_old(
#         hf_name: List[str],
#         tokenizer_name: str,
#         sequence_length: int,
#         batch_size: int,
#         num_proc: int = 4,
#         split: str = "train",
# ):
#     hf_ds = load_dataset(*hf_name, split=split, num_proc=num_proc)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     parent_ds = (
#         grain.MapDataset.source(hf_ds)
#         .map(lambda x: {"tokens": tokenizer.encode(x["text"], return_tensors="np")[0]})
        
#     )

#     ds = grain.experimental.ConcatThenSplitIterDataset(
#         parent=parent_ds,
#         length_struct={"tokens": sequence_length+1},
#     )

#     ds = ds.map(lambda x: (x['tokens'][:-1], x['tokens'][1:])).batch(batch_size)

#     return tokenizer, ds