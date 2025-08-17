from datasets import load_dataset
from transformers import AutoTokenizer
import grain
from typing import List
import numpy as np

def get_hf_dataset(
        hf_name: List[str],
        tokenizer_name: str,
        max_length: int,
        num_proc: int = 4,
        split: str = "train",
):
    hf_ds = load_dataset(*hf_name, split=split, num_proc=num_proc)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    parent_ds = (
        grain.MapDataset.source(hf_ds)
        .map(lambda x: {"tokens": tokenizer.encode(x["text"], return_tensors="np")[0]})
        
    )

    ds = grain.experimental.ConcatThenSplitIterDataset(
        parent=parent_ds,
        length_struct={"tokens": max_length+1},
    )

    ds = ds.map(lambda x: (x['tokens'][:-1], x['tokens'][1:]))

    return tokenizer, ds