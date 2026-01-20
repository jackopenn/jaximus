import pickle

import grain
import numpy as np


class ParseArrayRecord(grain.transforms.Map):
    def map(self, x):
        return pickle.loads(x)


class Tokenize(grain.transforms.Map):
    def __init__(self, tokenizer, prepend_bos=True):
        self.prepend_bos = prepend_bos
        self.tokenizer = tokenizer

    def map(self, x):
        tokens = self.tokenizer.encode(x["text"])
        if self.prepend_bos:
            return {"tokens": np.asarray([self.tokenizer.bos_token_id] + tokens, dtype=np.int32)}
        else:
            return {"tokens": np.asarray(tokens, dtype=np.int32)}


class Trim(grain.transforms.Map):
    def __init__(self, max_length):
        self.max_length = max_length

    def map(self, x):
        return {"tokens": x["tokens"][: self.max_length]}


class Shift(grain.transforms.Map):
    def map(self, x):
        return (x["tokens"][:-1], x["tokens"][1:])
