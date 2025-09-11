import os
import jax
import grain
from grain_transforms import ParseArrayRecord, Tokenize, Shift


def get_array_record_dataset(
    path,
    sequence_length,
    batch_size,
    tokenizer=None,
):
    """ array records should be in the format of {"tokens": np.array([int])} or {"text": str} """

    array_files = os.listdir(path)
    array_files = [f"{path}/{f}" for f in array_files if f.endswith(".array_record")]
    
    ds = (
        grain.MapDataset.source(grain.sources.ArrayRecordDataSource(array_files))
        .map(ParseArrayRecord())
    )

    # if no tokenizer, assume already tokenized
    if tokenizer:
        ds = ds.map(Tokenize(tokenizer, prepend_bos=True))

    # shard the dataset
    ds = ds[jax.process_index()::jax.process_count()]

    ds = grain.experimental.ConcatThenSplitIterDataset(
        parent=ds,
        length_struct={"tokens": sequence_length + 1},
    )

    assert batch_size % jax.process_count() == 0, "batch_size must be divisible by process_count"

    local_batch_size = batch_size // jax.process_count()
    ds = ds.map(Shift()).batch(local_batch_size, drop_remainder=True)

    return ds