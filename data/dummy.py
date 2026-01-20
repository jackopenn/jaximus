import grain
import jax


def get_dummy_dataset(max_length: int, batch_size: int):
    batch = (
        jax.random.randint(
            jax.random.PRNGKey(0),
            (max_length,),
            0,
            max_length,
        ),
        jax.random.randint(
            jax.random.PRNGKey(0),
            (max_length,),
            0,
            max_length,
        ),
    )

    return grain.MapDataset.source(batch).to_iter_dataset().batch(batch_size)
