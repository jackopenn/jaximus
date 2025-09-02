import jax
from jax import numpy as jnp
import grain

def get_dummy_dataset(max_length: int):
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
        )
    )

    return None, grain.MapDataset.source(batch).to_iter_dataset()