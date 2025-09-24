import jax
from flax import nnx

def get_nparams_and_flops(model):
    _, params = nnx.split(model, nnx.Param)

    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    embed_params = model.token_embedding.embedding.size

    try:
        embed_params += model.pos_embedding.embedding.size
    except:
        pass

    l, h, q, t = (
        model.config.num_layers,
        model.config.num_attention_heads,
        model.config.head_dim,
        model.config.max_seq_len,
    )

    nflops_per_token = 6 * (nparams - embed_params) + 12 * l * h * q * t

    return nparams, nflops_per_token


def get_gpu_peak_flops(gpu_name):   
    if gpu_name == "H100":
        return 1979e12/2 # SXM
        # return 1513e12/2 # PCIe
    elif gpu_name == "A100":
        return 624e12/2
    elif gpu_name == "5090":
        return 104.8 * 1e12
    elif gpu_name == "v6e":
        return 918e12
    else:
        raise ValueError(f"don't have peak flops for {gpu_name}")