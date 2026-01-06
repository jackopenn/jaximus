import re
from datetime import datetime
import jax
from flax import nnx

def get_num_params_and_flops(model):
    _, params = nnx.split(model, nnx.Param)

    nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
    embed_params = model.token_embedding.embedding.size

    try:
        embed_params += model.pos_embedding.embedding.size
    except AttributeError:
        pass

    l, h, q, t = (
        model.num_layers,
        model.num_attention_heads,
        model.head_dim,
        model.max_seq_len,
    )

    nflops_per_token = 6 * (nparams - embed_params) + 12 * l * h * q * t

    return nparams, nflops_per_token


def get_xpu_peak_flops(xpu_name):   
    if xpu_name == "H100":
        return 1979e12/2 # SXM
        # return 1513e12/2 # PCIe
    elif xpu_name == "A100":
        return 624e12/2
    elif xpu_name == "5090":
        return 104.8 * 1e12
    elif xpu_name == "v6e":
        return 918e12
    elif xpu_name == "v5p":
        return 459e12
    elif xpu_name == "v4":
        return 275e12
    else:
        raise ValueError(f"don't have peak flops for {xpu_name}")
    
def pretty_print_samples(samples):
    for prompt, samples_list in samples.items():
        print(f"prompt: {prompt}")
        for i, sample in enumerate(samples_list):
            clean = re.sub(r'^(?:<\|endoftext\|>)+', '', sample)
            print(f"sample {i}: {clean}")
        print()


# https://github.com/karpathy/nanochat/blob/bc51da8baca66c54606bdd75c861c82ced90dcb0/nanochat/common.py#L183C1-L190C13
class DummyWandb:
    def __init__(self):
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S")
    def log(self, *args, **kwargs):
        pass
    def log_artifact(self, *args, **kwargs):
        pass
    def log_model(self, *args, **kwargs):
        pass
    def finish(self):
        pass

class MetricLogger:
    def __init__(self, batch_size, accum_steps, sequence_length, num_flops_per_token, xpu_name, wandb_run):
        self.num_flops_per_token = float(num_flops_per_token)
        self.wandb_run = wandb_run
        self.tokens_per_batch = batch_size * accum_steps * sequence_length
        self.num_devices = jax.device_count()
        self.xpu_peak_flops = get_xpu_peak_flops(xpu_name) * self.num_devices
        self.prev_metrics = None
        self.tokens_consumed = 0


    def _human_format(self, num: float, divide_by_1024: bool = False) -> str:
        # https://github.com/huggingface/nanotron/blob/7bc9923285a03069ebffe994379a311aceaea546/src/nanotron/logging/base.py#L268
        if abs(num) < 1:
            return "{:.3g}".format(num)
        SIZES = ["", "K", "M", "B", "T", "P", "E"]
        num = float("{:.3g}".format(num))
        i = 0
        while abs(num) >= 1000 and i < len(SIZES) - 1:
            num /= 1000.0 if not divide_by_1024 else 1024.0
            i += 1
        return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[i])


    def _pretty_print(self, metrics, step):
        
        print_string = f"step: {step}, loss: {self._human_format(metrics['loss'])}"
        for k, v in metrics.items():
            if k != "loss":
                print_string += f", {k}: {self._human_format(v)}"
        print(print_string)


    def _log(self, metrics):
        step = metrics.pop("step")
        # move to cpu - to not block 
        metrics = jax.tree_util.tree_map(lambda x: float(x), metrics)
        metrics["tokens_consumed"] = self.tokens_consumed
        metrics["tokens_per_second"] = self.tokens_per_batch / metrics["step_time"]
        metrics["tokens_per_second_per_device"] = metrics["tokens_per_second"] / self.num_devices
        metrics["mfu"] = ((self.num_flops_per_token * metrics["tokens_per_second"]) / self.xpu_peak_flops) * 100
        self._pretty_print(metrics, step)
        self.wandb_run.log(metrics, step=step)

    def log(self, metrics):
        self.prev_metrics, log_metrics = metrics, self.prev_metrics 
        if log_metrics:
            self._log(log_metrics)
        self.tokens_consumed += self.tokens_per_batch

    def flush(self):
        if self.prev_metrics:
            self._log(self.prev_metrics)
