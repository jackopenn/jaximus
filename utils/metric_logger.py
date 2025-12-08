import jax
from utils.common import get_gpu_peak_flops

class MetricLogger:
    def __init__(self, batch_size, accum_steps, sequence_length, n_flops_per_token, gpu_name, optimizer_scheduler, wandb=None):
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.sequence_length = sequence_length
        self.n_flops_per_token = float(n_flops_per_token)
        self.gpu_name = gpu_name
        self.optimizer_scheduler = optimizer_scheduler
        self.wandb = wandb

        self.tokens_per_batch = batch_size * accum_steps * sequence_length
        self.num_devices = jax.device_count()
        self.gpu_peak_flops = get_gpu_peak_flops(gpu_name) * self.num_devices
        self.cpu_device = jax.devices("cpu")[0]
        # if isinstance(optimizer_scheduler, float):
        #     self.learning_rate = lambda step: optimizer_scheduler
        # else:
        #     # force on cpu to avoid blocking
        #     self.learning_rate = lambda step: optimizer_scheduler(step)

        self.prev_metrics = None
        self.tokens_consumed = 0


    def _human_format(self, num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    # https://github.com/huggingface/nanotron/blob/7bc9923285a03069ebffe994379a311aceaea546/src/nanotron/logging/base.py#L268
        if abs(num) < 1:
            return "{:.3g}".format(num)
        SIZES = ["", "K", "M", "B", "T", "P", "E"]
        num = float("{:.3g}".format(num))
        magnitude = 0
        i = 0
        while abs(num) >= 1000 and i < len(SIZES) - 1:
            magnitude += 1
            num /= 1000.0 if not divide_by_1024 else 1024.0
            i += 1
        return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])


    def _pretty_print(self, metrics, step):
        loss = metrics.pop("loss")
        print_string = f"step: {step}, loss: {self._human_format(loss)}"
        for k, v in metrics.items():
            print_string += f", {k}: {self._human_format(v)}"
        print(print_string)


    def log(self, metrics):
        self.prev_metrics, log_metrics = metrics, self.prev_metrics 
        self.tokens_consumed += self.tokens_per_batch

        if not log_metrics:
            return

        step = log_metrics.pop("step")
        # move to cpu - to not block 
        log_metrics = jax.tree_util.tree_map(lambda x: float(x), log_metrics)
        log_metrics["tokens_consumed"] = self.tokens_consumed
        log_metrics["tokens_per_second"] = self.tokens_per_batch / log_metrics["step_time"]
        log_metrics["tokens_per_second_per_device"] = log_metrics["tokens_per_second"] / self.num_devices
        log_metrics["mfu"] = ((self.n_flops_per_token * log_metrics["tokens_per_second"]) / self.gpu_peak_flops) * 100
        # log_metrics["lr"] = self.learning_rate(self.step)
        self._pretty_print(log_metrics, step)
        if self.wandb:
            self.wandb.log(log_metrics, step=step)
