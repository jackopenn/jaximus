
import optax
from sws import Config
from model_configs import get_gpt2_config

def get_config():
    cfg = Config()
    cfg.seed = 42
    cfg.exp_name = "gpt2-base"

    cfg.model = get_gpt2_config()

    cfg.data.hf_name = ["HuggingFaceFW/fineweb-edu", "sample-10BT"]
    cfg.data.tokenizer_name = "gpt2"
    cfg.data.max_length = 1024
    cfg.data.batch_size = 2

    cfg.optim.name = "adamw"
    cfg.optim.weight_decay = 0.1
    cfg.optim.betas = (0.9, 0.95)
    cfg.optim.grad_clip = 1.0
    cfg.optim.batch_size = lambda: cfg.data.batch_size
    cfg.optim.accum_steps = 1
    cfg.optim.eps = 1e-8
    cfg.optim.schedule.init_value = 0.0
    cfg.optim.schedule.peak_value = 1e-3
    cfg.optim.schedule.warmup_steps = 700
    cfg.optim.schedule.decay_steps = lambda: cfg.max_steps - cfg.optim.schedule.warmup_steps
    cfg.optim.schedule.end_value = 6e-5

    cfg.parallel.data_parallel = 1
    cfg.parallel.zero_stage = 3
    
    
    cfg.max_steps = 60_000
    cfg.generate_every = 500
    cfg.eval_every = -1
    cfg.save_every = 5000
    cfg.save_dir = "checkpoints"
    cfg.profile_dir= "profile"
    cfg.start_trace_micro_step = 10
    cfg.end_trace_micro_step = 20
    cfg.gpu = "H100"

    return cfg
