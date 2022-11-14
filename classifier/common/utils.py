import torch
from matches.loop import IterationType, Loop
from torch.optim import Optimizer


def log_optimizer_lrs(
    loop: Loop,
    optimizer: Optimizer,
    prefix: str = "lr",
    iteration: IterationType = IterationType.AUTO,
):
    for i, g in enumerate(optimizer.param_groups):
        loop.metrics.log(f"{prefix}/group_{i}", g["lr"], iteration)


def enumerate_normalized(it, len):
    for i, e in enumerate(it):
        yield i / len, e
