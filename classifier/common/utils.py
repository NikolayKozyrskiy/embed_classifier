import numpy as np
import torch
from torch.optim import Optimizer
from matches.loop import IterationType, Loop


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


def tensor_to_image(tensor: torch.Tensor, keepdim: bool = False) -> "np.ndarray":
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(tensor)}")

    if len(tensor.shape) > 4 or len(tensor.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional tensor")

    input_shape = tensor.shape
    image: "np.ndarray" = tensor.cpu().detach().numpy()

    if len(input_shape) == 2:
        # (H, W) -> (H, W)
        pass
    elif len(input_shape) == 3:
        # (C, H, W) -> (H, W, C)
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be (H,W)
            image = image.squeeze()
        else:
            image = image.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        image = image.transpose(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(f"Cannot process tensor with shape {input_shape}")

    image = (image * 255).clip(0, 255).astype(np.uint8)
    return image
