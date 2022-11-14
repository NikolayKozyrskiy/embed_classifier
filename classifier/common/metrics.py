import torch
from torch import Tensor


def psnr(x: Tensor, y: Tensor, max_val: float = 1.0) -> Tensor:
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3), keepdim=True)
    return 10 * torch.log10(max_val**2 / mse)
