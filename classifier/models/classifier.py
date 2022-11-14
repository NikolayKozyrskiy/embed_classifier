from typing import Callable, List, Type, TypeVar, Optional

from torch import Tensor
import torch.nn as nn

from ..config import EClrConfig


C = TypeVar("C", bound=Callable)


class MPL(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_size_lst: List[int],
        classes_num: int = 10,
        activation_fn: Type[C] = nn.ReLU,
        output_activation_fn: Optional[Type[C]] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_size_lst = feature_size_lst
        self.classes_num = classes_num

        mlp = [
            nn.Flatten(),
            nn.Linear(latent_dim, feature_size_lst[0]),
            activation_fn(),
            nn.Linear(feature_size_lst[1], feature_size_lst[2]),
            activation_fn(),
            nn.Linear(feature_size_lst[2], classes_num),
        ]
        if output_activation_fn is not None:
            mlp += [output_activation_fn()]

        self.mlp = nn.Sequential(*mlp)

    def forward(self, inp: Tensor) -> Tensor:
        return self.mlp(inp)
