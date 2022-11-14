from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    TypeVar,
    Type,
)

import ignite.distributed as idist
from pydantic import BaseModel, Field
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from matches.loop import Loop
from matches.shortcuts.optimizer import LRSchedulerProto


C = TypeVar("C", bound=Callable)


class DatasetName(str, Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


class EClrConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    data_root: str
    dataset_name: DatasetName = DatasetName.CIFAR10

    channels_num_lst: List[int] = [3, 16, 32, 32, 32, 32]
    latent_dim: int = 128
    encoder_activation_fn: Optional[Type[C]] = nn.ReLU
    decoder_activation_fn: Optional[Type[C]] = nn.ReLU
    decoder_out_activation_fn: Optional[Type[C]] = nn.Tanh

    batch_size: int = 200
    lr: float = 1e-2
    max_epoch: int = 100
    train_transforms: list[Callable] = []
    valid_transforms: list[Callable] = []

    comment: Optional[str] = "default_comment"
    data_loader_workers: int = 4
    single_pass_length: float = 1.0
    monitor = "valid/accuracy"
    resume_from_checkpoint: Optional[Path] = None
    shuffle_train: bool = True

    image_size = (32, 32)

    def optimizer(self, model: Module):
        return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return CosineAnnealingLR(optimizer, T_max=self.max_epoch)

    def resume(self, loop: Loop):
        if self.resume_from_checkpoint is not None:
            loop.state_manager.read_state(
                self.resume_from_checkpoint, skip_keys=["scheduler"]
            )


def pipeline_from_config():
    return None


def load_config(config_path: Path, desired_class):
    text = config_path.read_text()

    ctx = {}
    exec(text, ctx)

    config = ctx["config"]

    assert isinstance(config, desired_class), (config.__class__, desired_class)
    return config
