from concurrent.futures import Executor
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
from matches.shortcuts.optimizer import (
    LRSchedulerProto,
    LRSchedulerWrapper,
    SchedulerScopeType,
)

if TYPE_CHECKING:
    from .pipeline import EmbedClassifierPipeline


C = TypeVar("C", bound=Callable)


class DatasetName(str, Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


class TrainSetup(str, Enum):
    AE = "ae_only"
    CLR = "classifier_only"
    E2E = "end_to_end"


class EClrConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    data_root: str
    dataset_name: DatasetName = DatasetName.CIFAR10
    train_setup: TrainSetup = TrainSetup.AE

    loss_aggregation_weigths: Dict[str, float]
    metrics: List[str]

    ae_channels_num_lst: List[int] = [3, 16, 32, 32, 32, 32]
    latent_dim: int = 128
    encoder_activation_fn: Optional[Type[C]] = nn.ReLU
    decoder_activation_fn: Optional[Type[C]] = nn.ReLU
    decoder_out_activation_fn: Optional[Type[C]] = nn.Tanh
    ae_checkpoint_path: Optional[Union[Path, str]] = None

    mlp_feature_size_lst: List[int] = [256, 256, 128]
    mlp_activation_fn: Type[C] = nn.ReLU
    mlp_output_activation_fn: Optional[Type[C]] = None
    classifier_checkpoint_path: Optional[Union[Path, str]] = None

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
    output_config: list[
        Callable[["EmbedClassifierPipeline", Executor, Path], None]
    ] = []
    preview_image_fns: List[Callable] = []

    image_size = (32, 32)

    def optimizer(self, model: Module):
        return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.BATCH,
        )

    def resume(self, loop: Loop):
        if self.resume_from_checkpoint is not None:
            loop.state_manager.read_state(
                self.resume_from_checkpoint,
                skip_keys=[
                    "ae_scheduler",
                    "classifier_scheduler",
                ],
            )

    def postprocess(self, loop: Loop, pipeline: "EmbedClassifierPipeline"):
        pass


def pipeline_from_config():
    return None


def load_config(config_path: Path, desired_class):
    text = config_path.read_text()

    ctx = {}
    exec(text, ctx)

    config = ctx["config"]

    assert isinstance(config, desired_class), (config.__class__, desired_class)
    return config
