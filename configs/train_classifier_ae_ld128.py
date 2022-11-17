from pathlib import Path
from typing import Optional, TYPE_CHECKING

from matches.loop import Loop
from matches.shortcuts.optimizer import (
    LRSchedulerProto,
    LRSchedulerWrapper,
    SchedulerScopeType,
)
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from classifier.config import EClrConfig, DatasetName, TrainSetup, AEArchitecture
from classifier.transforms import train_basic_augs
from classifier.common.vis import images_gt, images_reconstructed

if TYPE_CHECKING:
    from classifier.pipeline import EmbedClassifierPipeline


class Config(EClrConfig):
    def optimizer(self, model: nn.Module):
        # return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-3)
        return Adam(model.parameters(), lr=self.lr)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.EPOCH,
        )

    # def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
    #     return LRSchedulerWrapper(
    #         StepLR(optimizer, step_size=25, gamma=0.5),
    #         scope_type=SchedulerScopeType.EPOCH,
    #     )

    def postprocess(self, loop: Loop, pipeline: "EmbedClassifierPipeline") -> None:
        loop.state_manager.read_state(
            self.ae_checkpoint_path,
            skip_keys=[
                "ae_scheduler",
                "ae_optimizer",
                "classifier_model",
                "classifier_optimizer",
            ],
        )


config = Config(
    comment="train_classifier",
    data_root="_data",
    dataset_name=DatasetName.CIFAR10,
    train_setup=TrainSetup.CLR,
    loss_aggregation_weigths={"clr/cross_entropy": 1.0},
    monitor="valid/clr/accuracy",
    metrics=["clr/accuracy"],
    ae_architecture=AEArchitecture.VANILLA,
    is_vae=False,
    ae_channels_num_lst=[3, 32, 64, 128, 256, 256],
    latent_dim=128,
    encoder_activation_fn=nn.ReLU,
    decoder_activation_fn=nn.ReLU,
    decoder_out_activation_fn=nn.Sigmoid,  # nn.Tanh, None
    ae_checkpoint_path="logs/ae_ld128_v3/221117_0234/last.pth",
    # ae_checkpoint_path="logs/3_ae/221115_2248/best.pth",
    mlp_feature_size_lst=[128, 64],
    mlp_activation_fn=nn.ReLU,
    mlp_output_activation_fn=None,
    classifier_checkpoint_path=None,
    batch_size=200,
    lr=1e-3,
    max_epoch=50,
    train_transforms=[],  # train_basic_augs(),
    valid_transforms=[],
    data_loader_workers=8,
    single_pass_length=1.0,
    shuffle_train=True,
    output_config=[],
    preview_image_fns=[],
    image_size=(32, 32),
)
