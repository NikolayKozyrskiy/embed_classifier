from pathlib import Path
from typing import Optional

from matches.shortcuts.optimizer import (
    LRSchedulerProto,
    LRSchedulerWrapper,
    SchedulerScopeType,
)
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from classifier.config import EClrConfig, DatasetName, TrainSetup
from classifier.transforms import train_basic_augs
from classifier.common.vis import images_gt, images_reconstructed


class Config(EClrConfig):
    def optimizer(self, model: nn.Module):
        # return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        return Adam(model.parameters(), lr=self.lr)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            # scope_type=SchedulerScopeType.BATCH,
            scope_type=SchedulerScopeType.EPOCH,
        )

    # def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
    #     return LRSchedulerWrapper(
    #         ReduceLROnPlateau(
    #             optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
    #         ),
    #         scope_type=SchedulerScopeType.EPOCH,
    #     )


config = Config(
    comment="train_ae",
    data_root="_data",
    dataset_name=DatasetName.CIFAR10,
    train_setup=TrainSetup.AE,
    loss_aggregation_weigths={"ae/mse": 1.0},
    monitor="valid/ae/mse",
    metrics=["ae/psnr"],
    ae_channels_num_lst=[3, 16, 32, 32, 64, 64],
    latent_dim=128,
    encoder_activation_fn=nn.ReLU,
    decoder_activation_fn=nn.ReLU,
    decoder_out_activation_fn=nn.Sigmoid,  # nn.Tanh, None
    ae_checkpoint_path=None,
    # mlp_feature_size_lst=[256, 256, 128],
    # mlp_activation_fn=nn.ReLU,
    # mlp_output_activation_fn=None,
    # classifier_checkpoint_path=None,
    batch_size=200,
    lr=1e-3,
    max_epoch=150,
    train_transforms=[],  # train_basic_augs(),
    valid_transforms=[],
    data_loader_workers=8,
    single_pass_length=1.0,
    shuffle_train=True,
    output_config=[],
    preview_image_fns=[images_gt, images_reconstructed],
    image_size=(32, 32),
)