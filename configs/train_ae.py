from pathlib import Path
from typing import Optional

from matches.shortcuts.optimizer import (
    LRSchedulerProto,
    LRSchedulerWrapper,
    SchedulerScopeType,
)
from torch import nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from classifier.config import EClrConfig, DatasetName, TrainSetup
from classifier.transforms import train_basic_augs


class Config(EClrConfig):
    def optimizer(self, model: nn.Module):
        return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            scope_type=SchedulerScopeType.BATCH,
        )


config = Config(
    comment="debug_train_ae",
    data_root="_data",
    dataset_name=DatasetName.CIFAR10,
    train_setup=TrainSetup.AE,
    loss_aggregation_weigths={"ae/mse": 1.0},
    monitor="valid/ae/mse",
    metrics=["ae/psnr"],
    channels_num_lst=[3, 16, 32, 32, 32, 32],
    latent_dim=128,
    encoder_activation_fn=nn.ReLU,
    decoder_activation_fn=nn.ReLU,
    decoder_out_activation_fn=nn.Tanh,  # nn.Tanh, None
    ae_checkpoint_path=None,
    # mlp_feature_size_lst=[256, 256, 128],
    # mlp_activation_fn=nn.ReLU,
    # mlp_output_activation_fn=None,
    # classifier_checkpoint_path=None,
    batch_size=3,
    lr=1e-2,
    max_epoch=100,
    train_transforms=train_basic_augs(),
    valid_transforms=[],
    data_loader_workers=10,
    single_pass_length=1.0,
    shuffle_train=True,
    output_config=[],
    image_size=(32, 32),
)
