from pathlib import Path
from typing import Optional

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
from classifier.common.vis import (
    images_gt,
    images_reconstructed,
    log_embeddings_to_tb_batch,
    log_generated_images_to_tb_batch,
    log_images_to_tb_batch,
)


class Config(EClrConfig):
    def optimizer(self, model: nn.Module):
        # return SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        return Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

    def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
        return LRSchedulerWrapper(
            CosineAnnealingLR(optimizer, T_max=self.max_epoch),
            # scope_type=SchedulerScopeType.BATCH,
            scope_type=SchedulerScopeType.EPOCH,
        )

    # def scheduler(self, optimizer: Optimizer) -> Optional[LRSchedulerProto]:
    #     return LRSchedulerWrapper(
    #         StepLR(optimizer, step_size=15, gamma=0.2),
    #         scope_type=SchedulerScopeType.EPOCH,
    #     )


config = Config(
    comment="train_ae",
    data_root="_data",
    dataset_name=DatasetName.CIFAR10,
    train_setup=TrainSetup.AE,
    loss_aggregation_weigths={"recon/mse": 1.0},
    monitor="valid/recon/mse",
    metrics=["recon/psnr"],
    ae_architecture=AEArchitecture.VANILLA,
    is_vae=False,
    ae_channels_num_lst=[3, 32, 64, 128, 256, 256],
    latent_dim=64,
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
    max_epoch=100,
    train_transforms=[],  # train_basic_augs(),
    valid_transforms=[],
    data_loader_workers=8,
    single_pass_length=1.0,
    shuffle_train=True,
    output_config=[],
    preview_image_fns=[images_gt, images_reconstructed],
    log_vis_fns=[
        log_images_to_tb_batch,
        log_generated_images_to_tb_batch,
        log_embeddings_to_tb_batch,
    ],
    image_size=(32, 32),
)
