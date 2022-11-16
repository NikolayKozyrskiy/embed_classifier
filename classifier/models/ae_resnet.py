from typing import Callable, List, Union, Type, TypeVar, Optional

import torch
from torch import Tensor
import torch.nn as nn

from ..config import EClrConfig
from .ae import Encoder, Decoder, AutoEncoder, VAE, AE
from .resnet import resnet18_encoder, resnet18_decoder

C = TypeVar("C", bound=Callable)


class ResnetEncoder(Encoder):
    def __init__(
        self,
        latent_dim: int,
        inp_shape: tuple = (32, 32),
        is_vae: bool = False,
    ):
        """
        Args:
            - latent_dim: Dimensionality of latent representation
            - inp_shape: Spatial sizes of input images
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.inp_shape = inp_shape
        self.is_vae = is_vae
        self.enc_out_dim = 512
        self.feature_extractor = resnet18_encoder(first_conv=False, maxpool1=False)
        if is_vae:
            self.layer_mu = nn.Linear(
                self.enc_out_dim,
                self.latent_dim,
            )
            self.layer_log_var = nn.Linear(
                self.enc_out_dim,
                self.latent_dim,
            )
        else:
            self.fc = nn.Linear(
                self.enc_out_dim,
                self.latent_dim,
            )

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        if self.is_vae:
            return self._vae_out(features)
        else:
            return self._ae_out(features)

    def _ae_out(self, features: Tensor) -> Tensor:
        return self.fc(features)

    def _vae_out(self, features: Tensor) -> Tensor:
        return self.layer_mu(features), self.layer_log_var(features)


class ResnetDecoder(Decoder):
    def __init__(
        self,
        latent_dim: int,
        inp_shape: tuple = (32, 32),
        output_activation_fn: Optional[Type[C]] = nn.Tanh,
        is_vae: bool = False,
    ):
        """
        Args:
            - channels_num_lst: Number of input channels for all conv layers
            - latent_dim: Dimensionality of latent representation
            - spatial_div: Divisor of spatial size
            - inp_shape: Spatial sizes of input images
            - activation_fn: Activation function used throughout the encoder network
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.inp_shape = inp_shape
        self.output_activation_fn = output_activation_fn
        self.is_vae = is_vae

        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=self.inp_shape[0],
            first_conv=False,
            maxpool1=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


def ae_from_config(config: EClrConfig) -> AE:
    encoder = ResnetEncoder(
        latent_dim=config.latent_dim,
        inp_shape=config.image_size,
        is_vae=config.is_vae,
    )
    decoder = ResnetDecoder(
        latent_dim=config.latent_dim,
        inp_shape=config.image_size,
        output_activation_fn=config.decoder_out_activation_fn,
        is_vae=config.is_vae,
    )
    return VAE(encoder, decoder) if config.is_vae else AutoEncoder(encoder, decoder)
