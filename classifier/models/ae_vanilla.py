from typing import Callable, List, Union, Type, TypeVar, Optional, NamedTuple

import torch
from torch import Tensor
import torch.nn as nn

from ..config import EClrConfig
from .ae import Encoder, Decoder, AutoEncoder, VAE, AE

C = TypeVar("C", bound=Callable)


def conv2d_act(
    inp_channel_num: int,
    out_channel_num: int,
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
    activation_fn: Optional[Type[C]] = nn.ReLU,
) -> List:
    conv = [
        nn.Conv2d(
            inp_channel_num,
            out_channel_num,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        ),
        nn.BatchNorm2d(out_channel_num),
    ]
    if activation_fn is not None:
        conv += [activation_fn()]
    return conv


def conv2dtr_act(
    inp_channel_num: int,
    out_channel_num: int,
    kernel_size: int = 3,
    output_padding: int = 1,
    padding: int = 1,
    stride: int = 2,
    activation_fn: Optional[Type[C]] = nn.ReLU,
) -> List:
    conv = [
        nn.ConvTranspose2d(
            inp_channel_num,
            out_channel_num,
            kernel_size=kernel_size,
            output_padding=output_padding,
            padding=padding,
            stride=stride,
        ),
        nn.BatchNorm2d(out_channel_num),
    ]
    if activation_fn is not None:
        conv += [activation_fn()]
    return conv


class VanillaEncoder(Encoder):
    def __init__(
        self,
        channels_num_lst: List[int],
        latent_dim: int,
        inp_shape: tuple = (32, 32),
        activation_fn: Type[C] = nn.ReLU,
        is_vae: bool = False,
    ):
        """
        Args:
            - channels_num_lst: Number of input channels for all conv layers
            - latent_dim: Dimensionality of latent representation
            - inp_shape: Spatial sizes of input images
            - activation_fn: Activation function used throughout the encoder network
        """
        super().__init__()
        assert (
            len(channels_num_lst) == 6
        ), f"Given channels_num_lst len {len(channels_num_lst)} is not supported!"
        self.channels_num_lst = channels_num_lst
        self.latent_dim = latent_dim
        self.inp_shape = inp_shape
        self.activation_fn = activation_fn
        self.is_vae = is_vae
        self.spatial_div = 2**3
        ft = self._get_convs_stack() + [nn.Flatten()]
        self.feature_extractor = nn.Sequential(*ft)
        if is_vae:
            self.layer_mu = nn.Linear(
                self.inp_shape[0]
                * self.inp_shape[1]
                * channels_num_lst[5]
                // (self.spatial_div**2),
                latent_dim,
            )
            self.layer_log_var = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.inp_shape[0]
                    * self.inp_shape[1]
                    * channels_num_lst[5]
                    // (self.spatial_div**2),
                    latent_dim,
                ),
            )
        else:
            self.fc = nn.Linear(
                self.inp_shape[0]
                * self.inp_shape[1]
                * channels_num_lst[5]
                // (self.spatial_div**2),
                latent_dim,
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

    def _get_convs_stack(self) -> List:
        convs = conv2d_act(
            self.channels_num_lst[0],
            self.channels_num_lst[1],
            kernel_size=3,
            padding=1,
            stride=2,
            activation_fn=self.activation_fn,
        )  # 32x32 => 16x16
        convs += conv2d_act(
            self.channels_num_lst[1],
            self.channels_num_lst[2],
            kernel_size=3,
            padding=1,
            stride=1,
            activation_fn=self.activation_fn,
        )
        convs += conv2d_act(
            self.channels_num_lst[2],
            self.channels_num_lst[3],
            kernel_size=3,
            padding=1,
            stride=2,
            activation_fn=self.activation_fn,
        )  # 16x16 => 8x8
        convs += conv2d_act(
            self.channels_num_lst[3],
            self.channels_num_lst[4],
            kernel_size=3,
            padding=1,
            stride=1,
            activation_fn=self.activation_fn,
        )
        convs += conv2d_act(
            self.channels_num_lst[4],
            self.channels_num_lst[5],
            kernel_size=3,
            padding=1,
            stride=2,
            activation_fn=self.activation_fn,
        )  # 8x8 => 4x4
        return convs


class VanillaDecoder(Decoder):
    def __init__(
        self,
        channels_num_lst: List[int],
        latent_dim: int,
        spatial_div: int,
        inp_shape: tuple = (32, 32),
        activation_fn: Type[C] = nn.ReLU,
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
        assert (
            len(channels_num_lst) == 6
        ), f"Given channels_num_lst len {len(channels_num_lst)} is not supported!"
        self.channels_num_lst = channels_num_lst
        self.latent_dim = latent_dim
        self.spatial_div = spatial_div
        self.inp_shape = inp_shape
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn
        self.is_vae = is_vae

        self.linear = nn.Sequential(
            nn.Linear(
                latent_dim,
                self.inp_shape[0]
                * self.inp_shape[1]
                * channels_num_lst[5]
                // (self.spatial_div**2),
            ),
            activation_fn(),
        )
        self.convs = nn.Sequential(*self._get_convs_stack())
        if is_vae:
            self.conv_mu = nn.Sequential(
                *conv2dtr_act(
                    self.channels_num_lst[1],
                    self.channels_num_lst[0],
                    activation_fn=self.output_activation_fn,
                )
            )
            # self.conv_log_std = nn.Sequential(
            #     *conv2dtr_act(
            #         self.channels_num_lst[1],
            #         self.channels_num_lst[0],
            #         activation_fn=None,
            #     )
            # )
        else:
            self.conv_out = nn.Sequential(
                *conv2dtr_act(
                    self.channels_num_lst[1],
                    self.channels_num_lst[0],
                    activation_fn=self.output_activation_fn,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        out = out.reshape(
            x.shape[0],
            -1,
            self.inp_shape[0] // self.spatial_div,
            self.inp_shape[1] // self.spatial_div,
        )
        out = self.convs(out)
        if self.is_vae:
            return self._vae_out(out)
        else:
            return self._ae_out(out)

    def _ae_out(self, x: Tensor) -> Tensor:
        return self.conv_out(x)

    def _vae_out(self, x: Tensor) -> Tensor:
        # mu = self.conv_mu(out)
        # log_std = self.conv_log_std(out)
        return self.conv_mu(x)

    def _get_convs_stack(self) -> List:
        convs = conv2dtr_act(
            self.channels_num_lst[5],
            self.channels_num_lst[4],
            activation_fn=self.activation_fn,
        )  # 4x4 => 8x8
        convs += conv2d_act(
            self.channels_num_lst[4],
            self.channels_num_lst[3],
            kernel_size=3,
            padding=1,
            stride=1,
            activation_fn=self.activation_fn,
        )
        convs += conv2dtr_act(
            self.channels_num_lst[3],
            self.channels_num_lst[2],
            activation_fn=self.activation_fn,
        )  # 8x8 => 16x16
        convs += conv2d_act(
            self.channels_num_lst[2],
            self.channels_num_lst[1],
            kernel_size=3,
            padding=1,
            stride=1,
            activation_fn=self.activation_fn,
        )
        # convs += conv2dtr_act(
        #     self.channels_num_lst[1],
        #     self.channels_num_lst[0],
        #     activation_fn=self.output_activation_fn,
        # )  # 16x16 => 32x32
        return convs


def ae_from_config(config: EClrConfig) -> AE:
    encoder = VanillaEncoder(
        channels_num_lst=config.ae_channels_num_lst,
        latent_dim=config.latent_dim,
        inp_shape=config.image_size,
        activation_fn=config.encoder_activation_fn,
        is_vae=config.is_vae,
    )
    decoder = VanillaDecoder(
        channels_num_lst=config.ae_channels_num_lst,
        latent_dim=config.latent_dim,
        spatial_div=encoder.spatial_div,
        inp_shape=config.image_size,
        activation_fn=config.decoder_activation_fn,
        output_activation_fn=config.decoder_out_activation_fn,
        is_vae=config.is_vae,
    )
    return VAE(encoder, decoder) if config.is_vae else AutoEncoder(encoder, decoder)
