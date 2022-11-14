from typing import Callable, List, Tuple, Type, TypeVar, Optional, NamedTuple

from torch import Tensor
import torch.nn as nn

from ..config import EClrConfig

C = TypeVar("C", bound=Callable)


class AutoEncoderOutput(NamedTuple):
    embeddings: Optional[Tensor]
    reconstructed_img: Optional[Tensor]


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
        )
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
        )
    ]
    if activation_fn is not None:
        conv += [activation_fn()]
    return conv


class Encoder(nn.Module):
    def __init__(
        self,
        channels_num_lst: List[int],
        latent_dim: int,
        inp_shape: tuple = (32, 32),
        activation_fn: Type[C] = nn.ReLU,
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
        self.spatial_div = 2**3
        encoder = self._get_convs_stack() + [
            nn.Flatten(),
            nn.Linear(
                self.inp_shape[0]
                * self.inp_shape[1]
                * channels_num_lst[5]
                // (self.spatial_div**2),
                latent_dim,
            ),
        ]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)

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


class Decoder(nn.Module):
    def __init__(
        self,
        channels_num_lst: List[int],
        latent_dim: int,
        spatial_div: int,
        inp_shape: tuple = (32, 32),
        activation_fn: Type[C] = nn.ReLU,
        output_activation_fn: Optional[Type[C]] = nn.Tanh,
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

    def forward(self, x):
        out = self.linear(x)
        out = out.reshape(
            x.shape[0],
            -1,
            self.inp_shape[0] // self.spatial_div,
            self.inp_shape[1] // self.spatial_div,
        )
        return self.convs(out)

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
        convs += conv2dtr_act(
            self.channels_num_lst[1],
            self.channels_num_lst[0],
            activation_fn=self.output_activation_fn,
        )  # 16x16 => 32x32
        return convs


class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def embeddings(self, inp: Tensor) -> AutoEncoderOutput:
        emb = self.encoder(inp)
        return AutoEncoderOutput(emb, None)

    def reconstruct(self, embeddings: Tensor) -> AutoEncoderOutput:
        recon_img = self.decoder(embeddings)
        return AutoEncoderOutput(None, recon_img)

    def forward(self, inp: Tensor) -> AutoEncoderOutput:
        embeddings = self.encoder(inp)
        out = self.decoder(embeddings)
        return AutoEncoderOutput(embeddings, out)


def ae_from_config(config: EClrConfig, device: str) -> AutoEncoder:
    encoder = Encoder(
        channels_num_lst=config.channels_num_lst,
        latent_dim=config.latent_dim,
        inp_shape=config.image_size,
        activation_fn=config.encoder_activation_fn,
    )
    decoder = Decoder(
        channels_num_lst=config.channels_num_lst,
        latent_dim=config.latent_dim,
        spatial_div=encoder.spatial_div,
        inp_shape=config.image_size,
        activation_fn=config.decoder_activation_fn,
        output_activation_fn=config.decoder_out_activation_fn,
    )
    return AutoEncoder(encoder, decoder).to(device)
