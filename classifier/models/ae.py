from abc import ABC, abstractmethod
from typing import (
    Callable,
    List,
    Union,
    Type,
    TypeVar,
    Optional,
    NamedTuple,
)

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal

# from ..config import EClrConfig

C = TypeVar("C", bound=Callable)


class AEOutput(NamedTuple):
    embeddings: Optional[Tensor]
    reconstructed_img: Optional[Tensor]
    mu: Optional[Tensor]
    log_var: Optional[Tensor]


class Encoder(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(x: Tensor) -> Union[Tensor, List[Tensor]]:
        raise NotImplementedError


class Decoder(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(x: Tensor) -> Union[Tensor, List[Tensor]]:
        raise NotImplementedError


class AE(nn.Module, ABC):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def embeddings(self, inp: Tensor) -> AEOutput:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, embeddings: Tensor) -> AEOutput:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inp: Tensor) -> AEOutput:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, num_samples: int, device: str) -> Tensor:
        z = torch.randn((num_samples, self.decoder.latent_dim)).to(device)
        return self.decoder(z)


class AutoEncoder(AE):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__(encoder, decoder)

    def embeddings(self, inp: Tensor) -> AEOutput:
        emb = self.encoder(inp)
        return AEOutput(emb, None, None, None)

    def reconstruct(self, embeddings: Tensor) -> AEOutput:
        recon_img = self.decoder(embeddings)
        return AEOutput(None, recon_img, None, None)

    def forward(self, inp: Tensor) -> AEOutput:
        embeddings = self.encoder(inp)
        out = self.decoder(embeddings)
        return AEOutput(embeddings, out, None, None)


class VAE(AE):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__(encoder, decoder)

    def embeddings(self, inp: Tensor) -> AEOutput:
        mu, log_var = self.encoder(inp)
        z = self._sample_z(mu, log_var)
        return AEOutput(z, None, mu, log_var)

    def reconstruct(self, embeddings: Tensor) -> AEOutput:
        recon_mu = self.decoder(embeddings)
        # recon_img = Normal(recon_mu, recon_log_scale).sample()
        return AEOutput(None, recon_mu, None, None)

    def forward(self, inp: Tensor) -> AEOutput:
        mu, log_var = self.encoder(inp)
        z = self._sample_z(mu, log_var)
        recon_mu = self.decoder(z)
        return AEOutput(z, recon_mu, mu, log_var)

    def _sample_z(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(log_var / 2.0)
        return Normal(mu, std).rsample()


# def ae_from_config(config: EClrConfig) -> AE:
#     encoder = Encoder(
#         channels_num_lst=config.ae_channels_num_lst,
#         latent_dim=config.latent_dim,
#         inp_shape=config.image_size,
#         activation_fn=config.encoder_activation_fn,
#         is_vae=config.is_vae,
#     )
#     decoder = Decoder(
#         channels_num_lst=config.ae_channels_num_lst,
#         latent_dim=config.latent_dim,
#         spatial_div=encoder.spatial_div,
#         inp_shape=config.image_size,
#         activation_fn=config.decoder_activation_fn,
#         output_activation_fn=config.decoder_out_activation_fn,
#         is_vae=config.is_vae,
#     )
#     return VAE(encoder, decoder) if config.is_vae else AutoEncoder(encoder, decoder)
