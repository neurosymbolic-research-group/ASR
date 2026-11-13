"""Conventional convolutional autoencoder used as the paper's *Baseline*.

The encoder maps a 256x256 image to a global latent vector (paper: 200-d)
through five conv/ELU/BatchNorm/max-pool blocks and a dense layer; the decoder
mirrors it with nearest-neighbour upsampling.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn


class EncodingBlock(nn.Module):
    """``[Conv3x3 → act → BatchNorm]*`` followed by 2x2 max pooling.

    Args:
        filters: ``(in, out)`` channel pairs, one per convolution.
        activation: Factory of the activation module.
    """

    def __init__(
        self, filters: list[tuple[int, int]], activation: Callable[[], nn.Module] = nn.ELU
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for c_in, c_out in filters:
            modules += [nn.Conv2d(c_in, c_out, 3, padding=1), activation(), nn.BatchNorm2d(c_out)]
        modules.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block (halves the spatial size)."""
        return self.block(x)


class DecodingBlock(nn.Module):
    """2x nearest-neighbour upsampling followed by ``[Conv3x3 → act → BatchNorm]*``.

    Args:
        filters: ``(in, out)`` channel pairs, one per convolution.
        activation: Factory of the activation module.
    """

    def __init__(
        self, filters: list[tuple[int, int]], activation: Callable[[], nn.Module] = nn.ReLU
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.UpsamplingNearest2d(scale_factor=2)]
        for c_in, c_out in filters:
            modules += [nn.Conv2d(c_in, c_out, 3, padding=1), activation(), nn.BatchNorm2d(c_out)]
        self.block = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block (doubles the spatial size)."""
        return self.block(x)


class AEEncoder(nn.Module):
    """Encoder ``(B, 3, 256, 256) → (B, latent_dim)`` with sigmoid output.

    Args:
        in_channels: Image channels.
        out_channels: Base channel width ``c``.
        latent_dim: Latent size.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16, latent_dim: int = 200):
        super().__init__()
        c = out_channels
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            EncodingBlock([(in_channels, 2 * c)]),  # 128
            EncodingBlock([(2 * c, 2 * c)]),  # 64
            EncodingBlock([(2 * c, 4 * c)]),  # 32
            EncodingBlock([(4 * c, 8 * c)]),  # 16
            EncodingBlock([(8 * c, 8 * c)]),  # 8
            nn.Flatten(),
            nn.Linear(8 * c * 8 * 8, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode images to latent vectors."""
        return self.net(x)


class AEDecoder(nn.Module):
    """Decoder ``(B, latent_dim) → (B, 3, 256, 256)`` with sigmoid output.

    Args:
        in_channels: Image channels.
        out_channels: Base channel width ``c``.
        latent_dim: Latent size.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 16, latent_dim: int = 200):
        super().__init__()
        c = out_channels
        self.out_channels = c
        self.linear = nn.Sequential(nn.Linear(latent_dim, 8 * c * 8 * 8), nn.ReLU())
        self.conv = nn.Sequential(
            DecodingBlock([(8 * c, 8 * c)]),  # 16
            DecodingBlock([(8 * c, 4 * c)]),  # 32
            DecodingBlock([(4 * c, 2 * c)]),  # 64
            DecodingBlock([(2 * c, c)]),  # 128
            DecodingBlock([(c, in_channels)]),  # 256
        )
        self.sig = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent vectors to images."""
        x = self.linear(z).view(-1, 8 * self.out_channels, 8, 8)
        return self.sig(self.conv(x))


class Autoencoder(nn.Module):
    """Encoder + decoder pair.

    Args:
        latent_dim: Latent size.
        base_channels: Base channel width.
    """

    def __init__(self, latent_dim: int = 200, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder = AEEncoder(out_channels=base_channels, latent_dim=latent_dim)
        self.decoder = AEDecoder(out_channels=base_channels, latent_dim=latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct ``x``."""
        return self.decoder(self.encoder(x))
