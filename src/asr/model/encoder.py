"""ASR encoder: a strided CNN whose last blocks feed per-scale modelers.

Each *latent* block emits a grid of primitives through a
:class:`~asr.model.modeler.DenseModeler`. With the default 256x256 input the
three latent blocks yield grids of 8x8, 4x4 and 2x2 primitives (84 in total)
with grid spacings of 32, 64 and 128 px. Scale indices follow encoder order:
**scale 0 is the finest grid (smallest primitives)**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor, nn

from asr.config import ModelerConfig
from asr.model.modeler import DenseModeler
from asr.model.receptive_field import LayerGeometry, image_info, propagate


@dataclass(frozen=True)
class ScaleInfo:
    """Geometry of one primitive scale, as needed by the renderer.

    Attributes:
        width: Number of primitives horizontally.
        height: Number of primitives vertically.
        rj: Rendering radius ``r_j = floor(receptive_field / 2)``; each
            primitive is rasterised in a ``(2 r_j + 1)²`` window.
        s: Distance in pixels between neighbouring primitives (effective stride).
    """

    width: int
    height: int
    rj: int
    s: int


class EncoderOutput(NamedTuple):
    """Result of an encoder forward pass.

    Attributes:
        latents: One tensor of shape ``(B, 6, N_j)`` per scale ``j``.
        background: Background RGB colour, shape ``(B, 3)``.
    """

    latents: list[Tensor]
    background: Tensor


class Conv2dBlock(nn.Module):
    """Conv → (MaxPool) → ReLU → (BatchNorm), optionally followed by a modeler.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        pool: Insert a 3x3/2 max-pool after the convolution.
        batch_norm: Apply batch normalisation after the activation.
        get_latent: Whether this block feeds a modeler (i.e. is a scale).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        pool: bool = False,
        batch_norm: bool = True,
        get_latent: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.get_latent = get_latent
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.xavier_uniform_(self.conv2d.weight, gain=nn.init.calculate_gain("relu"))
        if self.conv2d.bias is not None:
            nn.init.normal_(self.conv2d.bias)
        self.pool = pool
        if pool:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.modeler: DenseModeler | None = None

        self.geometry = [LayerGeometry(kernel_size, stride, padding)]
        if pool:
            self.geometry.append(LayerGeometry(3, 2, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block to ``x`` of shape ``(B, C_in, H, W)``."""
        out = self.conv2d(x)
        if self.pool:
            out = self.pooling(out)
        out = self.relu(out)
        if self.batch_norm:
            out = self.norm(out)
        return out


class BackgroundBlock(nn.Module):
    """Predicts a single RGB background colour from the last feature map.

    Args:
        in_features: Flattened size of the last feature map.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, 3)
        self.activation = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("sigmoid"))
        nn.init.ones_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, C, H, W)`` features to a ``(B, 3)`` colour in ``[0, 1]``."""
        return self.activation(self.linear(self.flatten(x)))


def _default_blocks(c: int) -> list[Conv2dBlock]:
    """The architecture used in the paper (base channel width ``c``)."""
    spec = [  # (in, out, stride, latent)
        (3, c, 2, False),
        (c, 2 * c, 2, False),
        (2 * c, 4 * c, 2, False),
        (4 * c, 4 * c, 1, False),
        (4 * c, 8 * c, 2, False),
        (8 * c, 8 * c, 2, True),
        (8 * c, 16 * c, 2, True),
        (16 * c, 16 * c, 2, True),
    ]
    return [
        Conv2dBlock(i, o, kernel_size=3, stride=s, padding=1, get_latent=lat)
        for i, o, s, lat in spec
    ]


class Encoder(nn.Module):
    """Multi-scale encoder producing primitive parameters and a background colour.

    Args:
        input_shape: Input shape ``(C, H, W)``.
        modelers_configs: Incremental schedule per scale index; scales without
            an entry are not frozen.
        base_channels: Channel width of the first block.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (3, 256, 256),
        modelers_configs: dict[int, ModelerConfig] | None = None,
        base_channels: int = 8,
    ) -> None:
        super().__init__()
        modelers_configs = modelers_configs or {}
        self.input_shape = input_shape
        self.layers = nn.ModuleList(_default_blocks(base_channels))

        self.zs_info: list[ScaleInfo] = []
        info = image_info(input_shape[1], input_shape[2])
        for block in self.layers:
            assert isinstance(block, Conv2dBlock)
            info = propagate(block.geometry, info)
            if block.get_latent:
                scale = len(self.zs_info)
                block.modeler = DenseModeler(
                    block.out_channels, config=modelers_configs.get(scale, ModelerConfig())
                )
                self.zs_info.append(
                    ScaleInfo(
                        width=info.width,
                        height=info.height,
                        rj=info.receptive_field // 2,
                        s=info.stride,
                    )
                )
        last = self.layers[-1]
        assert isinstance(last, Conv2dBlock)
        self.backgroundBlock = BackgroundBlock(last.out_channels * info.height * info.width)

    @property
    def n_scales(self) -> int:
        """Number of primitive scales."""
        return len(self.zs_info)

    @property
    def modelers(self) -> list[DenseModeler]:
        """Modelers in scale order."""
        return [
            b.modeler for b in self.layers if isinstance(b, Conv2dBlock) and b.modeler is not None
        ]

    def set_epoch(self, epoch: int) -> None:
        """Propagate the current epoch to all modelers (incremental schedule)."""
        for m in self.modelers:
            m.set_epoch(epoch)

    def forward(self, x: Tensor) -> EncoderOutput:
        """Encode a batch of images.

        Args:
            x: Images of shape ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns:
            Per-scale primitive parameters and the background colour.
        """
        latents: list[Tensor] = []
        for block in self.layers:
            x = block(x)
            if isinstance(block, Conv2dBlock) and block.modeler is not None:
                latents.append(block.modeler(x))
        return EncoderOutput(latents=latents, background=self.backgroundBlock(x))
