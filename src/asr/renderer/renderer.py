"""Differentiable renderer: primitive parameters → RGB image."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from asr.model.encoder import ScaleInfo
from asr.renderer.compose import combine_primitives, merge_canvases
from asr.renderer.primitives import affine_transform, colorize, fuzzy_disc


class RenderOutput(NamedTuple):
    """Rendered batch.

    Attributes:
        images: ``(B, H, W, 3)`` rendered images (channel-last).
        canvases: Per-scale canvases of the first image (coarsest first) and
            its background, or ``None`` if not requested.
    """

    images: Tensor
    canvases: tuple[list[Tensor], Tensor] | None


class Renderer:
    """Render images from the per-scale primitive parameters of the encoder.

    For every image, each scale is rendered on its own white canvas
    (coarsest scale first), then all canvases are multiplied into a
    uniformly coloured background.

    Args:
        img_size: Output ``(H, W)``.
        zs_info: Per-scale grid geometry, see :attr:`asr.model.encoder.Encoder.zs_info`.
        beta: Fuzziness β of the primitives.
    """

    def __init__(self, img_size: tuple[int, int], zs_info: list[ScaleInfo], beta: float = 1.0):
        self.img_size = img_size
        self.zs_info = zs_info
        self.beta = beta
        # The prototype disc depends only on (rj, beta): cache it per scale.
        self._discs = {info.rj: fuzzy_disc(info.rj, beta) for info in zs_info}

    def render_one(
        self, latents: list[Tensor], background_color: Tensor
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        """Render a single image.

        Args:
            latents: Per-scale parameters of one image, each ``(6, N_j)``.
            background_color: ``(3,)`` RGB background.

        Returns:
            ``(image, canvases, background)`` with ``image`` of shape ``(H, W, 3)``,
            the per-scale canvases (coarsest first) and the background canvas.
        """
        device = background_color.device
        canvases: list[Tensor] = []
        for scale in reversed(range(len(self.zs_info))):
            info = self.zs_info[scale]
            x_scale, y_scale, angle, red, green, blue = latents[scale]
            color = torch.stack([red, green, blue]).T
            window = 2 * info.rj + 1
            rasters = affine_transform(self._discs[info.rj], x_scale, y_scale, angle, device)
            prims = colorize(rasters, color)
            canvases.append(
                combine_primitives(prims, info.height, info.width, info.s, window, device)
            )
        background = torch.ones(*self.img_size, 3, device=device) * background_color
        return merge_canvases(canvases, background), canvases, background

    def __call__(
        self, latents: list[Tensor], background: Tensor, return_canvases: bool = False
    ) -> RenderOutput:
        """Render a batch.

        Args:
            latents: Per-scale parameters, each ``(B, 6, N_j)``.
            background: ``(B, 3)`` background colours.
            return_canvases: Also return the canvases of the first image (for plots).

        Returns:
            The rendered images and optionally the first image's canvases.
        """
        images: list[Tensor] = []
        first: tuple[list[Tensor], Tensor] | None = None
        for i in range(background.shape[0]):
            img, canvases, bg = self.render_one([z[i] for z in latents], background[i])
            if i == 0 and return_canvases:
                first = ([c.detach() for c in canvases], bg.detach())
            images.append(img)
        return RenderOutput(images=torch.stack(images), canvases=first)
