"""Rasterisation, affine transformation and colouring of elliptical primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def fuzzy_disc(rj: int, beta: float) -> Tensor:
    """Rasterise a fuzzy disc that serves as the prototype primitive.

    The membership of a pixel at distance ``ρ`` from the window centre is
    ``1 - 1 / (1 + exp((r - ρ) β)) = sigmoid((r - ρ) β)`` with radius
    ``r = rj // 2``: ≈1 inside the disc, ≈0 outside; higher ``β`` gives
    sharper edges.

    Note:
        Compositing (:func:`asr.renderer.compose.combine_primitives`) multiplies
        the white canvas by ``1 - a * membership``, so the RGB appearance ``a``
        acts as per-channel *absorption*: ``a = 0`` switches a primitive off.

    Args:
        rj: Rendering radius; the window is ``(2 rj + 1)`` pixels wide.
        beta: Fuzziness coefficient β.

    Returns:
        Tensor of shape ``(2 rj + 1, 2 rj + 1)`` on the CPU.
    """
    n = 2 * rj + 1
    r = rj // 2
    x, y = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    dist = torch.sqrt((x - (n - 1) / 2) ** 2 + (y - (n - 1) / 2) ** 2)
    img = 1 / (1 + torch.exp((r - dist) * beta))
    return 1 - img


def affine_transform(
    img: Tensor, x_scale: Tensor, y_scale: Tensor, angle: Tensor, device: torch.device | str
) -> Tensor:
    """Scale and rotate one prototype raster into ``N`` primitives.

    Builds ``M = diag(1/w, 1/h) · R(d)`` per primitive and resamples the
    prototype bilinearly with :func:`torch.nn.functional.grid_sample`.

    Args:
        img: Prototype raster ``(n, n)``.
        x_scale: Horizontal scale ``w`` per primitive, shape ``(N,)``.
        y_scale: Vertical scale ``h`` per primitive, shape ``(N,)``.
        angle: Rotation ``d`` in radians, shape ``(N,)``.
        device: Device for the output.

    Returns:
        Transformed rasters of shape ``(N, n, n)``.
    """
    n_prims = x_scale.shape[0]
    row_x = F.pad(1 / x_scale.reshape(-1, 1), (0, 1), "constant", 0)
    row_y = F.pad(1 / y_scale.reshape(-1, 1), (1, 0), "constant", 0)
    S = torch.stack([row_x, row_y], dim=1)

    row1 = torch.stack([torch.cos(angle), -torch.sin(angle)], dim=1)
    row2 = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
    R = torch.stack([row1, row2], dim=1)

    M = F.pad(torch.matmul(S, R), (0, 1))  # (N, 2, 3) as required by affine_grid
    grid = F.affine_grid(M, size=[n_prims, 1, *img.shape], align_corners=True)
    repeated = img.repeat(n_prims, 1, 1).unsqueeze(1).to(device)
    out = F.grid_sample(repeated, grid=grid, mode="bilinear", align_corners=True)
    return out.squeeze(1)


def colorize(rasters: Tensor, color: Tensor) -> Tensor:
    """Multiply each single-channel raster by its RGB appearance.

    Args:
        rasters: ``(N, n, n)`` rasters.
        color: ``(N, 3)`` RGB appearance.

    Returns:
        ``(N, n, n, 3)`` coloured rasters.
    """
    return rasters[..., None] * color[:, None, None, :]
