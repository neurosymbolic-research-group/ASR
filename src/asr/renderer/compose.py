"""Multiplicative compositing of primitives into canvases and of canvases into images."""

from __future__ import annotations

import torch
from torch import Tensor


def combine_primitives(
    prims: Tensor,
    n_vertical: int,
    n_horizontal: int,
    spacing: int,
    window: int,
    device: torch.device | str,
) -> Tensor:
    """Place a grid of coloured primitives on a white canvas (one scale).

    Each primitive window is multiplied into the canvas as ``canvas *= 1 - p``,
    so overlapping primitives combine multiplicatively. Primitives are indexed
    row-major, matching the modeler output.

    Args:
        prims: ``(N, window, window, 3)`` coloured rasters, ``N = n_vertical * n_horizontal``.
        n_vertical: Grid rows.
        n_horizontal: Grid columns.
        spacing: Distance in pixels between neighbouring primitive windows.
        window: Window size ``2 r_j + 1``.
        device: Device of the canvas.

    Returns:
        Canvas of shape ``((n_vertical-1)*spacing + window, (n_horizontal-1)*spacing + window, 3)``.
    """
    height = (n_vertical - 1) * spacing + window
    width = (n_horizontal - 1) * spacing + window
    canvas = torch.full((height, width, 3), 1.0, device=device)
    idx = 0
    for y in range(n_vertical):
        for x in range(n_horizontal):
            top, left = y * spacing, x * spacing
            sl = (slice(top, top + window), slice(left, left + window))
            canvas[sl] = canvas[sl].clone() * (1 - prims[idx])
            idx += 1
    return canvas


def merge_canvases(canvases: list[Tensor], background: Tensor) -> Tensor:
    """Centre-crop each canvas to the image size and multiply into the background.

    Args:
        canvases: Per-scale canvases ``(H_j, W_j, 3)``, each at least as large as
            the background.
        background: Background image ``(H, W, 3)``.

    Returns:
        The rendered image ``(H, W, 3)``.
    """
    bg_h, bg_w, _ = background.shape
    out = background
    for canvas in canvases:
        h, w, _ = canvas.shape
        top, left = (h - bg_h) // 2, (w - bg_w) // 2
        out = out * canvas[top : top + bg_h, left : left + bg_w, :]
    return out
