"""Masked MSE (MMSE): reconstruction loss ignoring an image border."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def border_mask(height: int, width: int, border: int) -> Tensor:
    """Binary ``(H, W)`` mask that is 0 on a ``border``-pixel frame and 1 inside."""
    mask = torch.ones(height - 2 * border, width - 2 * border)
    return F.pad(mask, [border] * 4, "constant", 0).float()


class MaskedMSELoss(nn.Module):
    """Mean squared error computed only inside a border mask.

    The per-pixel squared error is masked, averaged over the unmasked pixels
    of every channel, and then averaged over channels and batch.

    Args:
        output_dims: Image ``(H, W)``.
        n_border_ignored: Width of the ignored border in pixels.
    """

    mask: Tensor
    mask_sum: Tensor

    def __init__(self, output_dims: tuple[int, int], n_border_ignored: int) -> None:
        super().__init__()
        mask = border_mask(*output_dims, n_border_ignored)
        self.register_buffer("mask", mask, persistent=False)
        self.register_buffer("mask_sum", mask.sum(), persistent=False)
        self.n_border_ignored = n_border_ignored

    def __str__(self) -> str:
        """Human-readable name used in plot titles."""
        return f"MSELoss Masked Loss (n_border_ignored={self.n_border_ignored})"

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the loss.

        Args:
            pred: ``(B, C, H, W)`` predictions.
            target: ``(B, C, H, W)`` targets.

        Returns:
            Scalar loss.
        """
        se = F.mse_loss(pred, target, reduction="none") * self.mask
        return (se.sum((-2, -1)) / self.mask_sum).mean()
