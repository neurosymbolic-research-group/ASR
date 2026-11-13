"""Appearance regularisation (ARV) encouraging sparse use of primitives.

For each image, the RGB appearance values ``a`` of all primitives are raised
to the power ``α``, weighted per scale, summed and divided by the total number
of primitives. The batch mean is scaled by ``λ_a``. Small ``α < 1`` pushes
appearances towards exactly zero, i.e. towards switching primitives off.
"""

from __future__ import annotations

import torch
from torch import Tensor

LOSS_OFFSET = 1e-5
"""Small constant offset added to the loss."""


class AppearanceRegularizer:
    """Callable computing the ARV loss term.

    Args:
        n_scales: Number of primitive scales.
        lambda_a: Weight λ_a.
        alpha: Exponent α.
        scale_weights: Per-scale weights in encoder order; ``None`` = all ones.
    """

    def __init__(
        self,
        n_scales: int,
        lambda_a: float = 0.009,
        alpha: float = 0.75,
        scale_weights: list[float] | None = None,
    ) -> None:
        self.lambda_a = lambda_a
        self.alpha = alpha
        self.scale_weights = scale_weights or [1.0] * n_scales
        if len(self.scale_weights) != n_scales:
            raise ValueError(f"Expected {n_scales} scale weights, got {len(self.scale_weights)}")

    def __call__(self, latents: list[Tensor]) -> Tensor:
        """Compute the ARV loss.

        Args:
            latents: Per-scale parameters, each ``(B, 6, N_j)``.

        Returns:
            Scalar loss tensor.
        """
        per_image = torch.zeros(latents[0].shape[0], device=latents[0].device)
        total = 0
        for z, w in zip(latents, self.scale_weights, strict=True):
            per_image = per_image + (z[:, 3:, :].pow(self.alpha) * w).sum(dim=(1, 2))
            total += z.shape[-1]
        return (per_image / total).mean() * self.lambda_a + LOSS_OFFSET
