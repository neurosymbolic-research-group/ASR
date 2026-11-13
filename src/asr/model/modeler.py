"""Per-scale modeler mapping a feature map to elliptical primitive parameters."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from asr.config import ModelerConfig

LATENT_DIM = 64
"""Hidden width of the 1x1-convolution MLP."""

N_VARIABLES = 6
"""Parameters per primitive: ``[w, h, d, a_r, a_g, a_b]``.

``w``/``h`` are the horizontal/vertical scale in ``[0.1, 2.0]``, ``d`` is the
rotation angle in ``[0, 2π]`` and ``a_*`` is the RGB appearance in ``[0, 1]``.
"""

FEATURE_NAMES = ("w", "h", "d", "ar", "ag", "ab")
"""Short names of the primitive parameters, used for feature-table columns."""


def importance_at_epoch(config: ModelerConfig, epoch: int) -> float:
    """Importance multiplier of a modeler at a given epoch.

    The value starts at ``config.start_value`` and is incremented by
    ``config.increment_value`` (capped at 1) at the start of each epoch ``e``
    with ``e >= epoch_start`` and ``(e + 1) % epoch_period == 0``.

    Args:
        config: The modeler's schedule.
        epoch: Zero-based epoch index.

    Returns:
        The importance value used during ``epoch``; 1.0 when freezing is off.
    """
    if not config.freeze:
        return 1.0
    value = config.start_value
    for e in range(epoch + 1):
        if (e + 1) % config.epoch_period == 0 and e >= config.epoch_start:
            value = min(value + config.increment_value, 1.0)
    return value


class DenseModeler(nn.Module):
    """Two-layer 1x1-conv MLP producing one primitive per feature-map cell.

    The sigmoid output is affinely rescaled to the parameter ranges
    (``w, h ∈ [0.1, 2.0]``, ``d ∈ [0, 2π]``, appearance ∈ ``[0, 1]``) and then
    multiplied by the importance value of the incremental schedule (all six
    parameters are scaled).

    Args:
        in_channels: Channels of the incoming feature map.
        config: Incremental-learning schedule for this scale.
    """

    def __init__(self, in_channels: int, config: ModelerConfig | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.config = config or ModelerConfig()
        self.importance_value = importance_at_epoch(self.config, 0)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, LATENT_DIM, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(LATENT_DIM, N_VARIABLES, kernel_size=1),
            nn.Sigmoid(),
        )
        # Non-persistent buffers: follow .to(device) without entering the state dict.
        self.register_buffer(
            "mul_scaling",
            torch.tensor([1.9, 1.9, 2 * math.pi, 1.0, 1.0, 1.0]).unsqueeze(-1),
            persistent=False,
        )
        self.register_buffer(
            "add_scaling",
            torch.tensor([0.1, 0.1, 0.0, 0.0, 0.0, 0.0]).unsqueeze(-1),
            persistent=False,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)

    def set_epoch(self, epoch: int) -> None:
        """Update the importance value for the given (zero-based) epoch."""
        self.importance_value = importance_at_epoch(self.config, epoch)

    def forward(self, x: Tensor) -> Tensor:
        """Predict primitive parameters.

        Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Parameters of shape ``(B, 6, H * W)``; primitives are ordered
            row-major over the ``H x W`` grid.
        """
        _, _, h, w = x.shape
        x = self.layers(x).view(-1, N_VARIABLES, h * w)
        x = x * self.mul_scaling + self.add_scaling
        if self.config.freeze:
            x = x * self.importance_value
        return x
