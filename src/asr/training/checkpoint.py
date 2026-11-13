"""Saving and loading checkpoints, and early stopping."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    train_loss: float,
    valid_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint dictionary.

    Keys: ``epoch``, ``model_state_dict``, ``optimizer_state_dict``, ``loss``
    (training loss), ``valid_loss`` and any ``extra``.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "loss": train_loss,
            "valid_loss": valid_loss,
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Load a checkpoint, accepting both dict checkpoints and bare state dicts.

    Bare state dicts are wrapped as ``{"model_state_dict": ..., "epoch": None}``.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in ckpt:
        ckpt = {"model_state_dict": ckpt, "epoch": None}
    return ckpt


class EarlyStopping:
    """Stop when the monitored loss has not improved for ``patience`` calls.

    Args:
        patience: Number of non-improving calls tolerated.
        min_delta: Minimum decrease that counts as an improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = math.inf

    def __call__(self, loss: float) -> bool:
        """Register ``loss``; return ``True`` if training should stop."""
        if loss + self.min_delta < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss + self.min_delta > self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
