"""Diagnostic figures: reconstructions, per-scale canvases and loss curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def plot_reconstructions(
    inputs: np.ndarray, outputs: np.ndarray, path: Path, title: str, n: int = 10
) -> None:
    """Save a two-row figure: input images on top, reconstructions below.

    Args:
        inputs: ``(B, H, W, 3)`` images in ``[0, 1]``.
        outputs: ``(B, H, W, 3)`` reconstructions in ``[0, 1]``.
        path: Output file.
        title: Figure title.
        n: Maximum number of images.
    """
    n = min(n, len(outputs))
    fig = plt.figure(figsize=(n * 2, 5))
    fig.suptitle(title, size=20)
    for i in range(n):
        for row, (imgs, label) in enumerate(
            ((inputs, "Original images"), (outputs, "Rendered images"))
        ):
            ax = plt.subplot(2, n, i + 1 + row * n)
            ax.imshow((255 * np.clip(imgs[i], 0, 1)).astype(np.uint8))
            ax.axis("off")
            if i == n // 2:
                ax.set_title(label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_canvases(canvases: list[Tensor], background: Tensor, merged: Tensor, path: Path) -> None:
    """Save background, per-scale canvases (coarsest first) and the merged image.

    Args:
        canvases: Per-scale canvases ``(H_j, W_j, 3)``.
        background: Background canvas ``(H, W, 3)``.
        merged: Rendered image ``(H, W, 3)``.
        path: Output file.
    """
    n_scales = len(canvases)
    panels = [("Background", background)]
    panels += [(f"Scale {n_scales - 1 - k}", c) for k, c in enumerate(canvases)]
    panels.append(("Merged", merged))
    fig, axs = plt.subplots(1, len(panels), figsize=(len(panels) * 3, 3))
    for ax, (name, img) in zip(axs, panels, strict=True):
        arr = img.detach().cpu().numpy()
        ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(f"{name}\n{tuple(arr.shape)}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_loss(history: dict[str, list[float]], title: str, path: Path) -> None:
    """Save training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, values in history.items():
        ax.plot(values, label=key.replace("_", " "))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
