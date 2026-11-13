"""Statistics of primitive appearance ("shape absorption") per scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

CHANNELS = ("R", "G", "B")


class PrimitiveStats:
    """Accumulate appearance statistics of the primitives over a test set.

    Reported per scale ``j`` (encoder order, 0 = finest grid):

    * ``MEAN_<C>_S<j>``: mean appearance of channel ``C`` (mean of per-batch means).
    * ``MEAN_ACT_S<j>``: fraction of *active* primitives, i.e. with any
      appearance channel above :attr:`active_threshold`.

    Args:
        n_scales: Number of primitive scales.
        active_threshold: Appearance above which a primitive counts as active.
        n_bins: Histogram bins over ``[0, 1]``.
    """

    def __init__(self, n_scales: int, active_threshold: float = 0.1, n_bins: int = 20) -> None:
        self.n_scales = n_scales
        self.active_threshold = active_threshold
        self.n_bins = n_bins
        self._means: list[list[Tensor]] = [[] for _ in range(n_scales)]
        self._active: list[list[float]] = [[] for _ in range(n_scales)]
        self._hist = [torch.zeros(n_bins, dtype=torch.float64) for _ in range(n_scales)]

    def update(self, latents: list[Tensor]) -> None:
        """Add one batch of per-scale parameters, each ``(B, 6, N_j)``."""
        for j, z in enumerate(latents):
            a = z[:, 3:, :].detach().float().cpu()
            b, _, n = a.shape
            self._means[j].append(a.mean((0, 2)))
            self._active[j].append(
                int((a > self.active_threshold).any(dim=1).sum().item()) / (b * n)
            )
            self._hist[j] += torch.histc(a, bins=self.n_bins, min=0.0, max=1.0).double()

    def compute(self) -> dict[str, float]:
        """Aggregate the accumulated statistics into a flat dictionary."""
        out: dict[str, float] = {}
        for j in range(self.n_scales):
            means = torch.stack(self._means[j]).mean(0)
            for c, v in zip(CHANNELS, means.tolist(), strict=True):
                out[f"MEAN_{c}_S{j}"] = v
            out[f"MEAN_ACT_S{j}"] = sum(self._active[j]) / len(self._active[j])
        return out

    def plot_histogram(self, path: Path) -> None:
        """Save a per-scale log-histogram of appearance values to ``path``."""
        fig, axs = plt.subplots(self.n_scales, 1, sharex=True, sharey=True, figsize=(6, 7))
        axs = [axs] if self.n_scales == 1 else list(axs)
        edges = torch.linspace(0, 1, self.n_bins + 1)
        for j, ax in enumerate(axs):
            ax.stairs(self._hist[j].numpy(), edges.numpy(), fill=True, color="#2ab0ff")
            ax.set_title(f"Scale {j}")
            ax.set_yscale("log")
            ax.spines[["top", "right"]].set_visible(False)
            ax.xaxis.set_tick_params(labelbottom=True)
        fig.supxlabel("Appearance value")
        fig.supylabel("Count (log)")
        fig.suptitle("Histograms of primitive appearance values")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
