"""Layout of a run (results) directory."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """Paths inside ``<results_root>/<run_name>``.

    Layout::

        <run>/config.yaml
        <run>/models/{best,last}.pt
        <run>/plots/{train,valid,test}/{rendered_images,canvases}/
        <run>/metrics.csv
        <run>/classification/
    """

    root: Path

    @classmethod
    def of(cls, results_root: str | Path, run_name: str) -> RunPaths:
        """Construct from a results root and run name."""
        return cls(Path(results_root) / run_name)

    @property
    def models(self) -> Path:
        """Checkpoint directory."""
        return self.root / "models"

    @property
    def plots(self) -> Path:
        """Plot directory."""
        return self.root / "plots"

    @property
    def classification(self) -> Path:
        """Stage-2 outputs (feature tables, tree, reports)."""
        return self.root / "classification"

    def split_plots(self, split: str, kind: str) -> Path:
        """``plots/<split>/<kind>`` where kind is ``rendered_images`` or ``canvases``."""
        return self.plots / split / kind

    def prepare_training(self, overwrite: bool) -> None:
        """Create the directory tree for a new training run.

        Args:
            overwrite: Delete an existing run directory first. Without it an
                existing, non-empty run directory raises ``FileExistsError``.
        """
        if self.root.exists() and any(self.root.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.root} exists; pass --overwrite to replace it")
            shutil.rmtree(self.root)
        self.models.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            for kind in ("rendered_images", "canvases"):
                self.split_plots(split, kind).mkdir(parents=True, exist_ok=True)
