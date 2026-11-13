"""Bag-of-patches feature tables for Stage 2 (classification).

Patches of each slide are grouped into random bags of ``bag_size`` patches.
Every bag becomes one row whose features are statistics of the latent
representation over the bag:

* **ASR**: for each aggregation (``mean``, ``std``) and scale ``j``, the
  statistic of each primitive parameter ``(w, h, d, ar, ag, ab)`` over all
  patches and all primitives of that scale → ``2 x 3 x 6 = 36`` features named
  ``<agg>_scale_<j>_<param>``.
* **Autoencoder baseline**: the statistic of each latent dimension over the
  bag → ``200`` features named ``<agg>_f<i>`` (paper: mean only).

Feature precision is set by ``features.dtype`` (default float16).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm import tqdm

from asr.config import Config
from asr.data.datasets import (
    CLASS_NAMES,
    SPLITS,
    SlidePatches,
    default_workers,
    slide_dirs,
    slide_label,
)
from asr.model.encoder import Encoder
from asr.model.modeler import FEATURE_NAMES
from asr.utils.runtime import get_logger

logger = get_logger(__name__)

AGGREGATIONS: dict[str, Callable[..., Tensor]] = {"mean": torch.mean, "std": torch.std}
LABEL_COLUMN = "class"
SLIDE_COLUMN = "slide"


def asr_bag_features(encoder: Encoder, bag: Tensor, aggregations: Sequence[str]) -> np.ndarray:
    """Features of one bag for the ASR encoder.

    Args:
        encoder: Trained ASR encoder.
        bag: ``(n, 3, H, W)`` patches.
        aggregations: Names from :data:`AGGREGATIONS`.

    Returns:
        ``(len(aggregations) * n_scales * 6,)`` vector.
    """
    latents = encoder(bag).latents
    parts = [AGGREGATIONS[agg](z, (0, 2)).cpu().numpy() for agg in aggregations for z in latents]
    return np.concatenate(parts)


def asr_feature_names(n_scales: int, aggregations: Sequence[str]) -> list[str]:
    """Column names matching :func:`asr_bag_features`."""
    return [
        f"{agg}_scale_{j}_{p}"
        for agg in aggregations
        for j in range(n_scales)
        for p in FEATURE_NAMES
    ]


def ae_bag_features(encoder: nn.Module, bag: Tensor, aggregations: Sequence[str]) -> np.ndarray:
    """Features of one bag for the autoencoder baseline (``(len(aggs) * latent_dim,)``)."""
    z = encoder(bag)
    return np.concatenate([AGGREGATIONS[agg](z, 0).cpu().numpy() for agg in aggregations])


def ae_feature_names(latent_dim: int, aggregations: Sequence[str]) -> list[str]:
    """Column names matching :func:`ae_bag_features`."""
    return [f"{agg}_f{i}" for agg in aggregations for i in range(latent_dim)]


@torch.no_grad()
def build_feature_tables(
    config: Config,
    encoder: nn.Module,
    featurize: Callable[[Tensor], np.ndarray],
    columns: list[str],
    out_dir: Path,
    device: torch.device,
) -> dict[str, Path]:
    """Write ``<out_dir>/<split>_features.csv`` for train, valid and test.

    Args:
        config: Experiment configuration (dataset root, ``features`` section).
        encoder: Encoder module (its train/eval mode is set from
            ``config.features.encoder_train_mode``).
        featurize: Maps a ``(n, 3, H, W)`` bag on ``device`` to a feature vector.
        columns: Feature column names.
        out_dir: Output directory.
        device: Compute device.

    Returns:
        Mapping from split name to the written CSV path.
    """
    fc = config.features
    encoder.train(fc.encoder_train_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(fc.seed)
    workers = (
        config.training.num_workers
        if config.training.num_workers is not None
        else default_workers(device)
    )
    written: dict[str, Path] = {}
    for split in SPLITS:
        rows: list[np.ndarray] = []
        labels: list[str] = []
        slides: list[str] = []
        for slide in slide_dirs(config.dataset.root, split):
            dataset = SlidePatches(slide, config.dataset.resize_shape)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=fc.bag_size,
                shuffle=True,
                num_workers=workers,
                generator=generator,
            )
            label = CLASS_NAMES[slide_label(slide)]
            for bag in tqdm(loader, desc=f"{split}/{slide.name}", unit="bag", leave=False):
                rows.append(featurize(bag.to(device)).astype(fc.dtype))
                labels.append(label)
                slides.append(slide.name)
        df = pd.DataFrame(np.stack(rows), columns=columns)
        df[LABEL_COLUMN] = labels
        df[SLIDE_COLUMN] = slides
        path = out_dir / f"{split}_features.csv"
        df.to_csv(path, index=False)
        logger.info("Wrote %d bags to %s", len(df), path)
        written[split] = path
    return written
