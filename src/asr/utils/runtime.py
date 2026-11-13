"""Device selection, seeding and logging."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch


def get_device(name: str | None = None) -> torch.device:
    """Return ``torch.device(name)``, or CUDA if available when ``name`` is ``None``/``auto``."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Logger with a single timestamped stream handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
