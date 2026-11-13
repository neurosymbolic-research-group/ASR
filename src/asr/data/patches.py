"""Extract tissue patches from Aperio (``.svs``) whole-slide images.

A window slides over the slide with a fixed stride. Each window covers
``window_size * rescale_factor`` pixels of the full-resolution slide and is
read downscaled to ``window_size``. Windows are kept if, after removing
near-white background in HSV space, at least ``tissue_threshold`` percent of
the pixels are tissue.

Requires the optional ``patches`` extra (``slideio``, ``opencv-python-headless``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from asr.utils.runtime import get_logger

logger = get_logger(__name__)

LOWER_WHITE = np.array([0, 0, 200], dtype=np.uint8)
"""Lower HSV bound of background (white) pixels."""
UPPER_WHITE = np.array([180, 25, 255], dtype=np.uint8)
"""Upper HSV bound of background (white) pixels."""


def sliding_windows(
    scene: Any, stride: int, window_size: tuple[int, int], rescale_factor: int
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(x, y, window)`` over a ``slideio`` scene.

    Args:
        scene: ``slideio`` scene.
        stride: Step in full-resolution pixels.
        window_size: Output ``(width, height)`` of each window.
        rescale_factor: Full-resolution window size is ``window_size * rescale_factor``.
    """
    w, h = scene.size
    win_w, win_h = window_size[0] * rescale_factor, window_size[1] * rescale_factor
    for y in range(0, h - win_h + 1, stride):
        for x in range(0, w - win_w + 1, stride):
            yield x, y, scene.read_block((x, y, win_h, win_w), size=window_size)


def tissue_percentage(gray: np.ndarray) -> float:
    """Percentage of pixels brighter than 10 in a background-masked grayscale image."""
    return float(np.sum(gray > 10) / gray.size * 100)


def select_windows(
    windows: Iterator[tuple[int, int, np.ndarray]], tissue_threshold: float, n_windows: int
) -> list[tuple[int, int]]:
    """Coordinates of windows with enough tissue.

    Args:
        windows: HSV windows from :func:`sliding_windows`.
        tissue_threshold: Minimum tissue percentage.
        n_windows: At most ``n_windows - 1`` windows are selected.
    """
    import cv2

    coords: list[tuple[int, int]] = []
    for x, y, window in windows:
        mask_white = cv2.inRange(window, LOWER_WHITE, UPPER_WHITE)
        masked = cv2.bitwise_and(window, window, mask=~mask_white)
        gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
        if tissue_percentage(gray) >= tissue_threshold:
            coords.append((x, y))
        if len(coords) == n_windows - 1:
            break
    return coords


def extract_slide(
    slide_path: Path,
    out_dir: Path,
    window_size: tuple[int, int] = (256, 256),
    rescale_factor: int = 4,
    stride: int = 635,
    tissue_threshold: float = 80,
    n_windows: int = 5000,
) -> int:
    """Extract patches of one ``.svs`` slide into ``out_dir/<slide_name>/NNNN.png``.

    Returns:
        Number of saved patches.
    """
    import slideio

    slide = slideio.open_slide(str(slide_path), "SVS")
    scene = slide.get_scene(0)
    params = slideio.ColorTransformation()
    params.color_space = slideio.ColorSpace.HSV
    scene_hsv = scene.apply_transformation([params])
    coords = select_windows(
        sliding_windows(scene_hsv, stride, window_size, rescale_factor),
        tissue_threshold,
        n_windows,
    )
    target = out_dir / slide_path.stem
    target.mkdir(parents=True, exist_ok=True)
    win_w, win_h = window_size[0] * rescale_factor, window_size[1] * rescale_factor
    for i, (x, y) in enumerate(coords):
        window = scene.read_block((x, y, win_h, win_w), size=window_size)
        Image.fromarray(window).save(target / f"{i:04d}.png")
    logger.info("%s: %d patches", slide_path.name, len(coords))
    return len(coords)
