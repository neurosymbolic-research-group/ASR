"""Image transforms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torchvision.transforms import v2


def get_transform(resize_shape: tuple[int, int], augment: bool) -> Callable[[Any], Any]:
    """Resize + convert to a float tensor in ``[0, 1]``, optionally with augmentation.

    Args:
        resize_shape: Target ``(H, W)``.
        augment: Add random horizontal/vertical flips and a random rotation by
            0, 90, 180 or 270 degrees.

    Returns:
        A torchvision v2 transform.
    """
    to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    if not augment:
        return v2.Compose([v2.Resize(resize_shape), to_tensor])
    rotations = [
        v2.RandomRotation((a, a), v2.InterpolationMode.BILINEAR) for a in (0, 90, 180, 270)
    ]
    return v2.Compose(
        [
            v2.Resize(resize_shape),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice(rotations),
            to_tensor,
        ]
    )
