"""Analytic output size, receptive field and effective stride of conv stacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerGeometry:
    """Geometry of one spatial layer (convolution or pooling).

    Attributes:
        kernel_size: Square kernel size ``k``.
        stride: Stride ``s``.
        padding: Zero padding ``p``.
    """

    kernel_size: int
    stride: int
    padding: int


@dataclass(frozen=True)
class FeatureMapInfo:
    """Spatial properties of a feature map, relative to the input image.

    Attributes:
        height: Feature-map height.
        width: Feature-map width.
        receptive_field: Receptive field size in input pixels.
        stride: Effective stride (distance in input pixels between adjacent
            feature-map positions).
    """

    height: int
    width: int
    receptive_field: int
    stride: int


def image_info(height: int, width: int) -> FeatureMapInfo:
    """Geometry of the raw input image (receptive field 1, stride 1)."""
    return FeatureMapInfo(height=height, width=width, receptive_field=1, stride=1)


def propagate(layers: list[LayerGeometry], start: FeatureMapInfo) -> FeatureMapInfo:
    """Propagate spatial geometry through a sequence of layers.

    Uses the standard recurrences ``n' = floor((n + 2p - k) / s) + 1``,
    ``rf' = rf + (k - 1) * j`` and ``j' = j * s``.

    Args:
        layers: Layers in forward order.
        start: Geometry of the feature map entering the first layer, e.g.
            :func:`image_info` for the raw image.

    Returns:
        Geometry of the output feature map.
    """
    h, w, rf, j = start.height, start.width, start.receptive_field, start.stride
    for layer in layers:
        k, s, p = layer.kernel_size, layer.stride, layer.padding
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
        rf = rf + (k - 1) * j
        j = j * s
    return FeatureMapInfo(height=h, width=w, receptive_field=rf, stride=j)
