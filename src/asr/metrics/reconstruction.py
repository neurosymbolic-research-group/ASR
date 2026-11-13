"""Image reconstruction metrics (MSE, MAE, SSIM)."""

from __future__ import annotations

from collections.abc import Iterable

from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric, MetricCollection
from torchmetrics.image import StructuralSimilarityIndexMeasure


def build_metrics(names: Iterable[str]) -> MetricCollection:
    """Create a metric collection for channel-first images in ``[0, 1]``.

    Args:
        names: Any of ``mse``, ``mae``, ``ssim``.

    Returns:
        A :class:`torchmetrics.MetricCollection` keyed by the given names.
    """
    factories: dict[str, type[Metric]] = {
        "mse": MeanSquaredError,
        "mae": MeanAbsoluteError,
    }
    metrics: dict[str, Metric | MetricCollection] = {}
    for name in names:
        if name == "ssim":
            metrics[name] = StructuralSimilarityIndexMeasure(data_range=1.0)
        elif name in factories:
            metrics[name] = factories[name]()
        else:
            raise ValueError(f"Unknown metric: {name}")
    return MetricCollection(metrics)
