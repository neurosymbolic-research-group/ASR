"""Typed experiment configuration loaded from YAML.

Every runnable stage (ASR training/testing, autoencoder baseline, feature
extraction, decision-tree classification) reads a single YAML file validated by
the :class:`Config` model below. Unknown keys raise an error, so typos in a
config file are caught early.

See ``configs/`` for the paper's three ASR variants (*Base*, *Regularized*,
*Incremental*) and the autoencoder *Baseline*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

MetricName = Literal["mse", "mae", "ssim"]
AggregationName = Literal["mean", "std"]
Criterion = Literal["gini", "entropy", "log_loss"]


def _default_metrics() -> list[MetricName]:
    return ["mae", "mse", "ssim"]


def _default_aggregations() -> list[AggregationName]:
    return ["mean", "std"]


def _default_criteria() -> list[Criterion]:
    return ["gini", "entropy"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(_Strict):
    """Location and geometry of the image dataset.

    Attributes:
        root: Directory containing ``train/``, ``valid/`` and ``test/`` subfolders,
            each holding one folder of PNG patches per whole-slide image.
        img_size: Encoder input shape ``(C, H, W)``.
        resize_shape: Images are resized to ``(H, W)`` before entering the model.
    """

    root: Path = Path("dataset/images")
    img_size: tuple[int, int, int] = (3, 256, 256)
    resize_shape: tuple[int, int] = (256, 256)


class TrainingConfig(_Strict):
    """Stage-1 (reconstruction) training hyper-parameters.

    Attributes:
        batch_size: Images per batch.
        num_epochs: Maximum number of epochs.
        lr: Adam learning rate.
        patience: Early-stopping patience in epochs.
        early_stopping_on: Loss monitored by early stopping (``train`` or ``valid``).
        n_border_ignored: Width (px) of the image border excluded from the
            masked MSE (MMSE) loss.
        augment: Random flips and right-angle rotations.
        save_model: Save ``best.pt`` (lowest validation loss) and ``last.pt``.
        steps_per_epoch: Batches per "epoch". ``None`` means one full pass over
            the dataset. The paper uses 32 (1024 images).
        valid_full_pass: If ``False``, validation uses ``steps_per_epoch``
            randomly drawn batches per epoch; if ``True``, one full pass over the
            validation set.
        grad_clip_norm: Max gradient norm, ``None`` disables clipping.
        seed: Global random seed.
        num_workers: DataLoader workers; ``None`` = 2 on GPU, 0 on CPU.
    """

    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1.0e-3
    patience: int = 20
    early_stopping_on: Literal["train", "valid"] = "train"
    n_border_ignored: int = 16
    augment: bool = False
    save_model: bool = True
    steps_per_epoch: int | None = 32
    valid_full_pass: bool = False
    grad_clip_norm: float | None = 1.0
    seed: int = 2001
    num_workers: int | None = None


class RendererConfig(_Strict):
    """Differentiable renderer settings.

    Attributes:
        beta: Fuzziness (sigmoid steepness) of the elliptical primitive.
    """

    beta: float = 1.0


class ModelerConfig(_Strict):
    """Incremental-learning ("freezing") schedule of one scale's modeler.

    When ``freeze`` is true, the modeler output is multiplied by an importance
    value that starts at ``start_value`` and grows by ``increment_value`` at
    the start of every epoch ``e >= epoch_start`` with
    ``(e + 1) % epoch_period == 0``, capped at 1. In the paper these are the
    β (start) and γ (increment) coefficients of the *Incremental* variant.

    Attributes:
        freeze: Enable the schedule. If false, importance is always 1.
        start_value: Initial importance (must be > 0 for gradients to flow).
        increment_value: Importance increment.
        epoch_period: Increment every this many epochs.
        epoch_start: First epoch at which incrementing may happen.
    """

    freeze: bool = False
    start_value: float = 1.0
    increment_value: float = 0.0
    epoch_period: int = 1
    epoch_start: int = 0

    @field_validator("epoch_period")
    @classmethod
    def _positive_period(cls, v: int) -> int:
        if v < 1:
            raise ValueError("epoch_period must be >= 1")
        return v


class ARVConfig(_Strict):
    """Appearance-regularisation (ARV) loss, see :mod:`asr.losses.appearance`.

    Attributes:
        enabled: Add the ARV term to the reconstruction loss.
        start_epoch: First epoch at which the term is added (paper's
            *Incremental* variant: 35).
        lambda_a: Weight λ_a of the ARV term.
        alpha: Exponent α applied to appearance values (paper: α ∈ (0.5, 1)).
        scale_weights: Per-scale weights, in encoder order (finest grid first).
            ``None`` means all ones.
    """

    enabled: bool = False
    start_epoch: int = 0
    lambda_a: float = 0.009
    alpha: float = 0.75
    scale_weights: list[float] | None = None


class TestingConfig(_Strict):
    """Stage-1 evaluation settings.

    Attributes:
        batch_size: Images per batch.
        weights: Checkpoint filename inside ``<run>/models``.
        plot: Save rendered images and per-scale canvases for the first batch.
        metrics: Reconstruction metrics to report (``mse``, ``mae``, ``ssim``).
    """

    batch_size: int = 32
    weights: str = "best.pt"
    plot: bool = True
    metrics: list[MetricName] = Field(default_factory=_default_metrics)


class FeaturesConfig(_Strict):
    """Stage-2 bag-of-patches feature extraction.

    Attributes:
        bag_size: Patches per bag; each bag becomes one row of the tabular dataset.
        aggregations: Statistics aggregated over the patches and primitives of
            a bag. Paper: ``[mean, std]`` for ASR, ``[mean]`` for the baseline.
        encoder_train_mode: Run the encoder in training mode, so BatchNorm
            normalises with the statistics of each bag (setting used in the
            paper). ``false`` uses the running statistics.
        dtype: Precision of the stored features.
        seed: Seed controlling how patches are grouped into bags.
    """

    bag_size: int = 16
    aggregations: list[AggregationName] = Field(default_factory=_default_aggregations)
    encoder_train_mode: bool = True
    dtype: Literal["float16", "float32"] = "float16"
    seed: int = 2001


class ClassifierConfig(_Strict):
    """Decision-tree classifier (paper Sec. 4, Stage 2).

    Hyper-parameters are selected with ``cv``-fold grid search on the training
    set; afterwards the cost-complexity pruning parameter ``ccp_alpha`` is tuned
    on the validation set.

    Attributes:
        criterion: Impurity measures to search.
        max_depth: Depths to search (``null`` = unlimited).
        min_samples_leaf: Minimum leaf sizes to search.
        cv: Number of cross-validation folds.
        tune_ccp_alpha: Tune ``ccp_alpha`` on the validation set.
        seed: Random state of the tree and of the row shuffle.
    """

    criterion: list[Criterion] = Field(default_factory=_default_criteria)
    max_depth: list[int | None] = Field(default_factory=lambda: [3, 4, 5, 7, None])
    min_samples_leaf: list[int] = Field(default_factory=lambda: [5, 10, 20])
    cv: int = 5
    tune_ccp_alpha: bool = True
    seed: int = 0


class PlottingConfig(_Strict):
    """How often (in epochs) diagnostic figures are written during training."""

    canvas_plot_period: int = 2
    output_plot_period: int = 2


class AutoencoderConfig(_Strict):
    """Convolutional autoencoder baseline.

    Attributes:
        latent_dim: Size of the global latent vector (paper: 200).
        base_channels: Channel multiplier of the conv blocks.
        ssim_weight: Weight of the ``1 - SSIM`` term added to the MSE loss.
    """

    latent_dim: int = 200
    base_channels: int = 16
    ssim_weight: float = 0.05


class Config(_Strict):
    """Root configuration object."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    renderer: RendererConfig = Field(default_factory=RendererConfig)
    modelers: dict[int, ModelerConfig] = Field(default_factory=dict)
    arv: ARVConfig = Field(default_factory=ARVConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated configuration.
    """
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
