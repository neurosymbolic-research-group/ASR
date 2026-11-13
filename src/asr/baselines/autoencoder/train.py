"""Training and evaluation of the autoencoder baseline."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from piqa import SSIM
from torch import Tensor, nn
from tqdm import tqdm

from asr.baselines.autoencoder.model import Autoencoder
from asr.config import Config
from asr.data.datasets import cycle_batches, default_workers, image_folder, make_loader
from asr.losses.masked import MaskedMSELoss
from asr.metrics.reconstruction import build_metrics
from asr.training.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from asr.utils.paths import RunPaths
from asr.utils.runtime import get_logger
from asr.viz.plots import plot_loss, plot_reconstructions

logger = get_logger(__name__)


class AELoss(nn.Module):
    """``MSE + ssim_weight * (1 - SSIM)`` .

    Args:
        ssim_weight: Weight of the SSIM term.
    """

    def __init__(self, ssim_weight: float = 0.05) -> None:
        super().__init__()
        self.ssim = SSIM()
        self.ssim_weight = ssim_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the loss for ``(B, 3, H, W)`` images in ``[0, 1]``."""
        return F.mse_loss(pred, target) + self.ssim_weight * (1.0 - self.ssim(pred, target))


def build_autoencoder(config: Config) -> Autoencoder:
    """Construct the baseline autoencoder from ``config.autoencoder``."""
    return Autoencoder(config.autoencoder.latent_dim, config.autoencoder.base_channels)


def _run_epoch(
    model: Autoencoder,
    batches: Iterable[Tensor],
    loss_fn: AELoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    plot: tuple[RunPaths, str, int] | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    with torch.set_grad_enabled(training):
        for b, images in enumerate(tqdm(batches, leave=False, disable=not training)):
            images = images.to(device, non_blocking=True)
            out = model(images)
            loss = loss_fn(out, images)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if plot is not None and b == 0:
                paths, split, epoch = plot
                plot_reconstructions(
                    images.permute(0, 2, 3, 1).detach().cpu().numpy(),
                    out.permute(0, 2, 3, 1).detach().cpu().numpy(),
                    paths.split_plots(split, "rendered_images")
                    / f"{split}_epoch_{epoch + 1}_batch_1.png",
                    title=f"Sample reconstructions epoch {epoch + 1}",
                )
            losses.append(loss.item())
    return float(np.mean(losses))


def _take(it: Iterator[Tensor], n: int) -> Iterator[Tensor]:
    for _ in range(n):
        yield next(it)


def train_autoencoder(
    config: Config, paths: RunPaths, device: torch.device
) -> dict[str, list[float]]:
    """Train the baseline autoencoder; checkpoints/plots go to ``paths``.

    Training uses ``training.steps_per_epoch`` random batches per epoch, and
    validation is one full pass over the validation split.

    Returns:
        Loss history with keys ``train_loss`` and ``valid_loss``.
    """
    tc = config.training
    workers = tc.num_workers if tc.num_workers is not None else default_workers(device)
    pin = device.type == "cuda"
    root, shape = config.dataset.root, config.dataset.resize_shape
    train_loader = make_loader(
        image_folder(root, "train", shape, tc.augment), tc.batch_size, True, workers, pin
    )
    valid_loader = make_loader(
        image_folder(root, "valid", shape), tc.batch_size, True, workers, pin
    )
    train_it = cycle_batches(train_loader)
    steps = tc.steps_per_epoch or len(train_loader)

    model = build_autoencoder(config).to(device)
    loss_fn = AELoss(config.autoencoder.ssim_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)
    early_stopping = EarlyStopping(patience=tc.patience)
    history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}
    period = config.plotting.output_plot_period

    start = time.time()
    for epoch in range(tc.num_epochs):
        do_plot = epoch % period == 0 or epoch + 1 == tc.num_epochs
        train_loss = _run_epoch(
            model,
            _take(train_it, steps),
            loss_fn,
            device,
            optimizer,
            (paths, "train", epoch) if do_plot else None,
        )
        valid_loss = _run_epoch(
            model,
            (x for x, _ in valid_loader),
            loss_fn,
            device,
            None,
            (paths, "valid", epoch) if do_plot else None,
        )
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        logger.info(
            "EPOCH %d/%d train %.6f valid %.6f", epoch + 1, tc.num_epochs, train_loss, valid_loss
        )
        if tc.save_model and valid_loss <= min(history["valid_loss"]):
            save_checkpoint(
                paths.models / "best.pt", model, optimizer, epoch, train_loss, valid_loss
            )
        if early_stopping(train_loss if tc.early_stopping_on == "train" else valid_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break
    logger.info("Training time %.1fs", time.time() - start)
    if tc.save_model:
        save_checkpoint(
            paths.models / "last.pt",
            model,
            optimizer,
            epoch,
            history["train_loss"][-1],
            history["valid_loss"][-1],
        )
    plot_loss(history, "Autoencoder loss (MSE + SSIM term)", paths.plots / "loss_plot.png")
    return history


def load_trained_autoencoder(config: Config, weights: Path, device: torch.device) -> Autoencoder:
    """Load a trained autoencoder.

    Accepts both the full-model checkpoints written by :func:`train_autoencoder`
    and encoder-only state dicts (in which case the decoder stays randomly
    initialised and only feature extraction is meaningful).
    """
    model = build_autoencoder(config)
    state = load_checkpoint(weights, map_location=device)["model_state_dict"]
    if not any(k.startswith("encoder.") for k in state):
        model.encoder.load_state_dict(state)
        logger.warning("Loaded an encoder-only state dict; the decoder is untrained.")
    else:
        model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def evaluate_autoencoder(
    config: Config, paths: RunPaths, device: torch.device, weights: Path | None = None
) -> dict[str, float]:
    """Reconstruction metrics of the baseline on the test split (paper Table 1)."""
    tc = config.testing
    workers = (
        config.training.num_workers
        if config.training.num_workers is not None
        else default_workers(device)
    )
    loader = make_loader(
        image_folder(config.dataset.root, "test", config.dataset.resize_shape),
        tc.batch_size,
        True,
        workers,
        device.type == "cuda",
    )
    model = load_trained_autoencoder(config, weights or paths.models / tc.weights, device).eval()
    metrics = build_metrics(tc.metrics).to(device)
    mmse = MaskedMSELoss(config.dataset.resize_shape, config.training.n_border_ignored).to(device)
    mmse_values = []
    for b, (images, _) in enumerate(tqdm(loader, unit="batch")):
        images = images.to(device)
        out = model(images)
        metrics.update(out, images)
        mmse_values.append(mmse(out, images).item())
        if tc.plot and b == 0:
            plot_reconstructions(
                images.permute(0, 2, 3, 1).cpu().numpy(),
                out.permute(0, 2, 3, 1).cpu().numpy(),
                paths.split_plots("test", "rendered_images") / "test_batch_1.png",
                title="Test reconstructions",
            )
    results = {k: float(v) for k, v in metrics.compute().items()}
    results["mmse"] = float(np.mean(mmse_values))
    out_path = paths.root / "metrics.csv"
    out_path.write_text(
        ",".join(results) + "\n" + ",".join(f"{v:.6g}" for v in results.values()) + "\n"
    )
    return results
