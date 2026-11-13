"""Stage 1: train and evaluate the ASR encoder through the renderer."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from asr.config import Config
from asr.data.datasets import cycle_batches, default_workers, image_folder, make_loader
from asr.losses.appearance import AppearanceRegularizer
from asr.losses.masked import MaskedMSELoss
from asr.metrics.primitive_stats import PrimitiveStats
from asr.metrics.reconstruction import build_metrics
from asr.model.encoder import Encoder, EncoderOutput
from asr.renderer.renderer import Renderer
from asr.training.checkpoint import EarlyStopping, load_checkpoint, save_checkpoint
from asr.utils.paths import RunPaths
from asr.utils.runtime import get_logger
from asr.viz.plots import plot_canvases, plot_loss, plot_reconstructions

logger = get_logger(__name__)


def build_encoder(config: Config) -> Encoder:
    """Construct the encoder described by ``config``."""
    return Encoder(config.dataset.img_size, modelers_configs=config.modelers)


def build_renderer(config: Config, encoder: Encoder) -> Renderer:
    """Construct a renderer matching the encoder's scale geometry."""
    return Renderer(config.dataset.resize_shape, encoder.zs_info, beta=config.renderer.beta)


@dataclass
class Reconstruction:
    """Output of :func:`reconstruct`.

    Attributes:
        images: ``(B, 3, H, W)`` rendered images.
        encoded: Encoder output (latents and background colour).
        canvases: Per-scale canvases and background of the first image, if requested.
    """

    images: Tensor
    encoded: EncoderOutput
    canvases: tuple[list[Tensor], Tensor] | None


def reconstruct(
    encoder: Encoder, renderer: Renderer, images: Tensor, return_canvases: bool = False
) -> Reconstruction:
    """Encode ``images`` ``(B, 3, H, W)`` and render them back."""
    encoded = encoder(images)
    rendered = renderer(encoded.latents, encoded.background, return_canvases=return_canvases)
    images_chw = rendered.images.permute(0, 3, 1, 2).contiguous()
    return Reconstruction(images_chw, encoded, rendered.canvases)


def _save_plots(
    paths: RunPaths, split: str, epoch: int, batch: Tensor, rec: Reconstruction, n: int = 10
) -> None:
    inputs = batch.permute(0, 2, 3, 1).detach().cpu().numpy()[:n]
    outputs = rec.images.permute(0, 2, 3, 1).detach().cpu().numpy()[:n]
    plot_reconstructions(
        inputs,
        outputs,
        paths.split_plots(split, "rendered_images") / f"{split}_epoch_{epoch + 1}_batch_1.png",
        title=f"Sample rendered images epoch {epoch + 1}",
        n=n,
    )
    if rec.canvases is not None:
        canvases, bg = rec.canvases
        plot_canvases(
            canvases,
            bg,
            rec.images[0].permute(1, 2, 0),
            paths.split_plots(split, "canvases") / f"canvas_epoch{epoch + 1}_batch_el_1.png",
        )


def run_epoch(
    encoder: Encoder,
    renderer: Renderer,
    batches: Iterable[Tensor],
    loss_fn: MaskedMSELoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    arv: AppearanceRegularizer | None = None,
    grad_clip_norm: float | None = None,
    plot: tuple[RunPaths, str, int, bool, bool] | None = None,
) -> float:
    """Run one training (``optimizer`` given) or validation epoch.

    Args:
        encoder: The encoder.
        renderer: The renderer.
        batches: Finite iterable of image batches for this epoch.
        loss_fn: Reconstruction loss.
        device: Compute device.
        optimizer: If given, the encoder is trained; otherwise evaluated without grad.
        arv: Optional appearance regulariser added to the loss.
        grad_clip_norm: Gradient-norm clipping threshold.
        plot: ``(paths, split, epoch, plot_outputs, plot_canvases)`` to save
            figures of the first batch.

    Returns:
        Mean per-batch loss.
    """
    training = optimizer is not None
    encoder.train(training)
    losses: list[float] = []
    with torch.set_grad_enabled(training):
        for b, images in enumerate(tqdm(batches, leave=False, disable=not training)):
            images = images.to(device, non_blocking=True)
            want_canvases = plot is not None and plot[4] and b == 0
            rec = reconstruct(encoder, renderer, images, return_canvases=want_canvases)
            loss = loss_fn(rec.images, images)
            if arv is not None:
                loss = loss + arv(rec.encoded.latents)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip_norm)
                optimizer.step()
            if plot is not None and b == 0 and (plot[3] or plot[4]):
                paths, split, epoch, _, _ = plot
                _save_plots(paths, split, epoch, images, rec)
            losses.append(loss.item())
    return float(np.mean(losses))


def _take(it: Iterator[Tensor], n: int) -> Iterator[Tensor]:
    for _ in range(n):
        yield next(it)


def train_asr(config: Config, paths: RunPaths, device: torch.device) -> dict[str, list[float]]:
    """Train the ASR encoder (paper Stage 1) and save checkpoints/plots to ``paths``.

    Returns:
        Loss history with keys ``train_loss`` and ``valid_loss``.
    """
    tc, pc = config.training, config.plotting
    workers = tc.num_workers if tc.num_workers is not None else default_workers(device)
    pin = device.type == "cuda"
    root, shape = config.dataset.root, config.dataset.resize_shape
    train_loader = make_loader(
        image_folder(root, "train", shape, tc.augment), tc.batch_size, True, workers, pin
    )
    valid_loader = make_loader(
        image_folder(root, "valid", shape), tc.batch_size, not tc.valid_full_pass, workers, pin
    )
    train_it, valid_it = cycle_batches(train_loader), cycle_batches(valid_loader)
    steps = tc.steps_per_epoch or len(train_loader)

    encoder = build_encoder(config).to(device)
    renderer = build_renderer(config, encoder)
    loss_fn = MaskedMSELoss(shape, tc.n_border_ignored).to(device)
    arv = (
        AppearanceRegularizer(
            encoder.n_scales, config.arv.lambda_a, config.arv.alpha, config.arv.scale_weights
        )
        if config.arv.enabled
        else None
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=tc.lr)
    early_stopping = EarlyStopping(patience=tc.patience)
    history: dict[str, list[float]] = {"train_loss": [], "valid_loss": []}
    for i, info in enumerate(encoder.zs_info):
        logger.info("Scale %d: %s, modeler %s", i, info, config.modelers.get(i))

    best_epoch, epoch_times, start = -1, [], time.time()
    for epoch in range(tc.num_epochs):
        t0 = time.time()
        encoder.set_epoch(epoch)
        last = epoch + 1 == tc.num_epochs
        plot_out = last or epoch % pc.output_plot_period == 0
        plot_can = last or epoch % pc.canvas_plot_period == 0
        epoch_arv = arv if (arv is not None and epoch >= config.arv.start_epoch) else None

        train_loss = run_epoch(
            encoder,
            renderer,
            _take(train_it, steps),
            loss_fn,
            device,
            optimizer=optimizer,
            arv=epoch_arv,
            grad_clip_norm=tc.grad_clip_norm,
            plot=(paths, "train", epoch, plot_out, plot_can),
        )
        valid_batches = (
            (x for x, _ in valid_loader) if tc.valid_full_pass else _take(valid_it, steps)
        )
        valid_loss = run_epoch(
            encoder,
            renderer,
            valid_batches,
            loss_fn,
            device,
            arv=epoch_arv,
            plot=(paths, "valid", epoch, plot_out, plot_can),
        )
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        epoch_times.append(time.time() - t0)
        logger.info(
            "EPOCH %d/%d train %.6f valid %.6f (%.1fs)",
            epoch + 1,
            tc.num_epochs,
            train_loss,
            valid_loss,
            epoch_times[-1],
        )

        if tc.save_model and valid_loss <= min(history["valid_loss"]):
            best_epoch = epoch
            save_checkpoint(
                paths.models / "best.pt", encoder, optimizer, epoch, train_loss, valid_loss
            )
        if early_stopping(train_loss if tc.early_stopping_on == "train" else valid_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    logger.info("Average epoch time %.1fs", sum(epoch_times) / len(epoch_times))
    logger.info("Training time %.1fs; best model at epoch %d", time.time() - start, best_epoch + 1)
    if tc.save_model:
        save_checkpoint(
            paths.models / "last.pt",
            encoder,
            optimizer,
            epoch,
            history["train_loss"][-1],
            history["valid_loss"][-1],
        )
    plot_loss(history, f"Training and Validation {loss_fn}", paths.plots / "loss_plot.png")
    return history


def load_trained_encoder(config: Config, weights: Path, device: torch.device) -> Encoder:
    """Build the encoder, load ``weights`` and set the incremental schedule to the saved epoch."""
    encoder = build_encoder(config)
    ckpt = load_checkpoint(weights, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("epoch") is not None:
        encoder.set_epoch(int(ckpt["epoch"]))
    return encoder.to(device)


@torch.no_grad()
def evaluate_asr(
    config: Config, paths: RunPaths, device: torch.device, weights: Path | None = None
) -> dict[str, float]:
    """Evaluate reconstruction quality on the test split (paper Table 1).

    Reports MSE/MAE/SSIM, the masked MSE (``mmse``) and primitive
    statistics, writes ``<run>/metrics.csv`` and the appearance histogram.


    Returns:
        Flat metrics dictionary.
    """
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
    encoder = load_trained_encoder(config, weights or paths.models / tc.weights, device)
    encoder.eval()
    renderer = build_renderer(config, encoder)
    mmse = MaskedMSELoss(config.dataset.resize_shape, config.training.n_border_ignored).to(device)
    metrics = build_metrics(tc.metrics).to(device)
    stats = PrimitiveStats(encoder.n_scales)
    mmse_values = []
    for b, (images, _) in enumerate(tqdm(loader, unit="batch")):
        images = images.to(device)
        rec = reconstruct(encoder, renderer, images, return_canvases=tc.plot and b == 0)
        stats.update(rec.encoded.latents)
        metrics.update(rec.images, images)
        mmse_values.append(mmse(rec.images, images).item())
        if tc.plot and b == 0:
            _save_plots(paths, "test", 0, images, rec)
    results = {k: float(v) for k, v in metrics.compute().items()}
    results["mmse"] = float(np.mean(mmse_values))
    results.update(stats.compute())
    stats.plot_histogram(paths.plots / "appearance_hist.pdf")
    _write_metrics(paths.root / "metrics.csv", results)
    return results


def _write_metrics(path: Path, results: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(results)
    path.write_text(",".join(keys) + "\n" + ",".join(f"{results[k]:.6g}" for k in keys) + "\n")


def save_config(config: Config, path: Path) -> None:
    """Write the effective (validated) configuration as YAML."""
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
