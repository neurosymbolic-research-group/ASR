"""End-to-end smoke test on a tiny synthetic dataset (CPU)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asr.baselines.autoencoder.train import evaluate_autoencoder, train_autoencoder
from asr.classification.decision_tree import run_classification
from asr.classification.features import (
    ae_bag_features,
    ae_feature_names,
    asr_bag_features,
    asr_feature_names,
    build_feature_tables,
)
from asr.config import load_config
from asr.training.asr import evaluate_asr, load_trained_encoder, train_asr
from asr.utils.paths import RunPaths
from asr.utils.runtime import get_device


def _config(name, root):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    cfg.dataset.root = root
    cfg.training.batch_size = 2
    cfg.training.num_epochs = 2
    cfg.training.steps_per_epoch = 2
    cfg.testing.batch_size = 4
    cfg.features.bag_size = 2
    cfg.classifier.cv = 2
    cfg.classifier.min_samples_leaf = [1]
    return cfg


@pytest.mark.parametrize("variant", ["incremental"])
def test_asr_pipeline(tiny_dataset, tmp_path, variant):
    device = get_device("cpu")
    cfg = _config(variant, tiny_dataset)
    cfg.arv.start_epoch = 1
    paths = RunPaths.of(tmp_path / "results", variant)
    paths.prepare_training(overwrite=False)
    history = train_asr(cfg, paths, device)
    assert len(history["train_loss"]) == 2 and np.isfinite(history["valid_loss"]).all()
    assert (paths.models / "best.pt").exists() and (paths.models / "last.pt").exists()

    metrics = evaluate_asr(cfg, paths, device)
    assert {"mse", "mae", "ssim", "mmse", "MEAN_ACT_S0"} <= set(metrics)

    enc = load_trained_encoder(cfg, paths.models / "best.pt", device)
    cols = asr_feature_names(enc.n_scales, cfg.features.aggregations)
    written = build_feature_tables(
        cfg,
        enc,
        lambda b: asr_bag_features(enc, b, cfg.features.aggregations),
        cols,
        paths.classification,
        device,
    )
    df = pd.read_csv(written["train"])
    assert len(cols) == 36 and list(df.columns[:36]) == cols and len(df) == 12

    summary = run_classification(paths.classification, cfg.classifier)
    assert set(summary["accuracy"]) == {"train", "valid", "test"}
    assert json.loads((paths.classification / "results.json").read_text())["test"]["accuracy"] >= 0


def test_autoencoder_pipeline(tiny_dataset, tmp_path):
    device = get_device("cpu")
    cfg = _config("autoencoder", tiny_dataset)
    cfg.training.early_stopping_on = "valid"
    cfg.features.dtype = "float32"
    paths = RunPaths.of(tmp_path / "results", "ae")
    paths.prepare_training(overwrite=False)
    train_autoencoder(cfg, paths, device)
    metrics = evaluate_autoencoder(cfg, paths, device)
    assert {"mse", "mae", "ssim", "mmse"} <= set(metrics)

    from asr.baselines.autoencoder.train import load_trained_autoencoder

    enc = load_trained_autoencoder(cfg, paths.models / "best.pt", device).encoder
    cols = ae_feature_names(200, ["mean"])
    build_feature_tables(
        cfg, enc, lambda b: ae_bag_features(enc, b, ["mean"]), cols, paths.classification, device
    )
    run_classification(paths.classification, cfg.classifier)


def test_existing_run_requires_overwrite(tmp_path):
    paths = RunPaths.of(tmp_path, "run")
    paths.prepare_training(overwrite=False)
    (paths.root / "marker").write_text("x")
    with pytest.raises(FileExistsError):
        paths.prepare_training(overwrite=False)
    paths.prepare_training(overwrite=True)
    assert not (paths.root / "marker").exists()


ROOT = Path(__file__).resolve().parents[1]
