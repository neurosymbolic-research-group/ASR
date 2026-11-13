"""Decision-tree classification of bag features (paper Stage 2).

Model selection follows the paper: a grid search with k-fold cross-validation
on the training set over impurity criterion, maximum depth and minimum leaf
size, followed by tuning of the cost-complexity pruning parameter
``ccp_alpha`` on the validation set. The final tree is evaluated on all splits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

from asr.classification.features import LABEL_COLUMN, SLIDE_COLUMN
from asr.config import ClassifierConfig
from asr.data.datasets import CLASS_NAMES, SPLITS
from asr.utils.runtime import get_logger

logger = get_logger(__name__)
COLORS = ("darkorange", "mediumseagreen", "darkorchid")


@dataclass
class SplitData:
    """Features ``X`` and string labels ``y`` of one split."""

    X: pd.DataFrame
    y: pd.Series


def load_split(features_dir: Path, split: str, seed: int) -> SplitData:
    """Load ``<split>_features.csv`` and shuffle its rows with ``seed``."""
    df = pd.read_csv(features_dir / f"{split}_features.csv")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    X = df.drop(columns=[c for c in (LABEL_COLUMN, SLIDE_COLUMN) if c in df.columns])
    return SplitData(X=X, y=df[LABEL_COLUMN].astype(str))


def fit_tree(
    train: SplitData, valid: SplitData, cfg: ClassifierConfig
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    """Select and fit the decision tree.

    Args:
        train: Training split (grid search with ``cfg.cv``-fold CV).
        valid: Validation split (``ccp_alpha`` tuning).
        cfg: Search space and options.

    Returns:
        The fitted tree and a dictionary describing the selection.
    """
    grid = {
        "criterion": cfg.criterion,
        "max_depth": cfg.max_depth,
        "min_samples_leaf": cfg.min_samples_leaf,
    }
    search = GridSearchCV(DecisionTreeClassifier(random_state=cfg.seed), grid, cv=cfg.cv)
    search.fit(train.X, train.y)
    best: DecisionTreeClassifier = search.best_estimator_
    info: dict[str, Any] = {
        "param_grid": grid,
        "best_params": search.best_params_,
        "cv_accuracy": float(search.best_score_),
        "ccp_alpha": 0.0,
    }
    if cfg.tune_ccp_alpha:
        alphas = np.unique(
            np.clip(best.cost_complexity_pruning_path(train.X, train.y).ccp_alphas, 0, None)
        )
        scores = []
        for alpha in alphas:
            tree = clone(best).set_params(ccp_alpha=float(alpha)).fit(train.X, train.y)
            scores.append(accuracy_score(valid.y, tree.predict(valid.X)))
        # Highest validation accuracy; ties broken towards stronger pruning.
        best_idx = max(range(len(alphas)), key=lambda i: (scores[i], alphas[i]))
        info["ccp_alpha"] = float(alphas[best_idx])
        info["ccp_alpha_valid_accuracy"] = float(scores[best_idx])
        best = clone(best).set_params(ccp_alpha=info["ccp_alpha"]).fit(train.X, train.y)
    return best, info


def _plot_metric_bars(reports: dict[str, dict], path: Path, title: str) -> None:
    metric_names = ["precision", "recall", "f1-score"]
    splits = list(reports)
    fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 4))
    idx = np.arange(len(splits))
    width = 0.25
    for ax, metric in zip(axes, metric_names, strict=True):
        for i, c in enumerate(CLASS_NAMES):
            vals = [reports[s].get(c, {}).get(metric, 0.0) for s in splits]
            ax.bar(idx + i * width, vals, width, alpha=0.8, label=c, color=COLORS[i])
        ax.set_title(metric)
        ax.set_xticks(idx + width, splits)
        ax.set_ylim(0, 1)
    axes[-1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_roc(clf: DecisionTreeClassifier, data: SplitData, path: Path, split: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    proba = clf.predict_proba(data.X)
    classes = [str(c) for c in clf.classes_]
    for c, color in zip(CLASS_NAMES, COLORS, strict=True):
        if c not in classes:
            continue
        fpr, tpr, _ = roc_curve((data.y == c).to_numpy(), proba[:, classes.index(c)])
        ax.plot(fpr, tpr, color=color, label=f"{c} (AUC = {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title(f"One-vs-rest ROC ({split})")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def run_classification(features_dir: Path, cfg: ClassifierConfig) -> dict[str, Any]:
    """Fit, evaluate and report the decision tree on the feature tables in ``features_dir``.

    Writes ``results.json`` (per-split classification reports, accuracy and
    confusion matrices), ``tree_params.json``, ``predictions.json``,
    ``metrics_bars.png``, ``roc_{valid,test}.png`` and ``tree.pdf``.

    Returns:
        Summary with per-split accuracy and the selected hyper-parameters.
    """
    data = {s: load_split(features_dir, s, cfg.seed) for s in SPLITS}
    clf, info = fit_tree(data["train"], data["valid"], cfg)
    labels = list(CLASS_NAMES)
    reports: dict[str, dict] = {}
    results: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for split, d in data.items():
        pred = clf.predict(d.X)
        reports[split] = classification_report(
            d.y, pred, labels=labels, output_dict=True, zero_division=0
        )
        results[split] = {
            "accuracy": float(accuracy_score(d.y, pred)),
            "confusion_matrix": confusion_matrix(d.y, pred, labels=labels).tolist(),
            "report": reports[split],
        }
        predictions[split] = {
            "y_true": d.y.tolist(),
            "y_pred": pred.tolist(),
            "probabilities": clf.predict_proba(d.X).tolist(),
        }
        logger.info("%s accuracy: %.4f", split, results[split]["accuracy"])

    (features_dir / "results.json").write_text(json.dumps(results, indent=2))
    (features_dir / "tree_params.json").write_text(json.dumps(info, indent=2, default=str))
    (features_dir / "predictions.json").write_text(json.dumps(predictions))
    _plot_metric_bars(reports, features_dir / "metrics_bars.png", features_dir.parent.name)
    for split in ("valid", "test"):
        _plot_roc(clf, data[split], features_dir / f"roc_{split}.png", split)
    fig = plt.figure(figsize=(40, 20))
    plot_tree(
        clf,
        feature_names=list(data["train"].X.columns),
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        impurity=False,
        fontsize=12,
    )
    fig.savefig(features_dir / "tree.pdf")
    plt.close(fig)
    return {"accuracy": {s: results[s]["accuracy"] for s in SPLITS}, **info}
