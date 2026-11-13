"""``asr-classify``: decision-tree classification of the bag feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr.classification.decision_tree import run_classification
from asr.config import load_config
from asr.utils.paths import RunPaths


def main() -> None:
    """Fit/evaluate the tree on ``<run>/classification/*_features.csv`` and write reports there."""
    p = argparse.ArgumentParser(description="Decision-tree classification of bag features.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=None, help="Override classifier.seed")
    args = p.parse_args()
    cfg = load_config(args.config).classifier
    if args.seed is not None:
        cfg.seed = args.seed
    summary = run_classification(RunPaths.of(args.results_root, args.run).classification, cfg)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
