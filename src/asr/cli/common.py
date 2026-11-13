"""Shared command-line arguments."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr.config import Config, load_config
from asr.utils.paths import RunPaths


def base_parser(description: str) -> argparse.ArgumentParser:
    """Parser with ``--config``, ``--run``, ``--results-root``, ``--data-root`` and ``--device``."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True, help="YAML config file")
    p.add_argument("--run", required=True, help="Run name (subdirectory of --results-root)")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--data-root", type=Path, default=None, help="Override dataset.root")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    return p


def resolve(args: argparse.Namespace) -> tuple[Config, RunPaths]:
    """Load the config (applying CLI overrides) and the run paths."""
    config = load_config(args.config)
    if args.data_root is not None:
        config.dataset.root = args.data_root
    return config, RunPaths.of(args.results_root, args.run)
