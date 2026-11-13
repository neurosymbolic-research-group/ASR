"""``asr-patches``: extract tissue patches from ``.svs`` whole-slide images."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr.data.patches import extract_slide


def main() -> None:
    """Extract patches from every ``*.svs`` in ``--input`` into ``--output/<slide>/``."""
    p = argparse.ArgumentParser(description="Extract patches from Aperio .svs slides.")
    p.add_argument("--input", type=Path, default=Path("dataset/raw_gtex"))
    p.add_argument("--output", type=Path, default=Path("dataset/patched_gtex"))
    p.add_argument("--window-size", type=int, default=256)
    p.add_argument("--rescale-factor", type=int, default=4)
    p.add_argument("--stride", type=int, default=635)
    p.add_argument("--tissue-threshold", type=float, default=80)
    p.add_argument("--n-windows", type=int, default=5000)
    args = p.parse_args()
    slides = sorted(args.input.glob("*.svs"))
    if not slides:
        raise SystemExit(f"No .svs files in {args.input}")
    for slide in slides:
        extract_slide(
            slide,
            args.output,
            (args.window_size, args.window_size),
            args.rescale_factor,
            args.stride,
            args.tissue_threshold,
            args.n_windows,
        )


if __name__ == "__main__":
    main()
