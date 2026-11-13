"""``asr-test``: reconstruction metrics on the test split."""

from __future__ import annotations

import json
from pathlib import Path

from asr.baselines.autoencoder.train import evaluate_autoencoder
from asr.cli.common import base_parser, resolve
from asr.training.asr import evaluate_asr
from asr.utils.runtime import get_device, seed_everything


def main() -> None:
    """Evaluate a trained ASR encoder (or ``--model ae``) and write ``<run>/metrics.csv``."""
    p = base_parser("Evaluate reconstruction quality on the test split.")
    p.add_argument("--model", choices=["asr", "ae"], default="asr")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint (default: <run>/models/<testing.weights>)",
    )
    args = p.parse_args()
    config, paths = resolve(args)
    seed_everything(config.training.seed)
    device = get_device(args.device)
    evaluate = evaluate_asr if args.model == "asr" else evaluate_autoencoder
    print(json.dumps(evaluate(config, paths, device, args.weights), indent=2))


if __name__ == "__main__":
    main()
