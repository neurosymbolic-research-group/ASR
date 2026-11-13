"""``asr-train``: Stage-1 training of the ASR encoder."""

from __future__ import annotations

from asr.cli.common import base_parser, resolve
from asr.training.asr import save_config, train_asr
from asr.utils.runtime import get_device, seed_everything


def main() -> None:
    """Train ASR as specified by ``--config``; outputs go to ``<results-root>/<run>``."""
    p = base_parser("Train the ASR encoder by reconstruction.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing run directory")
    args = p.parse_args()
    config, paths = resolve(args)
    paths.prepare_training(args.overwrite)
    save_config(config, paths.root / "config.yaml")
    seed_everything(config.training.seed)
    train_asr(config, paths, get_device(args.device))


if __name__ == "__main__":
    main()
