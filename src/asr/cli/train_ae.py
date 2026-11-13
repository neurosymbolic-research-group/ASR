"""``asr-train-ae``: train the convolutional autoencoder baseline."""

from __future__ import annotations

from asr.baselines.autoencoder.train import train_autoencoder
from asr.cli.common import base_parser, resolve
from asr.training.asr import save_config
from asr.utils.runtime import get_device, seed_everything


def main() -> None:
    """Train the baseline as specified by ``--config``."""
    p = base_parser("Train the autoencoder baseline.")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing run directory")
    args = p.parse_args()
    config, paths = resolve(args)
    paths.prepare_training(args.overwrite)
    save_config(config, paths.root / "config.yaml")
    seed_everything(config.training.seed)
    train_autoencoder(config, paths, get_device(args.device))


if __name__ == "__main__":
    main()
