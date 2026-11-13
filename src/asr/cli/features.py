"""``asr-features``: build Stage-2 bag feature tables from a trained encoder."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from asr.baselines.autoencoder.train import load_trained_autoencoder
from asr.classification.features import (
    ae_bag_features,
    ae_feature_names,
    asr_bag_features,
    asr_feature_names,
    build_feature_tables,
)
from asr.cli.common import base_parser, resolve
from asr.training.asr import load_trained_encoder
from asr.utils.runtime import get_device


def main() -> None:
    """Write ``<run>/classification/{train,valid,test}_features.csv``."""
    p = base_parser("Extract bag-of-patches features for classification.")
    p.add_argument("--model", choices=["asr", "ae"], default="asr")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint (default: <run>/models/<testing.weights>)",
    )
    args = p.parse_args()
    config, paths = resolve(args)
    device = get_device(args.device)
    weights = args.weights or paths.models / config.testing.weights
    aggs = list(config.features.aggregations)
    if args.model == "asr":
        encoder = load_trained_encoder(config, weights, device)
        featurize = partial(asr_bag_features, encoder, aggregations=aggs)
        columns = asr_feature_names(encoder.n_scales, aggs)
    else:
        encoder = load_trained_autoencoder(config, weights, device).encoder
        featurize = partial(ae_bag_features, encoder, aggregations=aggs)
        columns = ae_feature_names(config.autoencoder.latent_dim, aggs)
    build_feature_tables(config, encoder, featurize, columns, paths.classification, device)


if __name__ == "__main__":
    main()
