from pathlib import Path

import pytest
from pydantic import ValidationError

from asr.config import Config, load_config


@pytest.mark.parametrize("name", ["base", "regularized", "incremental", "autoencoder"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.dataset.resize_shape == (256, 256)


def test_paper_settings():
    inc = load_config(ROOT / "configs" / "incremental.yaml")
    assert inc.training.num_epochs == 55 and inc.arv.enabled and inc.arv.start_epoch == 35
    assert inc.modelers[1].epoch_start == 8 and inc.modelers[0].epoch_start == 20
    ae = load_config(ROOT / "configs" / "autoencoder.yaml")
    assert ae.autoencoder.latent_dim == 200 and ae.features.aggregations == ["mean"]


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate({"training": {"batchsize": 3}})


ROOT = Path(__file__).resolve().parents[1]
