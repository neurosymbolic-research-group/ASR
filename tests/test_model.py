import dataclasses

import pytest
import torch

from asr.config import ModelerConfig
from asr.model.encoder import Encoder
from asr.model.modeler import DenseModeler, importance_at_epoch

# Reference geometry of the default 256x256 encoder.
REFERENCE_ZS_INFO = [
    {"width": 8, "height": 8, "rj": 39, "s": 32},
    {"width": 4, "height": 4, "rj": 71, "s": 64},
    {"width": 2, "height": 2, "rj": 135, "s": 128},
]


def test_scale_geometry():
    enc = Encoder((3, 256, 256))
    assert [dataclasses.asdict(z) for z in enc.zs_info] == REFERENCE_ZS_INFO


def test_two_encoders_do_not_share_state():
    a, b = Encoder(), Encoder()
    assert len(a.zs_info) == len(b.zs_info) == 3


def test_checkpoint_keys_stable():
    keys = set(Encoder().state_dict())
    for k in [
        "layers.0.conv2d.weight",
        "layers.0.norm.running_mean",
        "layers.5.modeler.layers.0.weight",
        "layers.7.modeler.layers.2.bias",
        "backgroundBlock.linear.weight",
    ]:
        assert k in keys
    assert not any("scaling" in k for k in keys)
    assert Encoder().backgroundBlock.linear.weight.shape == (3, 512)


def test_encoder_output_shapes_and_ranges():
    enc = Encoder().eval()
    out = enc(torch.rand(2, 3, 256, 256))
    assert [tuple(z.shape) for z in out.latents] == [(2, 6, 64), (2, 6, 16), (2, 6, 4)]
    assert out.background.shape == (2, 3)
    for z in out.latents:
        assert (z[:, :2] >= 0.1).all() and (z[:, :2] <= 2.0).all()
        assert (z[:, 3:] >= 0).all() and (z[:, 3:] <= 1).all()


def _stateful_schedule(cfg: ModelerConfig, epochs: int, steps: int = 32) -> list[float]:
    """Equivalent stateful formulation (counts 2*steps forward calls per epoch)."""
    importance, n_batch, n_epoch, allow = cfg.start_value, 0, 0.0, True
    values = []
    for _ in range(epochs):
        for _ in range(2 * steps):
            if (
                cfg.freeze
                and (n_epoch + 1) % cfg.epoch_period == 0
                and n_epoch >= cfg.epoch_start
                and allow
            ):
                importance = min(importance + cfg.increment_value, 1.0)
                allow = False
            n_batch += 1
            if n_batch == steps:
                n_epoch += 0.5
                n_batch = 0
                allow = True
        values.append(importance if cfg.freeze else 1.0)
    return values


@pytest.mark.parametrize(
    "cfg",
    [
        ModelerConfig(
            freeze=True, start_value=0.01, increment_value=0.1, epoch_period=1, epoch_start=20
        ),
        ModelerConfig(
            freeze=True, start_value=0.01, increment_value=0.1, epoch_period=1, epoch_start=8
        ),
        ModelerConfig(
            freeze=True, start_value=0.01, increment_value=0.1, epoch_period=10, epoch_start=0
        ),
        ModelerConfig(
            freeze=True, start_value=0.3, increment_value=0.25, epoch_period=3, epoch_start=5
        ),
        ModelerConfig(),
    ],
)
def test_importance_schedule(cfg):
    assert [importance_at_epoch(cfg, e) for e in range(60)] == _stateful_schedule(cfg, 60)


def test_modeler_applies_importance():
    cfg = ModelerConfig(freeze=True, start_value=0.5, increment_value=0.0)
    m = DenseModeler(8, cfg)
    x = torch.rand(1, 8, 2, 2)
    frozen = m(x)
    m.config = ModelerConfig()
    torch.testing.assert_close(frozen, m(x) * 0.5)
