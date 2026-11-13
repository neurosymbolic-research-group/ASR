import torch

from asr.baselines.autoencoder.model import Autoencoder


def test_parameter_count_matches_paper():
    model = Autoencoder(latent_dim=200, base_channels=16)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3_781_509


def test_reconstruction_shape():
    model = Autoencoder().eval()
    with torch.no_grad():
        assert model(torch.rand(2, 3, 256, 256)).shape == (2, 3, 256, 256)
