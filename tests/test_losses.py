import torch
import torch.nn.functional as F

from asr.losses.appearance import LOSS_OFFSET, AppearanceRegularizer
from asr.losses.masked import MaskedMSELoss


def test_masked_mse_ignores_border():
    loss = MaskedMSELoss((16, 16), n_border_ignored=4)
    target = torch.zeros(1, 3, 16, 16)
    pred = target.clone()
    pred[..., :4, :] = 1.0  # error only in the border
    assert loss(pred, target).item() == 0.0


def test_masked_mse_without_border_is_mse():
    loss = MaskedMSELoss((16, 16), n_border_ignored=0)
    a, b = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    torch.testing.assert_close(loss(a, b), F.mse_loss(a, b))


def test_appearance_regularizer_matches_definition():
    z = [torch.rand(2, 6, 4), torch.rand(2, 6, 1)]
    reg = AppearanceRegularizer(2, lambda_a=0.5, alpha=0.75, scale_weights=[1.0, 2.0])
    per_img = (z[0][:, 3:] ** 0.75).sum((1, 2)) + 2.0 * (z[1][:, 3:] ** 0.75).sum((1, 2))
    expected = (per_img / 5).mean() * 0.5 + LOSS_OFFSET
    torch.testing.assert_close(reg(z), expected)
