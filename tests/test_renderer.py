import torch

from asr.model.encoder import Encoder
from asr.renderer.compose import combine_primitives, merge_canvases
from asr.renderer.primitives import affine_transform, fuzzy_disc
from asr.renderer.renderer import Renderer


def test_fuzzy_disc_shape_and_center():
    img = fuzzy_disc(rj=5, beta=1.0)
    assert img.shape == (11, 11)
    assert img[5, 5] > 0.8 and img[0, 0] < 0.05  # disc membership: ~1 inside, ~0 outside


def test_affine_identity_preserves_image():
    img = fuzzy_disc(7, 1.0)
    out = affine_transform(img, torch.ones(2), torch.ones(2), torch.zeros(2), "cpu")
    torch.testing.assert_close(out, img.expand(2, -1, -1))


def test_single_primitive_does_not_collapse():
    img = fuzzy_disc(3, 1.0)
    out = affine_transform(img, torch.ones(1), torch.ones(1), torch.zeros(1), "cpu")
    assert out.shape == (1, 7, 7)


def test_compose_shapes():
    prims = torch.zeros(4, 5, 5, 3)
    canvas = combine_primitives(prims, 2, 2, spacing=3, window=5, device="cpu")
    assert canvas.shape == (8, 8, 3) and torch.all(canvas == 1)
    merged = merge_canvases([canvas], torch.full((4, 4, 3), 0.5))
    assert torch.all(merged == 0.5)


def test_renderer_end_to_end_differentiable():
    enc = Encoder()
    x = torch.rand(2, 3, 256, 256)
    out = enc(x)
    rendered = Renderer((256, 256), enc.zs_info)(out.latents, out.background, return_canvases=True)
    assert rendered.images.shape == (2, 256, 256, 3)
    assert rendered.canvases is not None and len(rendered.canvases[0]) == 3
    assert rendered.images.min() >= 0 and rendered.images.max() <= 1
    rendered.images.mean().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
