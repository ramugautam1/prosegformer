import torch

from prosegformer.model import DEFAULT, LITE, build_model, size_multiple


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_output_shapes():
    m = build_model().eval()
    for shape in [(1, 1, 13, 64, 64), (2, 1, 13, 32, 32)]:
        with torch.no_grad():
            y, aux = m(torch.randn(shape), aux=True)
        assert y.shape == (shape[0], 2) + shape[2:]
        assert [a.shape[-1] for a in aux] == [shape[-1] // 2, shape[-1] // 4]
        assert all(a.shape[2] == 13 for a in aux)  # depth is never downsampled


def test_lite_has_no_attention():
    full, lite = build_model(DEFAULT), build_model(LITE)
    assert not any(b.attn is not None for s in lite.stages for b in s)
    assert all(b.attn is not None for s in full.stages for b in s)
    assert round(n_params(full) / 1e6, 1) == 1.2
    assert round(n_params(lite) / 1e6, 1) == 1.0


def test_mask_token_replaces_stem_features():
    m = build_model().eval()
    x = torch.randn(1, 1, 13, 32, 32)
    mask = torch.zeros_like(x)
    mask[..., :16, :] = 1
    with torch.no_grad():
        a, b = m(x, mask=mask), m(x, mask=torch.zeros_like(x))
    assert not torch.allclose(a, b)


def test_size_multiple():
    assert size_multiple(DEFAULT) == 16
