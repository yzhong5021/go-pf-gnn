import pytest
import torch

from src.modules.adaptive_protein import AdaptiveProteinBlock


def test_adaptive_protein_forward_with_prior() -> None:
    torch.manual_seed(0)
    block = AdaptiveProteinBlock(d_in=5, d_attn=3, steps=2, p=0.8)
    x = torch.randn(4, 2, 5)
    prior = torch.rand(4, 4)

    out = block(x, prior=prior)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_adaptive_protein_mask_zeroes_outputs() -> None:
    torch.manual_seed(2)
    block = AdaptiveProteinBlock(d_in=4, steps=1)
    x = torch.randn(3, 2, 4)
    mask = torch.tensor([1, 0, 0], dtype=torch.bool)

    out = block(x, mask=mask)

    assert torch.allclose(out[1:], torch.zeros_like(out[1:]))


def test_adaptive_protein_bad_prior_shape() -> None:
    block = AdaptiveProteinBlock(d_in=3, steps=1)
    x = torch.randn(2, 1, 3)
    bad_prior = torch.rand(3, 3)

    with pytest.raises(ValueError):
        _ = block(x, prior=bad_prior)
