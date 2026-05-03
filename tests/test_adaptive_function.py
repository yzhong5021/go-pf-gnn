import pytest
import torch

from src.modules.adaptive_function import AdaptiveFunctionBlock


def test_adaptive_function_forward_shapes() -> None:
    torch.manual_seed(0)
    block = AdaptiveFunctionBlock(d_in=8, d_attn=4, steps=2, p=0.7)
    x = torch.randn(5, 3, 8)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, x)


def test_adaptive_function_prior_validation() -> None:
    block = AdaptiveFunctionBlock(d_in=6, steps=1)
    x = torch.randn(4, 2, 6)
    bad_prior = torch.rand(3, 3)

    with pytest.raises(ValueError):
        _ = block(x, bad_prior)


def test_adaptive_function_attention_rows_sum_to_one() -> None:
    torch.manual_seed(1)
    block = AdaptiveFunctionBlock(d_in=4, d_attn=3, steps=1, p=0.6)
    x = torch.randn(2, 3, 4)

    _ = block(x)

    assert block.last_attention is not None
    row_sums = block.last_attention.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
