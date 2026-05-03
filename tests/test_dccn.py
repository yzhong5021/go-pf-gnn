import pytest
import torch

from src.modules.dccn import DCCN_1D


def test_dccn_forward_mask() -> None:
    torch.manual_seed(0)
    model = DCCN_1D(embed_len=8, k_size=3, dilation=2, dropout=0.0)
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)

    out = model(x, mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_dccn_mask_shape_mismatch() -> None:
    model = DCCN_1D(embed_len=4)
    x = torch.randn(1, 4, 4)
    bad_mask = torch.ones(2, 4)

    with pytest.raises(ValueError):
        model(x, bad_mask)
