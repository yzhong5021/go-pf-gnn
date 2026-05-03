import pytest
import torch

from src.modules.seq_gating import SeqGating


def test_seq_gating_forward_with_lengths() -> None:
    torch.manual_seed(0)
    module = SeqGating(d_shared=16, d_esm=8, c_dcc=5)
    esm = torch.randn(2, 4, 8)
    dcc = torch.randn(2, 4, 5)
    lengths = torch.tensor([4, 2])

    out = module(esm, dcc, lengths=lengths)

    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_seq_gating_forward_with_mask() -> None:
    torch.manual_seed(1)
    module = SeqGating(d_shared=12, d_esm=7, c_dcc=3)
    esm = torch.randn(1, 3, 7)
    dcc = torch.randn(1, 3, 3)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)

    out = module(esm, dcc, mask=mask)

    assert out.shape == (1, 12)
    assert torch.isfinite(out).all()


def test_seq_gating_requires_valid_mask_or_lengths() -> None:
    module = SeqGating(d_shared=8, d_esm=6, c_dcc=2)
    esm = torch.randn(1, 2, 6)
    dcc = torch.randn(1, 2, 2)

    with pytest.raises(ValueError):
        module(esm, dcc)

    with pytest.raises(ValueError):
        module(esm, dcc, mask=torch.ones(1, 2, 1))
