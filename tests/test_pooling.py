"""Unit tests for the attention pooling module."""

import torch
import pytest

from src.modules.pooling import AdaptivePooling

def test_adaptive_pooling_reduces_dimension() -> None:
    torch.manual_seed(0)
    pooling = AdaptivePooling(embed_dim=16, attn_hidden=8, dropout=0.0)
    x = torch.randn(5, 7, 16)

    output = pooling(x)

    assert output.shape == (5, 16)
    assert torch.isfinite(output).all()


def test_adaptive_pooling_respects_mask() -> None:
    pooling = AdaptivePooling(embed_dim=8, attn_hidden=4, dropout=0.0)
    x = torch.ones(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)

    output = pooling(x, mask=mask)

    assert torch.isfinite(output).all()
    assert torch.allclose(output[1], torch.zeros(8))


def test_adaptive_pooling_validates_rank() -> None:
    pooling = AdaptivePooling(embed_dim=4)

    with pytest.raises(ValueError):
        _ = pooling(torch.randn(2, 4))
