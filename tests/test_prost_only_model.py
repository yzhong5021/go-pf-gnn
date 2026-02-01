from types import SimpleNamespace

import pytest
import torch

from src.model.prost_only_model import ProstOnlyPFAGCN


def _make_cfg(num_functions: int, dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(num_functions=num_functions),
        model=SimpleNamespace(prostt5_3di=SimpleNamespace(encoder_dim=dim)),
    )


def test_prost_only_shapes() -> None:
    torch.manual_seed(7)
    cfg = _make_cfg(num_functions=4, dim=8)
    model = ProstOnlyPFAGCN(cfg)
    prost = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 3])
    outputs = model(prost, lengths=lengths)
    assert outputs.logits.shape == (2, 4)


def test_prost_only_mask_mismatch() -> None:
    cfg = _make_cfg(num_functions=3, dim=8)
    model = ProstOnlyPFAGCN(cfg)
    prost = torch.randn(1, 4, 8)
    bad_mask = torch.ones(1, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask must match"):
        _ = model(prost, mask=bad_mask)


def test_prost_only_mlp_dimensions() -> None:
    cfg = _make_cfg(num_functions=6, dim=8)
    model = ProstOnlyPFAGCN(cfg)
    assert isinstance(model.mlp[0], torch.nn.Linear)
    assert model.mlp[0].out_features == 4
    assert isinstance(model.mlp[2], torch.nn.Linear)
    assert model.mlp[2].out_features == 2
    assert isinstance(model.mlp[4], torch.nn.Linear)
    assert model.mlp[4].out_features == 6
