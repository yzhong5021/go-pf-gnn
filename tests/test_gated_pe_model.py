from types import SimpleNamespace

import pytest
import torch

from src.model.gated_pe_model import GatedPEPFAGCN


def _make_cfg(
    num_functions: int = 5,
    raw_dim: int = 8,
    channels: int = 8,
    heads: int = 2,
    prost_dim: int = 8,
) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(num_functions=num_functions),
        model=SimpleNamespace(
            seq_embeddings=SimpleNamespace(raw_dim=raw_dim, feature_dim=raw_dim),
            sqb=SimpleNamespace(
                channels=channels,
                dccn=SimpleNamespace(kernel_size=3, dilation=1, dropout=0.0),
            ),
            gcn=SimpleNamespace(dropout=0.0, heads=heads),
            prostt5_3di=SimpleNamespace(encoder_dim=prost_dim),
            prost_graph=SimpleNamespace(enabled=True, dropout=0.0),
            gated_pe=SimpleNamespace(mlp_dropout=0.0),
        ),
    )


def _dense_adj(batch: int, length: int) -> torch.Tensor:
    eye = torch.eye(length, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
    return eye


def test_gated_pe_shapes() -> None:
    torch.manual_seed(11)
    cfg = _make_cfg(num_functions=7)
    model = GatedPEPFAGCN(cfg)
    seq = torch.randn(2, 5, 8)
    prost = torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 4])
    adj = _dense_adj(2, 5)
    outputs = model(seq, adj, prost, lengths=lengths)
    assert outputs.logits.shape == (2, 7)


def test_gated_pe_dim_mismatch() -> None:
    cfg = _make_cfg()
    model = GatedPEPFAGCN(cfg)
    seq = torch.randn(1, 4, 8)
    prost = torch.randn(1, 4, 6)
    adj = _dense_adj(1, 4)
    with pytest.raises(ValueError, match="encoder_dim"):
        _ = model(seq, adj, prost, lengths=torch.tensor([4]))


def test_gated_pe_mlp_input_dim() -> None:
    cfg = _make_cfg(channels=8, prost_dim=8, num_functions=3)
    model = GatedPEPFAGCN(cfg)
    assert model.stream_gate.shape == (4,)
    assert isinstance(model.mlp[0], torch.nn.Linear)
    assert model.mlp[0].in_features == 8
    assert model.mlp[0].out_features == 4
    assert isinstance(model.mlp[3], torch.nn.Linear)
    assert model.mlp[3].out_features == 4
    assert isinstance(model.mlp[6], torch.nn.Linear)
    assert model.mlp[6].out_features == 3
