import torch
import torch.nn as nn

from src.utils.adaptive_helpers import _build_top_p_attention, _init_attention_layers, _prepare_prior


def test_build_top_p_attention_row_sums() -> None:
    torch.manual_seed(0)
    d_in = 6
    d_attn = 4
    W1 = nn.Linear(d_in, d_attn, bias=False)
    W2 = nn.Linear(d_attn, d_attn, bias=False)
    W3 = nn.Linear(d_in, d_attn, bias=False)
    _init_attention_layers(W1, W2, W3)

    X = torch.randn(5, d_in)
    attn = _build_top_p_attention(W1, W2, W3, X, tau=1.0, p=0.6)

    assert attn.shape == (5, 5)
    assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-5)
    assert torch.all(attn >= 0)


def test_prepare_prior_unidirectional_and_bidirectional() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    rf = _prepare_prior(None, 3, device, dtype, bidirectional=False)
    assert torch.allclose(rf, torch.eye(3))

    prior = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    rf, rb = _prepare_prior(prior, 3, device, dtype, bidirectional=True)

    assert rf.shape == rb.shape == torch.Size([3, 3])
    assert torch.all(rf >= 0)
    assert torch.all(rb >= 0)
    assert torch.all(rf.diag() > 0)
    assert torch.all(rb.diag() > 0)
