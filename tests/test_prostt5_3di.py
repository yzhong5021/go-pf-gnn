import torch

from src.modules.prostt5_3di import (
    ProstT5QueryEncoder,
    compute_entropy_confidence,
    slice_logits_to_3di,
)


def test_slice_logits_to_3di() -> None:
    logits = torch.arange(0, 60, dtype=torch.float32).view(1, 2, 30)
    token_ids = list(range(20))
    sliced = slice_logits_to_3di(logits, token_ids)
    assert sliced.shape == (1, 2, 20)
    assert torch.allclose(sliced[0, 0], logits[0, 0, :20])


def test_entropy_confidence_bounds() -> None:
    uniform = torch.full((1, 2, 20), 1.0 / 20.0)
    confidence_uniform = compute_entropy_confidence(uniform)
    assert torch.allclose(confidence_uniform, torch.zeros_like(confidence_uniform), atol=1e-6)

    one_hot = torch.zeros((1, 1, 20))
    one_hot[0, 0, 3] = 1.0
    confidence_one_hot = compute_entropy_confidence(one_hot)
    assert torch.allclose(confidence_one_hot, torch.ones_like(confidence_one_hot), atol=1e-6)


def test_query_encoder_weights_sum_to_one() -> None:
    torch.manual_seed(5)
    batch, length, heads, head_dim, input_dim = 2, 4, 3, 6, 8
    embeddings = torch.rand(batch, length, input_dim)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    encoder = ProstT5QueryEncoder(heads=heads, head_dim=head_dim, input_dim=input_dim)
    queries, weights = encoder(embeddings, mask, return_weights=True)

    assert queries.shape == (batch, heads, head_dim)
    assert weights is not None
    assert weights.shape == (batch, heads, length)
    sums = torch.zeros(batch, heads)
    for b in range(batch):
        for h in range(heads):
            sums[b, h] = weights[b, h, mask[b]].sum()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
