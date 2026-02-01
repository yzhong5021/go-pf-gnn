import torch

from src.modules.attention import LearnedAttentionPooling, SingleQueryCrossAttention


def test_learned_attention_pooling_respects_mask() -> None:
    torch.manual_seed(0)
    features = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    pooling = LearnedAttentionPooling(input_dim=8)
    pooled, weights = pooling(features, mask, return_weights=True)

    assert pooled.shape == (2, 8)
    assert weights is not None
    assert weights.shape == (2, 4)
    sums = torch.stack([weights[i, mask[i]].sum() for i in range(weights.size(0))])
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_single_query_cross_attention_weights_sum_to_one() -> None:
    torch.manual_seed(1)
    batch, heads, length, head_dim = 2, 4, 5, 3
    channels = heads * head_dim
    keys = torch.randn(batch, heads, length, head_dim)
    values = torch.randn(batch, heads, length, head_dim)
    query = torch.randn(batch, heads, head_dim)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    cross_attn = SingleQueryCrossAttention(
        channels=channels,
        heads=heads,
        head_dim=head_dim,
        dropout=0.0,
    )
    output, attn = cross_attn(keys, values, query, mask, return_attention=True)

    assert output.shape == (batch, length, channels)
    assert attn is not None
    assert attn.shape == (batch, heads, length)
    sums = torch.zeros(batch, heads)
    for b in range(batch):
        for h in range(heads):
            sums[b, h] = attn[b, h, mask[b]].sum()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
