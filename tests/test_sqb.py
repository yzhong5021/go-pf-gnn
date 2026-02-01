import torch

from src.modules.sqb import SQBBlock


def test_sqb_block_shapes_and_mask() -> None:
    torch.manual_seed(2)
    batch, length, input_dim, channels = 2, 6, 12, 8
    seq_embeddings = torch.randn(batch, length, input_dim)
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True, True, True, True, False, False]]
    )
    sqb = SQBBlock(
        input_dim=input_dim,
        channels=channels,
        kernel_size=3,
        dilation=2,
        dropout=0.0,
    )
    output = sqb(seq_embeddings, mask)
    assert output.shape == (batch, length, channels)
    assert torch.all(output[~mask] == 0)
