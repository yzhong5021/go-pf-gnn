import torch

from src.modules.mock_esm import MockESM, N_AA


def test_mock_esm_forward_shape() -> None:
    model = MockESM(seq_len=10, hidden_len=16, embed_len=8, proj_len=6)
    tokens = torch.randint(0, N_AA, (2, 10))

    out = model(tokens)

    assert out.shape == (2, 10, 6)


def test_mock_esm_backward_pass_updates_embedding() -> None:
    model = MockESM(seq_len=5, hidden_len=12, embed_len=7, proj_len=4)
    tokens = torch.randint(0, N_AA, (1, 5))

    out = model(tokens)
    out.sum().backward()

    assert model.embed.weight.grad is not None
