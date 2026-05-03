import torch

from src.modules.seq_final import SeqFinal


def test_seq_final_shared_tensor_shape() -> None:
    torch.manual_seed(0)
    module = SeqFinal(in_dim=6, N_C=3, proj=4, out_ch=5)
    proteins = torch.randn(4, 6)

    out = module(proteins, mode="alternating")

    assert out.shape == (4, 3, 5)
    assert torch.isfinite(out).all()


def test_seq_final_decoupled_outputs() -> None:
    torch.manual_seed(1)
    module = SeqFinal(
        in_dim=5,
        N_C=2,
        proj=3,
        out_ch=4,
        decoupled=True,
        function_dim=6,
        protein_dim=7,
    )
    proteins = torch.randn(3, 5)

    function_tensor, protein_tensor = module(proteins, mode="decoupled")

    assert function_tensor.shape == (3, 2, 6)
    assert protein_tensor.shape == (3, 2, 7)
    assert torch.isfinite(function_tensor).all()
    assert torch.isfinite(protein_tensor).all()
