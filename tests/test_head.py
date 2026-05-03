import torch

from src.modules.head import ClassificationHead


def test_classification_head_pools_three_dimensional_inputs() -> None:
    torch.manual_seed(0)
    head = ClassificationHead(
        N_C=5,
        d_in=6,
        protein_input_dim=4,
        function_input_dim=3,
        dropout=0.0,
        attn_hidden=4,
    )
    functions = torch.randn(4, 5, 3)
    proteins = torch.randn(4, 5, 4)

    logits, protein_embeddings, function_embeddings = head(functions, proteins)

    assert logits.shape == (4, 5)
    assert protein_embeddings.shape == (4, 6)
    assert function_embeddings.shape == (5, 6)
    assert torch.isfinite(logits).all()


def test_classification_head_scaling_changes_output() -> None:
    torch.manual_seed(1)
    head = ClassificationHead(N_C=3, d_in=4)
    functions = torch.randn(3, 4)
    proteins = torch.randn(2, 4)

    baseline, _, _ = head(functions, proteins)
    head.log_tau.data.fill_(0.5)
    scaled, _, _ = head(functions, proteins)

    assert not torch.allclose(baseline, scaled)
