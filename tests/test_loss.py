import torch
import torch.nn.functional as F

from src.modules.loss import BCEWithLogits


def test_bce_with_logits_matches_functional() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 4)
    targets = torch.randint(0, 2, (3, 4), dtype=torch.float32)

    loss_module = BCEWithLogits(pos_weight=None)

    assert torch.isclose(loss_module(logits, targets), F.binary_cross_entropy_with_logits(logits, targets))


def test_bce_with_logits_pos_weight() -> None:
    logits = torch.tensor([[0.0, 1.0]])
    targets = torch.tensor([[0.0, 1.0]])
    pos_weight = torch.tensor([1.0, 2.0])

    loss_module = BCEWithLogits(pos_weight=pos_weight)
    loss = loss_module(logits, targets)

    expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none").mean()

    assert torch.isclose(loss, expected)
