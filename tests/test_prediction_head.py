import torch

from src.modules.prediction_head import PredictionHead


def test_prediction_head_shapes() -> None:
    torch.manual_seed(4)
    batch, length, channels, num_functions = 2, 5, 8, 6
    attn_features = torch.randn(batch, length, channels)
    gcn_features = torch.randn(batch, length, channels)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    head = PredictionHead(channels=channels, num_functions=num_functions)
    logits = head(attn_features, gcn_features, mask)
    assert logits.shape == (batch, num_functions)
