import torch

from src.modules.structural_gcn import StructuralGCNBlock


def test_structural_gcn_shapes() -> None:
    torch.manual_seed(3)
    batch, length, channels = 2, 4, 8
    heads, head_dim = 2, 4
    features = torch.randn(batch, length, channels)
    edge_indices = []
    edge_weights = []
    node_splits = [0]
    lengths = [3, 4]
    for size in lengths:
        offset = node_splits[-1]
        nodes = torch.arange(size)
        edge_index = torch.stack([nodes, nodes], dim=0) + offset
        edge_indices.append(edge_index)
        edge_weights.append(torch.ones(size))
        node_splits.append(offset + size)
    adjacency = {
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_weight": torch.cat(edge_weights, dim=0),
        "node_splits": torch.tensor(node_splits, dtype=torch.long),
    }
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    block = StructuralGCNBlock(
        channels=channels,
        heads=heads,
        head_dim=head_dim,
        dropout=0.0,
    )
    output = block(features, adjacency, mask)
    assert output.features.shape == (batch, length, channels)
    assert output.keys.shape == (batch, heads, length, head_dim)
    assert output.values.shape == (batch, heads, length, head_dim)
