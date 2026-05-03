import torch
import torch.nn.functional as F

from src.modules.prost_graph import ProstGraphBlock


def test_prost_graph_plddt_self_loops() -> None:
    torch.manual_seed(7)
    block = ProstGraphBlock(input_dim=4, dropout=0.0)
    with torch.no_grad():
        block.input_proj.weight.copy_(torch.eye(4))
        block.update_proj.weight.copy_(torch.eye(4))

    embeddings = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([0.5, 0.5, 0.25, 0.25])
    plddt = torch.tensor([0.2, 0.4, 0.6])
    graph = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "plddt": plddt,
        "node_splits": torch.tensor([0, 3], dtype=torch.long),
    }

    output = block(embeddings, graph)

    loop_index = torch.arange(3)
    full_edge_index = torch.cat(
        [edge_index, torch.stack([loop_index, loop_index], dim=0)], dim=1
    )
    full_edge_weight = torch.cat([edge_weight, plddt], dim=0)
    row = full_edge_index[0]
    col = full_edge_index[1]
    deg = torch.zeros(3)
    deg.index_add_(0, row, full_edge_weight)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    norm_weight = full_edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    normed = F.layer_norm(
        embeddings[0],
        (4,),
        weight=block.pre_norm.weight,
        bias=block.pre_norm.bias,
        eps=block.pre_norm.eps,
    )
    messages = normed[col] * norm_weight.unsqueeze(-1)
    aggregated = torch.zeros_like(embeddings[0])
    aggregated.index_add_(0, row, messages)
    expected = embeddings[0] + F.relu(aggregated)

    assert output.shape == embeddings.shape
    assert torch.allclose(output[0], expected, atol=1e-5)
