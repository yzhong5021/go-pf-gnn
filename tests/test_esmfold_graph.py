import torch

from src.modules.structure_graph import StructureGraphBuilder


def test_build_graph_from_ca_cutoff_topk_symmetry() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 20.0],
        ]
    )
    plddt = torch.tensor([100.0, 50.0, 80.0, 30.0])
    adj = StructureGraphBuilder.build_graph_from_ca(
        ca_coords=coords,
        plddt=plddt,
        distance_cutoff=10.0,
        top_k=2,
    )

    conf = torch.tensor([1.0, 0.5, 0.8, 0.3])
    assert torch.allclose(torch.diag(adj), conf)
    assert adj[0, 1].item() == 0.5
    assert adj[1, 0].item() == 0.5
    assert adj[1, 2].item() == 0.5
    assert adj[2, 1].item() == 0.5
    assert adj[0, 2].item() == 0.0
    assert adj[2, 0].item() == 0.0


def test_build_sparse_graph_from_ca_topk_union() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 9.0],
            [0.0, 0.0, 20.0],
        ]
    )
    plddt = torch.tensor([100.0, 50.0, 80.0, 30.0])
    edge_index, edge_weight = StructureGraphBuilder.build_sparse_graph_from_ca(
        ca_coords=coords,
        plddt=plddt,
        distance_cutoff=10.0,
        top_k=2,
    )
    pairs = set(map(tuple, edge_index.T.tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}
    weights = {tuple(edge_index[:, i].tolist()): edge_weight[i].item() for i in range(edge_weight.numel())}
    assert weights[(0, 1)] == 0.5
    assert weights[(1, 0)] == 0.5
    assert weights[(1, 2)] == 0.5
    assert weights[(2, 1)] == 0.5
