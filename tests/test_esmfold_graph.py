import torch

from src.modules.esmfold_graph import ESMFoldGraphBuilder


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
    adj = ESMFoldGraphBuilder.build_graph_from_ca(
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
