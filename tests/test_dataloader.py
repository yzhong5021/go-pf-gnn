import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.modules.dataloader import (
    ManifestDataset,
    SequenceAnnotationDataset,
    build_manifest_dataloader,
    build_sequence_dataloader,
    collate_manifest_batch,
    dataframe_to_multi_hot,
    load_ia_weights,
    load_information_accretion,
    load_npz_tensor,
    parse_fasta_sequences,
    parse_ground_truth_table,
)


def _write_neighbor_npz(manifest_path: Path, top_k: int) -> Path:
    stem = manifest_path.stem
    out = manifest_path.parent / f"{stem}_neighbors_top{top_k}.npz"
    n = 2
    neighbors = np.tile(np.arange(n, dtype=np.int64), (n, 1))[:, :top_k]
    weights = np.ones_like(neighbors, dtype=np.float16)
    np.savez_compressed(out, neighbors=neighbors, weights=weights)
    return out


def test_dataframe_to_multi_hot() -> None:
    annotations = pd.DataFrame(
        {
            "entry_id": ["A", "A", "B"],
            "term": ["GO:1", "GO:2", "GO:2"],
        }
    )
    vocab = ["GO:1", "GO:2", "GO:3"]

    label_map = dataframe_to_multi_hot(annotations, vocab)

    assert torch.equal(label_map["A"], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(label_map["B"], torch.tensor([0.0, 1.0, 0.0]))


def test_sequence_annotation_dataset_tensor_and_missing() -> None:
    sequences = pd.DataFrame(
        {"entry_id": ["A", "B"], "sequence": ["AA", "BB"], "header": ["hA", "hB"]}
    )
    annotations = {"A": torch.tensor([1.0, 0.0])}
    dataset = SequenceAnnotationDataset(sequences, annotations, term_to_index={"GO:1": 0, "GO:2": 1})

    sample_a = dataset[0]
    sample_b = dataset[1]

    assert torch.equal(sample_a["targets"], torch.tensor([1.0, 0.0]))
    assert torch.equal(sample_b["targets"], torch.zeros(2))
    assert sample_b["terms"] == []


def test_load_npz_tensor_variants(tmp_path: Path) -> None:
    npy_path = tmp_path / "tensor.npy"
    np.save(npy_path, np.arange(6, dtype=np.float32).reshape(2, 3))

    npz_path = tmp_path / "tensor.npz"
    np.savez(npz_path, arr_0=np.ones((2, 2), dtype=np.float32), custom=np.full((2, 2), 2.0, dtype=np.float32))

    pt_tensor_path = tmp_path / "tensor.pt"
    torch.save(torch.randn(2, 2), pt_tensor_path)

    pt_mapping_path = tmp_path / "tensor_map.pt"
    torch.save({"foo": torch.full((2, 1), 3.0)}, pt_mapping_path)

    adjacency_matrix = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32)
    go_prior_npz = tmp_path / "go_prior.npz"
    np.savez(go_prior_npz, adjacency=adjacency_matrix, terms=np.array([1, 2, 3], dtype=np.int32))

    graph_matrix = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
    graph_npz = tmp_path / "graph_prior.npz"
    np.savez(graph_npz, graph=graph_matrix, metadata=np.array([42], dtype=np.int32))

    assert load_npz_tensor(npy_path).shape == (2, 3)
    assert torch.equal(load_npz_tensor(npz_path, key="custom"), torch.full((2, 2), 2.0))
    assert load_npz_tensor(pt_tensor_path).dtype == torch.float32

    with pytest.raises(KeyError):
        load_npz_tensor(pt_mapping_path)

    mapping_tensor = load_npz_tensor(pt_mapping_path, key="foo")
    assert torch.equal(mapping_tensor, torch.full((2, 1), 3.0))
    go_prior_tensor = load_npz_tensor(go_prior_npz, dtype=torch.float16)
    assert torch.allclose(
        go_prior_tensor.float(), torch.from_numpy(adjacency_matrix), atol=1e-3, rtol=1e-3
    )
    assert go_prior_tensor.dtype == torch.float16

    with pytest.raises(KeyError):
        load_npz_tensor(graph_npz)
    resolved_graph = load_npz_tensor(graph_npz, key_priority=("graph",))
    assert torch.allclose(resolved_graph, torch.from_numpy(graph_matrix))


def test_load_npz_tensor_preserves_dtype(tmp_path: Path) -> None:
    half_path = tmp_path / "half.npy"
    np.save(half_path, np.ones((2, 2), dtype=np.float16))

    preserved = load_npz_tensor(half_path, dtype=None)
    assert preserved.dtype == torch.float16


def test_manifest_dataset_and_collate(tmp_path: Path) -> None:
    emb0 = tmp_path / "emb0.npy"
    emb1 = tmp_path / "emb1.npy"
    np.save(emb0, np.ones((3, 4), dtype="float32"))
    np.save(emb1, np.full((2, 4), 0.5, dtype="float32"))

    protein_prior_path = tmp_path / "protein_prior.npy"
    np.save(protein_prior_path, np.eye(3, dtype="float32"))

    go_prior_path = tmp_path / "go_prior.npy"
    np.save(go_prior_path, np.ones((2, 2), dtype="float32"))

    manifest_path = tmp_path / "manifest.json"
    records = [
        {
            "embedding_path": emb0.name,
            "labels": [1, 0, 1, 0],
            "protein_prior_path": protein_prior_path.name,
            "go_prior_path": go_prior_path.name,
        },
        {
            "embedding_path": emb1.name,
            "labels": [0, 1, 0, 1],
            "protein_prior_path": protein_prior_path.name,
            "go_prior_path": go_prior_path.name,
        },
    ]
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    dataset = ManifestDataset(manifest_path)
    sample0 = dataset[0]
    assert sample0["go_prior"].dtype == torch.float16
    sample1 = dataset[1]
    assert torch.equal(sample0["go_prior"], sample1["go_prior"])
    assert int(sample0["node_index"]) == 0

    torch.manual_seed(0)
    _write_neighbor_npz(manifest_path, top_k=2)
    loader = build_manifest_dataloader(
        str(manifest_path),
        {"neighbors": {"batch_size": 1, "top_k": 2, "fanout_1": 1, "fanout_2": 0}},
        base_dir=tmp_path,
        shuffle=False,
    )
    assert loader is not None
    batch = next(iter(loader))

    assert batch["seq_embeddings"].shape[0] == 2
    assert batch["target_mask"].tolist() == [True, False]
    assert batch["protein_prior"].shape == (2, 2)
    assert torch.allclose(batch["protein_prior"].diag(), torch.ones(2))
    assert batch["root_indices"].tolist() == [0]
    assert torch.count_nonzero(batch["protein_prior"]).item() >= 3


def test_manifest_dataset_go_prior_key_priority(tmp_path: Path) -> None:
    emb = tmp_path / "emb.npy"
    np.save(emb, np.random.randn(2, 4).astype("float32"))

    archive = tmp_path / "compound_prior.npz"
    adjacency = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
    np.savez(archive, graph=adjacency, stats=np.arange(2, dtype=np.int32))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "embedding_path": emb.name,
                    "labels": [1, 0],
                    "go_prior_path": archive.name,
                    "go_prior_key_priority": ["graph"],
                },
                {
                    "embedding_path": emb.name,
                    "labels": [0, 1],
                    "go_prior_path": archive.name,
                    "go_prior_key": "graph",
                },
            ]
        ),
        encoding="utf-8",
    )

    dataset = ManifestDataset(manifest_path)
    sample0 = dataset[0]
    sample1 = dataset[1]
    assert sample0["go_prior"].dtype == torch.float16
    assert torch.allclose(
        sample0["go_prior"].float(), torch.from_numpy(adjacency), atol=1e-3, rtol=1e-3
    )
    assert torch.equal(sample0["go_prior"], sample1["go_prior"])
    batch = [sample0, sample1]
    collated = collate_manifest_batch(batch)

    assert collated["seq_embeddings"].shape == (2, 2, 4)
    assert collated["targets"].shape == (2, 2)
    assert collated["go_prior"].shape == (2, 2)

    _write_neighbor_npz(manifest_path, top_k=1)
    loader = build_manifest_dataloader(
        str(manifest_path),
        {"neighbors": {"batch_size": 1, "top_k": 1, "fanout_1": 1, "fanout_2": 0}},
        base_dir=tmp_path,
        shuffle=False,
    )
    assert loader is not None
    assert len(loader.dataset) == 2


def test_parse_fasta_sequences_and_information_accretion(tmp_path: Path) -> None:
    fasta_path = tmp_path / "toy.fasta"
    fasta_path.write_text(
        ">sp|P1|DESC\nAAAA\n>sp|P2|DESC\nTTTT\n",
        encoding="utf-8",
    )

    df = parse_fasta_sequences(fasta_path)
    assert set(df["entry_id"]) == {"P1", "P2"}

    bad_fasta_path = tmp_path / "bad.fasta"
    bad_fasta_path.write_text(">badheader\nAAAA\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_fasta_sequences(bad_fasta_path)

    ia_path = tmp_path / "ia.txt"
    ia_path.write_text("GO:1 0.5\nGO:2 1.0\n", encoding="utf-8")
    ia_df = load_information_accretion(ia_path)
    assert list(ia_df["ia"]) == [0.5, 1.0]


def test_parse_ground_truth_table_with_header(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms.tsv"
    terms_path.write_text(
        "EntryID\tterm\taspect\n"
        "P1\tGO:0001\tc\n"
        "P2\tGO:0002\tBPO\n"
        "P3\tGO:0003\tmf\n"
        "P4\tGO:0004\tCCO\n"
        "P5\tGO:0005\tunknown\n",
        encoding="utf-8",
    )
    df = parse_ground_truth_table(terms_path)
    assert set(df.columns) == {"entry_id", "term", "aspect"}
    assert set(df["entry_id"]) == {"P1", "P2", "P3", "P4"}
    assert set(df["aspect"]) == {"C", "F", "P"}


def test_build_sequence_dataloader_and_load_ia_weights(tmp_path: Path) -> None:
    sequences = pd.DataFrame({"entry_id": ["A"], "sequence": ["AA"], "header": ["h"]})
    annotations = {"A": torch.tensor([1.0, 0.0])}

    loader = build_sequence_dataloader(sequences, annotations, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    assert batch["entry_id"][0] == "A"
    assert batch["targets"].shape == (1, 2)

    weights_array = np.array([0.1, 0.2], dtype=np.float32)
    weights_path = tmp_path / "weights.npz"
    np.savez(weights_path, weights=weights_array)

    cfg = {"evaluation": {"ia_weights_path": weights_path.name, "ia_weights_key": "weights"}}
    weights = load_ia_weights(cfg, base_dir=tmp_path)
    assert np.allclose(weights, weights_array)

    cfg_missing = {"data": {}}
    assert load_ia_weights(cfg_missing, base_dir=tmp_path) is None
