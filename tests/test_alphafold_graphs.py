"""Unit tests for AlphaFold graph preprocessing helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

af = import_module("scripts.precompute_alphafold_graphs")


def test_parse_pdb_ca_extracts_coords_and_plddt(tmp_path: Path) -> None:
    pdb_text = "\n".join(
        [
            "ATOM      1  N   MET A   1      17.000   7.000 -17.000  1.00 50.00           N  ",
            "ATOM      2  CA  MET A   1      17.330   7.577 -17.222  1.00 57.91           C  ",
            "ATOM      3  C   MET A   1      18.000   8.000 -16.500  1.00 60.00           C  ",
            "ATOM      4  CA  ALA A   2      19.000   9.000 -15.000  1.00 42.00           C  ",
            "END",
        ]
    )
    pdb_path = tmp_path / "sample.pdb"
    pdb_path.write_text(pdb_text, encoding="utf-8")

    coords, plddt = af._parse_pdb_ca(pdb_path)

    assert coords.shape == (2, 3)
    np.testing.assert_allclose(coords[0], np.array([17.33, 7.577, -17.222]), rtol=1e-4)
    np.testing.assert_allclose(plddt, np.array([57.91, 42.0]), rtol=1e-4)


def test_select_fragments_prefers_length_match() -> None:
    items = [
        {
            "uniprotAccession": "P12345-2",
            "modelEntityId": "AF-P12345-2-F1",
            "sequenceStart": 1,
            "sequenceEnd": 3,
            "pdbUrl": "https://example.com/frag1.pdb",
        },
        {
            "uniprotAccession": "P12345-2",
            "modelEntityId": "AF-P12345-2-F2",
            "sequenceStart": 4,
            "sequenceEnd": 5,
            "pdbUrl": "https://example.com/frag2.pdb",
        },
        {
            "uniprotAccession": "P12345-3",
            "modelEntityId": "AF-P12345-3-F1",
            "sequenceStart": 1,
            "sequenceEnd": 4,
            "pdbUrl": "https://example.com/frag3.pdb",
        },
    ]

    fragments = af._select_fragments_for_entry(items, entry_id="P12345", expected_length=5)

    assert len(fragments) == 2
    assert fragments[0].start == 1
    assert fragments[1].start == 4
    assert sum(fragment.length for fragment in fragments) == 5


def test_merge_sparse_graphs_offsets_edges() -> None:
    edge_index_1 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight_1 = np.array([0.1, 0.2], dtype=np.float32)
    plddt_1 = np.array([10.0, 20.0], dtype=np.float32)

    edge_index_2 = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_weight_2 = np.array([0.3, 0.4], dtype=np.float32)
    plddt_2 = np.array([30.0, 40.0, 50.0], dtype=np.float32)

    merged_index, merged_weight, merged_plddt = af._merge_sparse_graphs(
        [(edge_index_1, edge_weight_1, plddt_1), (edge_index_2, edge_weight_2, plddt_2)]
    )

    expected_index = np.array([[0, 1, 2, 3], [1, 0, 3, 4]], dtype=np.int64)
    np.testing.assert_array_equal(merged_index, expected_index)
    np.testing.assert_allclose(merged_weight, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    np.testing.assert_allclose(
        merged_plddt, np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    )


def test_fallback_to_contact_graph_symlink(tmp_path: Path) -> None:
    contact_dir = tmp_path / "esm2_contact_cache"
    contact_dir.mkdir()
    entry_id = "P12345"
    contact_path = contact_dir / af._entry_cache_filename(entry_id)
    contact_path.write_text("dummy", encoding="utf-8")

    output_dir = tmp_path / "af_graphs"
    output_dir.mkdir()
    cache_path = output_dir / af._entry_cache_filename(entry_id)

    af._fallback_to_contact_graph(
        entry_id,
        cache_path,
        contact_dir,
        reason="missing alphaFold",
    )

    assert cache_path.is_symlink()
    assert cache_path.resolve() == contact_path.resolve()
