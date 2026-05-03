from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.go_prior import (
    Go_Prior,
    GoTerm,
    _build_adjacency,
    _extract_go_terms,
    _row_normalise,
)


def test_extract_go_terms_variants() -> None:
    assert _extract_go_terms(["GO:1", "GO:2"]) == ["GO:1", "GO:2"]
    assert _extract_go_terms("['GO:3', 'X']") == ["GO:3"]
    assert _extract_go_terms("GO:4; GO:5") == ["GO:4", "GO:5"]
    assert _extract_go_terms(None) == []


def test_row_normalise_handles_empty() -> None:
    matrix = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    _row_normalise(matrix)
    assert np.allclose(matrix[0], [0.5, 0.5])
    assert np.allclose(matrix[1], [0.0, 0.0])


def test_go_prior_builds_expected_adjacency(tmp_path: Path) -> None:
    obo_path = tmp_path / "mini.obo"
    obo_path.write_text(
        """
[Term]
id: GO:0001
namespace: biological_process
is_a: GO:0003

[Term]
id: GO:0002
namespace: biological_process
relationship: part_of GO:0001

[Term]
id: GO:0003
namespace: biological_process
""".strip(),
        encoding="utf-8",
    )

    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"go_terms": ["['GO:0001', 'GO:0002']"]}).to_csv(csv_path, index=False)

    priors = Go_Prior(obo_path, csv_path, top_k_mf=None, top_k_bp=2, top_k_cc=None)

    bp_prior = priors["BP"]
    adjacency, terms, index_map = bp_prior.adjacency, bp_prior.terms, bp_prior.index_map

    assert set(terms) == {"GO:0001", "GO:0002", "GO:0003"}
    parent_idx = index_map["GO:0003"]
    child_idx = index_map["GO:0001"]
    assert adjacency[parent_idx, child_idx] > 0
    nonzero_rows = adjacency.sum(axis=1) > 0
    if np.any(nonzero_rows):
        assert np.allclose(adjacency[nonzero_rows].sum(axis=1), 1.0)

    ontology = {
        "GO:A": GoTerm(term_id="GO:A", aspect="BP", parents={"GO:B"}),
        "GO:B": GoTerm(term_id="GO:B", aspect="BP", parents=set()),
    }
    adjacency_manual, ordered_terms, manual_map = _build_adjacency({"GO:A", "GO:B"}, ontology)
    assert ordered_terms == sorted({"GO:A", "GO:B"})
    assert adjacency_manual[manual_map["GO:B"], manual_map["GO:A"]] == 1.0
