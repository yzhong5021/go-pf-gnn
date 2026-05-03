import importlib.util
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pf_agcn_scripts.preprocessing", PROJECT_ROOT / "scripts" / "preprocessing.py"
)
preprocessing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preprocessing
assert SPEC.loader is not None
SPEC.loader.exec_module(preprocessing)


def test_propagate_ancestor_labels_marks_all_parents() -> None:
    adjacency = np.zeros((3, 3), dtype=np.float32)
    adjacency[0, 1] = 1.0  # parent of term 1
    adjacency[1, 2] = 1.0  # parent of term 2

    parent_lookup = preprocessing._build_parent_lookup(adjacency)
    propagated = preprocessing._propagate_ancestor_labels([0.0, 0.0, 1.0], parent_lookup)

    assert propagated == [1.0, 1.0, 1.0]


def test_propagate_ancestor_labels_handles_normalised_edges() -> None:
    adjacency = np.zeros((3, 3), dtype=np.float32)
    adjacency[0, 1] = 0.5
    adjacency[0, 2] = 0.5
    adjacency[1, 2] = 1.0

    parent_lookup = preprocessing._build_parent_lookup(adjacency)
    propagated = preprocessing._propagate_ancestor_labels([0.0, 0.0, 1.0], parent_lookup)

    assert propagated == [1.0, 1.0, 1.0]
