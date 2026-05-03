"""
go_prior.py

generates the GO prior matrix: 
    - binary with 1 indicating present relationship (is_a, part_of)
    - normalized to be row-stochastic.
    - no self-loops
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import ast
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


ASPECT_MAP = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}


@dataclass
class GoAspectPrior:
    """Container for an aspect-specific GO adjacency and metadata."""

    adjacency: np.ndarray
    terms: Sequence[str]
    index_map: Dict[str, int]


@dataclass
class GoTerm:
    """Parsed metadata for a GO term in the ontology."""

    term_id: str
    aspect: Optional[str]
    parents: Set[str]


def Go_Prior(
    obo_path: Path | str,
    train_split_csv: Path | str,
    top_k_mf: Optional[int],
    top_k_bp: Optional[int],
    top_k_cc: Optional[int],
) -> Dict[str, GoAspectPrior]:
    """Create GO priors for Molecular Function (MF), Biological Process (BP), and Cellular Component (CC).

    Args:
        obo_path: Path to the GO ontology '.obo' file.
        train_split_csv: Path to the CSV describing the training split with GO annotations.
        top_k_mf: Cutoff for the top-K Molecular Function terms.
        top_k_bp: Cutoff for the top-K Biological Process terms.
        top_k_cc: Cutoff for the top-K Cellular Component terms.

    Returns:
        Dictionary keyed by aspect ('MF', 'BP', 'CC') whose values contain
        the adjacency matrix, GO set, and index mapping for that aspect.
    """

    obo_path = Path(obo_path)
    train_split_csv = Path(train_split_csv)

    go_terms = _parse_training_split(train_split_csv)
    ontology = _parse_obo(obo_path)

    frequencies = Counter(go_terms)
    if not frequencies:
        raise ValueError("No GO terms were extracted from the training split CSV.")

    aspect_topk = {"MF": top_k_mf, "BP": top_k_bp, "CC": top_k_cc}

    aspect_results: Dict[str, GoAspectPrior] = {}
    for aspect, top_k in aspect_topk.items():
        selected_terms = _select_terms_for_aspect(
            aspect=aspect,
            frequencies=frequencies,
            ontology=ontology,
            top_k=top_k,
        )

        adjacency, ordered_terms, index_map = _build_adjacency(selected_terms, ontology)
        aspect_results[aspect] = GoAspectPrior(
            adjacency=adjacency,
            terms=ordered_terms,
            index_map=index_map,
        )

    LOGGER.info("successfully created GO prior.")

    return aspect_results


def _parse_training_split(csv_path: Path) -> List[str]:
    """Extract GO identifiers from a training split CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Training split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    candidate_columns = [
        col for col in df.columns if "go" in col.lower() or col.lower().endswith("terms")
    ]
    if "go_terms" in df.columns:
        candidate_columns = ["go_terms"]

    if not candidate_columns:
        raise ValueError(
            "Could not identify GO term columns in the training split CSV. "
            "Expected a column named 'go_terms' or containing 'go'."
        )

    collected: List[str] = []
    for column in candidate_columns:
        series = df[column]
        for value in series:
            collected.extend(_extract_go_terms(value))

    return collected


def _extract_go_terms(value: object) -> List[str]:
    """Normalise a cell value into a list of GO identifiers."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        iterable = value
    else:
        text = str(value).strip()
        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, (list, tuple, set)):
            iterable = parsed
        else:
            separators = [";", ",", "|"]
            for sep in separators:
                if sep in text:
                    iterable = [item.strip() for item in text.split(sep)]
                    break
            else:
                iterable = [text]

    go_ids = [item.strip() for item in iterable if isinstance(item, str) and item.strip()]
    return [go_id for go_id in go_ids if go_id.upper().startswith("GO:")]


def _parse_obo(obo_path: Path) -> Dict[str, GoTerm]:
    """Read the ontology file and collect term metadata."""

    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}")

    ontology: Dict[str, GoTerm] = {}
    current_id: Optional[str] = None
    current_namespace: Optional[str] = None
    current_parents: Set[str] = set()
    is_obsolete = False

    def flush_current() -> None:
        nonlocal current_id, current_namespace, current_parents, is_obsolete
        if current_id and not is_obsolete:
            aspect = ASPECT_MAP.get(current_namespace)
            ontology[current_id] = GoTerm(
                term_id=current_id,
                aspect=aspect,
                parents=set(current_parents),
            )
        current_id = None
        current_namespace = None
        current_parents = set()
        is_obsolete = False

    with obo_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "[Term]":
                flush_current()
                continue
            if line == "[Typedef]":
                flush_current()
                current_id = None
                continue
            if line.startswith("id: "):
                current_id = line.split("id: ", 1)[1].strip()
                continue
            if line.startswith("namespace: "):
                current_namespace = line.split("namespace: ", 1)[1].strip()
                continue
            if line.startswith("is_a: "):
                parent = line.split("is_a: ", 1)[1].split()[0]
                current_parents.add(parent)
                continue
            if line.startswith("relationship: "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "part_of":
                    current_parents.add(parts[2])
                continue
            if line.startswith("is_obsolete: "):
                is_obsolete = line.endswith("true")

        flush_current()

    return ontology


def _select_terms_for_aspect(
    aspect: str,
    frequencies: Counter,
    ontology: Dict[str, GoTerm],
    top_k: Optional[int],
) -> Set[str]:
    """Choose the GO set for a specific aspect."""

    aspect_terms: Set[str] = set()
    for term_id in frequencies:
        term = ontology.get(term_id)
        if term is None:
            LOGGER.warning("GO term %s from training set missing in ontology", term_id)
            continue
        if term.aspect == aspect:
            aspect_terms.add(term_id)

    if not aspect_terms:
        return set()

    ranked = sorted(
        aspect_terms,
        key=lambda tid: (-frequencies[tid], tid),
    )

    if top_k is None or top_k <= 0 or top_k >= len(ranked):
        chosen = ranked
    else:
        chosen = ranked[:top_k]

    selected: Set[str] = set(chosen)
    for term_id in chosen:
        selected.update(
            _collect_ancestors(term_id, ontology, aspect_filter=aspect)
        )

    return selected


def _collect_ancestors(
    term_id: str,
    ontology: Dict[str, GoTerm],
    aspect_filter: Optional[str] = None,
) -> Set[str]:
    """Gather all ancestors reachable from 'term_id' respecting the aspect filter."""

    term = ontology.get(term_id)
    if term is None:
        LOGGER.warning("GO term %s not found in ontology while collecting ancestors", term_id)
        return set()

    ancestors: Set[str] = set()
    stack: List[str] = list(term.parents)

    while stack:
        parent_id = stack.pop()
        if parent_id in ancestors:
            continue
        parent_term = ontology.get(parent_id)
        if parent_term is None:
            LOGGER.debug("Skipping missing parent %s", parent_id)
            continue
        if aspect_filter is None or parent_term.aspect == aspect_filter:
            ancestors.add(parent_id)
        stack.extend(parent_term.parents)

    return ancestors


def _build_adjacency(
    selected_terms: Set[str],
    ontology: Dict[str, GoTerm],
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Construct the adjacency matrix for a set of GO terms."""

    ordered_terms = sorted(selected_terms)
    index_map = {term_id: idx for idx, term_id in enumerate(ordered_terms)}
    size = len(ordered_terms)
    adjacency = np.zeros((size, size), dtype=np.float32)

    for child_id in ordered_terms:
        child_term = ontology.get(child_id)
        if child_term is None:
            continue
        child_idx = index_map[child_id]
        for parent_id in child_term.parents:
            parent_idx = index_map.get(parent_id)
            if parent_idx is not None and parent_idx != child_idx:
                adjacency[parent_idx, child_idx] = 1.0

    _row_normalise(adjacency)
    return adjacency, ordered_terms, index_map


def _row_normalise(matrix: np.ndarray) -> None:
    """Normalise matrix rows in-place so they sum to 1 when possible."""

    if matrix.size == 0:
        return

    row_sums = matrix.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums.squeeze(-1) > 0
    if np.any(nonzero_rows):
        matrix[nonzero_rows] = matrix[nonzero_rows] / row_sums[nonzero_rows]
