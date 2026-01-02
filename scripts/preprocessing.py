"""Utilities to build aspect-specific manifests for PF-AGCN."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.dataloader import (
    dataframe_to_multi_hot,
    parse_fasta_sequences,
    parse_ground_truth_table,
)
from utils.esm_embed import ESM_Embed
from utils.prost_embed import ProstEmbed
from utils.go_prior import Go_Prior
from utils.prot_prior import prot_prior_blast

log = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path(
    os.environ.get("PF_AGCN_CACHE", "/orcd/home/002/lerchen/code/cafa_proj/data")
).expanduser()


def _normalize_cache_root(path: Optional[str | Path]) -> Path:
    base = Path(path or DEFAULT_CACHE_ROOT).expanduser()
    if not base.is_absolute():
        base = (PROJECT_ROOT / base).resolve()
    return base.resolve()


def _build_embed_cache_roots(base: Path) -> Dict[str, Path]:
    return {
        "esm": (base / "esm_cache").resolve(),
        "prost": (base / "prost_cache").resolve(),
    }


CACHE_ROOT = _normalize_cache_root(None)
EMBED_CACHE_ROOTS = _build_embed_cache_roots(CACHE_ROOT)


def set_cache_root(path: Optional[str | Path]) -> Path:
    """Update module-level cache roots (embeddings, splits, manifests)."""

    global CACHE_ROOT, EMBED_CACHE_ROOTS
    CACHE_ROOT = _normalize_cache_root(path)
    EMBED_CACHE_ROOTS = _build_embed_cache_roots(CACHE_ROOT)
    return CACHE_ROOT


ASPECT_CHOICES = {"MF", "BP", "CC"}
EMBED_BACKENDS = {"esm", "prost"}
EMBEDDER_FACTORIES = {
    "esm": ESM_Embed,
    "prost": ProstEmbed,
}
_EMBEDDER_SINGLETONS: Dict[str, Any] = {}
EMBED_BATCH_SIZE = max(1, int(os.environ.get("PF_AGCN_EMBED_BATCH", "8")))


@dataclass
class ManifestBundle:
    """Paths and metadata for an aspect-specific manifest set."""

    aspect: str
    train: Path
    val: Path
    test: Path
    num_functions: int
    go_prior_path: Path
    terms: Sequence[str]
    feature_dim: int
    embedding_backend: str


def _coerce_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _embedding_cache_path(entry_id: str, backend: str) -> Path:
    root = EMBED_CACHE_ROOTS.get(backend)
    if root is None:
        raise ValueError(f"Unsupported embedding backend '{backend}'.")
    root.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in entry_id)
    digest = hashlib.md5(entry_id.encode("utf-8")).hexdigest()[:8]
    filename = f"{safe_id or 'protein'}_{digest}.npz"
    return root / filename


def _write_embedding_cache(path: Path, array: np.ndarray) -> tuple[int, int]:
    """Persist embeddings with float16 compression."""

    cast = array.astype(np.float16, copy=False)
    np.savez_compressed(path, embeddings=cast)
    return cast.shape


def _read_embedding_metadata(path: Path) -> tuple[int, int]:
    """Return (length, dim) from an on-disk cache without loading to float32."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(path, mmap_mode="r", allow_pickle=False)
        key = "embeddings" if "embeddings" in archive.files else "arr_0"
        shape = archive[key].shape
        archive.close()
        return int(shape[0]), int(shape[1])
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return int(array.shape[0]), int(array.shape[1])
    raise ValueError(f"Unsupported embedding cache format: {path.suffix}")


def _model_cache_dir(backend: str) -> Path:
    root = EMBED_CACHE_ROOTS.get(backend)
    if root is None:
        raise ValueError(f"Unsupported embedding backend '{backend}'.")
    model_dir = (root / "models").resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _get_seq_embedder(max_length: Optional[int], backend: str):
    backend_key = backend.lower()
    if backend_key not in EMBEDDER_FACTORIES:
        raise ValueError(
            f"Unsupported embedding backend '{backend}'. Expected one of {sorted(EMBEDDER_FACTORIES)}"
        )
    embedder = _EMBEDDER_SINGLETONS.get(backend_key)
    if embedder is None:
        kwargs: Dict[str, Any] = {"cache_dir": _model_cache_dir(backend_key)}
        if backend_key == "esm" and max_length is not None:
            kwargs["max_len"] = int(max_length)
        embedder_cls = EMBEDDER_FACTORIES[backend_key]
        embedder = embedder_cls(**kwargs)
        _EMBEDDER_SINGLETONS[backend_key] = embedder
    return embedder


def _ensure_cached_embedding(
    entry_id: str,
    sequence: str,
    *,
    max_length: Optional[int],
    backend: str,
) -> tuple[Path, int, int]:
    metadata = _ensure_embeddings_for_entries(
        [(entry_id, sequence)],
        max_length=max_length,
        backend=backend,
        batch_size=1,
    )
    path, length, dim = metadata[entry_id]
    return path, length, dim


def _ensure_embeddings_for_entries(
    entries: Sequence[tuple[str, str]],
    *,
    max_length: Optional[int],
    backend: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> Dict[str, tuple[Path, int, int]]:
    """Ensure cached embeddings exist for provided entries.

    Generates embeddings in micro-batches for throughput while keeping the
    working set modest. Returns metadata required for manifest construction.
    """

    meta: Dict[str, tuple[Path, int, int]] = {}
    pending: list[tuple[str, str, Path]] = []
    for entry_id, sequence in entries:
        cache_path = _embedding_cache_path(entry_id, backend)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            length, dim = _read_embedding_metadata(cache_path)
            meta[entry_id] = (cache_path, length, dim)
            continue
        legacy_path = cache_path.with_suffix(".npy")
        if legacy_path.exists():
            array = np.load(legacy_path, mmap_mode="r", allow_pickle=False)
            length, dim = _write_embedding_cache(cache_path, array)
            meta[entry_id] = (cache_path, length, dim)
            continue
        pending.append((entry_id, sequence, cache_path))

    if not pending:
        return meta

    embedder = _get_seq_embedder(max_length, backend)
    chunk_size = max(1, batch_size)
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        sequences = [seq for _, seq, _ in chunk]
        embeddings, masks = embedder(sequences)
        embeddings = embeddings.detach().cpu()
        masks = masks.detach().cpu()
        for idx, (entry_id, _seq, cache_path) in enumerate(chunk):
            masked = embeddings[idx][masks[idx]]
            array = masked.numpy()
            length, dim = _write_embedding_cache(cache_path, array)
            meta[entry_id] = (cache_path, length, dim)

    return meta


def _resolve_split_paths(data_cfg: Mapping[str, Any]) -> Dict[str, Path]:
    defaults_root = (CACHE_ROOT / "splits").resolve()
    defaults_root.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidate = data_cfg.get(f"{split}_csv")
        if candidate:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()
        else:
            path = (defaults_root / f"{split}.csv").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved[split] = path
    return resolved


def _generate_split_assignments(
    entry_ids: Sequence[str],
    data_cfg: Mapping[str, Any],
) -> Dict[str, Sequence[str]]:
    unique_ids = sorted({str(e).strip() for e in entry_ids if str(e).strip()})
    if not unique_ids:
        raise ValueError("No entry IDs available for splitting.")

    split_cfg = data_cfg.get("split", {})
    ratios_cfg = split_cfg.get(
        "ratios",
        {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
    )
    ratios = {
        "train": float(ratios_cfg.get("train", 0.0)),
        "val": float(ratios_cfg.get("val", 0.0)),
        "test": float(ratios_cfg.get("test", 0.0)),
    }
    if ratios["train"] <= 0:
        raise ValueError("split.ratios.train must be greater than zero.")
    weights = np.array([ratios["train"], ratios["val"], ratios["test"]], dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    weights = weights / total_weight

    rng = np.random.default_rng(int(split_cfg.get("seed", 1337)))
    permuted = np.array(unique_ids, dtype=object)
    rng.shuffle(permuted)

    counts = np.floor(weights * len(permuted)).astype(int)
    remainder = len(permuted) - int(counts.sum())
    order = np.argsort(-weights)
    idx = 0
    while remainder > 0 and len(order) > 0:
        target = order[idx % len(order)]
        counts[target] += 1
        remainder -= 1
        idx += 1

    if len(permuted) > 0 and counts[0] == 0:
        donor_candidates = [i for i in order if i != 0 and counts[i] > 0]
        if donor_candidates:
            donor = donor_candidates[0]
            counts[0] += 1
            counts[donor] -= 1
        else:
            counts[0] = len(permuted)
            counts[1:] = 0

    splits: Dict[str, Sequence[str]] = {"train": [], "val": [], "test": []}
    start = 0
    split_order = ["train", "val", "test"]
    for idx, split in enumerate(split_order):
        end = start + int(counts[idx])
        splits[split] = permuted[start:end].tolist()
        start = end

    csv_paths = _resolve_split_paths(data_cfg)
    for split, ids in splits.items():
        df = pd.DataFrame({"entry_id": ids})
        df.to_csv(csv_paths[split], index=False)

    return splits


def _resolve_obo_path(go_path: Optional[Path], prior_cfg: Mapping[str, Any]) -> Path:
    candidate = go_path or _coerce_path(prior_cfg.get("obo_path"))
    if candidate and candidate.exists():
        return candidate
    raise FileNotFoundError("GO ontology (.obo) file not found in sources or go_prior config")


def _project_targets(
    entry_id: str,
    label_map: Mapping[str, Any],
    term_to_index: Mapping[str, int],
    selected_terms: Sequence[str],
) -> Sequence[float]:
    base = label_map.get(entry_id)
    if base is None:
        return [0.0] * len(selected_terms)
    values = base.tolist()
    return [float(values[term_to_index.get(term, -1)]) if term in term_to_index else 0.0 for term in selected_terms]


def _build_parent_lookup(adjacency: np.ndarray) -> list[list[int]]:
    """Precompute parent indices for each GO term given a parent->child adjacency."""

    if adjacency.size == 0:
        return []
    num_terms = int(adjacency.shape[1])
    parents: list[list[int]] = []
    for child_idx in range(num_terms):
        parent_indices = np.flatnonzero(adjacency[:, child_idx] > 0).astype(int).tolist()
        parents.append(parent_indices)
    return parents


def _propagate_ancestor_labels(
    targets: Sequence[float],
    parent_lookup: Sequence[Sequence[int]],
) -> list[float]:
    """Mark ancestor terms as positive if any descendant is positive."""

    if not targets:
        return []
    values = np.asarray(targets, dtype=np.float32)
    if parent_lookup and len(parent_lookup) != len(values):
        raise ValueError("parent lookup length must match number of targets")

    positive = list(np.flatnonzero(values > 0.0))
    visited = set(positive)
    while positive:
        child_idx = int(positive.pop())
        for parent_idx in parent_lookup[child_idx]:
            if values[parent_idx] < 0.5:
                values[parent_idx] = 1.0
            if parent_idx not in visited:
                positive.append(parent_idx)
                visited.add(parent_idx)
    return values.tolist()


def prepare_manifests(
    data_cfg: Mapping[str, Any],
    *,
    output_root: Path,
    aspect: str,
    feature_dim: int,
    max_length: Optional[int] = None,
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    embedding_backend: str = "esm",
    cache_root: Optional[Path | str] = None,
) -> ManifestBundle:
    set_cache_root(cache_root)
    aspect = aspect.upper()
    if aspect not in ASPECT_CHOICES:
        raise ValueError(f"Unsupported aspect '{aspect}'. Expected one of {sorted(ASPECT_CHOICES)}")

    backend = str(embedding_backend or "esm").lower()
    if backend not in EMBED_BACKENDS:
        raise ValueError(
            f"Unsupported embedding backend '{embedding_backend}'. Expected one of {sorted(EMBED_BACKENDS)}"
        )

    sources = data_cfg.get("sources", {})
    if not sources:
        raise ValueError("data configuration must define 'sources'")

    raw_dir = output_root / "raw" / aspect.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    prior_cfg = dict(protein_prior_cfg or {})
    prior_enabled = bool(prior_cfg.get("enabled", False))
    prior_method = str(prior_cfg.get("method", "cosine")).lower()
    use_blast_prior = prior_enabled and prior_method == "blast"
    blast_kwargs = {
        "evalue_threshold": float(prior_cfg.get("evalue_threshold", 1e-5)),
        "blastp_bin": str(prior_cfg.get("blastp_bin", "blastp")),
    }
    blast_exec = prior_cfg.get("blastp_exec")
    if blast_exec:
        blast_kwargs["blastp_exec"] = str(blast_exec)

    seqs_path = _coerce_path(sources.get("seqs_path") or sources.get("seqs_train_path"))
    terms_path = _coerce_path(sources.get("terms_path") or sources.get("terms_train_path"))

    if seqs_path is None or terms_path is None:
        raise ValueError("sources must include sequence and term paths")
    if not seqs_path.exists() or not terms_path.exists():
        raise FileNotFoundError("Training sequence or term file missing")

    seq_tables = [parse_fasta_sequences(seqs_path)]
    term_tables = [parse_ground_truth_table(terms_path)]

    sequences = (
        pd.concat(seq_tables, ignore_index=True)
        .drop_duplicates(subset="entry_id", keep="first")
        .reset_index(drop=True)
    )

    term_table = pd.concat(term_tables, ignore_index=True)
    aggregated = term_table.groupby("entry_id")["term"].agg(list).reset_index()
    aggregated["go_terms"] = aggregated["term"].apply(json.dumps)
    split_cache = raw_dir / "train_split_cache.csv"
    aggregated[["entry_id", "go_terms"]].to_csv(split_cache, index=False)

    vocab = sorted(term_table["term"].unique())
    label_map = dataframe_to_multi_hot(term_table, vocab)
    term_to_index = {term: idx for idx, term in enumerate(vocab)}

    split_assignments = _generate_split_assignments(
        sequences["entry_id"].astype(str).tolist(),
        data_cfg,
    )

    prior_cfg = data_cfg.get("go_prior", {})
    top_k = prior_cfg.get("top_k", {})
    split_source = prior_cfg.get("train_split_csv")
    candidate_path = _coerce_path(split_source) if split_source else None
    if candidate_path is None or not candidate_path.exists():
        if candidate_path is not None and not candidate_path.exists():
            log.warning(
                "train_split_csv %s missing; using cached aggregate %s",
                candidate_path,
                split_cache,
            )
        candidate_path = split_cache

    go_priors = Go_Prior(
        obo_path=_resolve_obo_path(_coerce_path(sources.get("go_path")), prior_cfg),
        train_split_csv=candidate_path,
        top_k_mf=top_k.get("MF"),
        top_k_bp=top_k.get("BP"),
        top_k_cc=top_k.get("CC"),
    )
    aspect_prior = go_priors[aspect]
    selected_terms = list(aspect_prior.terms)
    num_functions = len(selected_terms)
    parent_lookup = _build_parent_lookup(aspect_prior.adjacency)

    priors_dir = output_root / "priors" / aspect.lower()
    priors_dir.mkdir(parents=True, exist_ok=True)
    prior_path = priors_dir / f"{aspect.lower()}_prior.npz"
    np.savez_compressed(
        prior_path,
        adjacency=aspect_prior.adjacency,
        terms=np.array(selected_terms),
    )

    record_templates: Dict[str, Dict[str, Any]] = {}
    sequence_lookup: Dict[str, str] = {}
    ordered_entries: list[tuple[str, str]] = []
    for row in sequences.itertuples():
        entry_id = str(row.entry_id)
        sequence_lookup[entry_id] = row.sequence
        ordered_entries.append((entry_id, row.sequence))

    embedding_meta = _ensure_embeddings_for_entries(
        ordered_entries,
        max_length=max_length,
        backend=backend,
    )

    embedding_width: Optional[int] = None
    for entry_id, _sequence in ordered_entries:
        cache_path, _, dim = embedding_meta[entry_id]
        if embedding_width is None:
            embedding_width = dim
        elif embedding_width != dim:
            raise ValueError(
                f"Inconsistent {backend} embedding dimensionality encountered; check cache integrity."
            )
        targets = _project_targets(entry_id, label_map, term_to_index, selected_terms)
        targets = _propagate_ancestor_labels(targets, parent_lookup)
        record_templates[entry_id] = {
            "entry_id": entry_id,
            "embedding_path": cache_path,
            "targets": targets,
            "labels": targets,
        }

    if not record_templates:
        raise RuntimeError("No records were generated while building manifests")

    if embedding_width is None:
        raise RuntimeError("Failed to determine embedding dimensionality")

    if feature_dim != embedding_width:
        log.warning(
            "seq_embeddings.feature_dim=%s mismatches %s embedding dimension %s; hydra overrides will update it.",
            feature_dim,
            backend.upper(),
            embedding_width,
        )

    meta_template = {
        "feature_dim": embedding_width,
        "max_length": max_length,
        "num_functions": num_functions,
        "terms": selected_terms,
        "aspect": aspect,
        "embedding_backend": backend,
    }

    bundle_paths: Dict[str, Path] = {}
    for split, ids in split_assignments.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / f"{aspect.lower()}_manifest.json"
        selected_ids = ids if ids is not None else list(record_templates.keys())
        rel_go_prior = os.path.relpath(prior_path, split_dir).replace(os.sep, "/")
        records: list[Dict[str, Any]] = []
        manifest_ids: list[str] = []
        for entry_id in selected_ids:
            template = record_templates.get(str(entry_id))
            if template is None:
                continue
            manifest_ids.append(template["entry_id"])
            rel_embed = os.path.relpath(template["embedding_path"], split_dir)
            records.append(
                {
                    "entry_id": template["entry_id"],
                    "embedding_path": rel_embed.replace(os.sep, "/"),
                    "targets": template["targets"],
                    "labels": template.get("labels", template["targets"]),
                    "go_prior_path": rel_go_prior,
                }
            )
        if not records:
            records = []
            manifest_ids = []
            for template in record_templates.values():
                manifest_ids.append(template["entry_id"])
                rel_embed = os.path.relpath(template["embedding_path"], split_dir)
                records.append(
                    {
                        "entry_id": template["entry_id"],
                        "embedding_path": rel_embed.replace(os.sep, "/"),
                        "targets": template["targets"],
                        "labels": template.get("labels", template["targets"]),
                        "go_prior_path": rel_go_prior,
                    }
                )

        protein_prior_path: Optional[Path] = None
        if use_blast_prior and manifest_ids:
            sequences_for_prior: list[str] = []
            for entry_id in manifest_ids:
                seq = sequence_lookup.get(entry_id)
                if seq is None:
                    raise KeyError(
                        f"Sequence missing for entry_id '{entry_id}' required for BLAST prior"
                    )
                sequences_for_prior.append(seq)
            prior_tensor = prot_prior_blast(sequences_for_prior, **blast_kwargs)
            protein_prior_dir = (priors_dir / "protein").resolve()
            protein_prior_dir.mkdir(parents=True, exist_ok=True)
            split_prior_path = protein_prior_dir / f"{aspect.lower()}_{split}_blast_prior.npz"
            np.savez_compressed(
                split_prior_path,
                adjacency=prior_tensor.numpy(),
            )
            protein_prior_path = split_prior_path
            rel_protein_prior = os.path.relpath(split_prior_path, split_dir).replace(
                os.sep,
                "/",
            )
            for idx, record in enumerate(records):
                record["protein_prior_path"] = rel_protein_prior
                record["protein_prior_index"] = idx

        manifest_meta = {
            **meta_template,
            "go_prior_path": rel_go_prior,
        }
        if protein_prior_path is not None:
            manifest_meta["protein_prior_path"] = os.path.relpath(
                protein_prior_path, split_dir
            ).replace(os.sep, "/")
            manifest_meta["protein_prior_method"] = "blast"

        payload = {
            "meta": manifest_meta,
            "records": records,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        bundle_paths[split] = manifest_path

    return ManifestBundle(
        aspect=aspect,
        train=bundle_paths["train"],
        val=bundle_paths["val"],
        test=bundle_paths["test"],
        num_functions=num_functions,
        go_prior_path=prior_path,
        terms=selected_terms,
        feature_dim=embedding_width,
        embedding_backend=backend,
    )
