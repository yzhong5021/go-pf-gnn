"""Build stratified evaluation sets for PF-AGCN using CAFA6 Train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.dataloader import parse_fasta_sequences, parse_ground_truth_table
from utils.esm_embed import ESM_Embed
from utils.go_prior import Go_Prior

LOGGER = logging.getLogger(__name__)

ASPECT_MAP = {"F": "MF", "P": "BP", "C": "CC"}
BUCKET_NAMES = ("low", "med", "high")


@dataclass(frozen=True)
class BucketSpec:
    name: str
    ids: List[str]


def _safe_entry_id(entry_id: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in entry_id)


def _embedding_cache_path(root: Path, entry_id: str) -> Path:
    safe_id = _safe_entry_id(entry_id)
    digest = hashlib.md5(entry_id.encode("utf-8")).hexdigest()[:8]
    filename = f"{safe_id or 'protein'}_{digest}.npz"
    return root / filename


def _load_taxonomy(path: Path) -> Dict[str, str]:
    if not path.exists():
        LOGGER.warning("taxonomy file missing at %s", path)
        return {}
    df = pd.read_csv(path, sep="\t", header=None, names=["entry_id", "tax_id"], dtype=str)
    df["entry_id"] = df["entry_id"].astype(str).str.strip()
    df["tax_id"] = df["tax_id"].astype(str).str.strip()
    return dict(zip(df["entry_id"], df["tax_id"]))


def _terms_by_aspect_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    df = df.copy()
    df["aspect"] = df["aspect"].map(ASPECT_MAP).fillna(df["aspect"])
    by_aspect: Dict[str, Dict[str, List[str]]] = {}
    for aspect, group in df.groupby("aspect"):
        grouped = group.groupby("entry_id")["term"].apply(list)
        by_aspect[str(aspect)] = grouped.to_dict()
    return by_aspect


def _term_frequencies(by_aspect: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Counter]:
    freqs: Dict[str, Counter] = {}
    for aspect, mapping in by_aspect.items():
        counter: Counter = Counter()
        for terms in mapping.values():
            counter.update(terms)
        freqs[str(aspect)] = counter
    return freqs


def _normalize_bucket_fracs(fracs: Sequence[float] | None) -> Tuple[float, float, float]:
    if fracs is None:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    if len(fracs) != 3:
        raise ValueError("bucket_fracs must contain three values (low, med, high).")
    total = float(sum(fracs))
    if total <= 0.0:
        raise ValueError("bucket_fracs must sum to a positive value.")
    if abs(total - 1.0) > 1e-6:
        LOGGER.warning("bucket_fracs sum to %.4f; normalizing to 1.0", total)
    low, med, high = (float(val) / total for val in fracs)
    if low < 0 or med < 0 or high < 0:
        raise ValueError("bucket_fracs values must be non-negative.")
    return low, med, high


def _bucket_by_rank(
    ids: Sequence[str],
    scores: np.ndarray,
    fracs: Sequence[float] | None = None,
) -> Dict[str, BucketSpec]:
    order = np.argsort(scores)
    total = len(order)
    low_frac, med_frac, _high_frac = _normalize_bucket_fracs(fracs)
    split1 = int(total * low_frac)
    split2 = int(total * (low_frac + med_frac))
    split1 = max(0, min(split1, total))
    split2 = max(split1, min(split2, total))
    buckets = {
        "low": [ids[idx] for idx in order[:split1]],
        "med": [ids[idx] for idx in order[split1:split2]],
        "high": [ids[idx] for idx in order[split2:]],
    }
    return {name: BucketSpec(name=name, ids=ids_list) for name, ids_list in buckets.items()}


def _sample_bucket(
    bucket: Sequence[str],
    size: int,
    rng: np.random.Generator,
    prefer: Iterable[str] | None = None,
) -> List[str]:
    bucket_list = list(bucket)
    if not bucket_list:
        return []
    prefer_ids = set(prefer or [])
    preferred = [entry_id for entry_id in bucket_list if entry_id in prefer_ids]
    selected: List[str] = []
    if preferred:
        if len(preferred) >= size:
            return rng.choice(preferred, size=size, replace=False).tolist()
        selected.extend(preferred)
    remaining = [entry_id for entry_id in bucket_list if entry_id not in set(selected)]
    needed = size - len(selected)
    if needed <= 0:
        return selected
    if len(remaining) >= needed:
        selected.extend(rng.choice(remaining, size=needed, replace=False).tolist())
    else:
        LOGGER.warning("bucket size %d < needed %d; sampling with replacement", len(remaining), needed)
        if remaining:
            selected.extend(rng.choice(remaining, size=needed, replace=True).tolist())
        else:
            selected.extend(rng.choice(bucket_list, size=needed, replace=True).tolist())
    return selected


def _write_train_split_cache(train_terms: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated = train_terms.groupby("entry_id")["term"].agg(list).reset_index()
    aggregated["go_terms"] = aggregated["term"].apply(json.dumps)
    aggregated[["entry_id", "go_terms"]].to_csv(output_path, index=False)


def _build_parent_lookup(adjacency: np.ndarray) -> List[List[int]]:
    if adjacency.size == 0:
        return []
    num_terms = int(adjacency.shape[1])
    parents: List[List[int]] = []
    for child_idx in range(num_terms):
        parent_indices = np.flatnonzero(adjacency[:, child_idx] > 0).astype(int).tolist()
        parents.append(parent_indices)
    return parents


def _propagate_ancestor_labels(
    targets: Sequence[float],
    parent_lookup: Sequence[Sequence[int]],
) -> List[float]:
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


def _targets_for_entry(
    entry_terms: Sequence[str],
    term_index: Mapping[str, int],
    parent_lookup: Sequence[Sequence[int]],
) -> List[float]:
    if not term_index:
        return []
    targets = np.zeros(len(term_index), dtype=np.float32)
    for term in entry_terms:
        idx = term_index.get(term)
        if idx is not None:
            targets[idx] = 1.0
    return _propagate_ancestor_labels(targets.tolist(), parent_lookup)


def _compute_val_size(total: int, ratios: Mapping[str, float]) -> int:
    weights = np.array(
        [float(ratios.get("train", 0.8)), float(ratios.get("val", 0.1)), float(ratios.get("test", 0.1))],
        dtype=float,
    )
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("split ratios must sum to a positive value")
    weights = weights / total_weight
    counts = np.floor(weights * total).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-weights)
    idx = 0
    while remainder > 0 and len(order) > 0:
        target = order[idx % len(order)]
        counts[target] += 1
        remainder -= 1
        idx += 1
    return int(counts[1])


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _compute_similarity_scores(
    train_sequences: Sequence[str],
    candidate_ids: Sequence[str],
    candidate_sequences: Sequence[str],
    *,
    n_features: int,
    batch_size: int,
) -> Dict[str, float]:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        n_features=int(n_features),
        alternate_sign=False,
        norm="l2",
    )
    train_matrix = vectorizer.transform(train_sequences)
    centroid = train_matrix.mean(axis=0)
    centroid_vec = np.asarray(centroid).ravel()
    centroid_vec = normalize(centroid_vec.reshape(1, -1)).ravel()
    scores: Dict[str, float] = {}
    for ids_chunk, seq_chunk in zip(
        _chunked(candidate_ids, batch_size), _chunked(candidate_sequences, batch_size)
    ):
        batch_matrix = vectorizer.transform(seq_chunk)
        batch_scores = np.asarray(batch_matrix.dot(centroid_vec)).ravel()
        for entry_id, score in zip(ids_chunk, batch_scores):
            scores[str(entry_id)] = float(score)
    return scores


def _ensure_embeddings(
    entry_ids: Sequence[str],
    sequences: Mapping[str, str],
    *,
    cache_root: Path,
    reuse_root: Path | None,
    device: str,
    batch_size: int,
    model_cache: Path | None,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    reuse_root = reuse_root if reuse_root and reuse_root.exists() else None
    pending: List[str] = []
    for entry_id in entry_ids:
        dest = _embedding_cache_path(cache_root, entry_id)
        if dest.exists():
            continue
        if reuse_root is not None:
            src = _embedding_cache_path(reuse_root, entry_id)
            if src.exists():
                try:
                    os.link(src, dest)
                    continue
                except OSError:
                    shutil.copy2(src, dest)
                    continue
        pending.append(entry_id)

    if not pending:
        LOGGER.info("All embeddings already present under %s", cache_root)
        return

    embedder = ESM_Embed(cache_dir=model_cache, device=device)
    chunk_size = max(1, int(batch_size))
    processed = 0
    total = len(pending)
    for chunk in _chunked(pending, chunk_size):
        seqs = [sequences[entry_id] for entry_id in chunk]
        embeddings, masks = embedder(seqs)
        embeddings = embeddings.detach().cpu()
        masks = masks.detach().cpu()
        for idx, entry_id in enumerate(chunk):
            dest = _embedding_cache_path(cache_root, entry_id)
            masked = embeddings[idx][masks[idx]]
            np.savez_compressed(dest, embeddings=masked.numpy().astype(np.float16))
        processed += len(chunk)
        LOGGER.info("embedded %d/%d sequences", processed, total)


def _resolve_embed_dim_any(cache_root: Path, reuse_root: Path | None) -> int:
    for root in [cache_root, reuse_root]:
        if root is None or not root.exists():
            continue
        candidate = next(root.glob("*.npz"), None)
        if candidate is None:
            candidate = next(root.glob("*.npy"), None)
        if candidate is None:
            continue
        if candidate.suffix.lower() == ".npz":
            archive = np.load(candidate, mmap_mode="r", allow_pickle=False)
            key = "embeddings" if "embeddings" in archive.files else "arr_0"
            shape = archive[key].shape
            archive.close()
        else:
            array = np.load(candidate, mmap_mode="r", allow_pickle=False)
            shape = array.shape
        return int(shape[1])
    raise FileNotFoundError("Unable to determine embedding dimension from caches")


def _write_manifest(
    output_dir: Path,
    *,
    aspect: str,
    set_name: str,
    entry_ids: Sequence[str],
    labels: Mapping[str, Sequence[float]],
    embed_root: Path,
    prior_path: Path,
    feature_dim: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    rel_prior = os.path.relpath(prior_path, output_dir).replace(os.sep, "/")
    for entry_id in entry_ids:
        embed_path = _embedding_cache_path(embed_root, entry_id)
        rel_embed = os.path.relpath(embed_path, output_dir).replace(os.sep, "/")
        targets = labels.get(entry_id, [])
        records.append(
            {
                "entry_id": entry_id,
                "embedding_path": rel_embed,
                "targets": targets,
                "labels": targets,
                "go_prior_path": rel_prior,
            }
        )
    payload = {
        "meta": {
            "aspect": aspect,
            "set_name": set_name,
            "num_functions": len(next(iter(labels.values()))) if labels else 0,
            "feature_dim": feature_dim,
            "embedding_backend": "esm",
        },
        "records": records,
    }
    manifest_path = output_dir / f"{aspect.lower()}_{set_name}_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified eval sets for PF-AGCN")
    parser.add_argument("--train-seqs", type=Path, required=True)
    parser.add_argument("--full-seqs", type=Path, required=True)
    parser.add_argument("--train-terms", type=Path, required=True)
    parser.add_argument("--full-terms", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--obo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--embed-cache", type=Path, required=True)
    parser.add_argument("--reuse-cache", type=Path, default=None)
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-batch", type=int, default=4)
    parser.add_argument("--sim-batch", type=int, default=5000)
    parser.add_argument("--sim-features", type=int, default=2**16)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--top-k-mf", type=int, default=528)
    parser.add_argument("--top-k-bp", type=int, default=1024)
    parser.add_argument("--top-k-cc", type=int, default=528)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument(
        "--bucket-fracs",
        nargs=3,
        type=float,
        metavar=("LOW", "MED", "HIGH"),
        default=None,
        help="Fractions for low/med/high buckets (default: equal thirds).",
    )
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    train_seqs_df = parse_fasta_sequences(args.train_seqs)
    full_seqs_df = parse_fasta_sequences(args.full_seqs)
    train_ids = set(train_seqs_df["entry_id"].astype(str))

    full_seqs_df["entry_id"] = full_seqs_df["entry_id"].astype(str)
    full_seqs_df["sequence"] = full_seqs_df["sequence"].astype(str)
    candidate_df = full_seqs_df[~full_seqs_df["entry_id"].isin(train_ids)].reset_index(drop=True)

    train_terms_df = parse_ground_truth_table(args.train_terms)
    full_terms_df = parse_ground_truth_table(args.full_terms)

    train_terms_by_aspect = _terms_by_aspect_from_df(train_terms_df)
    full_terms_by_aspect = _terms_by_aspect_from_df(full_terms_df)
    term_freqs = _term_frequencies(train_terms_by_aspect)

    taxonomy = _load_taxonomy(args.taxonomy)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    val_size = _compute_val_size(len(train_ids), ratios)
    LOGGER.info("computed eval set size=%d (train=%d)", val_size, len(train_ids))

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_split_csv = raw_dir / "train_split_cache.csv"
    _write_train_split_cache(train_terms_df, train_split_csv)

    go_priors = Go_Prior(
        args.obo,
        train_split_csv,
        top_k_mf=args.top_k_mf,
        top_k_bp=args.top_k_bp,
        top_k_cc=args.top_k_cc,
    )

    candidate_ids = candidate_df["entry_id"].tolist()
    candidate_sequences = candidate_df["sequence"].tolist()
    train_sequences = train_seqs_df["sequence"].astype(str).tolist()
    candidate_index = {entry_id: idx for idx, entry_id in enumerate(candidate_ids)}

    similarity_scores = _compute_similarity_scores(
        train_sequences,
        candidate_ids,
        candidate_sequences,
        n_features=args.sim_features,
        batch_size=args.sim_batch,
    )

    rng = np.random.default_rng(args.seed)

    selections: Dict[str, Dict[str, List[str]]] = {}
    manifests: List[Path] = []
    all_selected: List[str] = []

    if args.feature_dim is not None:
        feature_dim = int(args.feature_dim)
    else:
        feature_dim = _resolve_embed_dim_any(args.embed_cache, args.reuse_cache)

    for aspect, prior in go_priors.items():
        aspect = str(aspect)
        term_index = {term: idx for idx, term in enumerate(prior.terms)}
        parent_lookup = _build_parent_lookup(prior.adjacency)

        aspect_terms = full_terms_by_aspect.get(aspect, {})
        freq_counter = term_freqs.get(aspect, Counter())
        freq_scores = np.array(
            [
                float(
                    np.mean([np.log1p(freq_counter.get(term, 0)) for term in aspect_terms.get(entry_id, [])])
                )
                if aspect_terms.get(entry_id)
                else 0.0
                for entry_id in candidate_ids
            ],
            dtype=np.float32,
        )
        sim_scores = np.array([similarity_scores.get(entry_id, 0.0) for entry_id in candidate_ids], dtype=np.float32)

        freq_buckets = _bucket_by_rank(candidate_ids, freq_scores, args.bucket_fracs)
        sim_buckets = _bucket_by_rank(candidate_ids, sim_scores, args.bucket_fracs)

        aspect_selection: Dict[str, List[str]] = {}
        reuse_pool: set[str] = set()

        for bucket in BUCKET_NAMES:
            set_name = f"go_freq_{bucket}"
            chosen = _sample_bucket(freq_buckets[bucket].ids, val_size, rng, prefer=reuse_pool)
            aspect_selection[set_name] = chosen
            reuse_pool.update(chosen)
            all_selected.extend(chosen)

        for bucket in BUCKET_NAMES:
            set_name = f"sim_{bucket}"
            chosen = _sample_bucket(sim_buckets[bucket].ids, val_size, rng, prefer=reuse_pool)
            aspect_selection[set_name] = chosen
            reuse_pool.update(chosen)
            all_selected.extend(chosen)

        selections[aspect] = aspect_selection

        prior_dir = output_root / "priors" / aspect.lower()
        prior_dir.mkdir(parents=True, exist_ok=True)
        prior_path = prior_dir / f"{aspect.lower()}_prior.npz"
        np.savez_compressed(
            prior_path,
            adjacency=prior.adjacency,
            terms=np.array(list(prior.terms)),
        )

        aspect_manifest_dir = output_root / aspect.lower()
        labels_cache: Dict[str, List[float]] = {}
        for set_name, entry_list in aspect_selection.items():
            for entry_id in entry_list:
                if entry_id in labels_cache:
                    continue
                entry_terms = aspect_terms.get(entry_id, [])
                labels_cache[entry_id] = _targets_for_entry(entry_terms, term_index, parent_lookup)
            manifest_path = _write_manifest(
                aspect_manifest_dir,
                aspect=aspect,
                set_name=set_name,
                entry_ids=entry_list,
                labels=labels_cache,
                embed_root=args.embed_cache,
                prior_path=prior_path,
                feature_dim=feature_dim,
            )
            manifests.append(manifest_path)

        selection_dir = output_root / "selections" / aspect.lower()
        selection_dir.mkdir(parents=True, exist_ok=True)
        for set_name, entry_list in aspect_selection.items():
            data = {
                "entry_id": entry_list,
                "tax_id": [taxonomy.get(entry_id, "") for entry_id in entry_list],
                "go_freq_score": [
                    float(freq_scores[candidate_index[entry_id]]) for entry_id in entry_list
                ],
                "similarity_score": [
                    float(sim_scores[candidate_index[entry_id]]) for entry_id in entry_list
                ],
            }
            pd.DataFrame(data).to_csv(selection_dir / f"{aspect.lower()}_{set_name}.csv", index=False)

    if not args.skip_embeddings:
        sequence_lookup = dict(zip(full_seqs_df["entry_id"], full_seqs_df["sequence"]))
        unique_selected = sorted(set(all_selected))
        _ensure_embeddings(
            unique_selected,
            sequence_lookup,
            cache_root=args.embed_cache,
            reuse_root=args.reuse_cache,
            device=args.device,
            batch_size=args.embed_batch,
            model_cache=args.model_cache,
        )
        sentinel = output_root / "embeddings_complete.txt"
        sentinel.write_text("embeddings complete\n", encoding="utf-8")

    manifest_index = output_root / "manifest_index.json"
    manifest_index.write_text(
        json.dumps({"manifests": [path.as_posix() for path in manifests]}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("wrote %d manifests to %s", len(manifests), output_root)


if __name__ == "__main__":
    main()
