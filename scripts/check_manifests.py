"""
Offline checker for manifest-backed cached tensors and priors.

The script assumes a manifests directory layout containing the split
subdirectories train/, val/, and test/, alongside priors/ and raw/.
Example:
    manifests/
        priors/<ontology>/<ontology>_prior.npz
        raw/<ontology>/train_split_cache.csv
        train/<ontology>_manifest.json
        val/<ontology>_manifest.json
        test/<ontology>_manifest.json

It reports common pitfalls that can introduce NaNs/Infs at validation time:
- missing or unreadable tensor files
- embeddings/labels with non-finite values
- labels with inconsistent dimensionality
- GO/protein priors that are non-square or contain non-finite values
- length annotations that do not match embedding lengths

Usage:
    python scripts/check_manifests.py [--manifest-dir /path/to/manifests]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


DEFAULT_KEY_PRIORITY: tuple[str, ...] = (
    "embeddings",
    "adjacency",
    "tensor",
    "matrix",
    "weights",
    "data",
    "arr_0",
)

EXPECTED_SPLITS: tuple[str, ...] = ("train", "test", "val")
EXPECTED_ROOT_ENTRIES: tuple[str, ...] = ("priors", "raw") + EXPECTED_SPLITS

NONFINITE_CHUNK_BYTES = 64 * 1024 * 1024  # 64MB per streaming chunk


@dataclass(frozen=True)
class TensorSummary:
    shape: tuple[int, ...]
    dtype: str
    has_nonfinite: bool

    @property
    def ndim(self) -> int:
        return len(self.shape)



@dataclass(frozen=True)
class TensorCacheKey:
    path: Path
    key: str | None
    key_priority: tuple[str, ...] | None



def _to_numpy(tensor_like: Any) -> np.ndarray:
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.detach().cpu().numpy()
    return np.asarray(tensor_like)


def _has_nonfinite(arr: np.ndarray, *, chunk_bytes: int = NONFINITE_CHUNK_BYTES) -> bool:
    if arr.size == 0:
        return False
    chunk_elems = max(1, chunk_bytes // max(arr.dtype.itemsize, 1))
    with np.nditer(
        arr,
        flags=["external_loop"],
        op_flags=["readonly"],
        order="C",
        buffersize=chunk_elems,
    ) as it:
        for chunk in it:
            if not np.isfinite(chunk).all():
                return True
    return False


def _summarize_array(arr: np.ndarray) -> TensorSummary:
    return TensorSummary(
        shape=tuple(int(dim) for dim in arr.shape),
        dtype=str(arr.dtype),
        has_nonfinite=_has_nonfinite(arr),
    )


def _load_npz_tensor(
    path: Path,
    key: str | None = None,
    *,
    key_priority: Sequence[str] | None = None,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    archive = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    try:
        if isinstance(archive, np.lib.npyio.NpzFile):
            if key is not None:
                if key not in archive.files:
                    raise KeyError(
                        f"Key '{key}' not in {path.name}; keys={list(archive.files)}"
                    )
                return archive[key]
            candidates: list[str] = []
            if key_priority:
                candidates.extend([k for k in key_priority if k])
            for fallback in DEFAULT_KEY_PRIORITY:
                if fallback not in candidates:
                    candidates.append(fallback)
            for candidate in candidates:
                if candidate in archive.files:
                    return archive[candidate]
            raise KeyError(
                f"Could not select array from {path.name}; keys={list(archive.files)}"
            )
        return archive
    finally:
        if hasattr(archive, "close"):
            archive.close()


def _resolve_tensor_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_tensor_from_path(
    base_dir: Path,
    value: str,
    *,
    key: str | None = None,
    key_priority: Sequence[str] | None = None,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    path = _resolve_tensor_path(base_dir, value)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".npy"}:
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    if suffix == ".npz":
        return _load_npz_tensor(
            path, key=key, key_priority=key_priority, mmap_mode=mmap_mode
        )
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, Mapping):
            if key is None:
                raise KeyError(
                    f"{path.name} stores a mapping; supply --key or set embedding_key/go_prior_key."
                )
            if key not in payload:
                raise KeyError(f"Key '{key}' missing from {path.name}")
            payload = payload[key]
        return _to_numpy(payload)
    raise ValueError(f"Unsupported tensor file extension for {path}")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "records" in raw and isinstance(raw["records"], list):
        return raw["records"]
    raise TypeError(f"Manifest {path} must be a list or contain a 'records' list.")


def _normalize_optional_key_sequence(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else None
    if isinstance(value, Iterable):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        return tuple(cleaned) if cleaned else None
    raise TypeError("key list must be a string or list/tuple of strings.")


def _summarize_tensor_from_path(
    cache: dict[TensorCacheKey, TensorSummary],
    base_dir: Path,
    value: str,
    *,
    key: str | None = None,
    key_priority: Sequence[str] | None = None,
) -> TensorSummary:
    resolved_path = _resolve_tensor_path(base_dir, value)
    cache_key = TensorCacheKey(
        path=resolved_path,
        key=key,
        key_priority=tuple(key_priority) if key_priority else None,
    )
    if cache_key in cache:
        return cache[cache_key]

    array = _load_tensor_from_path(
        base_dir, value, key=key, key_priority=key_priority, mmap_mode="r"
    )
    summary = _summarize_array(array)
    del array
    cache[cache_key] = summary
    return summary


def analyze_manifest(manifest_path: Path) -> list[str]:
    base_dir = manifest_path.parent
    issues: list[str] = []
    tensor_cache: dict[TensorCacheKey, TensorSummary] = {}
    try:
        records = _load_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        return [f"{manifest_path.name}: failed to read manifest: {exc}"]

    if not records:
        return [f"{manifest_path.name}: manifest contains no records."]

    label_dims: set[int] = set()

    for idx, record in enumerate(records):
        prefix = f"{manifest_path.name} [record {idx}]"
        try:
            emb_path = record.get("embedding_path")
            if not emb_path:
                issues.append(f"{prefix}: missing embedding_path.")
                continue
            emb_key = record.get("embedding_key")
            emb_summary = _summarize_tensor_from_path(
                tensor_cache, base_dir, emb_path, key=emb_key
            )
            if emb_summary.ndim != 2:
                issues.append(f"{prefix}: embeddings must be 2D, got {emb_summary.shape}.")
            if emb_summary.has_nonfinite:
                issues.append(f"{prefix}: embeddings contain NaN/Inf.")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{prefix}: failed to load embeddings: {exc}")
            continue

        try:
            labels_value = (
                record.get("labels")
                or record.get("labels_path")
                or record.get("targets")
                or record.get("targets_path")
            )
            if not labels_value:
                issues.append(f"{prefix}: missing labels or labels_path.")
            else:
                if isinstance(labels_value, (str, Path)):
                    labels_summary = _summarize_tensor_from_path(
                        tensor_cache, base_dir, str(labels_value)
                    )
                else:
                    labels_summary = _summarize_array(_to_numpy(labels_value))
                if labels_summary.ndim != 1:
                    issues.append(
                        f"{prefix}: labels must be 1D, got {labels_summary.shape}."
                    )
                else:
                    label_dims.add(int(labels_summary.shape[0]))
                if labels_summary.has_nonfinite:
                    issues.append(f"{prefix}: labels contain NaN/Inf.")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{prefix}: failed to load labels: {exc}")

        if "lengths" in record:
            try:
                lengths = record["lengths"]
                if isinstance(lengths, (str, Path)):
                    lengths_arr = _load_tensor_from_path(base_dir, str(lengths))
                else:
                    lengths_arr = _to_numpy(lengths)
                if lengths_arr.size == 1:
                    length_val = int(lengths_arr.reshape(-1)[0])
                    if emb_summary.shape[0] != length_val:
                        issues.append(
                            f"{prefix}: length mismatch (embedding len={emb_summary.shape[0]} vs lengths={length_val})."
                        )
                else:
                    issues.append(f"{prefix}: unexpected lengths shape {lengths_arr.shape}.")
            except Exception as exc:  # noqa: BLE001
                issues.append(f"{prefix}: failed to validate lengths: {exc}")

        if "go_prior" in record or "go_prior_path" in record:
            try:
                go_key = record.get("go_prior_key")
                go_key_priority = _normalize_optional_key_sequence(
                    record.get("go_prior_key_priority") or record.get("go_prior_candidates")
                )
                go_value = record.get("go_prior") or record.get("go_prior_path")
                go_summary = _summarize_tensor_from_path(
                    tensor_cache,
                    base_dir,
                    str(go_value),
                    key=go_key,
                    key_priority=go_key_priority,
                )
                if go_summary.ndim != 2 or go_summary.shape[0] != go_summary.shape[1]:
                    issues.append(f"{prefix}: GO prior is not square ({go_summary.shape}).")
                if go_summary.has_nonfinite:
                    issues.append(f"{prefix}: GO prior contains NaN/Inf.")
                if label_dims and go_summary.shape[0] not in label_dims:
                    issues.append(
                        f"{prefix}: GO prior size {go_summary.shape[0]} does not match label dimension(s) {sorted(label_dims)}."
                    )
            except Exception as exc:  # noqa: BLE001
                issues.append(f"{prefix}: failed to load GO prior: {exc}")

        if "protein_prior" in record or "protein_prior_path" in record:
            try:
                protein_value = record.get("protein_prior") or record.get("protein_prior_path")
                protein_summary = _summarize_tensor_from_path(
                    tensor_cache, base_dir, str(protein_value)
                )
                if (
                    protein_summary.ndim != 2
                    or protein_summary.shape[0] != protein_summary.shape[1]
                ):
                    issues.append(
                        f"{prefix}: protein prior is not square ({protein_summary.shape})."
                    )
                if protein_summary.has_nonfinite:
                    issues.append(f"{prefix}: protein prior contains NaN/Inf.")
                if "protein_prior_index" in record:
                    idx_val = int(record["protein_prior_index"])
                    if idx_val < 0 or idx_val >= protein_summary.shape[0]:
                        issues.append(
                            f"{prefix}: protein_prior_index {idx_val} out of bounds for prior of size {protein_summary.shape[0]}."
                        )
            except Exception as exc:  # noqa: BLE001
                issues.append(f"{prefix}: failed to load protein prior: {exc}")

    if len(label_dims) > 1:
        issues.append(
            f"{manifest_path.name}: inconsistent label dimensions detected: {sorted(label_dims)}."
        )

    return issues


def _collect_manifest_paths(manifest_dir: Path) -> list[Path]:
    """Gather manifest JSON files under the expected split layout."""
    manifest_paths: list[Path] = []
    for split in EXPECTED_SPLITS:
        split_dir = manifest_dir / split
        if split_dir.is_dir():
            manifest_paths.extend(sorted(split_dir.glob("*_manifest.json")))

    if not manifest_paths:
        manifest_paths.extend(sorted(manifest_dir.glob("*_manifest.json")))
    if not manifest_paths:
        manifest_paths.extend(sorted(manifest_dir.glob("*.json")))
    return manifest_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Check manifest tensors for NaNs/Infs and shape issues.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path.home() / "data" / "manifests",
        help="Directory containing manifest JSON files (default: ~/data/manifests).",
    )
    args = parser.parse_args()

    manifest_dir = args.manifest_dir.expanduser().resolve()
    if not manifest_dir.exists():
        print(f"[ERROR] Manifest directory not found: {manifest_dir}")
        return

    missing_entries = [
        name for name in EXPECTED_ROOT_ENTRIES if not (manifest_dir / name).exists()
    ]
    if missing_entries:
        print(
            "[WARN] Manifest root is missing expected entries: "
            f"{', '.join(missing_entries)}"
        )

    manifest_paths = _collect_manifest_paths(manifest_dir)
    if not manifest_paths:
        print(
            "[WARN] No manifest JSON files found. "
            "Expected *_manifest.json under train/, test/, and val/."
        )
        return

    total_issues = 0
    for manifest_path in manifest_paths:
        issues = analyze_manifest(manifest_path)
        total_issues += len(issues)
        if issues:
            print(f"\n[FAIL] {manifest_path} -> {len(issues)} issue(s)")
            for item in issues:
                print(f"  - {item}")
        else:
            print(f"[OK]   {manifest_path} (no issues found)")

    print("\nSummary:")
    print(f"  Manifests checked: {len(manifest_paths)}")
    print(f"  Total issues: {total_issues}")


if __name__ == "__main__":
    main()
