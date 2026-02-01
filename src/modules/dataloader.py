"""
dataloader.py

Data processing utilities. Contains all dataset- and dataloader-related logic. It provides helper
functions for reading raw CAFA-format data sources (ground truths, FASTA
sequences, IA weights) as well as utilities for loading cached
tensors via manifests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

log = None
_DEFAULT_NPZ_KEY_PRIORITY: Tuple[str, ...] = (
    "embeddings",
    "adjacency",
    "tensor",
    "matrix",
    "weights",
    "data",
    "arr_0",
)
_ASPECT_ALIASES = {
    "C": "C",
    "CC": "C",
    "CCO": "C",
    "CELLULARCOMPONENT": "C",
    "F": "F",
    "MF": "F",
    "MFO": "F",
    "MOLECULARFUNCTION": "F",
    "M": "F",
    "P": "P",
    "BP": "P",
    "BPO": "P",
    "BIOLOGICALPROCESS": "P",
    "B": "P",
}


def _ensure_logger() -> Any:
    global log
    if log is None:
        import logging

        log = logging.getLogger(__name__)
    return log


def _ensure_finite(
    tensor: torch.Tensor, *, name: str, path: Optional[Path] = None
) -> torch.Tensor:
    """Sanitize non-finite values in a tensor and warn if any are found."""

    if torch.isfinite(tensor).all():
        return tensor
    logger = _ensure_logger()
    location = f" ({path})" if path is not None else ""
    logger.warning("%s contains non-finite values%s; replacing with zeros.", name, location)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _record_length(record: Mapping[str, Any], manifest_dir: Path) -> Optional[int]:
    """Best-effort extraction of sequence length from a manifest record."""

    if "lengths" in record:
        lengths = record["lengths"]
        if isinstance(lengths, (int, float)):
            return int(lengths)
        if isinstance(lengths, (list, tuple)) and lengths:
            try:
                return int(lengths[0])
            except (TypeError, ValueError):
                return None
        if isinstance(lengths, (str, Path)):
            path = Path(lengths)
            if not path.is_absolute():
                path = (manifest_dir / path).resolve()
            try:
                arr = np.load(path, allow_pickle=False, mmap_mode="r")
                try:
                    return int(arr.reshape(-1)[0])
                finally:
                    if hasattr(arr, "close"):
                        arr.close()
            except Exception:  # noqa: BLE001
                return None

    emb_path = record.get("embedding_path")
    if emb_path:
        path = Path(emb_path)
        if not path.is_absolute():
            path = (manifest_dir / path).resolve()
        suffix = path.suffix.lower()
        if suffix in {".npy", ".npz"}:
            try:
                arr = np.load(path, allow_pickle=False, mmap_mode="r")
                try:
                    return int(arr.shape[0])
                finally:
                    if hasattr(arr, "close"):
                        arr.close()
            except Exception:  # noqa: BLE001
                return None
    return None


####### RAW DATA LOADERS #######

def parse_ground_truth_table(path: Path) -> pd.DataFrame:
    """Load CAFA ground-truth annotations into a dataframe.

    Expected format: header row with columns EntryID, term, aspect.
    Columns may be separated by tabs or whitespace. Returns a dataframe
    with canonical columns: entry_id, term, aspect.
    """

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        header=0,
        dtype=str,
    )
    # Map header variants to canonical names
    rename_map: Dict[str, str] = {}
    for col in list(df.columns):
        key = str(col).strip().lower()
        if key == "entryid":
            rename_map[col] = "entry_id"
        elif key == "term":
            rename_map[col] = "term"
        elif key == "aspect":
            rename_map[col] = "aspect"
    if rename_map:
        df = df.rename(columns=rename_map)
    req = {"entry_id", "term", "aspect"}
    if not req.issubset(df.columns):
        missing = req - set(df.columns)
        raise ValueError(f"Ground-truth file {path} missing columns: {sorted(missing)}")
    # Clean values
    df["entry_id"] = df["entry_id"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    cleaned = df["aspect"].astype(str).str.strip().str.replace(r"[^A-Za-z]", "", regex=True).str.upper()
    mapped = cleaned.map(_ASPECT_ALIASES).fillna(cleaned)
    df["aspect"] = mapped
    df = df[df["aspect"].isin(["C", "F", "P"])].reset_index(drop=True)
    _ensure_logger().info("Loaded %d ground-truth rows from %s", len(df), path)
    return df


def parse_fasta_sequences(path: Path) -> pd.DataFrame:
    """Read a FASTA file and return protein sequences with identifiers.

    Headers are expected in the CAFA format sp|P9WHI7|RECN_MYCT; the
    *EntryID* (e.g. P9WHI7) is the second pipe-delimited token. The output
    dataframe contains three columns: entry_id (protein identifier), the
    raw header, and the amino-acid sequence.
    """

    records: Dict[str, Dict[str, str]] = {}
    current_id: Optional[str] = None
    current_header: Optional[str] = None
    sequence_chunks: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = {
                        "entry_id": current_id,
                        "header": current_header or "",
                        "sequence": "".join(sequence_chunks),
                    }
                tokens = line[1:].split("|")
                if len(tokens) < 3:
                    raise ValueError(
                        f"Unexpected FASTA header format: '{line}' (expected sp|ID|DESC)"
                    )
                current_id = tokens[1]
                current_header = line[1:]
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    if current_id is not None:
        records[current_id] = {
            "entry_id": current_id,
            "header": current_header or "",
            "sequence": "".join(sequence_chunks),
        }

    df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)
    _ensure_logger().info("Loaded %d sequences from %s", len(df), path)
    return df


def load_information_accretion(path: Path) -> pd.DataFrame:
    """Load IA (information accretion) scores and return a dataframe.

    The CAFA IA.txt file contains whitespace-delimited term/ia
    pairs. The dataframe can be merged with ontology tables or passed directly
    to CAFAEval. The function also normalises missing values to 0.0 for
    convenience.
    """

    df = pd.read_csv(path, sep=r"\s+", names=["term", "ia"], dtype={"term": str, "ia": float})
    df["ia"] = df["ia"].fillna(0.0)
    _ensure_logger().info("Loaded IA weights for %d terms from %s", len(df), path)
    return df


def dataframe_to_multi_hot(
    annotations: pd.DataFrame,
    vocab: Sequence[str],
    entry_id_col: str = "entry_id",
    term_col: str = "term",
) -> Dict[str, torch.Tensor]:
    """Convert an annotation dataframe to multi-hot label tensors.

    Args:
        annotations: DataFrame with at least entry_id and term columns.
        vocab: Ordered iterable of GO terms forming the target vocabulary.
        entry_id_col: Column name holding protein identifiers.
        term_col: Column name holding GO term identifiers.

    Returns:
        Mapping from entry_id to a torch.FloatTensor of shape (len(vocab),) with
        1.0 for present terms and 0.0 otherwise.
    """

    term_to_index = {term: idx for idx, term in enumerate(vocab)}
    label_map: Dict[str, torch.Tensor] = {}

    grouped = annotations.groupby(entry_id_col)[term_col].agg(list)
    for entry_id, terms in grouped.items():
        vector = torch.zeros(len(vocab), dtype=torch.float32)
        for term in terms:
            idx = term_to_index.get(term)
            if idx is not None:
                vector[idx] = 1.0
        label_map[str(entry_id)] = vector
    return label_map


class SequenceAnnotationDataset(Dataset):
    """Dataset yielding raw sequences and GO annotations.

    The dataset accepts a dataframe containing sequences (from FASTA) and a
    mapping of entry IDs to GO term lists or multi-hot vectors.
    """

    def __init__(
        self,
        sequences: pd.DataFrame,
        annotations: Mapping[str, Sequence[str]] | Dict[str, torch.Tensor],
        term_to_index: Optional[Mapping[str, int]] = None,
    ) -> None:
        self.sequences = sequences.reset_index(drop=True)
        self.annotations = annotations
        self.term_to_index = term_to_index
        if term_to_index is not None:
            self.num_terms = len(term_to_index)
        else:
            self.num_terms = None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.sequences.iloc[index]
        entry_id = row["entry_id"]
        sample: Dict[str, Any] = {
            "entry_id": entry_id,
            "sequence": row["sequence"],
            "header": row.get("header", ""),
        }
        labels = self.annotations.get(entry_id)
        if labels is None:
            sample["terms"] = []
            if self.term_to_index is not None:
                sample["targets"] = torch.zeros(self.num_terms or 0, dtype=torch.float32)
            return sample

        if isinstance(labels, torch.Tensor):
            sample["targets"] = labels.to(dtype=torch.float32)
        else:
            sample["terms"] = list(labels)
            if self.term_to_index is not None:
                target = torch.zeros(self.num_terms or 0, dtype=torch.float32)
                for term in labels:
                    idx = self.term_to_index.get(term)
                    if idx is not None:
                        target[idx] = 1.0
                sample["targets"] = target
        return sample


###### HELPERS FOR CACHED DATA ######

def load_npz_tensor(
    path: Path,
    key: str | None = None,
    dtype: torch.dtype | None = torch.float32,
    key_priority: Sequence[str] | None = None,
) -> torch.Tensor:
    """Load a tensor from .npz/.npy/.pt files.

    The loader accepts numpy and torch serialisations, normalising everything to
    torch tensors and optionally casting to the requested dtype.

    Args:
        path: Location of the persisted tensor.
        key: Explicit array key to read from structured formats.
        dtype: Optional dtype override for the returned tensor.
        key_priority: Ordered list of fallback keys to try when `key` is not
            provided and the archive offers multiple named arrays.
    """

    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return _ensure_tensor_dtype(payload, dtype)
        if isinstance(payload, Mapping):
            if key is None:
                raise KeyError(
                    f"File {path} stores a mapping; please provide 'key' to select a tensor."
                )
            tensor = payload.get(key)
            if tensor is None:
                raise KeyError(f"Key '{key}' missing from {path.name}")
            return _ensure_tensor_dtype(tensor, dtype)
        raise TypeError(f"Unsupported payload in {path}: {type(payload)}")

    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return _ensure_tensor_dtype(torch.from_numpy(array), dtype)

    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            if key is not None:
                if key not in archive:
                    raise KeyError(
                        f"Key '{key}' not found in {path.name}; available={list(archive.files)}"
                    )
                array = archive[key]
            else:
                array = _select_npz_array(path, archive, key_priority)
        finally:
            archive.close()
        return _ensure_tensor_dtype(torch.from_numpy(array), dtype)

    raise ValueError(f"Unsupported tensor file extension: {suffix}")


def _select_npz_array(
    path: Path,
    archive: np.lib.npyio.NpzFile,
    key_priority: Sequence[str] | None,
) -> np.ndarray:
    """Resolve a best-effort array key from an .npz archive."""

    seen: set[str] = set()
    candidates: list[str] = []
    if key_priority:
        for candidate in key_priority:
            if candidate is None:
                continue
            cand = str(candidate).strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            candidates.append(cand)
    for fallback in _DEFAULT_NPZ_KEY_PRIORITY:
        if fallback in seen:
            continue
        seen.add(fallback)
        candidates.append(fallback)
    searched: list[str] = []
    for candidate in candidates:
        searched.append(candidate)
        if candidate in archive.files:
            return archive[candidate]
    raise KeyError(
        f"Could not locate a tensor in {path.name}; tried {searched} but archive "
        f"only provides {list(archive.files)}"
    )


def _ensure_tensor_dtype(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Cast tensor to dtype if requested; otherwise preserve on-disk dtype."""

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if dtype is None:
        return tensor
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype=dtype)


def _load_cached_protein_prior(path: Path) -> torch.Tensor:
    """Load and cache a protein prior adjacency matrix."""

    resolved = path.resolve()
    archive = np.load(resolved, allow_pickle=False)
    if isinstance(archive, np.lib.npyio.NpzFile):
        if "adjacency" not in archive.files:
            raise KeyError(
                f"adjacency missing from {resolved.name}; keys={list(archive.files)}"
            )
        array = archive["adjacency"]
        archive.close()
    else:
        array = archive
    tensor = torch.as_tensor(array, dtype=torch.float32)
    tensor = _ensure_finite(tensor, name="protein_prior", path=resolved)
    return tensor


def _load_cached_go_prior(
    path: Path,
    *,
    key: str | None = None,
    key_priority: Sequence[str] | None = None,
) -> torch.Tensor:
    """Load and cache GO prior adjacency matrices."""

    resolved = path.resolve()
    priority_tuple = tuple(key_priority) if key_priority else None
    tensor = load_npz_tensor(
        resolved,
        key=key,
        dtype=torch.float16,
        key_priority=priority_tuple or ("adjacency", "matrix"),
    )
    if tensor.ndim != 2:
        raise ValueError(f"GO prior at {path} must be a square matrix.")
    tensor = _ensure_finite(tensor, name="go_prior", path=resolved)
    return tensor


class ManifestDataset(Dataset):
    """Dataset backed by a JSON/JSONL manifest of cached embeddings.

    Each record must provide at least a labels field (multi-hot array or
    path to a persisted tensor) and one of embedding or embedding_path.
    Optional fields include lengths, protein_prior (or
    protein_prior_path/protein_prior_index), go_prior (or go_prior_path
    with optional go_prior_key/go_prior_key_priority hints), structure_path
    for cached sparse ESMFold graphs (edge_index/edge_weight/plddt), and
    prostt5_path for cached ProstT5 encoder embeddings.
    """

    def __init__(
        self,
        manifest_path: Path,
        min_length: Optional[int] = None,
        go_prior_enabled: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.go_prior_enabled = bool(go_prior_enabled)
        self._prostt5_infer_logged = False
        self._structure_infer_logged = False
        self._prostt5_align_logged = False
        self._structure_align_logged = False
        self._structure_fill_logged = False
        self._prostt5_embed_dim: Optional[int] = None
        self.fill_missing_structure = bool(
            int(os.environ.get("PF_AGCN_FILL_MISSING_STRUCTURE", "0"))
        )
        records = self._load_manifest(manifest_path)
        self.short_drop_count = 0
        self.original_size = len(records)
        self.original_indices = list(range(len(records)))
        if min_length is not None:
            filtered: list[Dict[str, Any]] = []
            kept_indices: list[int] = []
            for idx, record in enumerate(records):
                length_val = _record_length(record, manifest_path.parent)
                if length_val is not None and length_val < min_length:
                    self.short_drop_count += 1
                    continue
                filtered.append(record)
                kept_indices.append(idx)
            records = filtered
            self.original_indices = kept_indices
        self.records = records
        self.lengths: list[int] = []
        for record in self.records:
            length_val = _record_length(record, manifest_path.parent)
            if length_val is None:
                length_val = 0
            self.lengths.append(int(length_val))
        if not self.records:
            raise ValueError(f"Manifest {manifest_path} did not yield any records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        seq_embeddings = self._load_embedding(record)
        labels = self._load_tensor(record, key="labels")
        sample: Dict[str, torch.Tensor] = {
            "seq_embeddings": seq_embeddings,
            "targets": labels.to(dtype=torch.float32),
        }
        sample["node_index"] = torch.tensor(index, dtype=torch.long)

        if "lengths" in record:
            sample["lengths"] = self._load_tensor(record, key="lengths").to(torch.long)

        protein_prior = self._load_optional_tensor(record, base_key="protein_prior")
        if protein_prior is not None:
            sample["protein_prior"] = protein_prior

        prior_path = record.get("protein_prior_path")
        prior_index = record.get("protein_prior_index")
        if prior_path is not None and prior_index is not None:
            resolved = Path(prior_path)
            if not resolved.is_absolute():
                resolved = (self.manifest_path.parent / resolved).resolve()
            sample["protein_prior_path"] = resolved
            sample["protein_prior_index"] = int(prior_index)

        if self.go_prior_enabled:
            go_prior = self._load_optional_tensor(record, base_key="go_prior")
            if go_prior is not None:
                sample["go_prior"] = go_prior

        structure_graph = self._load_sparse_structure_graph(record)
        if structure_graph is not None:
            structure_graph = self._align_structure_graph(structure_graph, seq_embeddings.shape[0])
            sample["structure_graph"] = structure_graph
        elif self.fill_missing_structure:
            if not self._structure_fill_logged:
                _ensure_logger().warning(
                    "Missing structure graph; using empty self-loop graph fallback."
                )
                self._structure_fill_logged = True
            sample["structure_graph"] = self._empty_structure_graph(seq_embeddings.shape[0])

        seq_len = seq_embeddings.shape[0]
        lengths_tensor = sample.get("lengths")
        if lengths_tensor is not None:
            length_val = int(lengths_tensor.reshape(-1)[0].item())
            if length_val != seq_len:
                raise ValueError(
                    f"Length mismatch for {self.manifest_path.name}: "
                    f"manifest length={length_val}, embedding len={seq_len}"
                )
        else:
            sample["lengths"] = torch.tensor([seq_len], dtype=torch.long)

        prost_probs = self._load_optional_array(
            record,
            path_key="prostt5_path",
            key="embeddings",
            dtype=torch.float32,
        )
        if prost_probs is None and self._prostt5_cache_dir_exists(record):
            if not self._prostt5_infer_logged:
                _ensure_logger().warning(
                    "Missing ProstT5 cache entries; filling missing residues with zeros."
                )
                self._prostt5_infer_logged = True
            embed_dim = self._infer_prostt5_embedding_dim(record)
            prost_probs = torch.zeros((seq_len, embed_dim), dtype=torch.float32)
        if prost_probs is not None:
            if prost_probs.ndim != 2:
                raise ValueError("prostt5_probs must be shaped (length, dim).")
            prost_probs = self._align_prostt5_probs(prost_probs, seq_embeddings.shape[0])
            sample["prostt5_probs"] = _ensure_finite(
                prost_probs, name="prostt5_probs"
            )

        return sample

    def _load_manifest(self, path: Path) -> list[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._read_json(path)
        raise ValueError(f"Unsupported manifest format: {path.suffix}")

    def _read_json(self, path: Path) -> list[Dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "records" in data:
            records = data["records"]
            if not isinstance(records, list):
                raise TypeError("records must be a list of manifest entries")
            return records
        raise TypeError("JSON manifest must be a list or wrap a 'records' list")

    def _load_embedding(self, record: Mapping[str, Any]) -> torch.Tensor:
        if "embedding_path" in record:
            emb_path = Path(record["embedding_path"])
            tensor = self._load_array_from_path(
                emb_path, key=record.get("embedding_key"), dtype=None
            )
        else:
            raise KeyError("Manifest record must include embedding info")

        if tensor.ndim != 2:
            raise ValueError("Embeddings must be 2D (length, feature_dim)")
        return _ensure_finite(tensor, name="embeddings", path=emb_path if "emb_path" in locals() else None)

    def _load_tensor(
        self,
        record: Mapping[str, Any],
        key: str,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        if key in record:
            value = record[key]
            if isinstance(value, (list, tuple)):
                return torch.tensor(value)
            if isinstance(value, (int, float)):
                return torch.tensor([value])
            if isinstance(value, str):
                return self._load_array_from_path(Path(value), dtype=dtype)
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            if isinstance(value, torch.Tensor):
                return value
            raise TypeError(f"Unsupported value type for {key}: {type(value)}")
        path_key = f"{key}_path"
        if path_key in record:
            return self._load_array_from_path(Path(record[path_key]), dtype=dtype)
        raise KeyError(f"Manifest record missing '{key}' or '{path_key}'")

    def _load_optional_tensor(
        self, record: Mapping[str, Any], base_key: str
    ) -> Optional[torch.Tensor]:
        if base_key == "go_prior":
            tensor = self._load_go_prior(record)
            if tensor is not None:
                return tensor
        try:
            return self._load_tensor(record, key=base_key, dtype=None).to(dtype=torch.float32)
        except KeyError:
            return None

    def _load_go_prior(self, record: Mapping[str, Any]) -> Optional[torch.Tensor]:
        if "go_prior" in record:
            tensor = torch.as_tensor(record["go_prior"])
            return tensor.to(dtype=torch.float16)
        path = record.get("go_prior_path")
        if path is None:
            return None
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (self.manifest_path.parent / resolved).resolve()
        key = record.get("go_prior_key")
        if key is not None:
            if not isinstance(key, str):
                raise TypeError("go_prior_key must be a string.")
            key = key.strip() or None
        priority_field: Optional[Tuple[str, Any]] = None
        if "go_prior_key_priority" in record:
            priority_field = ("go_prior_key_priority", record["go_prior_key_priority"])
        elif "go_prior_candidates" in record:
            priority_field = ("go_prior_candidates", record["go_prior_candidates"])
        key_priority: Optional[Tuple[str, ...]] = None
        if priority_field is not None:
            field_name, raw_value = priority_field
            key_priority = _normalize_optional_key_sequence(raw_value, field_name)
        return _load_cached_go_prior(resolved, key=key, key_priority=key_priority)

    def _load_optional_array(
        self,
        record: Mapping[str, Any],
        *,
        path_key: str,
        key: Optional[str],
        dtype: torch.dtype | None = torch.float32,
    ) -> Optional[torch.Tensor]:
        path = record.get(path_key)
        if path is None and path_key == "prostt5_path":
            inferred = self._infer_prostt5_path(record)
            if inferred is not None:
                return self._load_array_from_path(inferred, key=key, dtype=dtype)
        if path is None:
            return None
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (self.manifest_path.parent / resolved).resolve()
        return self._load_array_from_path(resolved, key=key, dtype=dtype)

    def _infer_prostt5_path(self, record: Mapping[str, Any]) -> Optional[Path]:
        emb_path_value = record.get("embedding_path")
        if not emb_path_value:
            return None
        emb_path = Path(emb_path_value)
        if not emb_path.is_absolute():
            emb_path = (self.manifest_path.parent / emb_path).resolve()
        stem = emb_path.stem
        candidate_roots = [emb_path.parent.parent]
        manifest_root = self.manifest_path.parent.parent.parent
        if manifest_root not in candidate_roots:
            candidate_roots.append(manifest_root)
        extra_roots = []
        test_cache_root = emb_path.parent.parent / "esm_final" / "test_cache"
        if test_cache_root.exists():
            extra_roots.append(test_cache_root)
        alt_test_cache = emb_path.parent.parent / "test_cache"
        if alt_test_cache.exists():
            extra_roots.append(alt_test_cache)
        for root in extra_roots:
            if root not in candidate_roots:
                candidate_roots.append(root)
        for root in candidate_roots:
            candidate = root / "prostt5_3di_cache" / f"{stem}.npz"
            if candidate.exists():
                if not self._prostt5_infer_logged:
                    _ensure_logger().warning(
                        "Manifest missing prostt5_path; inferring from embedding_path."
                    )
                    self._prostt5_infer_logged = True
                return candidate
        return None

    def _prostt5_cache_dir_exists(self, record: Mapping[str, Any]) -> bool:
        emb_path_value = record.get("embedding_path")
        candidate_roots = []
        if emb_path_value:
            emb_path = Path(emb_path_value)
            if not emb_path.is_absolute():
                emb_path = (self.manifest_path.parent / emb_path).resolve()
            candidate_roots.append(emb_path.parent.parent)
        candidate_roots.append(self.manifest_path.parent.parent.parent)
        test_cache_root = None
        if emb_path_value:
            test_cache_root = emb_path.parent.parent / "esm_final" / "test_cache"
        if test_cache_root is not None and test_cache_root.exists():
            candidate_roots.append(test_cache_root)
        alt_test_cache = None
        if emb_path_value:
            alt_test_cache = emb_path.parent.parent / "test_cache"
        if alt_test_cache is not None and alt_test_cache.exists():
            candidate_roots.append(alt_test_cache)
        for root in candidate_roots:
            if (root / "prostt5_3di_cache").exists():
                return True
        return False

    def _infer_prostt5_embedding_dim(self, record: Mapping[str, Any]) -> int:
        if self._prostt5_embed_dim is not None:
            return self._prostt5_embed_dim
        cache_dir = self._resolve_prostt5_cache_dir(record)
        if cache_dir is not None:
            sample = next(cache_dir.glob("*.npz"), None)
            if sample is not None:
                try:
                    tensor = load_npz_tensor(sample, key="embeddings", dtype=None)
                except KeyError:
                    tensor = load_npz_tensor(sample, key=None, dtype=None)
                if tensor.ndim == 2:
                    self._prostt5_embed_dim = int(tensor.size(1))
                    return self._prostt5_embed_dim
        self._prostt5_embed_dim = 1024
        return self._prostt5_embed_dim

    def _resolve_prostt5_cache_dir(self, record: Mapping[str, Any]) -> Optional[Path]:
        emb_path_value = record.get("embedding_path")
        if not emb_path_value:
            return None
        emb_path = Path(emb_path_value)
        if not emb_path.is_absolute():
            emb_path = (self.manifest_path.parent / emb_path).resolve()
        candidate_roots = [emb_path.parent.parent, self.manifest_path.parent.parent.parent]
        test_cache_root = emb_path.parent.parent / "esm_final" / "test_cache"
        if test_cache_root.exists():
            candidate_roots.append(test_cache_root)
        alt_test_cache = emb_path.parent.parent / "test_cache"
        if alt_test_cache.exists():
            candidate_roots.append(alt_test_cache)
        for root in candidate_roots:
            cache_dir = root / "prostt5_3di_cache"
            if cache_dir.exists():
                return cache_dir
        return None

    def _load_array_from_path(
        self,
        path: Path,
        key: Optional[str] = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        return load_npz_tensor(path, key, dtype=dtype)

    def _load_sparse_structure_graph(
        self, record: Mapping[str, Any]
    ) -> Optional[Dict[str, torch.Tensor]]:
        path = record.get("structure_path")
        if path is None:
            resolved = self._infer_structure_path(record)
            if resolved is None:
                return None
        else:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = (self.manifest_path.parent / resolved).resolve()
        archive = np.load(resolved, allow_pickle=False)
        try:
            edge_index = archive["edge_index"]
            edge_weight = archive["edge_weight"]
            plddt = archive["plddt"]
        finally:
            archive.close()
        edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index_t.ndim != 2 or edge_index_t.size(0) != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        edge_weight_t = _ensure_finite(
            torch.as_tensor(edge_weight, dtype=torch.float32),
            name="structure_edge_weight",
            path=resolved,
        )
        if edge_weight_t.ndim != 1 or edge_weight_t.numel() != edge_index_t.size(1):
            raise ValueError("edge_weight must have shape (edges,).")
        plddt_t = _ensure_finite(
            torch.as_tensor(plddt, dtype=torch.float32),
            name="structure_plddt",
            path=resolved,
        )
        if plddt_t.ndim != 1:
            raise ValueError("plddt must have shape (length,).")
        return {
            "edge_index": edge_index_t,
            "edge_weight": edge_weight_t,
            "plddt": plddt_t,
            "num_nodes": int(plddt_t.numel()),
        }

    def _infer_structure_path(self, record: Mapping[str, Any]) -> Optional[Path]:
        emb_path_value = record.get("embedding_path")
        if not emb_path_value:
            return None
        emb_path = Path(emb_path_value)
        if not emb_path.is_absolute():
            emb_path = (self.manifest_path.parent / emb_path).resolve()
        stem = emb_path.stem
        candidate_roots = [emb_path.parent.parent]
        manifest_root = self.manifest_path.parent.parent.parent
        if manifest_root not in candidate_roots:
            candidate_roots.append(manifest_root)
        extra_roots = []
        test_cache_root = emb_path.parent.parent / "esm_final" / "test_cache"
        if test_cache_root.exists():
            extra_roots.append(test_cache_root)
        alt_test_cache = emb_path.parent.parent / "test_cache"
        if alt_test_cache.exists():
            extra_roots.append(alt_test_cache)
        for root in extra_roots:
            if root not in candidate_roots:
                candidate_roots.append(root)
        for root in candidate_roots:
            candidate = root / "af_graphs" / f"{stem}.npz"
            if candidate.exists():
                if not self._structure_infer_logged:
                    _ensure_logger().warning(
                        "Manifest missing structure_path; inferring from embedding_path."
                    )
                    self._structure_infer_logged = True
                return candidate
        return None

    def _align_prostt5_probs(self, probs: torch.Tensor, seq_len: int) -> torch.Tensor:
        length = probs.size(0)
        if length == seq_len:
            return probs
        if not self._prostt5_align_logged:
            _ensure_logger().warning(
                "Aligning ProstT5 length (%d) to seq embedding length (%d).",
                length,
                seq_len,
            )
            self._prostt5_align_logged = True
        if length > seq_len:
            return probs[:seq_len]
        pad = probs.new_zeros((seq_len - length, probs.size(1)))
        return torch.cat([probs, pad], dim=0)

    def _align_structure_graph(
        self, graph: Dict[str, torch.Tensor], seq_len: int
    ) -> Dict[str, torch.Tensor]:
        num_nodes = int(graph.get("num_nodes", 0))
        if num_nodes == seq_len or num_nodes <= 0:
            return graph
        if not self._structure_align_logged:
            _ensure_logger().warning(
                "Aligning structure graph nodes (%d) to seq embedding length (%d).",
                num_nodes,
                seq_len,
            )
            self._structure_align_logged = True
        edge_index = graph["edge_index"]
        edge_weight = graph["edge_weight"]
        plddt = graph["plddt"]
        if num_nodes > seq_len:
            keep = (edge_index[0] < seq_len) & (edge_index[1] < seq_len)
            edge_index = edge_index[:, keep]
            edge_weight = edge_weight[keep]
            plddt = plddt[:seq_len]
        else:
            pad = plddt.new_zeros((seq_len - num_nodes,))
            plddt = torch.cat([plddt, pad], dim=0)
        return {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "plddt": plddt,
            "num_nodes": seq_len,
        }

    @staticmethod
    def _empty_structure_graph(seq_len: int) -> Dict[str, torch.Tensor]:
        return {
            "edge_index": torch.zeros((2, 0), dtype=torch.long),
            "edge_weight": torch.zeros((0,), dtype=torch.float32),
            "plddt": torch.zeros((seq_len,), dtype=torch.float32),
            "num_nodes": int(seq_len),
        }


def _collate_sparse_structure_graphs(
    graphs: Sequence[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    node_counts = [int(graph["num_nodes"]) for graph in graphs]
    node_splits = [0]
    for count in node_counts:
        node_splits.append(node_splits[-1] + count)
    edge_indices = []
    edge_weights = []
    plddts = []
    for offset, graph in zip(node_splits[:-1], graphs):
        edge_index = graph["edge_index"] + int(offset)
        edge_indices.append(edge_index)
        edge_weights.append(graph["edge_weight"])
        plddts.append(graph["plddt"])
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_weight = torch.cat(edge_weights, dim=0)
        plddt = torch.cat(plddts, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float32)
        plddt = torch.zeros((0,), dtype=torch.float32)
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "plddt": plddt,
        "node_splits": torch.tensor(node_splits, dtype=torch.long),
    }


def collate_manifest_batch(
    batch: Sequence[Dict[str, Any]],
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    min_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Pad variable-length sequences and stack optional priors."""

    seqs = [item["seq_embeddings"] for item in batch]
    lengths_list = []
    for item, seq in zip(batch, seqs):
        length_val = int(item.get("lengths", torch.tensor([seq.shape[0]]))[0].item())
        if length_val <= 0:
            _ensure_logger().warning("Non-positive length detected; using embedding length instead.")
            length_val = seq.shape[0]
        if length_val != seq.shape[0]:
            raise ValueError(
                f"Embedding length mismatch in batch: lengths={length_val}, "
                f"embedding_len={seq.shape[0]}"
            )
        lengths_list.append(length_val)
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    lengths = torch.clamp(lengths, min=0, max=padded.size(1))
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    targets = torch.stack([item["targets"] for item in batch])
    collated: Dict[str, torch.Tensor] = {
        "seq_embeddings": padded,
        "targets": targets,
        "lengths": lengths,
        "mask": mask,
    }

    if any("structure_graph" in item for item in batch):
        graphs = [item.get("structure_graph") for item in batch]
        if all(graph is not None for graph in graphs):
            collated["structure_graph"] = _collate_sparse_structure_graphs(graphs)

    if any("prostt5_probs" in item for item in batch):
        probs_list = [item.get("prostt5_probs") for item in batch]
        if all(probs is not None for probs in probs_list):
            max_len = padded.size(1)
            last_dim = probs_list[0].size(1)
            prob_batch = torch.zeros(
                (len(probs_list), max_len, last_dim),
                dtype=probs_list[0].dtype,
            )
            for idx, probs in enumerate(probs_list):
                length = probs.size(0)
                prob_batch[idx, :length] = probs
            collated["prostt5_probs"] = prob_batch

    if any("protein_prior" in item for item in batch):
        priors = [item.get("protein_prior") for item in batch]
        if all(p is not None for p in priors):
            collated["protein_prior"] = torch.stack([p for p in priors if p is not None])
    else:
        indexed_priors = [
            (item.get("protein_prior_path"), item.get("protein_prior_index"))
            for item in batch
        ]
        if all(path is not None and index is not None for path, index in indexed_priors):
            resolved_paths = []
            indices = []
            for path_value, index_value in indexed_priors:
                resolved = Path(path_value) if not isinstance(path_value, Path) else path_value
                resolved_paths.append(resolved.resolve())
                indices.append(int(index_value))
            unique_paths = set(resolved_paths)
            if len(unique_paths) != 1:
                raise ValueError(
                    "Mixed protein prior sources within a batch are not supported."
                )
            prior_matrix = _load_cached_protein_prior(unique_paths.pop())
            selector = torch.tensor(indices, dtype=torch.long)
            submatrix = prior_matrix.index_select(0, selector).index_select(1, selector)
            collated["protein_prior"] = submatrix

    if any("go_prior" in item for item in batch):
        go_priors = [item.get("go_prior") for item in batch]
        if all(p is not None for p in go_priors):
            collated["go_prior"] = go_priors[0]

    return collated


class CachedBatchSampler(Sampler[list[int]]):
    """Batch sampler backed by precomputed index lists."""

    def __init__(self, batches: Sequence[Sequence[int]]) -> None:
        self.batches = [list(map(int, batch)) for batch in batches]

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)


def _resolve_batch_cache_path(
    manifest_path: Path,
    data_cfg: Mapping[str, Any],
    split: Optional[str],
    batch_size: int,
) -> Optional[Path]:
    path_value = data_cfg.get("batch_cache_path")
    if path_value:
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        return path
    cache_dir = data_cfg.get("batch_cache_dir")
    if cache_dir:
        root = Path(str(cache_dir)).expanduser()
        if not root.is_absolute():
            root = (manifest_path.parent / root).resolve()
        suffix = f"{split}_batches_bs{batch_size}.npz" if split else f"batches_bs{batch_size}.npz"
        return (root / suffix).resolve()
    return (manifest_path.parent / f"{manifest_path.stem}_batches_bs{batch_size}.npz").resolve()


def _save_batch_cache(
    path: Path,
    batches: Sequence[Sequence[int]],
    *,
    batch_size: int,
    seed: int,
) -> None:
    if batches:
        indices = np.concatenate([np.asarray(batch, dtype=np.int64) for batch in batches])
    else:
        indices = np.array([], dtype=np.int64)
    offsets = np.zeros(len(batches) + 1, dtype=np.int64)
    cursor = 0
    for idx, batch in enumerate(batches):
        offsets[idx] = cursor
        cursor += len(batch)
    offsets[-1] = cursor
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=indices, offsets=offsets, batch_size=batch_size, seed=seed)


def _load_batch_cache(path: Path) -> list[list[int]]:
    archive = np.load(path, allow_pickle=False)
    try:
        indices = archive["indices"].astype(np.int64)
        offsets = archive["offsets"].astype(np.int64)
    finally:
        archive.close()
    batches: list[list[int]] = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        batches.append(indices[int(start) : int(end)].tolist())
    return batches


def _build_length_buckets(
    lengths: Sequence[int],
    batch_size: int,
    *,
    drop_last: bool,
) -> list[list[int]]:
    order = np.argsort(np.asarray(lengths, dtype=np.int64))
    batches = [
        order[idx : idx + batch_size].tolist()
        for idx in range(0, len(order), batch_size)
    ]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    return batches


def build_manifest_dataloader(
    manifest: Optional[str],
    data_cfg: Mapping[str, Any],
    base_dir: Path,
    shuffle: bool,
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    go_prior_cfg: Optional[Mapping[str, Any]] = None,
    min_length: Optional[int] = None,
    split: Optional[str] = None,
) -> Optional[DataLoader]:
    """Create a manifest-backed dataloader if a path is provided."""

    if not manifest:
        return None
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = (base_dir / manifest_path).resolve()
    go_prior_enabled = True
    if isinstance(go_prior_cfg, Mapping):
        go_prior_enabled = bool(go_prior_cfg.get("enabled", True))
    dataset = ManifestDataset(
        manifest_path,
        min_length=min_length,
        go_prior_enabled=go_prior_enabled,
    )

    batch_size = int(data_cfg.get("batch_size", 64))
    drop_last = bool(data_cfg.get("drop_last", False))
    batch_seed = int(
        data_cfg.get("batch_seed", data_cfg.get("split", {}).get("seed", 1337))
    )
    cache_path = _resolve_batch_cache_path(manifest_path, data_cfg, split, batch_size)
    batches: list[list[int]]
    if cache_path and cache_path.exists():
        batches = _load_batch_cache(cache_path)
    else:
        batches = _build_length_buckets(
            dataset.lengths,
            batch_size,
            drop_last=drop_last,
        )
        if cache_path:
            _save_batch_cache(cache_path, batches, batch_size=batch_size, seed=batch_seed)
    if shuffle and batches:
        rng = np.random.default_rng(int(batch_seed))
        rng.shuffle(batches)
    batch_sampler = CachedBatchSampler(batches)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        collate_fn=lambda items: collate_manifest_batch(
            items, protein_prior_cfg=protein_prior_cfg
        ),
    )


def build_sequence_dataloader(
    sequences: pd.DataFrame,
    annotations: Mapping[str, Sequence[str]] | Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    term_to_index: Optional[Mapping[str, int]] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap raw sequence annotations in a DataLoader."""

    dataset = SequenceAnnotationDataset(
        sequences=sequences,
        annotations=annotations,
        term_to_index=term_to_index,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_ia_weights(cfg: Mapping[str, Any], base_dir: Path) -> Optional[np.ndarray]:
    """Resolve IA weights path from config sections and load as np.ndarray."""

    path_str = None
    if "ia_weights_path" in cfg.get("data", {}):
        path_str = cfg["data"]["ia_weights_path"]
    elif "ia_weights_path" in cfg.get("evaluation", {}):
        path_str = cfg["evaluation"]["ia_weights_path"]
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        _ensure_logger().warning("IA weights file not found at %s", path)
        return None
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        key = cfg.get("evaluation", {}).get("ia_weights_key", "weights")
        if key not in data:
            raise KeyError(
                f"IA weight key '{key}' missing from {path.name}: keys={list(data.keys())}"
            )
        weights = data[key]
    else:
        weights = data
    return np.asarray(weights, dtype=np.float32)
def _normalize_optional_key_sequence(value: Any, field_name: str) -> Optional[Tuple[str, ...]]:
    """Convert optional manifest key lists to a canonical tuple."""

    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise TypeError(f"{field_name} must be a string or a list/tuple of strings.")
    cleaned: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        cleaned.append(text)
    if not cleaned:
        return None
    return tuple(cleaned)
