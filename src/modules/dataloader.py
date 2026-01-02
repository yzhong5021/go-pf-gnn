"""
dataloader.py

Data processing utilities. Contains all dataset- and dataloader-related logic. It provides helper
functions for reading raw CAFA-format data sources (ground truths, FASTA
sequences, IA weights) as well as utilities for loading cached
tensors via manifests.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import MutableMapping
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

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


def _log_cuda_memory(label: str) -> None:  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
    logger = _ensure_logger()  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
    if not torch.cuda.is_available():  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        return  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
    mem_mb = torch.cuda.memory_allocated() / 1e6  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
    logger.info(  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
        "CUDA memory %s at %s: %.2f MB", label, timestamp, mem_mb  # TEMPORARY; ONLY FOR ASSESSING MEMORY USE.
    )


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


def _save_neighbor_npz(path: Path, neighbors: torch.Tensor, weights: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        neighbors=neighbors.cpu().numpy(),
        weights=weights.cpu().numpy(),
    )


def _load_neighbor_npz(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    archive = np.load(path, allow_pickle=False)
    try:
        neighbors = torch.as_tensor(archive["neighbors"], dtype=torch.long)
        weights = torch.as_tensor(archive["weights"], dtype=torch.float16)
    finally:
        if hasattr(archive, "close"):
            archive.close()
    if neighbors.ndim != 2 or weights.shape != neighbors.shape:
        raise ValueError(f"Neighbor archive at {path} must contain matching 2D arrays.")
    return neighbors, weights


def _streaming_topk_similarities(
    normed: torch.Tensor,
    top_k: int,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-row top-k cosine sims without materialising full matrix."""

    if normed.ndim != 2:
        raise ValueError("normed embeddings must be 2D (N, D)")
    n, _ = normed.shape
    if n == 0:
        raise ValueError("Cannot build neighbor graph with zero proteins.")
    if n == 1:
        neighbors = torch.full((1, 0), -1, dtype=torch.long)
        weights = torch.empty((1, 0), dtype=torch.float16)
        return neighbors, weights

    effective_k = min(max(1, top_k), max(1, n - 1))
    neighbors = torch.full((n, effective_k), -1, dtype=torch.long)
    weights = torch.zeros((n, effective_k), dtype=torch.float16)

    chunk = max(1, chunk_size)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        block = normed[start:end]
        sims = torch.matmul(block, normed.T)
        row_idx = torch.arange(end - start)
        diag_idx = start + row_idx
        sims[row_idx, diag_idx] = -1e9
        vals, inds = torch.topk(sims, k=effective_k, dim=1)
        neighbors[start:end] = inds.cpu()
        weights[start:end] = vals.to(dtype=torch.float16).cpu()

    return neighbors, weights


def _resolve_neighbors_cfg(data_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize neighbour-sampling settings from the data config."""

    neighbors_cfg = data_cfg.get("neighbors", {}) if isinstance(data_cfg, Mapping) else {}
    if not isinstance(neighbors_cfg, Mapping):
        neighbors_cfg = {}

    defaults = {
        "batch_size": 64,
        "top_k": 20,
        "cosine_cutoff": 0.3,
        "max_fanout_1": 20,
        "max_fanout_2": 10,
        "global_path": None,
        "global_train_path": None,
        "global_val_path": None,
        "global_test_path": None,
    }
    cutoff_value = neighbors_cfg.get("cosine_cutoff", defaults["cosine_cutoff"])
    if cutoff_value is None:
        cutoff_value = defaults["cosine_cutoff"]
    resolved: Dict[str, Any] = {
        "batch_size": int(neighbors_cfg.get("batch_size", defaults["batch_size"])),
        "top_k": int(neighbors_cfg.get("top_k", defaults["top_k"])),
        "cosine_cutoff": float(cutoff_value),
        "max_fanout_1": int(neighbors_cfg.get("max_fanout_1", defaults["max_fanout_1"])),
        "max_fanout_2": int(neighbors_cfg.get("max_fanout_2", defaults["max_fanout_2"])),
        "global_path": neighbors_cfg.get("global_path", defaults["global_path"]),
        "global_train_path": neighbors_cfg.get(
            "global_train_path", defaults["global_train_path"]
        ),
        "global_val_path": neighbors_cfg.get(
            "global_val_path", defaults["global_val_path"]
        ),
        "global_test_path": neighbors_cfg.get(
            "global_test_path", defaults["global_test_path"]
        ),
    }
    if "max_fanout_1" not in neighbors_cfg and "fanout_1" in neighbors_cfg:
        resolved["max_fanout_1"] = int(neighbors_cfg["fanout_1"])
    if "max_fanout_2" not in neighbors_cfg and "fanout_2" in neighbors_cfg:
        resolved["max_fanout_2"] = int(neighbors_cfg["fanout_2"])

    legacy_batch = data_cfg.get("batch_size") if isinstance(data_cfg, Mapping) else None
    if legacy_batch is not None and "batch_size" not in neighbors_cfg:
        resolved["batch_size"] = int(legacy_batch)
        _ensure_logger().warning(
            "data_config.batch_size is deprecated; use data_config.neighbors.batch_size instead."
        )

    if resolved["cosine_cutoff"] < 0.0 or resolved["cosine_cutoff"] > 1.0:
        raise ValueError("neighbors.cosine_cutoff must be between 0 and 1.")
    if resolved["top_k"] < resolved["max_fanout_1"]:
        raise ValueError("neighbors.top_k must be >= neighbors.max_fanout_1")
    if resolved["top_k"] < resolved["max_fanout_2"]:
        raise ValueError("neighbors.top_k must be >= neighbors.max_fanout_2")

    return resolved


class ProteinNeighborGraph:
    """Memory-efficient adjacency built from pooled ESM embeddings."""

    def __init__(
        self,
        neighbors: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        if neighbors.shape != weights.shape:
            raise ValueError("neighbors and weights must have the same shape.")
        self.neighbors = neighbors.to(dtype=torch.long, device="cpu")
        self.weights = weights.to(dtype=torch.float16, device="cpu")
        self.valid = (self.neighbors >= 0) & torch.isfinite(self.weights)
        self.num_nodes = self.neighbors.size(0)

    @classmethod
    def from_embeddings(
        cls,
        pooled_embeddings: torch.Tensor,
        top_k: int,
        *,
        chunk_size: int = 256,
    ) -> "ProteinNeighborGraph":
        if pooled_embeddings.ndim != 2:
            raise ValueError("pooled_embeddings must be 2D (N, D)")
        with torch.no_grad():
            pooled = pooled_embeddings.to(dtype=torch.float32, device="cpu")
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
            normed = F.normalize(pooled, dim=1)
            normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
            neighbors, weights = _streaming_topk_similarities(normed, top_k, chunk_size=chunk_size)
            weights = torch.clamp(weights, min=0.0)
            return cls(neighbors, weights)

    @classmethod
    def from_npz(cls, path: Path) -> "ProteinNeighborGraph":
        neighbors, weights = _load_neighbor_npz(path)
        return cls(neighbors, weights)

    def save(self, path: Path) -> None:
        _save_neighbor_npz(path, self.neighbors, self.weights)

    def sample_neighbors(
        self, roots: Sequence[int], max_fanout: int, cosine_cutoff: float
    ) -> list[int]:
        neighbors: list[int] = []
        if max_fanout <= 0:
            return neighbors
        _log_cuda_memory("sample_neighbors:start")
        cutoff = float(cosine_cutoff)
        for root in roots:
            if root < 0 or root >= self.num_nodes:
                continue
            idx_tensor = self.neighbors[root]
            weight_tensor = self.weights[root]
            mask = self.valid[root] & (weight_tensor >= cutoff) & (idx_tensor != root)
            idx_tensor = idx_tensor[mask]
            weight_tensor = weight_tensor[mask].to(dtype=torch.float32)
            if idx_tensor.numel() == 0:
                continue
            if idx_tensor.numel() > max_fanout:
                _, selected = torch.topk(weight_tensor, k=max_fanout, largest=True)
                idx_tensor = idx_tensor[selected]
            for candidate in idx_tensor.tolist():
                candidate = int(candidate)
                if candidate not in neighbors:
                    neighbors.append(candidate)
        _log_cuda_memory("sample_neighbors:done")
        return neighbors

    def subgraph_adjacency(self, nodes: Sequence[int]) -> torch.Tensor:
        node_map = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        adjacency = torch.zeros((size, size), dtype=torch.float32)
        _log_cuda_memory("subgraph_adjacency:start")

        for local_idx, global_idx in enumerate(nodes):
            if global_idx < 0 or global_idx >= self.num_nodes:
                continue
            neighbors = self.neighbors[global_idx]
            weights = self.weights[global_idx]
            mask = self.valid[global_idx]
            if mask.any():
                neighbors = neighbors[mask]
                weights = weights[mask]
            for neighbor, weight in zip(neighbors.tolist(), weights.tolist()):
                target = node_map.get(int(neighbor))
                if target is None:
                    continue
                value = float(weight)
                adjacency[local_idx, target] = max(adjacency[local_idx, target].item(), value)
                adjacency[target, local_idx] = max(adjacency[target, local_idx].item(), value)

        if size > 0:
            adjacency.fill_diagonal_(1.0)
        _log_cuda_memory("subgraph_adjacency:done")
        return adjacency


class NeighborSubgraphCollator:
    """Collate manifest samples into neighbour-sampled protein subgraphs."""

    def __init__(
        self,
        dataset: "ManifestDataset",
        neighbors_cfg: Mapping[str, Any],
        protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.dataset = dataset
        self.protein_prior_cfg = protein_prior_cfg or {}
        self.root_batch_size = int(neighbors_cfg.get("batch_size", 64))
        self.top_k = int(neighbors_cfg.get("top_k", 20))
        self.cosine_cutoff = float(neighbors_cfg.get("cosine_cutoff", 0.3))
        self.max_fanout_1 = int(neighbors_cfg.get("max_fanout_1", 20))
        self.max_fanout_2 = int(neighbors_cfg.get("max_fanout_2", 10))
        self.global_path = neighbors_cfg.get("global_path")
        if self.top_k < self.max_fanout_1:
            raise ValueError("neighbors.top_k must be >= neighbors.max_fanout_1")
        if self.top_k < self.max_fanout_2:
            raise ValueError("neighbors.top_k must be >= neighbors.max_fanout_2")

        logger = _ensure_logger()
        logger.info(
            "Preparing global protein neighbor graph with top_k=%d over %d proteins",
            self.top_k,
            len(self.dataset),
        )
        self.graph = self._prepare_graph(logger)
        self._generator = torch.Generator().manual_seed(torch.initial_seed())

    def _pool_embeddings(self) -> torch.Tensor:
        _log_cuda_memory("pool_embeddings:start")
        pooled: list[torch.Tensor] = []
        for record in self.dataset.records:
            emb = self.dataset._load_embedding(record)
            pooled.append(_ensure_finite(emb, name="embeddings").mean(dim=0))
        stacked = torch.stack(pooled, dim=0)
        _log_cuda_memory("pool_embeddings:stacked")
        return stacked

    def _prepare_graph(self, logger: Any) -> ProteinNeighborGraph:
        path: Optional[Path] = None
        if self.global_path:
            path = Path(self.global_path)
            if not path.is_absolute():
                path = (self.dataset.manifest_path.parent / path).resolve()
        else:
              stem = self.dataset.manifest_path.stem
              stem = re.sub(r"^(mf|bp|cc)[_-]", "", stem, flags=re.IGNORECASE)
              stem = re.sub(r"[_-](mf|bp|cc)$", "", stem, flags=re.IGNORECASE)
              path = self.dataset.manifest_path.parent / f"{stem}_neighbors_top{self.top_k}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Neighbor cache not found at {path}; provide a precomputed neighbors npz via "
                "data_config.neighbors.global_*_path."
            )

        logger.info("Loading cached protein neighbors from %s", path)
        neighbors_graph = ProteinNeighborGraph.from_npz(path)
        self.global_path = str(path)
        return neighbors_graph

    def __call__(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
        _log_cuda_memory("subgraph_collate:start")
        root_indices = [int(item.get("node_index", torch.tensor(0)).item()) for item in batch]
        root_lookup = {int(item.get("node_index", torch.tensor(0)).item()): item for item in batch}
        root_set = set(root_indices)

        hop1_counts: list[int] = []
        hop2_counts: list[int] = []
        first_hop_all: list[int] = []
        second_hop_all: list[int] = []
        seeds_with_zero_second = 0

        for root in root_indices:
            first = self.graph.sample_neighbors(
                [root], self.max_fanout_1, self.cosine_cutoff
            )
            hop1_counts.append(len(first))
            first_hop_all.extend(first)
            second = self.graph.sample_neighbors(first, self.max_fanout_2, self.cosine_cutoff)
            hop2_counts.append(len(second))
            if len(second) == 0:
                seeds_with_zero_second += 1
            second_hop_all.extend(second)

        first_hop = list(dict.fromkeys(first_hop_all))
        _log_cuda_memory("subgraph_collate:after_first_hop")
        second_hop = list(dict.fromkeys(second_hop_all))
        _log_cuda_memory("subgraph_collate:after_second_hop")

        nodes: list[int] = []
        seen: set[int] = set()

        def _append(idx: int) -> None:
            if idx in seen:
                return
            seen.add(idx)
            nodes.append(idx)

        for idx in root_indices:
            _append(idx)
        for idx in first_hop:
            _append(idx)
        for idx in second_hop:
            _append(idx)

        go_prior = batch[0].get("go_prior") if batch else None
        samples: list[Dict[str, torch.Tensor]] = []

        for idx in nodes:
            if idx in root_lookup:
                sample = dict(root_lookup[idx])
            else:
                sample = dict(self.dataset[idx])
            sample.pop("protein_prior", None)
            sample.pop("protein_prior_path", None)
            sample.pop("protein_prior_index", None)
            if go_prior is not None and "go_prior" not in sample:
                sample["go_prior"] = go_prior
            samples.append(sample)

        collated = collate_manifest_batch(samples, protein_prior_cfg=self.protein_prior_cfg)
        collated.pop("protein_prior", None)

        adjacency = self.graph.subgraph_adjacency(nodes)
        _log_cuda_memory("subgraph_collate:after_adjacency")
        collated["protein_prior"] = adjacency
        collated["node_indices"] = torch.tensor(nodes, dtype=torch.long)
        root_mask = torch.tensor([idx in root_set for idx in nodes], dtype=torch.bool)
        collated["target_mask"] = root_mask

        root_positions = [nodes.index(idx) for idx in root_indices]
        collated["root_indices"] = torch.tensor(root_positions, dtype=torch.long)
        mean_hop1 = float(np.mean(hop1_counts)) if hop1_counts else 0.0
        mean_hop2 = float(np.mean(hop2_counts)) if hop2_counts else 0.0
        edge_count = int(torch.count_nonzero(adjacency > 0).item())
        edge_count = max(edge_count - len(nodes), 0)
        zero_second_pct = float(seeds_with_zero_second) / float(len(root_indices)) if root_indices else 0.0
        collated["graph_stats"] = {
            "nodes": float(len(nodes)),
            "edges": float(edge_count),
            "mean_hop1": mean_hop1,
            "mean_hop2": mean_hop2,
            "pct_seed_zero_second": zero_second_pct,
        }
        return collated


class ManifestDataset(Dataset):
    """Dataset backed by a JSON/JSONL manifest of cached embeddings.

    Each record must provide at least a labels field (multi-hot array or
    path to a persisted tensor) and one of embedding or embedding_path.
    Optional fields include lengths, protein_prior (or
    protein_prior_path/protein_prior_index), and go_prior (or go_prior_path
    with optional go_prior_key/go_prior_key_priority hints).
    """

    def __init__(self, manifest_path: Path, min_length: Optional[int] = None) -> None:
        self.manifest_path = manifest_path
        records = self._load_manifest(manifest_path)
        self.short_drop_count = 0
        if min_length is not None:
            filtered: list[Dict[str, Any]] = []
            for record in records:
                length_val = _record_length(record, manifest_path.parent)
                if length_val is not None and length_val < min_length:
                    self.short_drop_count += 1
                    continue
                filtered.append(record)
            records = filtered
        self.records = records
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

        go_prior = self._load_optional_tensor(record, base_key="go_prior")
        if go_prior is not None:
            sample["go_prior"] = go_prior

        seq_len = seq_embeddings.shape[0]
        lengths_tensor = sample.get("lengths")
        if lengths_tensor is not None:
            length_val = int(lengths_tensor.reshape(-1)[0].item())
            if length_val != seq_len:
                _ensure_logger().warning(
                    "Length mismatch for %s: manifest length=%s, embedding len=%s; using embedding length.",
                    self.manifest_path.name,
                    length_val,
                    seq_len,
                )
                sample["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        else:
            sample["lengths"] = torch.tensor([seq_len], dtype=torch.long)

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

    def _load_array_from_path(
        self,
        path: Path,
        key: Optional[str] = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor:
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        return load_npz_tensor(path, key, dtype=dtype)



def collate_manifest_batch(
    batch: Sequence[Dict[str, torch.Tensor]],
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    min_length: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences and stack optional priors."""

    seqs = [item["seq_embeddings"] for item in batch]
    lengths_list = []
    for item, seq in zip(batch, seqs):
        length_val = int(item.get("lengths", torch.tensor([seq.shape[0]]))[0].item())
        if length_val <= 0:
            _ensure_logger().warning("Non-positive length detected; using embedding length instead.")
            length_val = seq.shape[0]
        if length_val != seq.shape[0]:
            length_val = min(length_val, seq.shape[0])
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


def build_manifest_dataloader(
    manifest: Optional[str],
    data_cfg: Mapping[str, Any],
    base_dir: Path,
    shuffle: bool,
    protein_prior_cfg: Optional[Mapping[str, Any]] = None,
    min_length: Optional[int] = None,
    split: Optional[str] = None,
) -> Optional[DataLoader]:
    """Create a manifest-backed dataloader if a path is provided."""

    if not manifest:
        return None
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = (base_dir / manifest_path).resolve()
    dataset = ManifestDataset(manifest_path, min_length=min_length)

    neighbors_cfg = _resolve_neighbors_cfg(data_cfg)
    split_key = f"global_{str(split).lower()}_path" if split else None
    chosen_path = neighbors_cfg.get(split_key) if split_key else None
    if not chosen_path:
        chosen_path = neighbors_cfg.get("global_path")
    neighbors_cfg["global_path"] = chosen_path

    collate = NeighborSubgraphCollator(
        dataset,
        neighbors_cfg,
        protein_prior_cfg=protein_prior_cfg,
    )

    if isinstance(data_cfg, MutableMapping):
        neighbors_node = data_cfg.setdefault("neighbors", {})
        if isinstance(neighbors_node, MutableMapping):
            resolved_path = getattr(collate, "global_path", chosen_path)
            if split_key and split_key in neighbors_node:
                neighbors_node[split_key] = resolved_path
            if "global_path" in neighbors_node:
                neighbors_node["global_path"] = resolved_path
        else:
            try:
                data_cfg["neighbors"] = {"global_path": getattr(collate, "global_path", None)}
                if split_key:
                    data_cfg["neighbors"][split_key] = getattr(collate, "global_path", None)
            except Exception:  # noqa: BLE001
                pass

    neighbors_cfg["global_path"] = getattr(collate, "global_path", neighbors_cfg.get("global_path"))

    return DataLoader(
        dataset,
        batch_size=int(neighbors_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        collate_fn=collate,
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
