"""Minimal PF-AGCN prediction runner for CAFA-style TSV outputs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.dataloader import dataframe_to_multi_hot, parse_ground_truth_table
from utils.go_prior import Go_Prior
import utils.go_prior as go_prior_module
from src.model.structural_model import StructuralPFAGCN
from src.train.training import compute_cafa_metrics
import preprocessing as preprocessing_module

log = logging.getLogger(__name__)

OBO_PATH = Path("/home/lerchen/code/cafa_proj/cafa6/Train/go-basic.obo")
NOISE_FLOOR = 1e-4
MAX_TERMS = 1500
DEFAULT_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
ASPECTS = ("MF", "BP", "CC")
ASPECT_TO_CODE = {"MF": "F", "BP": "P", "CC": "C"}


@dataclass(frozen=True)
class EntryRecord:
    """Cached input paths for a single protein entry."""

    entry_id: str
    length: int
    embedding_path: Path
    structure_path: Path
    prostt5_path: Optional[Path] = None


class PredictionDataset(Dataset):
    """Dataset that loads cached embeddings and structure graphs by entry."""

    def __init__(self, records: Sequence[EntryRecord], require_prostt5: bool) -> None:
        self.records = list(records)
        self.require_prostt5 = bool(require_prostt5)
        if not self.records:
            raise ValueError("Prediction dataset received no records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        embeddings = _load_npz_array(record.embedding_path, key="embeddings")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must have shape (length, dim).")
        if embeddings.shape[0] != record.length:
            raise ValueError(
                f"Embedding length mismatch for {record.entry_id}: "
                f"expected {record.length}, got {embeddings.shape[0]}"
            )
        structure_graph = _load_structure_graph(record.structure_path)
        prostt5_probs = None
        if self.require_prostt5:
            if record.prostt5_path is None:
                raise FileNotFoundError(
                    f"Missing ProstT5 cache for {record.entry_id}"
                )
            prostt5_probs = _load_npz_array(record.prostt5_path, key="embeddings")
            if prostt5_probs.ndim != 2:
                raise ValueError("ProstT5 embeddings must have shape (length, dim).")
            if prostt5_probs.shape[0] != record.length:
                raise ValueError(
                    f"ProstT5 length mismatch for {record.entry_id}: "
                    f"expected {record.length}, got {prostt5_probs.shape[0]}"
                )
        return {
            "entry_id": record.entry_id,
            "seq_embeddings": torch.from_numpy(embeddings),
            "structure_graph": structure_graph,
            "prostt5_probs": torch.from_numpy(prostt5_probs) if prostt5_probs is not None else None,
            "length": record.length,
        }


class ArrayStore:
    """Disk-backed storage for large probability/target matrices."""

    def __init__(self, num_terms: int) -> None:
        self.num_terms = int(num_terms)
        self.samples = 0
        self._prob_file = tempfile.NamedTemporaryFile(delete=False)
        self._target_file = tempfile.NamedTemporaryFile(delete=False)
        self._closed = False

    def append(self, probs: np.ndarray, targets: np.ndarray) -> None:
        if probs.shape != targets.shape:
            raise ValueError("Probabilities and targets must share shape.")
        if probs.shape[1] != self.num_terms:
            raise ValueError("Term dimension mismatch in metrics store.")
        self._prob_file.write(probs.astype(np.float32, copy=False).tobytes())
        self._target_file.write(targets.astype(np.float32, copy=False).tobytes())
        self.samples += int(probs.shape[0])

    def materialize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.samples == 0:
            empty = np.zeros((0, self.num_terms), dtype=np.float32)
            return empty, empty
        self._close_handles()
        shape = (self.samples, self.num_terms)
        probs = np.memmap(self._prob_file.name, dtype=np.float32, mode="r", shape=shape)
        targets = np.memmap(self._target_file.name, dtype=np.float32, mode="r", shape=shape)
        return probs, targets

    def cleanup(self) -> None:
        self._close_handles()
        for file_obj in (self._prob_file, self._target_file):
            try:
                os.unlink(file_obj.name)
            except FileNotFoundError:
                continue

    def _close_handles(self) -> None:
        if self._closed:
            return
        self._prob_file.close()
        self._target_file.close()
        self._closed = True


class GoOntology:
    """GO ontology loader for ancestor propagation."""

    def __init__(self, obo_path: Path) -> None:
        ontology = go_prior_module._parse_obo(obo_path)
        self.parents = {term_id: list(term.parents) for term_id, term in ontology.items()}
        self.aspect = {term_id: term.aspect for term_id, term in ontology.items()}
        self._ancestor_cache: Dict[str, List[str]] = {}

    def ancestors(self, term_id: str) -> List[str]:
        if term_id in self._ancestor_cache:
            return self._ancestor_cache[term_id]
        if term_id not in self.parents:
            self._ancestor_cache[term_id] = []
            return []
        aspect = self.aspect.get(term_id)
        visited: set[str] = set()
        stack = list(self.parents.get(term_id, []))
        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            if self.aspect.get(parent) == aspect:
                visited.add(parent)
            stack.extend(self.parents.get(parent, []))
        ancestors = sorted(visited)
        self._ancestor_cache[term_id] = ancestors
        return ancestors

    def propagate_scores(self, scores: Dict[str, float]) -> None:
        items = list(scores.items())
        for term_id, score in items:
            for parent in self.ancestors(term_id):
                if score > scores.get(parent, 0.0):
                    scores[parent] = score



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PF-AGCN predictor")
    parser.add_argument("--ckpt-bp", required=True, type=Path)
    parser.add_argument("--ckpt-cc", required=True, type=Path)
    parser.add_argument("--ckpt-mf", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--metrics-out", required=True, type=Path)
    parser.add_argument("--terms-tsv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def iter_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    entry_id = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if entry_id is not None:
                    yield entry_id, "".join(chunks)
                header = line[1:].strip()
                tokens = header.split("|")
                entry_id = tokens[1].strip() if len(tokens) >= 3 else header.split()[0]
                chunks = []
            else:
                chunks.append(line)
        if entry_id is not None:
            yield entry_id, "".join(chunks)


def load_fasta_entries(path: Path) -> List[Tuple[str, str]]:
    seen: Dict[str, str] = {}
    entries: List[Tuple[str, str]] = []
    for entry_id, seq in iter_fasta(path):
        if entry_id in seen:
            if seen[entry_id] != seq:
                raise ValueError(f"Conflicting sequences for entry_id={entry_id}.")
            continue
        seen[entry_id] = seq
        entries.append((entry_id, seq))
    if not entries:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return entries


def resolve_cache_root() -> Path:
    value = os.environ.get("PF_AGCN_CACHE")
    cache_root = Path(value or "/orcd/home/002/lerchen/code/cafa_proj/data").expanduser()
    if not cache_root.is_absolute():
        cache_root = (PROJECT_ROOT / cache_root).resolve()
    return cache_root.resolve()


def resolve_structure_dir(cache_root: Path) -> Path:
    override = os.environ.get("PF_AGCN_STRUCTURE_DIR")
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else (cache_root / path).resolve()
    candidates = [
        cache_root / "af_graphs",
        cache_root / "esmfold_cache",
        cache_root / "esm2_contact_cache",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No structure graph cache directory found.")


def resolve_terms_tsv(path: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit
    env_path = os.environ.get("PF_AGCN_TERMS_TSV")
    if env_path:
        return Path(env_path).expanduser()
    stem = path.stem.replace("sequences", "terms")
    candidate = path.with_name(f"{stem}.tsv")
    if candidate.exists():
        return candidate
    fallback = path.parent / "train_terms.tsv"
    if fallback.exists():
        return fallback
    return None


def _load_npz_array(path: Path, *, key: str) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            if key not in archive and "arr_0" in archive:
                key_name = "arr_0"
            else:
                key_name = key
            return np.asarray(archive[key_name])
        finally:
            archive.close()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    raise ValueError(f"Unsupported cache format: {path}")


def _load_structure_graph(path: Path) -> Dict[str, torch.Tensor]:
    archive = np.load(path, allow_pickle=False)
    try:
        edge_index = torch.from_numpy(np.asarray(archive["edge_index"]).astype(np.int64))
        edge_weight = torch.from_numpy(np.asarray(archive["edge_weight"]).astype(np.float32))
        plddt = torch.from_numpy(np.asarray(archive["plddt"]).astype(np.float32))
    finally:
        archive.close()
    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "plddt": plddt,
    }


def _collate_structure_graphs(graphs: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    node_splits = [0]
    for graph in graphs:
        count = int(graph["plddt"].numel())
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


def collate_prediction_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    entry_ids = [str(item["entry_id"]) for item in batch]
    seqs = [item["seq_embeddings"] for item in batch]
    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    lengths = torch.clamp(lengths, min=0, max=padded.size(1))
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    graphs = [item["structure_graph"] for item in batch]
    structure_graph = _collate_structure_graphs(graphs)

    collated: Dict[str, object] = {
        "entry_ids": entry_ids,
        "seq_embeddings": padded,
        "structure_graph": structure_graph,
        "lengths": lengths,
        "mask": mask,
    }

    if any(item.get("prostt5_probs") is not None for item in batch):
        probs_list = [item.get("prostt5_probs") for item in batch]
        if any(probs is None for probs in probs_list):
            raise ValueError("Missing ProstT5 embeddings for at least one entry.")
        max_len = padded.size(1)
        last_dim = probs_list[0].size(1)
        prob_batch = torch.zeros((len(probs_list), max_len, last_dim), dtype=probs_list[0].dtype)
        for idx, probs in enumerate(probs_list):
            prob_batch[idx, : probs.size(0)] = probs
        collated["prostt5_probs"] = prob_batch
    return collated


def move_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()}
        else:
            moved[key] = value
    return moved


def build_records(
    entries: Sequence[Tuple[str, str]],
    cache_root: Path,
    structure_dir: Path,
    require_prostt5: bool,
) -> List[EntryRecord]:
    preprocessing_module.set_cache_root(cache_root)
    records: List[EntryRecord] = []
    missing: List[str] = []
    for entry_id, seq in entries:
        embedding_path = preprocessing_module._embedding_cache_path(entry_id, "esm")
        if not embedding_path.exists():
            missing.append(entry_id)
            continue
        structure_path = preprocessing_module._entry_cache_path(entry_id, structure_dir, mkdir=False)
        if not structure_path.exists():
            missing.append(entry_id)
            continue
        prostt5_path = None
        if require_prostt5:
            prostt5_path = preprocessing_module._embedding_cache_path(entry_id, "prostt5_3di")
            if not prostt5_path.exists():
                missing.append(entry_id)
                continue
        records.append(
            EntryRecord(
                entry_id=entry_id,
                length=len(seq),
                embedding_path=embedding_path,
                structure_path=structure_path,
                prostt5_path=prostt5_path,
            )
        )
    if missing:
        sample = ", ".join(missing[:5])
        raise FileNotFoundError(
            "Missing cached inputs for entries (showing up to 5): "
            f"{sample}."
        )
    return records


def load_checkpoint_state(path: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {path} did not contain a state dict.")
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        cleaned[clean_key] = value
    return cleaned


def infer_model_config(state: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    channels = None
    if "sqb.dccn_norm.weight" in state:
        channels = int(state["sqb.dccn_norm.weight"].shape[0])
    elif "structural_gcn.norms.0.weight" in state:
        channels = int(state["structural_gcn.norms.0.weight"].shape[0])
    if channels is None:
        raise ValueError("Unable to infer channel dimension from checkpoint.")

    if "sqb.input_proj.weight" in state:
        raw_dim = int(state["sqb.input_proj.weight"].shape[1])
    else:
        raw_dim = channels

    if "head.mlp.2.weight" not in state:
        raise ValueError("Missing prediction head weights in checkpoint.")
    num_functions = int(state["head.mlp.2.weight"].shape[0])

    prost_enabled = any(key.startswith("prost_query.") for key in state)
    if prost_enabled and "prost_query.pool_weights" in state:
        heads, head_dim = state["prost_query.pool_weights"].shape
    else:
        heads = 4
        if channels % heads != 0:
            heads = 1
        head_dim = channels // heads

    prost_input_dim = 1024
    if prost_enabled and "prost_query.projections.0.weight" in state:
        prost_input_dim = int(state["prost_query.projections.0.weight"].shape[1])

    kernel_size = 3
    conv_key = "sqb.dccn.convs.0.weight"
    if conv_key in state:
        kernel_size = int(state[conv_key].shape[-1])

    hidden_dim = int(state["head.mlp.0.weight"].shape[0])

    return {
        "task": {"num_functions": num_functions},
        "model": {
            "seq_embeddings": {"raw_dim": raw_dim, "feature_dim": raw_dim},
            "sqb": {
                "channels": channels,
                "dccn": {"kernel_size": kernel_size, "dilation": 2, "dropout": 0.1},
            },
            "cross_attention": {"heads": int(heads), "dropout": 0.1},
            "gcn": {"dropout": 0.1},
            "prost_attention": {"enabled": prost_enabled},
            "prostt5_3di": {"encoder_dim": prost_input_dim},
            "prediction_head": {"mlp_hidden_dim": hidden_dim},
        },
    }


def load_model(path: Path, device: torch.device) -> StructuralPFAGCN:
    state = load_checkpoint_state(path)
    cfg = infer_model_config(state)
    model = StructuralPFAGCN(cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_terms_from_prior(cache_root: Path, aspect: str) -> Optional[Tuple[List[str], np.ndarray]]:
    prior_path = cache_root / "manifests" / "priors" / aspect.lower() / f"{aspect.lower()}_prior.npz"
    if not prior_path.exists():
        return None
    with np.load(prior_path, allow_pickle=False) as archive:
        terms = [str(term) for term in archive["terms"].tolist()]
        adjacency = np.asarray(archive["adjacency"], dtype=np.float32)
    return terms, adjacency


def build_terms_from_tsv(obo_path: Path, terms_path: Path, aspect: str) -> Tuple[List[str], np.ndarray]:
    df = parse_ground_truth_table(terms_path)
    aggregated = df.groupby("entry_id")["term"].agg(list).reset_index()
    aggregated["go_terms"] = aggregated["term"].apply(json.dumps)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        aggregated[["entry_id", "go_terms"]].to_csv(tmp_file.name, index=False)
        priors = Go_Prior(
            obo_path=obo_path,
            train_split_csv=tmp_file.name,
            top_k_mf=None,
            top_k_bp=None,
            top_k_cc=None,
        )
        aspect_prior = priors[aspect]
        return list(aspect_prior.terms), np.asarray(aspect_prior.adjacency, dtype=np.float32)
    finally:
        tmp_file.close()
        try:
            os.unlink(tmp_file.name)
        except FileNotFoundError:
            pass


def build_parent_lookup(adjacency: np.ndarray) -> List[List[int]]:
    if adjacency.size == 0:
        return []
    num_terms = int(adjacency.shape[1])
    parents: List[List[int]] = []
    for child_idx in range(num_terms):
        parent_indices = np.flatnonzero(adjacency[:, child_idx] > 0).astype(int).tolist()
        parents.append(parent_indices)
    return parents


def propagate_scores_matrix(scores: np.ndarray, parent_lookup: Sequence[Sequence[int]]) -> np.ndarray:
    if scores.size == 0 or not parent_lookup:
        return scores
    values = scores.astype(np.float32, copy=True)
    term_active = np.any(values > 0.0, axis=0)
    pending = [idx for idx, active in enumerate(term_active) if active and parent_lookup[idx]]
    in_queue = [False] * len(parent_lookup)
    for idx in pending:
        in_queue[idx] = True
    while pending:
        child_idx = pending.pop()
        in_queue[child_idx] = False
        parents = parent_lookup[child_idx]
        if not parents:
            continue
        child_vals = values[:, child_idx]
        if not np.any(child_vals > 0.0):
            continue
        parent_idx = np.asarray(parents, dtype=np.int64)
        parent_vals = values[:, parent_idx]
        updated = np.maximum(parent_vals, child_vals[:, None])
        if np.any(updated > parent_vals):
            values[:, parent_idx] = updated
            for parent in parents:
                if parent_lookup[parent] and not in_queue[parent]:
                    pending.append(parent)
                    in_queue[parent] = True
    return values


def propagate_targets_matrix(targets: np.ndarray, parent_lookup: Sequence[Sequence[int]]) -> np.ndarray:
    if targets.size == 0 or not parent_lookup:
        return targets
    values = targets.astype(np.float32, copy=True)
    term_active = np.any(values > 0.0, axis=0)
    pending = [idx for idx, active in enumerate(term_active) if active and parent_lookup[idx]]
    in_queue = [False] * len(parent_lookup)
    for idx in pending:
        in_queue[idx] = True
    while pending:
        child_idx = pending.pop()
        in_queue[child_idx] = False
        parents = parent_lookup[child_idx]
        if not parents:
            continue
        child_mask = values[:, child_idx] > 0.0
        if not np.any(child_mask):
            continue
        parent_idx = np.asarray(parents, dtype=np.int64)
        parent_vals = values[:, parent_idx]
        updated = parent_vals.copy()
        updated[child_mask, :] = 1.0
        if np.any(updated > parent_vals):
            values[:, parent_idx] = updated
            for parent in parents:
                if parent_lookup[parent] and not in_queue[parent]:
                    pending.append(parent)
                    in_queue[parent] = True
    return values


def format_score(score: float) -> Optional[str]:
    if not np.isfinite(score):
        return None
    if score <= 0.0:
        return None
    score = min(score, 1.0)
    text = f"{score:.3g}"
    if text in {"0", "0.0", "0.00"}:
        return None
    return text


def build_targets_for_batch(
    entry_ids: Sequence[str],
    label_map: Mapping[str, torch.Tensor],
    num_terms: int,
) -> np.ndarray:
    targets = np.zeros((len(entry_ids), num_terms), dtype=np.float32)
    for idx, entry_id in enumerate(entry_ids):
        labels = label_map.get(entry_id)
        if labels is None:
            continue
        targets[idx] = labels.numpy()
    return targets


def select_top_predictions(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    filtered = [(term, score) for term, score in scores.items() if score >= NOISE_FLOOR]
    filtered.sort(key=lambda item: (-item[1], item[0]))
    if len(filtered) > MAX_TERMS:
        filtered = filtered[:MAX_TERMS]
    return filtered


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    fasta_path = args.fasta.resolve()
    entries = load_fasta_entries(fasta_path)
    entry_ids = [entry_id for entry_id, _seq in entries]

    cache_root = resolve_cache_root()
    structure_dir = resolve_structure_dir(cache_root)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    models = {
        "BP": load_model(args.ckpt_bp, device),
        "CC": load_model(args.ckpt_cc, device),
        "MF": load_model(args.ckpt_mf, device),
    }

    require_prostt5 = any(model.prost_attention_enabled for model in models.values())
    records = build_records(entries, cache_root, structure_dir, require_prostt5)

    batch_size = int(args.batch_size or int(os.environ.get("PF_AGCN_PRED_BATCH", "1")))
    dataloader = DataLoader(
        PredictionDataset(records, require_prostt5=require_prostt5),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_prediction_batch,
    )

    terms_path = resolve_terms_tsv(fasta_path, args.terms_tsv)
    terms_df = None
    if terms_path is not None and terms_path.exists():
        terms_df = parse_ground_truth_table(terms_path)
    elif terms_path is not None:
        log.warning("Terms TSV not found at %s; metrics will default to zeros.", terms_path)

    terms_by_aspect: Dict[str, List[str]] = {}
    adjacency_by_aspect: Dict[str, np.ndarray] = {}
    for aspect in ASPECTS:
        prior = load_terms_from_prior(cache_root, aspect)
        if prior is not None:
            terms, adjacency = prior
        elif terms_df is not None:
            terms, adjacency = build_terms_from_tsv(OBO_PATH, terms_path, aspect)
        else:
            raise FileNotFoundError(
                f"Missing terms for {aspect}. Provide --terms-tsv or precomputed priors."
            )
        terms_by_aspect[aspect] = terms
        adjacency_by_aspect[aspect] = adjacency

    for aspect, model in models.items():
        num_functions = model.num_functions
        if num_functions != len(terms_by_aspect[aspect]):
            raise ValueError(
                f"Checkpoint term mismatch for {aspect}: "
                f"model outputs {num_functions}, terms list has {len(terms_by_aspect[aspect])}."
            )

    parent_lookup_by_aspect = {
        aspect: build_parent_lookup(adjacency_by_aspect[aspect]) for aspect in ASPECTS
    }

    label_maps: Dict[str, Dict[str, torch.Tensor]] = {}
    if terms_df is not None:
        entry_set = set(entry_ids)
        for aspect in ASPECTS:
            code = ASPECT_TO_CODE[aspect]
            subset = terms_df[terms_df["aspect"] == code]
            subset = subset[subset["entry_id"].isin(entry_set)]
            if subset.empty:
                label_maps[aspect] = {}
            else:
                label_maps[aspect] = dataframe_to_multi_hot(subset, terms_by_aspect[aspect])
    else:
        for aspect in ASPECTS:
            label_maps[aspect] = {}

    metric_stores = {
        aspect: ArrayStore(len(terms_by_aspect[aspect])) for aspect in ASPECTS
    }

    ontology = GoOntology(OBO_PATH)

    predictions_path = args.predictions_out
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_out
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Starting prediction for %d sequences.", len(records))
    with predictions_path.open("w", encoding="utf-8") as pred_handle:
        with torch.no_grad():
            for batch in dataloader:
                entry_batch = batch["entry_ids"]
                batch = move_to_device(batch, device)
                seq_embeddings = batch["seq_embeddings"].to(dtype=torch.float32)
                structure_graph = batch["structure_graph"]
                prostt5_probs = batch.get("prostt5_probs")
                lengths = batch["lengths"]
                mask = batch["mask"]

                probs_by_aspect: Dict[str, np.ndarray] = {}
                for aspect, model in models.items():
                    if model.prost_attention_enabled:
                        output = model(
                            seq_embeddings=seq_embeddings,
                            structure_graph=structure_graph,
                            prostt5_probs=prostt5_probs,
                            lengths=lengths,
                            mask=mask,
                        )
                    else:
                        output = model(
                            seq_embeddings=seq_embeddings,
                            structure_graph=structure_graph,
                            lengths=lengths,
                            mask=mask,
                        )
                    logits = output.logits if hasattr(output, "logits") else output
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs_by_aspect[aspect] = probs

                for aspect, probs in probs_by_aspect.items():
                    parent_lookup = parent_lookup_by_aspect[aspect]
                    probs_by_aspect[aspect] = propagate_scores_matrix(probs, parent_lookup)

                for aspect, probs in probs_by_aspect.items():
                    targets = build_targets_for_batch(
                        entry_batch,
                        label_maps.get(aspect, {}),
                        len(terms_by_aspect[aspect]),
                    )
                    targets = propagate_targets_matrix(targets, parent_lookup_by_aspect[aspect])
                    metric_stores[aspect].append(probs, targets)

                for row_idx, entry_id in enumerate(entry_batch):
                    combined_scores: Dict[str, float] = {}
                    for aspect in ASPECTS:
                        probs = probs_by_aspect[aspect][row_idx]
                        terms = terms_by_aspect[aspect]
                        keep = np.flatnonzero(probs >= NOISE_FLOOR)
                        for idx in keep.tolist():
                            term_id = terms[idx]
                            score = float(probs[idx])
                            if score > combined_scores.get(term_id, 0.0):
                                combined_scores[term_id] = min(score, 1.0)

                    ontology.propagate_scores(combined_scores)
                    selected = select_top_predictions(combined_scores)
                    for term_id, score in selected:
                        formatted = format_score(score)
                        if formatted is None:
                            continue
                        pred_handle.write(f"{entry_id}\t{term_id}\t{formatted}\n")

    metrics_rows: Dict[str, Dict[str, float]] = {}
    for aspect in ASPECTS:
        probs, targets = metric_stores[aspect].materialize()
        metrics = compute_cafa_metrics(probs, targets, DEFAULT_THRESHOLDS, ia_weights=None)
        metrics_rows[aspect] = metrics
        metric_stores[aspect].cleanup()

    metric_keys = sorted({key for row in metrics_rows.values() for key in row.keys()})
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("aspect\t" + "\t".join(metric_keys) + "\n")
        for aspect in ASPECTS:
            row = metrics_rows.get(aspect, {})
            values = [str(row.get(key, "")) for key in metric_keys]
            handle.write(aspect + "\t" + "\t".join(values) + "\n")

    log.info("Wrote predictions to %s", predictions_path)
    log.info("Wrote metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
