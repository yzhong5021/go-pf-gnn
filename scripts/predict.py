"""MLflow-backed inference helper for PF-AGCN."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules.dataloader import build_manifest_dataloader
from utils.system_runtime import apply_system_env, merged_mlflow_settings
from src.model.structural_model import StructuralPFAGCN

log = logging.getLogger(__name__)


def _build_parent_lookup(adjacency: np.ndarray) -> List[List[int]]:
    """Precompute parent indices for each GO term from parent->child adjacency."""
    if adjacency.size == 0:
        return []
    num_terms = int(adjacency.shape[1])
    parents: List[List[int]] = []
    for child_idx in range(num_terms):
        parent_indices = np.flatnonzero(adjacency[:, child_idx] > 0).astype(int).tolist()
        parents.append(parent_indices)
    return parents


def _load_parent_lookup(manifest_path: Path) -> Tuple[List[List[int]], int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
        meta = {}
    elif isinstance(payload, dict):
        records = payload.get("records", [])
        meta = payload.get("meta", {})
    else:
        log.warning("Manifest %s has unsupported format; skipping ancestor propagation.", manifest_path)
        return [], 0
    if not records or not isinstance(records[0], dict):
        log.warning("Manifest %s has no records; skipping ancestor propagation.", manifest_path)
        return [], 0

    prior_rel = records[0].get("go_prior_path") or meta.get("go_prior_path")
    if not prior_rel:
        log.warning("Manifest %s missing go_prior_path; skipping ancestor propagation.", manifest_path)
        return [], 0

    prior_path = Path(prior_rel)
    if not prior_path.is_absolute():
        prior_path = (manifest_path.parent / prior_path).resolve()
    if not prior_path.exists():
        log.warning("GO prior %s not found; skipping ancestor propagation.", prior_path)
        return [], 0

    with np.load(prior_path, allow_pickle=False) as archive:
        if "adjacency" not in archive:
            log.warning("GO prior %s missing adjacency; skipping ancestor propagation.", prior_path)
            return [], 0
        adjacency = np.asarray(archive["adjacency"], dtype=np.float32)

    parent_lookup = _build_parent_lookup(adjacency)
    return parent_lookup, int(adjacency.shape[0])


def _propagate_ancestor_scores(
    scores: np.ndarray,
    parent_lookup: Sequence[Sequence[int]],
) -> np.ndarray:
    """Propagate scores to ancestors using iterative max aggregation."""
    if scores.size == 0 or not parent_lookup:
        return scores
    values = scores.astype(np.float32, copy=True)
    if values.shape[1] != len(parent_lookup):
        raise ValueError("parent lookup length must match number of terms")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PF-AGCN inference")
    parser.add_argument("mlflow_run", type=str, help="Path to MLflow run directory or run ID")
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON/JSONL manifest with cached embeddings for inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Destination CSV file for sigmoid scores",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Optional Hydra config to supply data loader parameters",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size used for inference",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def _move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(val, device) for key, val in value.items()}
    return value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    run_path = Path(args.mlflow_run)
    if run_path.is_dir():
        model_uri = run_path.as_uri()
    else:
        model_uri = f"runs:/{args.mlflow_run}/model"

    log.info("Loading PF-AGCN model from %s", model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_path = Path(args.config)
    cfg = load_config(config_path)
    apply_system_env(cfg)
    mlflow_settings = merged_mlflow_settings(cfg)
    tracking_uri = mlflow_settings.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    data_cfg = OmegaConf.to_container(cfg.get("data_config"), resolve=True) if cfg.get("data_config") else {}
    model_cfg = OmegaConf.to_container(cfg.get("model"), resolve=True) if cfg.get("model") else {}
    protein_prior_cfg = model_cfg.get("prot_prior") if isinstance(model_cfg, dict) else None
    go_prior_cfg = model_cfg.get("go_prior") if isinstance(model_cfg, dict) else None
    if args.batch_size is not None:
        data_cfg = dict(data_cfg)
        data_cfg["batch_size"] = int(args.batch_size)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parent / manifest_path).resolve()

    dataloader = build_manifest_dataloader(
        manifest=manifest_path.as_posix(),
        data_cfg=data_cfg,
        base_dir=config_path.parent.resolve(),
        shuffle=False,
        protein_prior_cfg=protein_prior_cfg,
        go_prior_cfg=go_prior_cfg,
    )
    if dataloader is None:
        raise RuntimeError("Inference manifest is required.")

    records = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_to_device(batch, device)
            if isinstance(model, StructuralPFAGCN):
                output = model(
                    seq_embeddings=batch["seq_embeddings"],
                    structure_graph=batch["structure_graph"],
                    prostt5_probs=batch.get("prostt5_probs"),
                    lengths=batch.get("lengths"),
                    mask=batch.get("mask"),
                )
            else:
                output = model(
                    seq_embeddings=batch["seq_embeddings"],
                    lengths=batch.get("lengths"),
                    mask=batch.get("mask"),
                    protein_prior=batch.get("protein_prior"),
                    go_prior=batch.get("go_prior"),
                )
            logits = output.logits if hasattr(output, "logits") else output
            target_mask = batch.get("target_mask")
            if target_mask is not None:
                logits = logits[target_mask]
            probs = torch.sigmoid(logits).cpu().numpy()
            records.append(probs)

    predictions = np.concatenate(records, axis=0)
    parent_lookup, prior_terms = _load_parent_lookup(manifest_path)
    if parent_lookup and predictions.shape[1] == prior_terms:
        predictions = _propagate_ancestor_scores(predictions, parent_lookup)
    elif parent_lookup:
        log.warning(
            "Skipping ancestor propagation for %s due to term mismatch (prior=%d, preds=%d).",
            manifest_path,
            prior_terms,
            predictions.shape[1],
        )
    output_path = Path(args.output)
    np.savetxt(output_path, predictions, delimiter=",", fmt="%.6f")
    log.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
