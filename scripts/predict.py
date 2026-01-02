"""MLflow-backed inference helper for PF-AGCN."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

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

log = logging.getLogger(__name__)


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
    if args.batch_size is not None:
        data_cfg = dict(data_cfg)
        neighbors_cfg = dict(data_cfg.get("neighbors", {}))
        neighbors_cfg["batch_size"] = args.batch_size
        data_cfg["neighbors"] = neighbors_cfg

    dataloader = build_manifest_dataloader(
        manifest=args.manifest,
        data_cfg=data_cfg,
        base_dir=config_path.parent.resolve(),
        shuffle=False,
        protein_prior_cfg=protein_prior_cfg,
    )
    if dataloader is None:
        raise RuntimeError("Inference manifest is required.")

    records = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
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
    output_path = Path(args.output)
    np.savetxt(output_path, predictions, delimiter=",", fmt="%.6f")
    log.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
