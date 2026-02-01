"""Training entry point for PF-AGCN using PyTorch Lightning.

Hydra-configured trainer with MLflow tracking and CAFA metric logging.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import hydra
from hydra.utils import get_original_cwd
import mlflow
import mlflow.pytorch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, StepLR

from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from cafaeval.evaluation import compute_metrics as cafa_compute_metrics
from cafaeval.evaluation import normalize as cafa_normalize
from cafaeval.graph import GroundTruth as CafaGroundTruth
from cafaeval.graph import Prediction as CafaPrediction

from src.model.model import PFAGCN
from src.model.structural_model import StructuralPFAGCN
from src.model.prost_only_model import ProstOnlyPFAGCN
from src.model.gated_pe_model import GatedPEPFAGCN
from src.modules.dataloader import build_manifest_dataloader, load_ia_weights
from src.modules.loss import AsymmetricLoss, BCEWithLogits, FocalLoss
from src.utils.system_runtime import apply_system_env, merged_mlflow_settings
from src.utils.manifest_paths import resolve_manifest_path_template

log = logging.getLogger(__name__)


###### Config utils ######


def flatten_config(config: Mapping[str, Any], parent: str = "") -> Dict[str, Any]:
    """Flatten nested mappings to MLflow-friendly key/value pairs."""

    items: Dict[str, Any] = {}
    for key, value in config.items():
        composite = f"{parent}.{key}" if parent else str(key)
        composite = composite.replace("@", "_")
        composite = composite.replace("/", ".")
        if isinstance(value, Mapping):
            items.update(flatten_config(value, composite))
        elif isinstance(value, (list, tuple)):
            items[composite] = json.dumps(value)
        else:
            items[composite] = value
    return items


def to_namespace(data: Any) -> Any:
    """Recursively convert dictionaries to SimpleNamespace objects."""

    if isinstance(data, Mapping):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in data.items()})
    if isinstance(data, list):
        return [to_namespace(v) for v in data]
    return data


def build_model_config(cfg: DictConfig) -> Any:
    """Create a model config object compatible with PFAGCN."""

    container = {
        "task": OmegaConf.to_container(cfg.task, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
    }
    try:
        from model import config as model_config

        task_cls = getattr(model_config, "TaskConfig")
        model_cls = getattr(model_config, "PFAGCNModelConfig")
        config_cls = getattr(model_config, "PFAGCNConfig")

        task_cfg = task_cls(**container["task"])
        model_cfg = model_cls(**container["model"])
        return config_cls(task=task_cfg, model=model_cfg)
    except (ImportError, AttributeError, TypeError):
        log.debug("Falling back to SimpleNamespace-based model config")
        return to_namespace(container)


def build_model(cfg: DictConfig) -> nn.Module:
    """Instantiate the configured PF-AGCN model."""

    model_cfg = cfg.get("model", {})
    arch = str(getattr(model_cfg, "arch", "pfagcn") or "pfagcn").lower()
    if arch in {"prost_only", "prost"}:
        return ProstOnlyPFAGCN(cfg)
    if arch in {"gated_pe"}:
        return GatedPEPFAGCN(cfg)
    if arch in {"structural", "structural_prost", "structural_pfagcn"}:
        return StructuralPFAGCN(cfg)
    model_config = build_model_config(cfg)
    return PFAGCN(model_config)


def build_loss(cfg: DictConfig) -> nn.Module:
    """Instantiate the configured criterion."""

    loss_cfg = cfg.model.loss
    name = str(loss_cfg.name).lower()
    if name == "bce_with_logits":
        pos_weight = loss_cfg.pos_weight
        tensor_weight = None
        if pos_weight is not None:
            tensor_weight = torch.tensor(pos_weight, dtype=torch.float32)
        return BCEWithLogits(pos_weight=tensor_weight)
    if name in {"focal", "focal_loss"}:
        focusing = float(getattr(loss_cfg, "focusing", 2.0))
        balancing = float(getattr(loss_cfg, "balancing", 0.25))
        return FocalLoss(focusing=focusing, balancing=balancing)
    if name in {"asymmetric", "asymmetric_loss"}:
        gamma_positive = float(getattr(loss_cfg, "gamma_positive", 0.0))
        gamma_negative = float(getattr(loss_cfg, "gamma_negative", 4.0))
        clip = float(getattr(loss_cfg, "clip", 0.025))
        return AsymmetricLoss(
            gamma_positive=gamma_positive,
            gamma_negative=gamma_negative,
            clip=clip,
        )
    raise ValueError(f"Unsupported loss: {loss_cfg.name}")


def build_optimizer(cfg: DictConfig, parameters: Iterable[nn.Parameter]) -> Optimizer:
    """Create the optimiser defined in the config."""

    optim_cfg = cfg.optimizer
    name = str(optim_cfg.name).lower()
    if name == "adamw":
        betas = tuple(optim_cfg.betas) if optim_cfg.get("betas") else (0.9, 0.999)
        return torch.optim.AdamW(
            parameters,
            lr=float(optim_cfg.lr),
            betas=betas,
            weight_decay=float(optim_cfg.weight_decay),
        )
    if name == "adam":
        betas = tuple(optim_cfg.betas) if optim_cfg.get("betas") else (0.9, 0.999)
        return torch.optim.Adam(
            parameters,
            lr=float(optim_cfg.lr),
            betas=betas,
            weight_decay=float(optim_cfg.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {optim_cfg.name}")


def build_scheduler(cfg: DictConfig, optimizer: Optimizer) -> Optional[LambdaLR]:
    """Configure learning-rate scheduling with optional warmup."""

    sched_cfg = cfg.get("scheduler") if hasattr(cfg, "get") else getattr(cfg, "scheduler", {})
    sched_cfg = sched_cfg or {}
    getter = getattr(sched_cfg, "get", None)

    def sched_value(key: str, default: Any) -> Any:
        if callable(getter):
            try:
                return getter(key, default)
            except Exception:
                return default
        return getattr(sched_cfg, key, default)

    name_val = sched_value("name", "none")
    name = str(name_val or "none").lower()
    if name in {"none", "null"}:
        return None
    total_epochs = int(cfg.training.max_epochs)

    if name == "step":
        gamma = float(sched_value("gamma", 0.95))
        step_size_default = max(1, total_epochs // 3)
        step_size = int(sched_value("step_size", step_size_default))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name_val}")

    warmup_epochs = int(sched_value("warmup_epochs", 0))
    decay_epochs = int(sched_value("decay_epochs", total_epochs))
    decay_epochs = max(warmup_epochs + 1, min(decay_epochs, total_epochs))
    min_lr = float(sched_value("min_lr", 0.0))
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def make_lambda(base_lr: float):
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(current_epoch: int) -> float:
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                return (current_epoch + 1) / float(max(1, warmup_epochs))
            progress = (current_epoch - warmup_epochs) / float(
                max(1, decay_epochs - warmup_epochs)
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda

    lambdas = [make_lambda(lr) for lr in base_lrs]
    return LambdaLR(optimizer, lr_lambda=lambdas)


######### CAFA metrics ###########


def _sanitize_metric_value(name: str, value: float) -> float:
    """Ensure metric values are finite floats before logging."""

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        log.warning("Metric %s is non-numeric; replacing with 0.0.", name)
        return 0.0
    if not np.isfinite(value_float):
        log.warning("Metric %s is non-finite; replacing with 0.0.", name)
        return 0.0
    return value_float


def compute_cafa_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: Sequence[float],
    ia_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute CAFA metrics via cafaeval utilities."""

    def _nan_safe_array(name: str, arr: np.ndarray) -> np.ndarray:
        if np.isfinite(arr).all():
            return arr
        log.warning(  # FOR NAN DEBUGGING ONLY
            "%s contains non-finite values; replacing with zeros.", name
        )
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if probabilities.size == 0 or targets.size == 0:
        return {
            "fmax": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "pr_auc": 0.0,
            "ap": 0.0,
            "ia_fmax": 0.0,
            "ia_precision": 0.0,
            "ia_recall": 0.0,
            "ia_threshold": thresholds[0] if thresholds else 0.5,
        }

    probabilities = _nan_safe_array("probabilities", probabilities.astype(np.float32, copy=False))
    targets = _nan_safe_array("targets", targets.astype(np.float32, copy=False))

    tau_arr = np.asarray(list(thresholds) if thresholds else [0.5], dtype=np.float32)
    ids = {str(idx): idx for idx in range(probabilities.shape[0])}
    prediction = CafaPrediction(ids=ids, matrix=probabilities.astype(np.float32))
    ground_truth = CafaGroundTruth(ids=ids, matrix=targets.astype(bool))
    toi = np.arange(probabilities.shape[1])
    ne = np.full(tau_arr.shape[0], ground_truth.matrix.shape[0])
    cpu_workers = max(1, int(os.cpu_count() or 1))

    metrics_df = cafa_normalize(
        cafa_compute_metrics(
            prediction,
            ground_truth,
            tau_arr,
            toi,
            ic_arr=None,
            n_cpu=cpu_workers,
        ),
        "mock",
        tau_arr,
        ne,
        normalization="cafa",
    )
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    metrics_df = metrics_df.dropna(subset=["f"], how="all")

    if metrics_df.empty:
        best_precision = best_recall = best_fmax = 0.0
        best_tau = float(tau_arr[0])
    else:
        best_idx = metrics_df["f"].astype(float).idxmax()
        best_row = metrics_df.loc[best_idx]
        best_precision = float(best_row.get("pr", 0.0))
        best_recall = float(best_row.get("rc", 0.0))
        best_fmax = float(best_row.get("f", 0.0))
        best_tau = float(best_row.get("tau", tau_arr[0]))

    ia_fmax = best_fmax
    ia_precision = best_precision
    ia_recall = best_recall
    ia_threshold = best_tau

    if ia_weights is not None:
        ia_df = cafa_normalize(
            cafa_compute_metrics(
                prediction,
                ground_truth,
                tau_arr,
                toi,
                ic_arr=ia_weights,
                n_cpu=cpu_workers,
            ),
            "mock",
            tau_arr,
            ne,
            normalization="cafa",
        )
        ia_df = ia_df.replace([np.inf, -np.inf], np.nan)
        ia_df = ia_df.dropna(subset=["f"], how="all")
        if not ia_df.empty:
            ia_best_idx = ia_df["f"].astype(float).idxmax()
            ia_row = ia_df.loc[ia_best_idx]
            ia_fmax = float(ia_row.get("f", ia_fmax))
            ia_precision = float(ia_row.get("pr", ia_precision))
            ia_recall = float(ia_row.get("rc", ia_recall))
            ia_threshold = float(ia_row.get("tau", ia_threshold))

    per_term_ap: List[float] = []
    for term_idx in range(targets.shape[1]):
        term_true = targets[:, term_idx]
        if float(term_true.sum()) < 1.0:
            continue
        term_scores = probabilities[:, term_idx]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            per_term_ap.append(float(average_precision_score(term_true, term_scores)))
    macro_ap = float(np.mean(per_term_ap)) if per_term_ap else 0.0

    roc_auc: Optional[float]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            roc_auc = float(roc_auc_score(targets, probabilities, average="macro"))
    except ValueError:
        roc_auc = None

    metrics = {
        "fmax": best_fmax,
        "precision": best_precision,
        "recall": best_recall,
        "pr_auc": macro_ap,
        "ap": macro_ap,
        "ia_fmax": ia_fmax,
        "ia_precision": ia_precision,
        "ia_recall": ia_recall,
        "ia_threshold": ia_threshold,
    }
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    for key, value in list(metrics.items()):
        metrics[key] = _sanitize_metric_value(key, value)
    return metrics


class _ValidationArrayStore:
    """Persist validation tensors to disk-backed buffers to cap RAM usage."""

    def __init__(self, num_terms: int) -> None:
        self.num_terms = int(num_terms)
        self.samples = 0
        self._prob_file = tempfile.NamedTemporaryFile(delete=False)
        self._target_file = tempfile.NamedTemporaryFile(delete=False)
        self._closed = False

    def append(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        probs_cpu = probs.detach().to(device="cpu", dtype=torch.float32)
        targets_cpu = targets.detach().to(device="cpu", dtype=torch.float32)
        if probs_cpu.shape[1] != self.num_terms or targets_cpu.shape[1] != self.num_terms:
            raise ValueError("Validation batches produced inconsistent dimensions")
        self._prob_file.write(probs_cpu.numpy().tobytes())
        self._target_file.write(targets_cpu.numpy().tobytes())
        self.samples += int(probs_cpu.shape[0])

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


####### Lightning Module #######

class PFAGCNLightningModule(LightningModule):
    """Lightning wrapper around PF-AGCN with CAFA evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        thresholds: Sequence[float],
        ia_weights: Optional[np.ndarray],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.thresholds = list(thresholds)
        self.ia_weights = ia_weights
        self.model = build_model(cfg)
        self.criterion = build_loss(cfg)
        self.best_ia_fmax = -float("inf")
        self.final_ia_fmax = float("nan")
        self._metric_modes = {  # ablation study
            "train/loss": "min",
            "val/loss": "min",
            "cafa/pr_auc": "max",
            "cafa/ap": "max",
            "cafa/precision": "max",
            "cafa/recall": "max",
            "cafa/ia_precision": "max",
            "cafa/ia_recall": "max",
            "cafa/ia_threshold": "max",
            "cafa/roc_auc": "max",
        }
        self.best_metrics = {  # ablation study
            name: (float("inf") if mode == "min" else -float("inf"))
            for name, mode in self._metric_modes.items()
        }
        self.final_metrics: Dict[str, float] = {}  # ablation study
        self._val_store: Optional[_ValidationArrayStore] = None
        self._val_losses: List[float] = []
        self._last_train_loss: Optional[float] = None

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(self.model, StructuralPFAGCN):
            outputs = self.model(
                seq_embeddings=batch["seq_embeddings"],
                structure_graph=batch["structure_graph"],
                prostt5_probs=batch.get("prostt5_probs"),
                lengths=batch.get("lengths"),
                mask=batch.get("mask"),
            )
        elif isinstance(self.model, GatedPEPFAGCN):
            outputs = self.model(
                seq_embeddings=batch["seq_embeddings"],
                structure_graph=batch["structure_graph"],
                prostt5_probs=batch["prostt5_probs"],
                lengths=batch.get("lengths"),
                mask=batch.get("mask"),
            )
        elif isinstance(self.model, ProstOnlyPFAGCN):
            outputs = self.model(
                prostt5_probs=batch["prostt5_probs"],
                structure_graph=batch.get("structure_graph"),
                lengths=batch.get("lengths"),
                mask=batch.get("mask"),
            )
        else:
            outputs = self.model(
                seq_embeddings=batch["seq_embeddings"],
                lengths=batch.get("lengths"),
                mask=batch.get("mask"),
                protein_prior=batch.get("protein_prior"),
                go_prior=batch.get("go_prior"),
            )
        return outputs.logits

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:  # noqa: ARG002
        logits = self.forward(batch)
        target_mask = batch.get("target_mask")
        targets = batch["targets"]
        if target_mask is not None:
            targets = targets[target_mask]
            logits = logits[target_mask]
        logits = logits.float()
        targets = targets.float()
        logits, targets = self._sanitize_logits_and_targets(logits, targets, stage="train")
        loss = self.criterion(logits, targets)
        if not torch.isfinite(loss):
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Non-finite loss detected during train; replacing with zeros."
            )
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        batch_size = targets.size(0)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=self._sync_dist,
        )
        return loss

    def on_train_epoch_end(self) -> None:  # ablation study
        train_loss = None
        if self.trainer and self.trainer.callback_metrics:
            train_loss = self.trainer.callback_metrics.get("train/loss")
        if train_loss is None:
            return
        if torch.is_tensor(train_loss):
            value = float(train_loss.detach().cpu().item())
        else:
            value = float(train_loss)
        self._update_metric_summary("train/loss", value)  # ablation study
        self._last_train_loss = value
        if self.trainer and getattr(self.trainer, "sanity_checking", False):
            return
        if self.trainer and not self.trainer.is_global_zero:
            return
        num_val = getattr(self.trainer, "num_val_batches", None)
        has_val = False
        if isinstance(num_val, (list, tuple)):
            has_val = sum(int(v) for v in num_val) > 0
        elif num_val is not None:
            has_val = int(num_val) > 0
        if not has_val:
            progress_metrics: Dict[str, Any] = {}
            if self.trainer is not None:
                progress_metrics.update(self.trainer.progress_bar_metrics or {})
            progress_metrics.setdefault("train/loss", value)
            message = self._format_epoch_metrics(progress_metrics)
            if message:
                log.info("Epoch %d %s", int(self.current_epoch), message)

    def on_train_epoch_start(self) -> None:
        self._last_train_loss = None

    def on_validation_epoch_start(self) -> None:
        if self._val_store is not None:
            self._val_store.cleanup()
        self._val_store = _ValidationArrayStore(self.model.num_functions)
        self._val_losses = []

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> None:  # noqa: ARG002
        logits = self.forward(batch)
        targets = batch["targets"]
        target_mask = batch.get("target_mask")
        if target_mask is not None:
            targets = targets[target_mask]
            logits = logits[target_mask]
        logits = logits.float()
        targets = targets.float()
        logits, targets = self._sanitize_logits_and_targets(logits, targets, stage="val")
        loss = self.criterion(logits, targets)
        if not torch.isfinite(loss):
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Non-finite loss detected during val; replacing with zeros."
            )
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        probabilities = torch.sigmoid(logits)
        if not torch.isfinite(probabilities).all():
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Non-finite probabilities detected during validation; sanitizing outputs."
            )
            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        if self._val_store is None:
            self._val_store = _ValidationArrayStore(self.model.num_functions)
        self._val_store.append(probabilities, targets)
        self._val_losses.append(float(loss.detach().cpu().item()))

    def on_validation_epoch_end(self) -> None:
        if self._val_store is None:
            return
        probs_np, targets_np = self._val_store.materialize()
        if not (np.isfinite(probs_np).all() and np.isfinite(targets_np).all()):
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Validation store contains non-finite values; sanitizing before metrics."
            )
            probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=0.0, neginf=0.0)
            targets_np = np.nan_to_num(targets_np, nan=0.0, posinf=0.0, neginf=0.0)
        metrics = compute_cafa_metrics(
            probabilities=probs_np,
            targets=targets_np,
            thresholds=self.thresholds,
            ia_weights=self.ia_weights,
        )
        mean_loss = float(np.mean(self._val_losses)) if self._val_losses else 0.0
        self._val_store.cleanup()
        self._val_store = None

        self.log(
            "val/loss",
            mean_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log(
            "cafa/ia_fmax",
            metrics["ia_fmax"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        self.log(
            "cafa/threshold",
            metrics["ia_threshold"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/precision",
            metrics["ia_precision"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/recall",
            metrics["ia_recall"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/pr_auc",
            metrics["pr_auc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "cafa/ap",
            metrics["ap"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        if "roc_auc" in metrics:
            self.log(
                "cafa/roc_auc",
                metrics["roc_auc"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.final_ia_fmax = float(metrics["ia_fmax"])
        self.best_ia_fmax = max(self.best_ia_fmax, metrics["ia_fmax"])
        self._update_metric_summary("val/loss", mean_loss)  # ablation study
        for key in (
            "pr_auc",
            "ap",
            "precision",
            "recall",
            "ia_precision",
            "ia_recall",
            "ia_threshold",
            "roc_auc",
        ):
            if key in metrics:
                self._update_metric_summary(f"cafa/{key}", float(metrics[key]))  # ablation study
        if self.trainer and getattr(self.trainer, "sanity_checking", False):
            return
        if self.trainer and not self.trainer.is_global_zero:
            return
        train_loss = self._last_train_loss
        progress_metrics: Dict[str, Any] = {}
        if self.trainer is not None:
            progress_metrics.update(self.trainer.progress_bar_metrics or {})
        if train_loss is not None:
            progress_metrics.setdefault("train/loss", train_loss)
        progress_metrics.setdefault("val/loss", mean_loss)
        progress_metrics.setdefault("cafa/ia_fmax", metrics.get("ia_fmax", 0.0))
        progress_metrics.setdefault("cafa/pr_auc", metrics.get("pr_auc", 0.0))
        message = self._format_epoch_metrics(progress_metrics)
        if message:
            log.info("Epoch %d %s", int(self.current_epoch), message)

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(self.cfg, self.parameters())
        scheduler = build_scheduler(self.cfg, optimizer)
        if scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @property
    def _sync_dist(self) -> bool:
        return bool(self.trainer and self.trainer.num_devices > 1)

    def _sanitize_logits_and_targets(
        self, logits: torch.Tensor, targets: torch.Tensor, *, stage: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clamp non-finite logits/targets to safe values to avoid metric explosions."""

        if not torch.isfinite(logits).all():
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Non-finite logits detected during %s; replacing with zeros.", stage
            )
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(targets).all():
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Non-finite targets detected during %s; replacing with zeros.", stage
            )
            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        return logits, targets

    def _has_non_finite_gradients(self) -> bool:
        for param in self.parameters():
            grad = param.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                return True
        return False

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        *args: Any,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
        optimizer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            optimizer_idx = kwargs.get("optimizer_idx", optimizer_idx)
            gradient_clip_val = kwargs.get("gradient_clip_val", gradient_clip_val)
            gradient_clip_algorithm = kwargs.get(
                "gradient_clip_algorithm", gradient_clip_algorithm
            )
        if args:
            # Support both Lightning signatures: (optimizer, clip_val, clip_alg) and
            # (optimizer, optimizer_idx, clip_val, clip_alg).
            if len(args) == 1:
                if gradient_clip_val is None and not isinstance(args[0], str):
                    gradient_clip_val = args[0]
                else:
                    optimizer_idx = args[0]
            elif len(args) == 2:
                if (
                    isinstance(args[0], int)
                    and gradient_clip_val is None
                    and gradient_clip_algorithm is None
                    and not isinstance(args[1], str)
                ):
                    optimizer_idx = args[0]
                    gradient_clip_val = args[1]
                else:
                    gradient_clip_val, gradient_clip_algorithm = args
            elif len(args) == 3:
                optimizer_idx, gradient_clip_val, gradient_clip_algorithm = args
            else:
                raise TypeError(
                    "configure_gradient_clipping received unexpected positional args"
                )
        del optimizer_idx, gradient_clip_algorithm
        clip_val = float(gradient_clip_val or 0.0)
        if clip_val <= 0.0:
            return
        if self._has_non_finite_gradients():
            log.warning(  # FOR NAN DEBUGGING ONLY
                "Non-finite gradients detected before clipping."
            )
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip_val)

    def _coerce_metric_value(self, value: Any) -> Optional[float]:
        """Convert a metric value to a finite float when possible.

        Args:
            value: Metric value emitted by Lightning.

        Returns:
            Float value if it can be interpreted as finite; otherwise None.
        """

        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        elif isinstance(value, (np.floating, np.integer)):
            value = float(value)
        elif not isinstance(value, (float, int)):
            return None
        if not np.isfinite(value):
            return None
        return float(value)

    def _format_epoch_metrics(self, metrics: Mapping[str, Any]) -> str:
        """Build a single-line epoch summary for console logs.

        Args:
            metrics: Mapping of metric names to values.

        Returns:
            Formatted metric summary string.
        """

        if not metrics:
            return ""
        priority_keys = ("train/loss", "val/loss", "cafa/ia_fmax", "cafa/pr_auc")
        parts: List[str] = []
        used = set()
        for key in priority_keys:
            metric_value = self._coerce_metric_value(metrics.get(key))
            if metric_value is None:
                continue
            parts.append(f"{key}={metric_value:.4f}")
            used.add(key)
        for key, value in metrics.items():
            if key in used:
                continue
            metric_value = self._coerce_metric_value(value)
            if metric_value is None:
                continue
            parts.append(f"{key}={metric_value:.4f}")
        return " ".join(parts)

    def _update_metric_summary(self, name: str, value: float) -> None:  # ablation study
        if not np.isfinite(value):
            return
        self.final_metrics[name] = float(value)
        mode = self._metric_modes.get(name)
        if mode == "min":
            current = self.best_metrics.get(name, float("inf"))
            if value < current:
                self.best_metrics[name] = float(value)
        elif mode == "max":
            current = self.best_metrics.get(name, -float("inf"))
            if value > current:
                self.best_metrics[name] = float(value)

    def summary_metrics(self) -> Dict[str, float]:  # ablation study
        summary: Dict[str, float] = {}
        for name, value in self.best_metrics.items():
            if np.isfinite(value):
                summary[f"{name}_best"] = float(value)
        for name, value in self.final_metrics.items():
            if np.isfinite(value):
                summary[f"{name}_final"] = float(value)
        return summary


# ---------------------------------------------------------------------------
# MLflow integrations
# ---------------------------------------------------------------------------


def _resolve_cfg_value(cfg: Any, key: str) -> Any:
    """Safely pull a value from either a DictConfig or a namespace-like object."""

    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except AttributeError:
            pass
    return getattr(cfg, key, None)


_ASPECT_SUFFIXES = {
    "MF": "_mf",
    "MOLECULARFUNCTION": "_mf",
    "F": "_mf",
    "BP": "_bp",
    "BIOLOGICALPROCESS": "_bp",
    "P": "_bp",
    "CC": "_cc",
    "CELLULARCOMPONENT": "_cc",
    "C": "_cc",
}


def _aspect_suffix(cfg: Any) -> str:
    """Return the GO-aspect suffix (_mf/_bp/_cc) based on cfg.aspect."""

    aspect_value = _resolve_cfg_value(cfg, "aspect")
    if not aspect_value:
        return ""
    cleaned = re.sub(r"[^A-Za-z]", "", str(aspect_value)).upper()
    return _ASPECT_SUFFIXES.get(cleaned, "")


def _resolve_mlflow_names(cfg: Any, mlflow_cfg: Mapping[str, Any]) -> Tuple[str, str]:
    """Derive experiment and run names with aspect-aware suffixing."""

    experiment_name = (
        mlflow_cfg.get("experiment_name")
        or _resolve_cfg_value(cfg, "experiment_name")
        or "pfagcn"
    )
    experiment_name = str(experiment_name)
    suffix = _aspect_suffix(cfg)
    run_name_override = mlflow_cfg.get("run_name_override")
    if run_name_override:
        run_name = str(run_name_override)
    else:
        run_name = f"{experiment_name}{suffix}"
    return experiment_name, run_name


def _with_active_mlflow_run(logger: MLFlowLogger, action: Callable[[], None]) -> None:
    """Execute an MLflow action with the logger's run active."""

    if not isinstance(logger, MLFlowLogger):
        return
    run_id = getattr(logger, "run_id", None)
    tracking_uri = getattr(logger, "_tracking_uri", None)
    if not run_id or not tracking_uri:
        return

    mlflow.set_tracking_uri(tracking_uri)
    active_run = mlflow.active_run()
    started_run = False
    if active_run is None or active_run.info.run_id != run_id:
        mlflow.start_run(run_id=run_id)
        started_run = True
    try:
        action()
    finally:
        if started_run:
            mlflow.end_run()


class MLflowModelSaver(Callback):
    """Callback to log the trained model to MLflow once per run."""

    def __init__(self) -> None:
        super().__init__()
        self.logged = False

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.logged:
            return
        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            return
        _with_active_mlflow_run(
            logger,
            lambda: mlflow.pytorch.log_model(
                pl_module.model,
                name="model",
                signature=False,  # Skip signature inference; model needs multiple tensor inputs.
            ),
        )
        self.logged = True


def _prepare_mlflow_logger(cfg: DictConfig, base_dir: Path) -> MLFlowLogger:
    mlflow_cfg = merged_mlflow_settings(cfg)
    tracking_uri = mlflow_cfg.get("tracking_uri")
    artifact_location = mlflow_cfg.get("artifact_root")
    if not tracking_uri:
        tracking_dir = (base_dir / "mlruns").resolve()
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"file:{tracking_dir.as_posix()}"
        artifact_location = artifact_location or tracking_uri
    experiment_name, run_name = _resolve_mlflow_names(cfg, mlflow_cfg)
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        artifact_location=artifact_location,
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(flatten_config(resolved_cfg))
    logger.experiment.log_dict(logger.run_id, resolved_cfg, artifact_file="hydra_config.json")

    log.info(                                                                                                                                                                                                          
          "MLflow run initialised: experiment=%s run=%s (tracking_uri=%s)",                                                                                                                                              
          experiment_name,                                                                                                                                                                                               
          logger.run_id,                                                                                                                                                                                                 
          tracking_uri,                                                                                                                                                                                                  
      )

    return logger


def _log_terminal_cafa_metrics(
    logger: MLFlowLogger, *, best_ia_fmax: float, final_ia_fmax: float
) -> None:
    """Write summary IA F-max values to MLflow at the end of training."""

    metrics = {
        "cafa/ia_fmax_best": _sanitize_metric_value("cafa/ia_fmax_best", best_ia_fmax),
        "cafa/ia_fmax_final": _sanitize_metric_value(
            "cafa/ia_fmax_final", final_ia_fmax
        ),
    }
    _with_active_mlflow_run(logger, lambda: mlflow.log_metrics(metrics))


def _log_terminal_summary_metrics(  # ablation study
    logger: MLFlowLogger, metrics: Mapping[str, float]
) -> None:
    """Write summary best/final metrics to MLflow at the end of training."""

    if not metrics:
        return
    sanitized = {
        key: _sanitize_metric_value(key, value) for key, value in metrics.items()
    }
    _with_active_mlflow_run(logger, lambda: mlflow.log_metrics(sanitized))


def _precision_arg(cfg: DictConfig) -> Any:
    precision_cfg = cfg.training.get("precision", 32)
    if isinstance(precision_cfg, str):
        precision_clean = precision_cfg.strip()
        try:
            precision_cfg = int(precision_clean)
        except (TypeError, ValueError):
            return precision_clean
    else:
        try:
            precision_cfg = int(precision_cfg)
        except (TypeError, ValueError):
            return precision_cfg
    if precision_cfg == 16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        if torch.cuda.is_available():
            return "16-mixed"
        return 32
    return precision_cfg


def _configure_tensor_core_precision() -> None:
    """Enable Tensor Core matmul optimizations when available."""

    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    if set_precision is None:
        return
    if torch.cuda.is_available():
        set_precision("medium")


def _disable_mpi_if_unavailable() -> None:
    """Disable Lightning MPI environment if mpi4py fails to import."""

    try:
        from mpi4py import MPI  # noqa: F401
    except Exception:
        try:
            from lightning.fabric.plugins.environments import mpi as mpi_env

            mpi_env._MPI4PY_AVAILABLE = False
        except Exception:
            return


def _resolve_output_dir(cfg: DictConfig) -> Path:
    output_dir = OmegaConf.select(cfg, "hydra.runtime.output_dir", default=None)
    if output_dir:
        return Path(str(output_dir)).expanduser()
    return Path(get_original_cwd())


def _configure_logging(cfg: DictConfig) -> Path:
    log_dir = _resolve_output_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / "train.log").resolve()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    log_to_stdout = bool(os.environ.get("SLURM_JOB_ID")) or os.environ.get(
        "PF_AGCN_LOG_STDOUT", ""
    ).lower() in {"1", "true", "yes"}

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO if log_to_stdout else logging.WARNING)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    log.setLevel(logging.INFO)
    return log_path


###### Hydra main ######


def run_training(cfg: DictConfig, extra_callbacks: Optional[List[Callback]] = None) -> float:
    """Execute PF-AGCN training with a pre-resolved Hydra config.

    Args:
        cfg: Fully composed Hydra config.
        extra_callbacks: Additional Lightning callbacks (e.g., Optuna pruning).

    Returns:
        Best IA F-max achieved during validation (float) for Optuna sweeps.
    """

    _configure_logging(cfg)
    apply_system_env(cfg)
    _configure_tensor_core_precision()
    _disable_mpi_if_unavailable()
    base_dir = Path(get_original_cwd())
    seed_everything(int(cfg.training.get("seed", 42)), workers=True)
    prot_prior_cfg = OmegaConf.to_container(
        getattr(cfg.model, "prot_prior", {}), resolve=True
    )
    go_prior_cfg = OmegaConf.to_container(
        getattr(cfg.model, "go_prior", {}), resolve=True
    )

    min_length_cfg = cfg.data_config.get("min_length", 10)
    min_length = int(min_length_cfg) if min_length_cfg is not None else None

    aspect = str(cfg.get("aspect", "") or "")
    train_manifest = resolve_manifest_path_template(
        cfg.data_config.get("train_manifest"),
        aspect=aspect,
    )
    val_manifest = resolve_manifest_path_template(
        cfg.data_config.get("val_manifest"),
        aspect=aspect,
    )
    test_manifest = resolve_manifest_path_template(
        cfg.data_config.get("test_manifest"),
        aspect=aspect,
    )

    train_loader = build_manifest_dataloader(
        train_manifest,
        cfg.data_config,
        base_dir,
        shuffle=True,
        protein_prior_cfg=prot_prior_cfg,
        go_prior_cfg=go_prior_cfg,
        min_length=min_length,
        split="train",
    )
    if train_loader is None:
        raise RuntimeError("Training manifest is required to start training.")
    val_loader = build_manifest_dataloader(
        val_manifest,
        cfg.data_config,
        base_dir,
        shuffle=False,
        protein_prior_cfg=prot_prior_cfg,
        go_prior_cfg=go_prior_cfg,
        min_length=None,
        split="val",
    )

    thresholds = list(cfg.evaluation.get("threshold_grid", [0.5]))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ia_weights = load_ia_weights(cfg_dict, base_dir)

    model = PFAGCNLightningModule(
        cfg=cfg,
        thresholds=thresholds,
        ia_weights=ia_weights,
    )

    train_dataset = getattr(train_loader, "dataset", None)
    if train_dataset is not None and hasattr(train_dataset, "short_drop_count"):
        dropped = int(getattr(train_dataset, "short_drop_count", 0))
        min_len_log = min_length if min_length is not None else 0
        log.info(
            "Filtered %d training samples shorter than %d amino acids (min_length).",
            dropped,
            min_len_log,
        )
    
    mlflow_logger = _prepare_mlflow_logger(cfg, base_dir)

    model_saver = MLflowModelSaver()
    callbacks = [
        ModelCheckpoint(
            monitor="cafa/ia_fmax",
            mode="max",
            filename="epoch{epoch:03d}-iafmax{cafa/ia_fmax:.4f}",
            save_top_k=1,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        model_saver,
    ]
    early_stop_metric = cfg.training.get("early_stop") if hasattr(cfg.training, "get") else getattr(
        cfg.training, "early_stop", None
    )
    if early_stop_metric:
        monitor = str(early_stop_metric)
        lower = monitor.lower()
        cafa_metrics = {
            "ia_fmax",
            "fmax",
            "precision",
            "recall",
            "pr_auc",
            "ap",
            "ia_precision",
            "ia_recall",
            "ia_threshold",
            "roc_auc",
        }
        if "/" not in monitor and lower in cafa_metrics:
            monitor = f"cafa/{monitor}"
        mode = "min" if "loss" in lower else "max"
        patience_val = (
            int(cfg.training.get("patience", 5))
            if hasattr(cfg.training, "get")
            else int(getattr(cfg.training, "patience", 5))
        )
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=patience_val,
                verbose=True,
            )
        )
    if extra_callbacks:
        callbacks.extend(list(extra_callbacks))

    log_interval_cfg = cfg.training.get("log_interval", 50)
    try:
        log_interval = int(log_interval_cfg)
    except (TypeError, ValueError):
        log_interval = 0
    if log_interval <= 0:
        log_interval = max(1, len(train_loader))

    trainer_kwargs: Dict[str, Any] = {
        "logger": mlflow_logger,
        "max_epochs": int(cfg.training.max_epochs),
        "precision": _precision_arg(cfg),
        "gradient_clip_val": float(cfg.training.get("gradient_clip", 0.0)),
        "accumulate_grad_batches": int(cfg.training.get("accumulate_batches", 1)),
        "log_every_n_steps": log_interval,
        "check_val_every_n_epoch": int(cfg.evaluation.get("val_interval", 1)),
        "callbacks": callbacks,
        "default_root_dir": str(base_dir),
        "enable_progress_bar": False,
    }
    if ("SLURM_JOB_ID" in os.environ or "SLURM_NTASKS" in os.environ) and "plugins" not in trainer_kwargs:
        trainer_kwargs["plugins"] = [SLURMEnvironment(auto_requeue=False)]
        log.info("Disabling Lightning SLURM auto-requeue to honor SIGTERM.")

    optional_keys = {
        "devices": "devices",
        "accelerator": "accelerator",
        "strategy": "strategy",
        "num_nodes": "num_nodes",
        "deterministic": "deterministic",
        "limit_train_batches": "limit_train_batches",
        "limit_val_batches": "limit_val_batches",
        "fast_dev_run": "fast_dev_run",
    }
    for cfg_key, trainer_key in optional_keys.items():
        if cfg.training.get(cfg_key) is not None:
            trainer_kwargs[trainer_key] = cfg.training[cfg_key]

    trainer = Trainer(**trainer_kwargs)
    try:                                                                                                                                                                                                                   
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)                                                                                                                                     
    except Exception:                                                                                                                                                                                                      
        model_saver.on_train_end(trainer, model)                                                                                                                                                                    
        raise

    best_metric = float(model.best_ia_fmax)
    callback_metric = trainer.callback_metrics.get("cafa/ia_fmax") if trainer.callback_metrics else None
    if not np.isfinite(best_metric) and callback_metric is not None:
        best_metric = float(callback_metric)
    if not np.isfinite(best_metric):
        best_metric = 0.0
    final_metric = float(getattr(model, "final_ia_fmax", float("nan")))
    if not np.isfinite(final_metric) and callback_metric is not None:
        final_metric = float(callback_metric)
    if not np.isfinite(final_metric):
        final_metric = best_metric

    _log_terminal_cafa_metrics(
        mlflow_logger,
        best_ia_fmax=best_metric,
        final_ia_fmax=final_metric,
    )
    summary_metrics = model.summary_metrics()  # ablation study
    if summary_metrics:
        _log_terminal_summary_metrics(mlflow_logger, summary_metrics)  # ablation study

    log.info("Best IA F-max achieved: %.4f (final=%.4f)", best_metric, final_metric)
    return best_metric


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def main(cfg: DictConfig) -> float:
    """Hydra entry point when invoking `python -m src.train.training`."""
    return run_training(cfg)


if __name__ == "__main__":
    main()
