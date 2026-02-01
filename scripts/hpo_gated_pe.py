"""Optuna HPO runner for gated_pe with MLflow + Lightning."""

from __future__ import annotations

import argparse
import math
import re
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import optuna
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import main as main_cli
from src.modules.dataloader import build_manifest_dataloader, load_ia_weights
from src.train.training import PFAGCNLightningModule, _precision_arg, _resolve_output_dir
from src.utils.system_runtime import apply_system_env, merged_mlflow_settings
from utils.manifest_paths import resolve_manifest_path_template

log = logging.getLogger(__name__)


@dataclass
class TrialResult:
    best_value: float
    best_epoch: int
    best_val_loss: float
    best_checkpoint: Optional[Path]


class BestMetricTracker(Callback):
    """Track best ia_fmax and the corresponding val loss/epoch."""

    def __init__(self, metric: str = "cafa/ia_fmax", loss_metric: str = "val/loss") -> None:
        super().__init__()
        self.metric = metric
        self.loss_metric = loss_metric
        self.best_value = -float("inf")
        self.best_epoch = -1
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics or {}
        value = metrics.get(self.metric)
        loss = metrics.get(self.loss_metric)
        if value is None or loss is None:
            return
        try:
            value_float = float(value)
            loss_float = float(loss)
        except (TypeError, ValueError):
            return
        if value_float > self.best_value:
            self.best_value = value_float
            self.best_epoch = int(trainer.current_epoch)
            self.best_val_loss = loss_float


class OptunaLossPruningCallback(Callback):
    """Optuna pruning based on val/loss with optional warmup epochs."""

    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val/loss",
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.warmup_epochs = int(max(0, warmup_epochs))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics or {}
        value = metrics.get(self.monitor)
        if value is None:
            return
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return
        epoch = int(trainer.current_epoch)
        if epoch + 1 < self.warmup_epochs:
            return
        # Study direction is maximize; invert loss for pruning decisions.
        self.trial.report(-value_float, step=epoch + 1)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned on {self.monitor}={value_float:.6f} at epoch {epoch}")


class HPOPFAGCNLightningModule(PFAGCNLightningModule):
    """Override optimizer/scheduler to support eps and warmup_lr."""

    def configure_optimizers(self) -> Any:
        optim_cfg = self.cfg.optimizer
        betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
        eps = float(optim_cfg.get("eps", 1e-8))
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(optim_cfg.lr),
            betas=betas,
            weight_decay=float(optim_cfg.weight_decay),
            eps=eps,
        )
        scheduler = build_hpo_scheduler(self.cfg, optimizer)
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


def build_hpo_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
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
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name_val}")

    warmup_epochs = int(sched_value("warmup_epochs", 0))
    decay_epochs = int(sched_value("decay_epochs", total_epochs))
    decay_epochs = max(warmup_epochs + 1, min(decay_epochs, total_epochs))
    min_lr = float(sched_value("min_lr", 0.0))
    warmup_lr = sched_value("warmup_lr", None)

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def make_lambda(base_lr: float):
        if base_lr <= 0:
            return lambda _epoch: 1.0
        min_factor = min_lr / base_lr if min_lr > 0 else 0.0
        if warmup_lr is None:
            start_factor = 1.0 / float(max(1, warmup_epochs)) if warmup_epochs > 0 else 1.0
        else:
            start_factor = float(warmup_lr) / base_lr
            start_factor = max(0.0, min(start_factor, 1.0))

        def lr_lambda(current_epoch: int) -> float:
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                if warmup_epochs == 1:
                    return 1.0
                progress = current_epoch / float(max(1, warmup_epochs - 1))
                return start_factor + progress * (1.0 - start_factor)
            progress = (current_epoch - warmup_epochs) / float(
                max(1, decay_epochs - warmup_epochs)
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_factor + (1.0 - min_factor) * cosine)

        return lr_lambda

    lambdas = [make_lambda(lr) for lr in base_lrs]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def _prepare_mlflow_logger(cfg: DictConfig, base_dir: Path) -> MLFlowLogger:
    mlflow_cfg = merged_mlflow_settings(cfg)
    tracking_uri = mlflow_cfg.get("tracking_uri")
    artifact_location = mlflow_cfg.get("artifact_root")
    if tracking_uri is None:
        tracking_dir = (base_dir / "mlruns").resolve()
        tracking_uri = f"file:{tracking_dir}"
    experiment_name = (
        mlflow_cfg.get("experiment_name")
        or cfg.get("experiment_name")
        or mlflow_cfg.get("experiment")
        or "pfagcn_hpo"
    )
    run_name_override = mlflow_cfg.get("run_name_override")
    run_name = run_name_override or mlflow_cfg.get("run_name")
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        artifact_location=artifact_location,
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(flatten_config(resolved_cfg))
    log.info(
        "MLflow run initialised: experiment=%s run=%s (tracking_uri=%s)",
        experiment_name,
        run_name,
        tracking_uri,
    )
    return logger


def flatten_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten nested mappings to MLflow-friendly key/value pairs."""

    flat: Dict[str, Any] = {}
    invalid_key = re.compile(r"[^A-Za-z0-9_\\-\\. /:]")

    def _flatten(node: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in node.items():
            name = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, Mapping):
                _flatten(value, name)
            else:
                sanitized = name.replace("/", "_")
                sanitized = invalid_key.sub("_", sanitized)
                flat[sanitized] = value

    _flatten(config)
    return flat


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


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trial_dir(study_dir: Path, trial_number: int) -> Path:
    return (study_dir / f"trial_{trial_number:04d}").resolve()


def _sample_param(
    trial: optuna.Trial, definition: Mapping[str, Any], *, low: Optional[float] = None, high: Optional[float] = None
) -> Any:
    name = str(definition["name"])
    dist_type = str(definition.get("type", "float")).lower()
    if dist_type == "float":
        low_val = float(low if low is not None else definition["low"])
        high_val = float(high if high is not None else definition["high"])
        if high_val <= low_val:
            return low_val
        return trial.suggest_float(name, low_val, high_val, log=bool(definition.get("log", False)))
    if dist_type == "int":
        low_val = int(definition["low"])
        high_val = int(definition["high"])
        if high_val <= low_val:
            return low_val
        return trial.suggest_int(
            name,
            low_val,
            high_val,
            step=int(definition.get("step", 1)),
            log=bool(definition.get("log", False)),
        )
    if dist_type == "categorical":
        choices = list(definition.get("choices", []))
        if not choices:
            raise ValueError(f"No choices provided for categorical parameter {name}")
        return trial.suggest_categorical(name, choices)
    raise ValueError(f"Unsupported parameter type: {dist_type}")


def _apply_trial_params(cfg: DictConfig, params: Mapping[str, Any]) -> None:
    with read_write(cfg):
        with open_dict(cfg):
            if "optimizer.lr" in params:
                cfg.optimizer.lr = float(params["optimizer.lr"])
            if "optimizer.weight_decay" in params:
                cfg.optimizer.weight_decay = float(params["optimizer.weight_decay"])
            if "optimizer.beta2" in params:
                beta2 = float(params["optimizer.beta2"])
                cfg.optimizer.betas = [0.9, beta2]
            if "optimizer.eps" in params:
                cfg.optimizer.eps = float(params["optimizer.eps"])
            if "scheduler.warmup_lr" in params:
                cfg.scheduler.warmup_lr = float(params["scheduler.warmup_lr"])
            if "scheduler.min_lr" in params:
                cfg.scheduler.min_lr = float(params["scheduler.min_lr"])
            if "model.dropout" in params:
                dropout = float(params["model.dropout"])
                cfg.model.gcn.dropout = dropout
                cfg.model.prost_graph.dropout = dropout
                cfg.model.gated_pe.mlp_dropout = dropout
                cfg.model.sqb.dccn.dropout = dropout
            if "model.loss.focusing" in params:
                cfg.model.loss.focusing = float(params["model.loss.focusing"])
            if "model.loss.balancing" in params:
                cfg.model.loss.balancing = float(params["model.loss.balancing"])


def _build_study(cfg: DictConfig, worker_id: int) -> optuna.study.Study:
    study_cfg = OmegaConf.to_container(cfg.hpo.study, resolve=True)
    study_name = str(study_cfg.get("name"))
    storage = study_cfg.get("storage")
    direction = study_cfg.get("direction", "maximize")
    seed = study_cfg.get("seed")

    sampler_cfg = OmegaConf.to_container(cfg.hpo.sampler, resolve=True)
    sampler = optuna.samplers.TPESampler(
        seed=int(seed) + int(worker_id) if seed is not None else None,
        multivariate=bool(sampler_cfg.get("multivariate", True)),
        group=bool(sampler_cfg.get("group", False)),
        constant_liar=bool(sampler_cfg.get("constant_liar", True)),
        n_startup_trials=int(sampler_cfg.get("n_startup_trials", 10)),
    )

    pruner_cfg = OmegaConf.to_container(cfg.hpo.pruner, resolve=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=int(pruner_cfg.get("min_resource", 4)),
        max_resource=int(cfg.training.max_epochs),
        reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
    )

    _ensure_storage_dir(storage)
    return optuna.create_study(
        study_name=study_name,
        direction=str(direction),
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


def _ensure_storage_dir(storage: Optional[str]) -> None:
    if not storage:
        return
    if not storage.startswith("sqlite:"):
        return
    if storage.startswith("sqlite:///"):
        path = storage[len("sqlite:///"):]
    else:
        path = storage[len("sqlite:"):]
    if not path:
        return
    db_path = Path(path).expanduser()
    if not db_path.is_absolute():
        db_path = (PROJECT_ROOT / db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_manifests(cfg: DictConfig, aspect: str) -> DictConfig:
    cache_root = main_cli._resolve_cache_root(cfg)
    manifest_overrides = main_cli._ensure_manifests(cfg, aspect, cache_root)
    if manifest_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(manifest_overrides))
    return cfg


def _train_once(cfg: DictConfig, trial: optuna.Trial, run_dir: Path) -> TrialResult:
    with read_write(cfg):
        with open_dict(cfg):
            cfg.hydra.runtime.output_dir = str(run_dir)
            cfg.training.deterministic = False
            cfg.data_config.batch_cache_dir = str(run_dir / "batch_cache")
    _configure_logging(cfg)
    apply_system_env(cfg)

    seed_everything(int(cfg.training.get("seed", 42)), workers=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    aspect = str(cfg.get("aspect", ""))
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

    base_dir = run_dir
    min_length_cfg = cfg.data_config.get("min_length", 10)
    min_length = int(min_length_cfg) if min_length_cfg is not None else None

    prot_prior_cfg = OmegaConf.to_container(
        getattr(cfg.model, "prot_prior", {}), resolve=True
    )
    go_prior_cfg = OmegaConf.to_container(
        getattr(cfg.model, "go_prior", {}), resolve=True
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
        min_length=min_length,
        split="val",
    )
    if val_loader is None:
        raise RuntimeError("Validation manifest is required to start training.")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ia_weights = load_ia_weights(cfg_dict, base_dir)
    thresholds = cfg.evaluation.get("threshold_grid", [0.5])

    mlflow_logger = _prepare_mlflow_logger(cfg, base_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor="cafa/ia_fmax",
        mode="max",
        filename="epoch{epoch:03d}-iafmax{cafa/ia_fmax:.4f}",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    best_tracker = BestMetricTracker()
    pruner_cfg = OmegaConf.to_container(cfg.hpo.pruner, resolve=True)
    pruning_cb = OptunaLossPruningCallback(
        trial,
        monitor="val/loss",
        warmup_epochs=int(pruner_cfg.get("warmup_epochs", 0)),
    )

    callbacks: list[Callback] = [checkpoint_cb, lr_monitor, best_tracker, pruning_cb]
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
        "default_root_dir": str(run_dir),
        "enable_progress_bar": False,
        "deterministic": False,
    }
    if ("SLURM_JOB_ID" in os.environ or "SLURM_NTASKS" in os.environ) and "plugins" not in trainer_kwargs:
        trainer_kwargs["plugins"] = [SLURMEnvironment(auto_requeue=False)]
        log.info("Disabling Lightning SLURM auto-requeue to honor SIGTERM.")

    optional_keys = {
        "devices": "devices",
        "accelerator": "accelerator",
        "strategy": "strategy",
        "num_nodes": "num_nodes",
        "limit_train_batches": "limit_train_batches",
        "limit_val_batches": "limit_val_batches",
        "fast_dev_run": "fast_dev_run",
    }
    for cfg_key, trainer_key in optional_keys.items():
        if cfg.training.get(cfg_key) is not None:
            trainer_kwargs[trainer_key] = cfg.training[cfg_key]

    trainer = Trainer(**trainer_kwargs)
    model = HPOPFAGCNLightningModule(cfg, thresholds=thresholds, ia_weights=ia_weights)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_value = best_tracker.best_value
    if not isinstance(best_value, float) or not (best_value == best_value):
        best_value = float(getattr(model, "best_ia_fmax", 0.0))
    try:
        mlflow_logger.log_metrics(
            {
                "hpo/best_ia_fmax": float(best_value),
                "hpo/best_epoch": float(best_tracker.best_epoch),
                "hpo/best_val_loss": float(best_tracker.best_val_loss),
            },
            step=best_tracker.best_epoch if best_tracker.best_epoch >= 0 else None,
        )
    except Exception as exc:
        log.warning("Failed to log HPO summary metrics to MLflow: %s", exc)

    return TrialResult(
        best_value=float(best_value),
        best_epoch=best_tracker.best_epoch,
        best_val_loss=float(best_tracker.best_val_loss),
        best_checkpoint=Path(checkpoint_cb.best_model_path) if checkpoint_cb.best_model_path else None,
    )


def _cleanup_pruned_trial(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _select_top_trials(study: optuna.study.Study, top_k: int) -> List[optuna.trial.FrozenTrial]:
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials.sort(key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    return trials[:top_k]


def _copy_top_checkpoints(study: optuna.study.Study, study_dir: Path, top_k: int) -> None:
    top_trials = _select_top_trials(study, top_k)
    dest_dir = study_dir / "top_checkpoints"
    dest_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for rank, trial in enumerate(top_trials, 1):
        checkpoint_path = trial.user_attrs.get("best_checkpoint")
        if not checkpoint_path:
            continue
        src = Path(checkpoint_path)
        if not src.exists():
            continue
        dest = dest_dir / f"rank{rank:02d}_trial{trial.number:04d}.ckpt"
        shutil.copy2(src, dest)
        summary.append(
            {
                "rank": rank,
                "trial": trial.number,
                "value": trial.value,
                "checkpoint": dest.as_posix(),
                "params": trial.params,
                "best_epoch": trial.user_attrs.get("best_epoch"),
                "best_val_loss": trial.user_attrs.get("best_val_loss"),
            }
        )
    summary_path = dest_dir / "top_trials.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_multi_seed(
    base_cfg: DictConfig,
    study: optuna.study.Study,
    study_dir: Path,
    seeds: Sequence[int],
    top_k: int,
) -> None:
    top_trials = _select_top_trials(study, top_k)
    if not top_trials:
        return
    multi_dir = study_dir / "multi_seed"
    multi_dir.mkdir(parents=True, exist_ok=True)
    for trial in top_trials:
        trial_dir = multi_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
            _apply_trial_params(cfg, trial.params)
            with read_write(cfg):
                with open_dict(cfg):
                    cfg.training.seed = int(seed)
                    cfg.mlflow.run_name_override = (
                        f"gated_pe_hpo_seed{seed}_trial{trial.number:04d}"
                    )
            seed_dir = trial_dir / f"seed_{seed:02d}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            try:
                _train_once(cfg, optuna.trial.FixedTrial(trial.params), seed_dir)
            except Exception as exc:
                log.warning("Multi-seed run failed for trial %s seed %s: %s", trial.number, seed, exc)


def _trial_objective(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    study_dir: Path,
    worker_id: int,
) -> float:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    search_space = OmegaConf.to_container(cfg.hpo.search_space, resolve=True) or []
    params: Dict[str, Any] = {}

    base_lr_def = next((d for d in search_space if d.get("name") == "optimizer.lr"), None)
    if base_lr_def:
        base_lr = _sample_param(trial, base_lr_def)
        params["optimizer.lr"] = base_lr
    else:
        base_lr = float(cfg.optimizer.lr)

    for definition in search_space:
        name = str(definition.get("name"))
        if name in {"optimizer.lr"}:
            continue
        if name == "scheduler.warmup_lr":
            low = float(definition.get("low", 1e-8))
            high = float(definition.get("high", base_lr))
            capped_high = min(high, base_lr * 0.9)
            params[name] = _sample_param(trial, definition, low=low, high=max(low * 1.01, capped_high))
            continue
        if name == "scheduler.min_lr":
            low = float(definition.get("low", 1e-8))
            high = float(definition.get("high", base_lr))
            capped_high = min(high, base_lr * 0.9)
            params[name] = _sample_param(trial, definition, low=low, high=max(low * 1.01, capped_high))
            continue
        params[name] = _sample_param(trial, definition)

    _apply_trial_params(cfg, params)

    with read_write(cfg):
        with open_dict(cfg):
            cfg.mlflow.run_name_override = f"gated_pe_hpo_trial{trial.number:04d}_w{worker_id}"

    run_dir = _trial_dir(study_dir, trial.number)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = _train_once(cfg, trial, run_dir)
    except optuna.TrialPruned:
        _cleanup_pruned_trial(run_dir)
        trial.set_user_attr("pruned", True)
        raise

    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("best_epoch", result.best_epoch)
    trial.set_user_attr("best_val_loss", result.best_val_loss)
    if result.best_checkpoint is not None:
        trial.set_user_attr("best_checkpoint", str(result.best_checkpoint))

    return result.best_value


def _should_stop(
    study: optuna.study.Study,
    *,
    max_failures: int,
    target_trials: int,
    baseline_failures: int = 0,
) -> bool:
    failed_total = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    failed = max(0, failed_total - int(baseline_failures))
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if failed >= max_failures:
        log.warning("Stopping study after %d failures.", failed)
        return True
    if completed >= target_trials:
        return True
    return False


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna HPO runner for gated_pe")
    parser.add_argument("--config-path", type=str, default="configs")
    parser.add_argument("--config-name", type=str, default="gated_pe_hpo")
    parser.add_argument(
        "--aspect",
        type=str,
        required=True,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        help="GO aspect to train (mf, bp, or cc)",
    )
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a 1-trial smoke check")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip optimization and only copy top checkpoints + run multi-seed.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    main_cli._register_resolvers()

    aspect_upper = args.aspect.upper()
    overrides: list[str] = [f"+aspect={aspect_upper}"]
    if args.max_epochs is not None:
        overrides.append(f"training.max_epochs={int(args.max_epochs)}")
    if args.n_trials is not None:
        overrides.append(f"hpo.study.n_trials={int(args.n_trials)}")
    if args.seed is not None:
        overrides.append(f"training.seed={int(args.seed)}")
    if args.patience is not None:
        overrides.append(f"training.patience={int(args.patience)}")
    if args.storage is not None:
        overrides.append(f"hpo.study.storage={args.storage}")
    if args.study_name is not None:
        overrides.append(f"hpo.study.name={args.study_name}")
    if args.smoke:
        overrides.extend(
            [
                "training.max_epochs=1",
                "training.limit_train_batches=1",
                "training.limit_val_batches=1",
                "training.accelerator=cpu",
                "training.devices=1",
                "training.precision=32",
                "data_config.min_length=1",
                "model.structural_graph.device=cpu",
                "model.prostt5_3di.device=cpu",
                "hpo.study.n_trials=1",
            ]
        )

    cfg = main_cli._compose_config(args.config_path, args.config_name, overrides)
    main_cli._finalize_hydra_runtime(cfg, args.config_path, args.config_name)

    cfg = _prepare_manifests(cfg, aspect_upper)
    with read_write(cfg):
        with open_dict(cfg):
            cfg.aspect = aspect_upper
            cfg.data_config.batch_size = 64

    study = _build_study(cfg, args.worker_id)
    baseline_failures = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    )

    results_root = Path(cfg.system.paths.results_root)
    study_dir = results_root / "hpo_gated_pe" / aspect_upper.lower() / study.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        top_k = int(cfg.hpo.checkpoints.get("top_k", 3))
        _copy_top_checkpoints(study, study_dir, top_k)
        multi_seed_cfg = OmegaConf.to_container(cfg.hpo.multi_seed, resolve=True)
        if multi_seed_cfg.get("enabled", False):
            seeds = multi_seed_cfg.get("seeds", [])
            top_k_ms = int(multi_seed_cfg.get("top_k", top_k))
            _run_multi_seed(cfg, study, study_dir, seeds=seeds, top_k=top_k_ms)
        return

    study_cfg = OmegaConf.to_container(cfg.hpo.study, resolve=True)
    target_trials = int(study_cfg.get("n_trials", 50))
    timeout_hours = float(study_cfg.get("timeout_hours", 0))
    timeout_seconds = int(timeout_hours * 3600) if timeout_hours > 0 else None
    max_failures = int(study_cfg.get("max_failures", 3))

    start_ts = time.time()
    while True:
        if timeout_seconds is not None and (time.time() - start_ts) >= timeout_seconds:
            log.info("Stopping due to timeout.")
            break
        if _should_stop(
            study,
            max_failures=max_failures,
            target_trials=target_trials,
            baseline_failures=baseline_failures,
        ):
            break
        study.optimize(
            lambda trial: _trial_objective(trial, cfg, study_dir, args.worker_id),
            n_trials=1,
            catch=(Exception,),
        )


if __name__ == "__main__":
    main()
