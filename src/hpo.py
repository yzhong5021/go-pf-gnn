"""Optuna-driven hyperparameter optimisation for PF-AGCN."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from omegaconf import DictConfig, OmegaConf, open_dict, read_write

from src.train.training import run_training
from src.utils.system_runtime import apply_system_env

log = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE: List[Mapping[str, Any]] = [
    {"name": "optimizer.lr", "type": "float", "low": 1e-5, "high": 5e-3, "log": True},
    {"name": "optimizer.weight_decay", "type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    {"name": "model.seq_gating.shared_dim", "type": "categorical", "choices": [128, 256, 384]},
    {"name": "model.graph.protein_stacks", "type": "int", "low": 2, "high": 4, "step": 1},
    {"name": "model.seq_gating.dropout", "type": "float", "low": 0.0, "high": 0.5},
    {
        "name": "data_config.batch_size",
        "type": "categorical",
        "choices": [32, 64, 96],
    },
    {"name": "scheduler.name", "type": "categorical", "choices": ["step"]},
    {"name": "scheduler.gamma", "type": "float", "low": 0.85, "high": 0.98},
]


def return_objective(
    base_cfg: DictConfig, search_space: Sequence[Mapping[str, Any]], max_epochs: int
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective that trains PF-AGCN and returns IA F-max."""

    def _suggest_parameter(trial: optuna.Trial, definition: Mapping[str, Any]) -> Any:
        name = str(definition["name"])
        dist_type = str(definition.get("type", "float")).lower()
        if dist_type == "float":
            return trial.suggest_float(
                name,
                float(definition["low"]),
                float(definition["high"]),
                log=bool(definition.get("log", False)),
            )
        if dist_type == "int":
            return trial.suggest_int(
                name,
                int(definition["low"]),
                int(definition["high"]),
                step=int(definition.get("step", 1)),
                log=bool(definition.get("log", False)),
            )
        if dist_type == "categorical":
            choices = list(definition["choices"])
            if not choices:
                raise ValueError(f"No choices provided for categorical parameter {name}")
            return trial.suggest_categorical(name, choices)
        raise ValueError(f"Unsupported parameter type: {dist_type}")

    base_container = OmegaConf.to_container(base_cfg, resolve=True)

    def objective(trial: optuna.Trial) -> float:
        cfg = OmegaConf.create(base_container)
        trial_overrides = [f"training.max_epochs={max_epochs}"]
        for definition in search_space:
            value = _suggest_parameter(trial, definition)
            trial_overrides.append(f"{definition['name']}={value}")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(trial_overrides))
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="cafa/ia_fmax")
        return run_training(cfg, extra_callbacks=[pruning_callback])

    return objective


def run_hpo_command(
    config_path: str,
    config_name: str,
    aspect: str,
    n_trials: int,
    max_epochs: int,
    hydra_overrides: Optional[List[str]],
    compose_config_fn: Callable[[str | Path, str, Optional[List[str]]], DictConfig],
    finalize_config_fn: Callable[[DictConfig, str | Path, str], None],
    resolve_cache_root_fn: Callable[[DictConfig], Path],
    ensure_manifests_fn: Callable[[DictConfig, str, Path], List[str]],
) -> optuna.study.Study:
    """Entrypoint for Optuna-based hyperparameter optimisation."""

    def _search_space_from_config(cfg: DictConfig) -> List[Mapping[str, Any]]:
        configured = OmegaConf.select(cfg, "hpo.search_space", default=None)
        if configured is None:
            return DEFAULT_SEARCH_SPACE
        container = OmegaConf.to_container(configured, resolve=True)
        if isinstance(container, list):
            definitions: List[Mapping[str, Any]] = []
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                if "name" not in item:
                    raise ValueError("Each search space entry must include a 'name' field.")
                definitions.append(dict(item))
            return definitions or DEFAULT_SEARCH_SPACE
        if isinstance(container, Mapping):
            definitions = []
            for name, definition in container.items():
                if not isinstance(definition, Mapping):
                    continue
                new_def = dict(definition)
                new_def["name"] = name
                definitions.append(new_def)
            return definitions or DEFAULT_SEARCH_SPACE
        raise ValueError("hpo.search_space must be a list or mapping of parameter definitions")

    def _study_timeout_seconds(study_cfg: Mapping[str, Any]) -> Optional[int]:
        timeout_seconds = study_cfg.get("timeout_seconds")
        if timeout_seconds is not None:
            return int(timeout_seconds)
        timeout_minutes = study_cfg.get("timeout_minutes")
        if timeout_minutes is None:
            return None
        return int(timeout_minutes) * 60

    aspect_upper = aspect.upper()
    overrides: List[str] = list(hydra_overrides or [])
    overrides.extend([f"+aspect={aspect_upper}", f"training.max_epochs={max_epochs}"])
    cfg = compose_config_fn(config_path, config_name, overrides)
    finalize_config_fn(cfg, config_path, config_name)
    cache_root = resolve_cache_root_fn(cfg)
    apply_system_env(cfg)
    manifest_overrides = ensure_manifests_fn(cfg, aspect_upper, cache_root)
    if manifest_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(manifest_overrides))
    with read_write(cfg):
        with open_dict(cfg):
            cfg["aspect"] = aspect_upper
            cfg.training["max_epochs"] = max_epochs
            hpo_node = cfg.get("hpo")
            if isinstance(hpo_node, DictConfig):
                with open_dict(hpo_node):
                    study_node = hpo_node.get("study")
                    if isinstance(study_node, DictConfig):
                        with open_dict(study_node):
                            study_node["n_trials"] = n_trials

    hpo_cfg = OmegaConf.to_container(OmegaConf.select(cfg, "hpo", default={}), resolve=True) or {}
    hpo_cfg = hpo_cfg if isinstance(hpo_cfg, Mapping) else {}
    study_cfg = hpo_cfg.get("study") if isinstance(hpo_cfg, Mapping) else {}
    study_cfg = study_cfg if isinstance(study_cfg, Mapping) else {}

    search_space = _search_space_from_config(cfg)
    sampler_seed = study_cfg.get("seed")
    sampler = TPESampler(
        multivariate=True,
        n_startup_trials=10,
        seed=int(sampler_seed) if sampler_seed is not None else None,
    )
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study_name = study_cfg.get("name") or f"pfagcn_hpo_{aspect_upper}"
    storage = study_cfg.get("storage")
    timeout = _study_timeout_seconds(study_cfg)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = return_objective(cfg, search_space, max_epochs)
    study.optimize(objective, n_trials=int(n_trials), timeout=timeout)
    try:
        log.info("Best trial %.4f with params: %s", study.best_value, study.best_trial.params)
    except ValueError:
        log.warning("No completed trials were recorded; check pruning or training failures.")
    return study
