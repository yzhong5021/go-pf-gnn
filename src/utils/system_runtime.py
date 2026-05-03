"""Helpers for applying Hydra system configs at runtime."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping

from omegaconf import DictConfig, OmegaConf


def _to_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        data = OmegaConf.to_container(node, resolve=True)
        return dict(data or {})
    if isinstance(node, Mapping):
        return dict(node)
    return {}


def _section(cfg: Any, key: str) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        value = cfg.get(key)
    elif isinstance(cfg, Mapping):
        value = cfg.get(key)
    else:
        value = getattr(cfg, key, None)
    return _to_dict(value)


def apply_system_env(cfg: Any) -> None:
    """Export environment variables declared under cfg.system.env."""

    system_cfg = _section(cfg, "system")
    env_cfg = _section(system_cfg, "env")
    for key, value in env_cfg.items():
        if value is None:
            os.environ.pop(str(key), None)
        else:
            os.environ[str(key)] = str(value)


def merged_mlflow_settings(cfg: Any) -> Dict[str, Any]:
    """Combine MLflow settings from cfg.system.mlflow and top-level cfg.mlflow."""

    merged: Dict[str, Any] = {}
    system_cfg = _section(cfg, "system")
    merged.update(_section(system_cfg, "mlflow"))
    merged.update(_section(cfg, "mlflow"))
    return merged
