"""Convenience CLI to train, tune, or run inference for PF-AGCN."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import hydra
from hydra import compose, initialize_config_dir, version as hydra_version
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.conf import ConfigSourceInfo
from hydra.types import RunMode
from omegaconf import DictConfig, MISSING, OmegaConf, open_dict, read_write

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.hpo import run_hpo_command
from preprocessing import ManifestBundle, prepare_manifests, set_cache_root
from train.training import run_training
from utils.system_runtime import apply_system_env

log = logging.getLogger(__name__)
DEFAULT_CACHE_PATH = Path("/orcd/home/002/lerchen/code/cafa_proj/data")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PF-AGCN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Launch Hydra-configured training")
    train_parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to Hydra config directory",
    )
    train_parser.add_argument(
        "--config-name",
        type=str,
        default="default_config",
        help="Hydra config name",
    )
    train_parser.add_argument(
        "--aspect",
        type=str,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        required=True,
        help="GO aspect to train (mf, bp, or cc)",
    )

    hpo_parser = subparsers.add_parser("hpo", help="Run Optuna hyperparameter search")
    hpo_parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to Hydra config directory",
    )
    hpo_parser.add_argument(
        "--config-name",
        type=str,
        default="default_config",
        help="Hydra config name",
    )
    hpo_parser.add_argument(
        "--aspect",
        type=str,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        required=True,
        help="GO aspect to train during HPO (mf, bp, or cc)",
    )
    hpo_parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials to run",
    )
    hpo_parser.add_argument(
        "--max-epochs",
        type=int,
        default=45,
        help="Max epochs per Optuna trial (overrides config)",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained model")
    predict_parser.add_argument("run", type=str, help="MLflow run ID or run directory")
    predict_parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest for inference (auto-generated if omitted)",
    )
    predict_parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Destination CSV for sigmoid scores",
    )
    predict_parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Hydra config driving manifest discovery",
    )
    predict_parser.add_argument(
        "--aspect",
        type=str,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        default=None,
        help="GO aspect for inference (mf, bp, or cc)",
    )
    predict_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size",
    )

    args, hydra_overrides = parser.parse_known_args(argv)
    setattr(args, "hydra_overrides", hydra_overrides)
    return args


def _resolve_config_dir(config_path: str | Path) -> Path:
    """Resolve config directory relative to the project root and validate it exists."""

    config_path_str = str(config_path).strip()
    if not config_path_str:
        raise ValueError("config_path must be a non-empty string")
    normalised_path = config_path_str.replace(chr(92), "/")
    config_dir = Path(normalised_path).expanduser()
    if not config_dir.is_absolute():
        config_dir = (PROJECT_ROOT / config_dir).resolve()
    else:
        config_dir = config_dir.resolve()
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found at {config_dir}. "
            "Set --config-path relative to the project root or provide an absolute path."
        )
    return config_dir


def _compose_config(
    config_dir: str | Path, config_name: str, overrides: list[str] | None = None
) -> DictConfig:
    GlobalHydra.instance().clear()
    resolved_dir = _resolve_config_dir(config_dir)
    with initialize_config_dir(
        config_dir=str(resolved_dir), job_name="pf_agcn_scripts", version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [], return_hydra_config=True)
    return cfg


def _resolve_cache_root(cfg: DictConfig) -> Path:
    value = OmegaConf.select(cfg, "cache_path", default=None)
    if value is None:
        value = OmegaConf.select(cfg, "system.paths.cache_path", default=None)
    fallback = os.environ.get("PF_AGCN_CACHE", str(DEFAULT_CACHE_PATH))
    base = Path(str(value or fallback)).expanduser()
    if not base.is_absolute():
        base = (PROJECT_ROOT / base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    set_cache_root(base)
    return base


def _finalize_hydra_runtime(cfg: DictConfig, config_path: str | Path, config_name: str) -> None:
    """Populate hydra.runtime fields so hydra.* interpolations resolve when composed manually."""

    hydra_cfg = cfg.hydra
    runtime = hydra_cfg.runtime
    with read_write(hydra_cfg):
        with open_dict(hydra_cfg):
            hydra_cfg.mode = hydra_cfg.mode or RunMode.RUN
            with read_write(runtime):
                with open_dict(runtime):
                    runtime.cwd = runtime.cwd or os.getcwd()
                    runtime.version = runtime.version or hydra.__version__
                    runtime.version_base = runtime.version_base or hydra_version.getbase()
                    if not runtime.config_sources or runtime.config_sources in (None, "???", MISSING):
                        config_dir = _resolve_config_dir(config_path)
                        runtime.config_sources = [
                            ConfigSourceInfo(path=str(config_dir), schema="file", provider="main")
                        ]
                    if runtime.choices in (None, "???", MISSING):
                        runtime.choices = {}
            with read_write(hydra_cfg.job):
                with open_dict(hydra_cfg.job):
                    job_name = OmegaConf.select(hydra_cfg, "job.name", default=None)
                    if not job_name or job_name in (None, "???", MISSING):
                        hydra_cfg.job.name = config_name
                    job_id = OmegaConf.select(hydra_cfg, "job.id", default=None)
                    if not job_id or job_id in (None, "???", MISSING):
                        hydra_cfg.job.id = "manual"
                    job_num = OmegaConf.select(hydra_cfg, "job.num", default=None)
                    if job_num in (None, "???", MISSING):
                        hydra_cfg.job.num = 0
                    job_cfg_name = OmegaConf.select(hydra_cfg, "job.config_name", default=None)
                    if not job_cfg_name or job_cfg_name in (None, "???", MISSING):
                        hydra_cfg.job.config_name = config_name

    HydraConfig.instance().set_config(cfg)

    run_dir = OmegaConf.select(cfg, "hydra.run.dir")
    if not run_dir:
        run_dir = os.path.join(runtime.cwd, "outputs", "manual")
    run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
    with read_write(runtime):
        with open_dict(runtime):
            current_output_dir = OmegaConf.select(cfg, "hydra.runtime.output_dir", default=None)
            if not current_output_dir or current_output_dir in (None, "???", MISSING):
                runtime.output_dir = run_dir

    HydraConfig.instance().set_config(cfg)


def _load_manifest_meta(manifest_path: Path) -> Dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("meta", {})


def _ensure_manifests(cfg: DictConfig, aspect: str, cache_root: Path) -> List[str]:
    overrides: List[str] = []
    aspect_upper = aspect.upper()

    data_cfg = OmegaConf.to_container(cfg.data_config, resolve=True)
    seq_cfg = OmegaConf.to_container(cfg.model.seq_embeddings, resolve=True)
    prot_prior_cfg = OmegaConf.to_container(cfg.model.prot_prior, resolve=True)
    feature_dim = int(seq_cfg["feature_dim"])
    max_len_val = seq_cfg.get("seq_len")
    max_length = int(max_len_val) if max_len_val else None
    backend = str(seq_cfg.get("backend", "esm") or "esm").lower()

    existing_manifest = data_cfg.get("train_manifest")
    if existing_manifest:
        manifest_path = Path(existing_manifest)
        if not manifest_path.is_absolute():
            candidate = (cache_root / manifest_path).resolve()
            manifest_path = candidate if candidate.exists() else (PROJECT_ROOT / manifest_path).resolve()
        meta = _load_manifest_meta(manifest_path)
        manifest_aspect = str(meta.get("aspect", aspect_upper)).upper()
        if manifest_aspect != aspect_upper:
            raise ValueError(f"Manifest aspect {manifest_aspect} does not match requested {aspect_upper}")
        num_functions = meta.get("num_functions")
        if num_functions is None:
            raise ValueError(f"Manifest at {manifest_path} is missing 'num_functions' metadata")
        overrides.append(f"task.num_functions={int(num_functions)}")
        feature_dim_meta = meta.get("feature_dim")
        backend_meta = meta.get("embedding_backend")
        if backend_meta:
            overrides.append(f"model.seq_embeddings.backend={backend_meta}")
        effective_backend = str(backend_meta or backend).lower()
        if feature_dim_meta is not None:
            feature_dim_meta = int(feature_dim_meta)
            if effective_backend == "esm" and feature_dim_meta != feature_dim:
                overrides.append(f"model.seq_embeddings.raw_dim={feature_dim_meta}")
            else:
                overrides.append(f"model.seq_embeddings.feature_dim={feature_dim_meta}")
        if not data_cfg.get("val_manifest"):
            overrides.append(f"data_config.val_manifest={manifest_path.as_posix()}")
        if not data_cfg.get("test_manifest"):
            overrides.append(f"data_config.test_manifest={manifest_path.as_posix()}")
        return overrides

    manifest_root = (cache_root / "manifests").resolve()
    bundle: ManifestBundle = prepare_manifests(
        data_cfg=data_cfg,
        output_root=manifest_root,
        aspect=aspect_upper,
        feature_dim=feature_dim,
        max_length=max_length,
        protein_prior_cfg=prot_prior_cfg,
        embedding_backend=backend,
        cache_root=cache_root,
    )
    overrides.extend(
        [
            f"data_config.train_manifest={bundle.train.as_posix()}",
            f"data_config.val_manifest={bundle.val.as_posix()}",
            f"data_config.test_manifest={bundle.test.as_posix()}",
            f"task.num_functions={bundle.num_functions}",
            f"model.seq_embeddings.backend={bundle.embedding_backend}",
        ]
    )
    bundle_backend = str(bundle.embedding_backend).lower()
    if bundle_backend == "esm" and bundle.feature_dim != feature_dim:
        overrides.append(f"model.seq_embeddings.raw_dim={bundle.feature_dim}")
    else:
        overrides.append(f"model.seq_embeddings.feature_dim={bundle.feature_dim}")
    return overrides


def run_train_command(
    config_path: str, config_name: str, aspect: str, hydra_overrides: Optional[List[str]] = None
) -> None:
    aspect_upper = aspect.upper()
    overrides: List[str] = []
    if hydra_overrides:
        overrides.extend(hydra_overrides)
    overrides.append(f"+aspect={aspect_upper}")
    cfg = _compose_config(config_path, config_name, overrides)
    _finalize_hydra_runtime(cfg, config_path, config_name)
    cache_root = _resolve_cache_root(cfg)
    apply_system_env(cfg)
    manifest_overrides = _ensure_manifests(cfg, aspect_upper, cache_root)
    if manifest_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(manifest_overrides))
    run_training(cfg)


def _resolve_manifest_for_predict(args: argparse.Namespace) -> str:
    config_path = Path(args.config).resolve()
    cli_overrides: list[str] = []
    if args.aspect:
        cli_overrides.append(f"aspect={args.aspect.upper()}")
    cfg = _compose_config(config_path.parent, config_path.stem, cli_overrides)
    cache_root = _resolve_cache_root(cfg)
    apply_system_env(cfg)
    data_cfg = OmegaConf.to_container(cfg.data_config, resolve=True)
    seq_cfg = OmegaConf.to_container(cfg.model.seq_embeddings, resolve=True)
    prot_prior_cfg = OmegaConf.to_container(cfg.model.prot_prior, resolve=True)
    feature_dim = int(seq_cfg["feature_dim"])
    max_len_val = seq_cfg.get("seq_len")
    max_length = int(max_len_val) if max_len_val else None
    backend = str(seq_cfg.get("backend", "esm") or "esm").lower()

    aspect = (args.aspect or cfg.get("aspect") or "").upper()

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            candidate = (cache_root / manifest_path).resolve()
            manifest_path = candidate if candidate.exists() else (PROJECT_ROOT / manifest_path).resolve()
        else:
            manifest_path = manifest_path.resolve()
        meta = _load_manifest_meta(manifest_path)
        manifest_aspect = str(meta.get("aspect", aspect or "")).upper()
        if aspect and manifest_aspect and manifest_aspect != aspect:
            raise ValueError(
                f"Provided manifest aspect {manifest_aspect} does not match requested {aspect}"
            )
        return manifest_path.as_posix()

    for candidate in (
        data_cfg.get("test_manifest"),
        data_cfg.get("val_manifest"),
        data_cfg.get("train_manifest"),
    ):
        if not candidate:
            continue
        manifest_path = Path(candidate)
        if not manifest_path.is_absolute():
            candidate_path = (cache_root / manifest_path).resolve()
            manifest_path = (
                candidate_path if candidate_path.exists() else (PROJECT_ROOT / manifest_path).resolve()
            )
        else:
            manifest_path = manifest_path.resolve()
        meta = _load_manifest_meta(manifest_path)
        manifest_aspect = str(meta.get("aspect", aspect or "")).upper()
        if aspect and manifest_aspect and manifest_aspect != aspect:
            continue
        if not aspect:
            aspect = manifest_aspect
        return manifest_path.as_posix()

    if not aspect:
        raise ValueError("Aspect must be specified when auto-generating inference manifests.")

    manifest_root = (cache_root / "manifests").resolve()
    bundle = prepare_manifests(
        data_cfg=data_cfg,
        output_root=manifest_root,
        aspect=aspect,
        feature_dim=feature_dim,
        max_length=max_length,
        protein_prior_cfg=prot_prior_cfg,
        embedding_backend=backend,
        cache_root=cache_root,
    )
    log.info("Generated inference manifest at %s", bundle.test)
    return bundle.test.as_posix()


def run_predict(args: argparse.Namespace) -> None:
    from scripts.predict import main as predict_main

    manifest_path = _resolve_manifest_for_predict(args)
    predict_argv = [
        args.run,
        "--manifest",
        manifest_path,
        "--output",
        args.output,
        "--config",
        args.config,
    ]
    if args.batch_size is not None:
        predict_argv.extend(["--batch-size", str(args.batch_size)])
    predict_main(predict_argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv or sys.argv[1:])
    if args.command == "train":
        run_train_command(args.config_path, args.config_name, args.aspect, args.hydra_overrides)
    elif args.command == "hpo":
        run_hpo_command(
            args.config_path,
            args.config_name,
            args.aspect,
            args.n_trials,
            args.max_epochs,
            args.hydra_overrides,
            _compose_config,
            _finalize_hydra_runtime,
            _resolve_cache_root,
            _ensure_manifests,
        )
    elif args.command == "predict":
        if getattr(args, "hydra_overrides", None):
            raise ValueError("Hydra overrides are only supported for the 'train' command.")
        run_predict(args)


if __name__ == "__main__":
    main()
