"""Precompute ESM2 contact graphs and ProstT5 3Di caches for PF-AGCN."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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

import preprocessing as preprocessing_module
from preprocessing import _coerce_path, _resolve_structure_npz_dir, prepare_manifests, set_cache_root
from scripts.precompute_esm2_contact_graphs import main as precompute_esm2_main
from utils.system_runtime import apply_system_env

log = logging.getLogger(__name__)
DEFAULT_CACHE_PATH = Path("/orcd/home/002/lerchen/code/cafa_proj/data")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute ESM2 contact graphs and ProstT5 3Di caches."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to Hydra config directory",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="hpc_structural_config",
        help="Hydra config name",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--aspect",
        type=str,
        choices=["MF", "BP", "CC", "mf", "bp", "cc"],
        help="GO aspect to precompute (mf, bp, or cc).",
    )
    group.add_argument(
        "--all-aspects",
        action="store_true",
        help="Build manifests for MF/BP/CC after precomputing shared caches.",
    )
    parser.add_argument(
        "--skip-second-fwd",
        action="store_true",
        help="Use generate() scores to avoid a second ProstT5 forward pass.",
    )
    parser.add_argument(
        "--prostt5-device",
        default=None,
        help="Override ProstT5 device (e.g., cuda, cpu).",
    )
    return parser.parse_args(argv)


def _resolve_config_dir(config_dir: str | Path) -> Path:
    config_dir = Path(config_dir).expanduser()
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
        config_dir=str(resolved_dir), job_name="pf_agcn_precompute", version_base=None
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
    """Populate hydra.runtime fields so hydra.* interpolations resolve."""

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


def _resolve_sequences_path(data_cfg: Mapping[str, Any]) -> Path:
    sources = data_cfg.get("sources", {})
    if not isinstance(sources, Mapping):
        raise ValueError("data_config.sources must be a mapping.")
    candidate = sources.get("seqs_path") or sources.get("seqs_train_path")
    seqs_path = _coerce_path(candidate)
    if seqs_path is None:
        raise ValueError("data_config.sources.seqs_path is required for precomputation.")
    if not seqs_path.exists():
        raise FileNotFoundError(f"Sequence FASTA not found at {seqs_path}")
    return seqs_path


def _maybe_precompute_esm2(
    *,
    data_cfg: Mapping[str, Any],
    structure_cfg: Mapping[str, Any],
    cache_root: Path,
) -> None:
    graph_source = str(structure_cfg.get("graph_source", "")).lower()
    if not structure_cfg.get("enabled", False):
        log.info("Structural graphs disabled; skipping ESM2 contact precompute.")
        return
    if not (graph_source.startswith("esm2") or "contact" in graph_source):
        log.info("Graph source %s does not use ESM2 contacts; skipping.", graph_source)
        return

    seqs_path = _resolve_sequences_path(data_cfg)
    graph_dir = _resolve_structure_npz_dir(data_cfg, cache_root, structure_cfg)
    embed_dir = preprocessing_module.EMBED_CACHE_ROOTS.get("esm", cache_root / "esm_cache")
    log.info("ESM2 embedding cache dir: %s", embed_dir)
    log.info("ESM2 contact graph cache dir: %s", graph_dir)

    args = [
        "--seqs-path",
        seqs_path.as_posix(),
        "--output-dir",
        cache_root.as_posix(),
        "--embed-dir",
        embed_dir.as_posix(),
        "--graph-dir",
        graph_dir.as_posix(),
    ]

    model_name = structure_cfg.get("contact_model_name") or structure_cfg.get("esm2_model_name")
    if model_name in {"esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D"}:
        args.extend(["--model-name", str(model_name)])

    for key, flag in (
        ("contact_batch_size", "--batch-size"),
        ("contact_top_k", "--top-k"),
        ("contact_min_prob", "--min-prob"),
        ("contact_band", "--band"),
        ("contact_device", "--device"),
        ("contact_emb_dtype", "--emb-dtype"),
        ("contact_model_cache_dir", "--model-cache-dir"),
    ):
        value = structure_cfg.get(key)
        if value is not None:
            args.extend([flag, str(value)])

    if structure_cfg.get("contact_bf16"):
        args.append("--bf16")
    if structure_cfg.get("contact_symmetrize"):
        args.append("--symmetrize")
    if structure_cfg.get("contact_mutual"):
        args.append("--mutual")

    log.info("Precomputing ESM2 contacts with args: %s", " ".join(args))
    precompute_esm2_main(args)


def _log_sequence_embedding_paths(
    *,
    backend: str,
    cache_root: Path,
) -> None:
    embed_dir = preprocessing_module.EMBED_CACHE_ROOTS.get(backend, cache_root / "esm_cache")
    log.info("Sequence embedding backend: %s", backend)
    log.info("Sequence embedding cache dir: %s", embed_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.aspect:
        aspects = [args.aspect.upper()]
    else:
        if not args.all_aspects:
            log.info("No --aspect provided; defaulting to all aspects (MF/BP/CC).")
        aspects = ["MF", "BP", "CC"]

    precomputed_cache_root: Optional[Path] = None
    for aspect in aspects:
        overrides = [f"+aspect={aspect}"]
        cfg = _compose_config(args.config_path, args.config_name, overrides)
        _finalize_hydra_runtime(cfg, args.config_path, args.config_name)
        cache_root = _resolve_cache_root(cfg)
        apply_system_env(cfg)
        log.info("Precompute cache root: %s", cache_root)

        data_cfg = OmegaConf.to_container(cfg.data_config, resolve=True) or {}
        seq_cfg = OmegaConf.to_container(cfg.model.seq_embeddings, resolve=True) or {}
        prot_prior_cfg = OmegaConf.to_container(cfg.model.prot_prior, resolve=True) or {}
        structure_cfg = (
            OmegaConf.to_container(cfg.model.structural_graph, resolve=True)
            if getattr(cfg.model, "structural_graph", None) is not None
            else {}
        )
        prostt5_cfg = (
            OmegaConf.to_container(cfg.model.prostt5_3di, resolve=True)
            if getattr(cfg.model, "prostt5_3di", None) is not None
            else {}
        )
        if args.skip_second_fwd:
            prostt5_cfg = dict(prostt5_cfg)
            prostt5_cfg["skip_second_fwd"] = True
        if args.prostt5_device:
            prostt5_cfg = dict(prostt5_cfg)
            prostt5_cfg["device"] = str(args.prostt5_device)
        feature_dim = int(seq_cfg.get("feature_dim", 0))
        backend = str(seq_cfg.get("backend", "esm") or "esm").lower()
        _log_sequence_embedding_paths(backend=backend, cache_root=cache_root)

        if precomputed_cache_root is None or cache_root != precomputed_cache_root:
            _maybe_precompute_esm2(
                data_cfg=data_cfg,
                structure_cfg=structure_cfg,
                cache_root=cache_root,
            )
            precomputed_cache_root = cache_root

        manifest_root = (cache_root / "manifests").resolve()
        log.info(
            "Starting manifest build (aspect=%s, seq_backend=%s).",
            aspect,
            backend,
        )
        log.info("Manifest output dir: %s", manifest_root)
        bundle = prepare_manifests(
            data_cfg=data_cfg,
            output_root=manifest_root,
            aspect=aspect,
            feature_dim=feature_dim,
            protein_prior_cfg=prot_prior_cfg,
            embedding_backend=backend,
            structure_cfg=structure_cfg,
            prostt5_cfg=prostt5_cfg,
            cache_root=cache_root,
        )
        log.info("Precomputed manifests: %s, %s, %s", bundle.train, bundle.val, bundle.test)
        log.info("Finished manifest build for aspect=%s.", aspect)


if __name__ == "__main__":
    main()
