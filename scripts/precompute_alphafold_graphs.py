"""Precompute AlphaFold residue graphs for PF-AGCN.

Downloads AlphaFold prediction metadata and PDBs via aria2c, extracts CA
coordinates and per-residue pLDDT, and writes sparse adjacency caches (.npz)
compatible with the structural graph dataloader.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def _parse_fasta_sequences_flexible(path: Path) -> pd.DataFrame:
    """Read FASTA sequences with CAFA or whitespace-delimited headers.

    Supports CAFA headers like "sp|P9WHI7|RECN_MYCT" and headers like
    "A0A0C5B5G6 9606" by taking the second pipe-delimited token or the first
    whitespace token as the entry ID.
    """

    records: dict[str, dict[str, str]] = {}
    current_id: Optional[str] = None
    current_header: Optional[str] = None
    sequence_chunks: list[str] = []

    def _flush() -> None:
        if current_id is None:
            return
        records[current_id] = {
            "entry_id": current_id,
            "header": current_header or "",
            "sequence": "".join(sequence_chunks),
        }

    def _extract_entry_id(header: str) -> str:
        tokens = header.split("|")
        if len(tokens) >= 3 and tokens[1].strip():
            return tokens[1].strip()
        space_tokens = header.split()
        if not space_tokens or not space_tokens[0].strip():
            raise ValueError(f"Unexpected FASTA header format: '{header}'")
        return space_tokens[0].strip()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                header = line[1:].strip()
                current_id = _extract_entry_id(header)
                current_header = header
                sequence_chunks = []
            else:
                sequence_chunks.append(line)

    _flush()
    df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)
    log.info("Loaded %d sequences from %s", len(df), path)
    return df

log = logging.getLogger(__name__)


class AlphaFoldDownloadError(RuntimeError):
    """Error raised when aria2c download fails."""

    def __init__(self, uri: str, output_path: Path, stdout: str, stderr: str) -> None:
        message = stderr.strip() or stdout.strip() or "aria2c download failed"
        super().__init__(f"{message} (uri={uri}, path={output_path})")
        self.uri = uri
        self.output_path = output_path
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class AlphaFoldFragment:
    """Metadata for a single AlphaFold fragment."""

    accession: str
    model_id: str
    pdb_url: str
    start: int
    end: int
    length: int


def _entry_cache_filename(entry_id: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in entry_id)
    digest = hashlib.md5(entry_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe_id or 'protein'}_{digest}.npz"


def _entry_cache_path(entry_id: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / _entry_cache_filename(entry_id)


def _contact_graph_path(entry_id: str, root: Path) -> Path:
    return root / _entry_cache_filename(entry_id)


def _load_id_filter(path: Optional[str]) -> Optional[set[str]]:
    if not path:
        return None
    ids_path = Path(path).expanduser()
    if not ids_path.is_absolute():
        ids_path = (PROJECT_ROOT / ids_path).resolve()
    if not ids_path.exists():
        raise FileNotFoundError(f"ID filter CSV not found: {ids_path}")
    frame = pd.read_csv(ids_path)
    if "entry_id" not in frame.columns:
        raise ValueError("ID filter CSV must include an 'entry_id' column.")
    return set(frame["entry_id"].astype(str).str.strip())


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_fragment_bounds(item: Mapping[str, Any]) -> tuple[int, int, int]:
    start = _coerce_int(item.get("sequenceStart")) or _coerce_int(item.get("uniprotStart"))
    end = _coerce_int(item.get("sequenceEnd")) or _coerce_int(item.get("uniprotEnd"))
    seq = item.get("sequence")
    seq_len = len(seq) if isinstance(seq, str) else None

    if start is not None and end is not None:
        length = end - start + 1
        if length <= 0:
            raise ValueError(f"Invalid fragment bounds: start={start}, end={end}")
        if seq_len is not None and seq_len != length:
            log.warning(
                "Sequence length mismatch in AlphaFold metadata: bounds=%d, seq_len=%d.",
                length,
                seq_len,
            )
        return start, end, length

    if seq_len is None:
        raise ValueError("AlphaFold metadata missing sequence bounds and sequence.")
    start = 1 if start is None else start
    end = start + seq_len - 1 if end is None else end
    return start, end, seq_len


def _group_fragments(
    items: Sequence[Mapping[str, Any]],
    *,
    entry_id: str,
) -> dict[str, list[AlphaFoldFragment]]:
    grouped: dict[str, list[AlphaFoldFragment]] = {}
    for item in items:
        pdb_url = item.get("pdbUrl")
        if not pdb_url:
            raise ValueError("AlphaFold metadata missing pdbUrl.")
        accession = (
            str(item.get("uniprotAccession") or item.get("entryId") or item.get("uniprotId") or entry_id)
        )
        model_id = str(item.get("modelEntityId") or item.get("entryId") or accession)
        start, end, length = _infer_fragment_bounds(item)
        fragment = AlphaFoldFragment(
            accession=accession,
            model_id=model_id,
            pdb_url=str(pdb_url),
            start=int(start),
            end=int(end),
            length=int(length),
        )
        grouped.setdefault(accession, []).append(fragment)
    return grouped


def _select_fragments_for_entry(
    items: Sequence[Mapping[str, Any]],
    *,
    entry_id: str,
    expected_length: int,
) -> list[AlphaFoldFragment]:
    grouped = _group_fragments(items, entry_id=entry_id)
    candidates: list[tuple[str, list[AlphaFoldFragment], int]] = []
    for accession, fragments in grouped.items():
        ordered = sorted(fragments, key=lambda frag: frag.start)
        total = sum(frag.length for frag in ordered)
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt.start <= prev.end:
                log.warning(
                    "Overlapping AlphaFold fragments for %s: %s and %s.",
                    entry_id,
                    prev.model_id,
                    nxt.model_id,
                )
            if nxt.start > prev.end + 1:
                log.warning(
                    "Gap between AlphaFold fragments for %s: %d..%d then %d..%d.",
                    entry_id,
                    prev.start,
                    prev.end,
                    nxt.start,
                    nxt.end,
                )
        if total == expected_length:
            candidates.append((accession, ordered, total))

    if not candidates:
        lengths = {acc: sum(frag.length for frag in frags) for acc, frags in grouped.items()}
        raise ValueError(
            f"No AlphaFold isoform matches expected length {expected_length} for {entry_id}. "
            f"Available lengths: {lengths}"
        )

    if len(candidates) > 1:
        log.warning(
            "Multiple AlphaFold isoforms match length for %s; selecting best match.", entry_id
        )

    def _priority(acc: str) -> tuple[int, str]:
        if acc == entry_id:
            return (0, acc)
        if acc.startswith(entry_id):
            return (1, acc)
        return (2, acc)

    candidates.sort(key=lambda item: _priority(item[0]))
    return candidates[0][1]


def _parse_pdb_ca(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[float, float, float]] = []
    plddt: list[float] = []
    parsed_any = False

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            parsed_any = True
            if len(line) < 66:
                raise ValueError(f"Unexpected PDB line length ({len(line)}): {line.strip()}")
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b_factor = float(line[60:66])
            except ValueError as exc:
                raise ValueError(f"Failed to parse PDB CA line: {line.strip()}") from exc
            coords.append((x, y, z))
            plddt.append(b_factor)

    if not parsed_any:
        raise ValueError(f"No CA atoms found in PDB file {pdb_path}")
    coords_array = np.asarray(coords, dtype=np.float32)
    plddt_array = np.asarray(plddt, dtype=np.float32)
    return coords_array, np.clip(plddt_array, 0.0, 100.0)


def _merge_sparse_graphs(
    graphs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_indices: list[np.ndarray] = []
    edge_weights: list[np.ndarray] = []
    plddt_list: list[np.ndarray] = []
    offset = 0
    for edge_index, edge_weight, plddt in graphs:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges).")
        edge_indices.append(edge_index + offset)
        edge_weights.append(edge_weight)
        plddt_list.append(plddt)
        offset += int(plddt.shape[0])
    if edge_indices:
        merged_index = np.concatenate(edge_indices, axis=1)
        merged_weight = np.concatenate(edge_weights, axis=0)
        merged_plddt = np.concatenate(plddt_list, axis=0)
    else:
        merged_index = np.zeros((2, 0), dtype=np.int64)
        merged_weight = np.zeros((0,), dtype=np.float32)
        merged_plddt = np.zeros((0,), dtype=np.float32)
    return merged_index, merged_weight, merged_plddt


def _cleanup_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("Failed to clean up %s.", path)


def _is_resource_not_found(message: str) -> bool:
    lowered = message.lower()
    return (
        "resource not found" in lowered
        or "errorcode=3" in lowered
        or "404" in lowered
        or "not found" in lowered
    )


def _fallback_to_contact_graph(
    entry_id: str,
    cache_path: Path,
    contact_graph_dir: Path,
    *,
    reason: str,
) -> None:
    if cache_path.exists():
        log.info("Skipping fallback for %s; cache already exists.", entry_id)
        return
    contact_path = _contact_graph_path(entry_id, contact_graph_dir)
    if not contact_path.exists():
        raise FileNotFoundError(
            f"Missing ESM2 contact graph for {entry_id} at {contact_path}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.symlink_to(contact_path)
    log.warning("Fell back to ESM2 contact graph for %s (%s).", entry_id, reason)


def _run_aria2c(
    uri: str,
    output_path: Path,
    *,
    aria2c_path: str,
    headers: Sequence[str] | None = None,
    extra_args: Sequence[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        aria2c_path,
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--summary-interval=0",
        "--console-log-level=warn",
        "-d",
        str(output_path.parent),
        "-o",
        output_path.name,
    ]
    if headers:
        for header in headers:
            cmd.append(f"--header={header}")
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd + [uri], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AlphaFoldDownloadError(uri, output_path, result.stdout, result.stderr)


def _build_sparse_graph(
    coords: np.ndarray,
    plddt: np.ndarray,
    *,
    distance_cutoff: float,
    top_k: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    from modules.structure_graph import StructureGraphBuilder

    ca_tensor = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
    plddt_tensor = torch.from_numpy(plddt).to(device=device, dtype=torch.float32)
    edge_index, edge_weight = StructureGraphBuilder.build_sparse_graph_from_ca(
        ca_coords=ca_tensor,
        plddt=plddt_tensor,
        distance_cutoff=distance_cutoff,
        top_k=top_k,
    )
    return (
        edge_index.detach().cpu().numpy().astype(np.int64, copy=False),
        edge_weight.detach().cpu().numpy().astype(np.float16, copy=False),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute AlphaFold sparse graph caches.")
    parser.add_argument("--seqs-path", required=True, help="FASTA file with input sequences.")
    parser.add_argument("--output-dir", required=True, help="Directory to write .npz caches.")
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Optional directory for temporary JSON/PDB downloads.",
    )
    parser.add_argument(
        "--fallback-graph-dir",
        default=None,
        help="Directory holding ESM2 contact graphs for fallback symlinks.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for distance computations (cpu or cuda).",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff (Angstroms) for residue edges.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K neighbors retained per residue.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for download/processing.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for distance computation.",
    )
    parser.add_argument(
        "--aria2c",
        default="aria2c",
        help="Path to aria2c executable.",
    )
    parser.add_argument(
        "--aria2c-extra",
        action="append",
        default=[],
        help="Extra arguments passed to aria2c (repeatable).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable certificate checks for aria2c downloads.",
    )
    parser.add_argument(
        "--ids-csv",
        default=None,
        help="Optional CSV with entry_id column to filter sequences.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignored; existing .npz files are never regenerated.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N sequences.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"AlphaFold metadata must be a list, got {type(data)}.")
    return data


def _process_entry(
    entry_id: str,
    sequence: str,
    cache_path: Path,
    *,
    download_root: Path,
    aria2c_path: str,
    aria2c_args: Sequence[str],
    distance_cutoff: float,
    top_k: int,
    device: torch.device,
    fallback_graph_dir: Path,
) -> None:
    if cache_path.exists():
        log.info("Skipping %s; cache already exists.", entry_id)
        return

    expected_length = len(sequence)
    safe_id = cache_path.stem
    json_path = download_root / f"{safe_id}.json"
    try:
        _run_aria2c(
            f"https://alphafold.ebi.ac.uk/api/prediction/{entry_id}",
            json_path,
            aria2c_path=aria2c_path,
            headers=("accept: application/json",),
            extra_args=aria2c_args,
        )
    except AlphaFoldDownloadError as exc:
        _cleanup_paths([json_path])
        if _is_resource_not_found(str(exc)):
            _fallback_to_contact_graph(
                entry_id,
                cache_path,
                fallback_graph_dir,
                reason="prediction metadata not found",
            )
            return
        raise

    metadata = _load_json(json_path)
    if not metadata:
        _cleanup_paths([json_path])
        _fallback_to_contact_graph(
            entry_id,
            cache_path,
            fallback_graph_dir,
            reason="empty prediction metadata",
        )
        return

    try:
        fragments = _select_fragments_for_entry(
            metadata,
            entry_id=entry_id,
            expected_length=expected_length,
        )
    except ValueError as exc:
        _cleanup_paths([json_path])
        _fallback_to_contact_graph(
            entry_id,
            cache_path,
            fallback_graph_dir,
            reason=str(exc),
        )
        return

    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frag_idx, fragment in enumerate(fragments):
        pdb_path = download_root / f"{safe_id}_frag{frag_idx + 1}.pdb"
        try:
            _run_aria2c(
                fragment.pdb_url,
                pdb_path,
                aria2c_path=aria2c_path,
                headers=None,
                extra_args=aria2c_args,
            )
        except AlphaFoldDownloadError as exc:
            _cleanup_paths([json_path, pdb_path])
            if _is_resource_not_found(str(exc)):
                _fallback_to_contact_graph(
                    entry_id,
                    cache_path,
                    fallback_graph_dir,
                    reason="PDB not found",
                )
                return
            raise
        coords, plddt = _parse_pdb_ca(pdb_path)
        if coords.shape[0] != fragment.length:
            raise ValueError(
                f"AlphaFold length mismatch for {entry_id} ({fragment.model_id}): "
                f"expected {fragment.length}, got {coords.shape[0]}"
            )
        edge_index, edge_weight = _build_sparse_graph(
            coords,
            plddt,
            distance_cutoff=distance_cutoff,
            top_k=top_k,
            device=device,
        )
        graphs.append((edge_index, edge_weight, plddt.astype(np.float32, copy=False)))
        pdb_path.unlink(missing_ok=True)

    edge_index, edge_weight, plddt = _merge_sparse_graphs(graphs)
    if plddt.shape[0] != expected_length:
        raise ValueError(
            f"AlphaFold total length mismatch for {entry_id}: "
            f"expected {expected_length}, got {plddt.shape[0]}"
        )
    np.savez_compressed(
        cache_path,
        edge_index=edge_index.astype(np.int64, copy=False),
        edge_weight=edge_weight.astype(np.float16, copy=False),
        plddt=plddt.astype(np.float16, copy=False),
    )
    json_path.unlink(missing_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)

    if shutil.which(args.aria2c) is None:
        raise FileNotFoundError(f"aria2c not found at '{args.aria2c}'.")

    seqs_path = Path(args.seqs_path).expanduser()
    if not seqs_path.is_absolute():
        seqs_path = (PROJECT_ROOT / seqs_path).resolve()
    if not seqs_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {seqs_path}")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_download: Optional[tempfile.TemporaryDirectory] = None
    download_dir = Path(args.download_dir).expanduser() if args.download_dir else None
    if download_dir is None:
        temp_download = tempfile.TemporaryDirectory(prefix="af_graphs_")
        download_dir = Path(temp_download.name)
    if not download_dir.is_absolute():
        download_dir = (PROJECT_ROOT / download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        id_filter = _load_id_filter(args.ids_csv)
        sequences = _parse_fasta_sequences_flexible(seqs_path)
        if id_filter is not None:
            sequences = sequences[sequences["entry_id"].astype(str).isin(id_filter)]
            sequences = sequences.reset_index(drop=True)
        if sequences.empty:
            raise ValueError("No sequences selected for AlphaFold preprocessing.")

        pending: list[tuple[str, str, Path]] = []
        skipped = 0
        for row in sequences.itertuples():
            entry_id = str(row.entry_id)
            sequence = str(row.sequence)
            cache_path = _entry_cache_path(entry_id, output_dir)
            if cache_path.exists():
                skipped += 1
                if args.overwrite:
                    log.warning("Existing cache for %s preserved; overwrite ignored.", entry_id)
                continue
            pending.append((entry_id, sequence, cache_path))

        log.info(
            "Loaded %d sequences (%d pending, %d cached).",
            len(sequences),
            len(pending),
            skipped,
        )
        if not pending:
            log.info("No new sequences to process; exiting.")
            return

        device = torch.device(args.device)
        torch.set_num_threads(max(1, int(args.torch_threads)))

        aria2c_args = list(args.aria2c_extra or [])
        if args.insecure:
            aria2c_args.append("--check-certificate=false")

        if device.type == "cuda" and args.workers > 1:
            log.warning("CUDA selected; forcing workers=1 to avoid GPU contention.")
            args.workers = 1

        fallback_dir = Path(args.fallback_graph_dir).expanduser() if args.fallback_graph_dir else None
        if fallback_dir is None:
            fallback_dir = (output_dir.parent / "esm2_contact_cache").resolve()
        if not fallback_dir.is_absolute():
            fallback_dir = (PROJECT_ROOT / fallback_dir).resolve()
        if not fallback_dir.exists():
            log.warning("Fallback ESM2 contact cache dir does not exist: %s", fallback_dir)

        processed = 0
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(args.workers))
        ) as executor:
            future_map = {}
            for entry_id, sequence, cache_path in pending:
                future = executor.submit(
                    _process_entry,
                    entry_id,
                    sequence,
                    cache_path,
                    download_root=download_dir,
                    aria2c_path=args.aria2c,
                    aria2c_args=aria2c_args,
                    distance_cutoff=args.distance_cutoff,
                    top_k=args.top_k,
                    device=device,
                    fallback_graph_dir=fallback_dir,
                )
                future_map[future] = entry_id

            for future in concurrent.futures.as_completed(future_map):
                entry_id = future_map[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    log.error("AlphaFold preprocessing failed for %s: %s", entry_id, exc)
                    failures.append(entry_id)
                processed += 1
                if args.log_every and processed % int(args.log_every) == 0:
                    log.info("Processed %d/%d sequences.", processed, len(pending))

        if failures:
            raise RuntimeError(
                f"AlphaFold preprocessing failed for {len(failures)} entries; "
                f"first failures: {', '.join(failures[:5])}"
            )
        log.info("Finished writing AlphaFold caches to %s.", output_dir)
    finally:
        if temp_download is not None:
            temp_download.cleanup()


if __name__ == "__main__":
    main()
