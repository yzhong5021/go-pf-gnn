"""Precompute ESM2 embeddings and sparse contact graphs for PF-AGCN.

Reads sequences from a FASTA file and writes:
  - embeddings: <embed_dir>/<entry_id>_<hash>.npz with key "embeddings"
  - graphs:     <graph_dir>/<entry_id>_<hash>.npz with edge_index/edge_weight/plddt

Contacts are attention-derived scores from the Meta ESM2 models.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:  # pragma: no cover - optional dependency
    import esm  # type: ignore
except ImportError as exc:  # pragma: no cover
    esm = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

log = logging.getLogger(__name__)


def _entry_cache_filename(entry_id: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in entry_id)
    digest = hashlib.md5(entry_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe_id or 'protein'}_{digest}.npz"


def _entry_cache_path(entry_id: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / _entry_cache_filename(entry_id)


def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _chunked(
    items: Sequence[tuple[str, str, Path, Path, bool, bool]],
    size: int,
) -> Iterable[Sequence[tuple[str, str, Path, Path, bool, bool]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    seq_id = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(chunks)
                header = line[1:].strip()
                seq_id = _entry_id_from_header(header)
                chunks = []
            else:
                chunks.append(line)
        if seq_id is not None:
            yield seq_id, "".join(chunks)


def _entry_id_from_header(header: str) -> str:
    tokens = header.split("|")
    if len(tokens) >= 3:
        return tokens[1]
    return header.split()[0]


def _mask_local_band(contacts: torch.Tensor, band: int) -> torch.Tensor:
    masked = contacts.clone()
    masked.fill_diagonal_(0)
    for delta in range(1, int(band) + 1):
        masked.diagonal(delta).zero_()
        masked.diagonal(-delta).zero_()
    return masked


def _sparsify_contacts(
    contacts: torch.Tensor,
    *,
    top_k: int,
    min_prob: float,
    band: int,
    symmetrize: bool,
    mutual: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    length = contacts.size(0)
    graph = contacts
    if symmetrize:
        graph = 0.5 * (graph + graph.t())
    graph = _mask_local_band(graph, band)
    if min_prob > 0:
        graph = graph.masked_fill(graph < float(min_prob), 0.0)

    k = min(int(top_k), max(1, length - 1))
    values, indices = torch.topk(graph, k=k, dim=-1)
    keep = values > 0
    if keep.sum().item() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=graph.device),
            torch.empty((0,), dtype=torch.float32, device=graph.device),
        )
    src = torch.arange(length, device=graph.device).unsqueeze(1).expand(length, k)
    src = src[keep]
    dst = indices[keep]
    weights = values[keep].float()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)

    if mutual and edge_index.numel() > 0:
        keys = edge_index[0] * length + edge_index[1]
        reverse = edge_index[1] * length + edge_index[0]
        keys_sorted, _ = torch.sort(keys)
        positions = torch.searchsorted(keys_sorted, reverse)
        in_bounds = (positions >= 0) & (positions < keys_sorted.numel())
        matched = torch.zeros_like(in_bounds, dtype=torch.bool)
        matched[in_bounds] = keys_sorted[positions[in_bounds]] == reverse[in_bounds]
        edge_index = edge_index[:, matched]
        weights = weights[matched]
    return edge_index, weights


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute ESM2 embeddings and sparse contact graph caches."
    )
    parser.add_argument("--seqs-path", required=True, help="FASTA file with input sequences.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for embedding/graph caches.",
    )
    parser.add_argument(
        "--embed-dir",
        default=None,
        help="Override embedding cache directory (default: <output-dir>/esm_cache).",
    )
    parser.add_argument(
        "--graph-dir",
        default=None,
        help="Override graph cache directory (default: <output-dir>/esm2_contact_cache).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run ESM2 on (default: cuda if available).",
    )
    parser.add_argument(
        "--model-name",
        default="esm2_t33_650M_UR50D",
        choices=["esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D"],
        help="ESM2 model variant.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Sequences per forward pass (sorted by length for padding efficiency).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 autocast on CUDA.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Top-K neighbors retained per residue.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="Drop edges with contact score < min-prob before top-k.",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=3,
        help="Mask |i-j| <= band to remove trivial local adjacency.",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help="Symmetrize contact scores before sparsification.",
    )
    parser.add_argument(
        "--mutual",
        action="store_true",
        help="Keep only mutual edges (i->j and j->i).",
    )
    parser.add_argument(
        "--emb-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Dtype for stored embeddings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N sequences.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Optional cache directory for ESM model weights (torch hub).",
    )
    return parser.parse_args(argv)


def _load_model(model_name: str) -> Tuple[torch.nn.Module, object, int]:
    if esm is None:  # pragma: no cover
        raise ImportError(
            "esm package is required for ESM2 contact extraction (pip install fair-esm)."
        ) from _IMPORT_ERROR
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        layer = 33
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        layer = 30
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model, alphabet, layer


def _slice_repr(repr_tensor: torch.Tensor, length: int) -> torch.Tensor:
    rep_len = repr_tensor.size(0)
    if rep_len == length + 2:
        return repr_tensor[1 : length + 1]
    if rep_len == length:
        return repr_tensor[:length]
    raise ValueError(f"Unexpected representation length {rep_len} for sequence length {length}.")


def _slice_contacts(contacts: torch.Tensor, length: int) -> torch.Tensor:
    cont_len = contacts.size(0)
    if cont_len == length + 2:
        return contacts[1 : length + 1, 1 : length + 1]
    if cont_len == length:
        return contacts[:length, :length]
    raise ValueError(f"Unexpected contact length {cont_len} for sequence length {length}.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)

    seqs_path = Path(args.seqs_path).expanduser()
    if not seqs_path.is_absolute():
        seqs_path = (PROJECT_ROOT / seqs_path).resolve()
    if not seqs_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {seqs_path}")

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    embed_dir = Path(args.embed_dir).expanduser() if args.embed_dir else output_root / "esm_cache"
    graph_dir = Path(args.graph_dir).expanduser() if args.graph_dir else output_root / "esm2_contact_cache"
    if not embed_dir.is_absolute():
        embed_dir = (output_root / embed_dir).resolve()
    if not graph_dir.is_absolute():
        graph_dir = (output_root / graph_dir).resolve()
    embed_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    if args.model_cache_dir:
        cache_dir = Path(args.model_cache_dir).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = (PROJECT_ROOT / cache_dir).resolve()
        torch.hub.set_dir(str(cache_dir))

    sequences = list(_read_fasta(seqs_path))
    if not sequences:
        raise ValueError("No sequences selected for ESM2 preprocessing.")

    sequences.sort(key=lambda pair: len(pair[1]))
    pending: list[tuple[str, str, Path, Path, bool, bool]] = []
    skipped = 0
    for entry_id, sequence in sequences:
        emb_path = _entry_cache_path(entry_id, embed_dir)
        graph_path = _entry_cache_path(entry_id, graph_dir)
        emb_exists = emb_path.exists()
        graph_exists = graph_path.exists()
        need_emb = args.overwrite or not emb_exists
        need_graph = args.overwrite or not graph_exists
        if not need_emb and not need_graph:
            skipped += 1
            continue
        pending.append((entry_id, sequence, emb_path, graph_path, need_emb, need_graph))

    log.info(
        "Loaded %d sequences (%d pending, %d cached).",
        len(sequences),
        len(pending),
        skipped,
    )
    if not pending:
        log.info("No new sequences to process; exiting.")
        return

    device = _resolve_device(args.device)
    log.info(
        "Running ESM2 (%s) on device=%s with batch_size=%d.",
        args.model_name,
        device,
        args.batch_size,
    )

    model, alphabet, layer = _load_model(args.model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)

    emb_dtype = np.float16 if args.emb_dtype == "float16" else np.float32
    autocast_enabled = device.type == "cuda" and args.bf16

    processed = 0
    for chunk in _chunked(pending, max(1, int(args.batch_size))):
        data = [(entry_id, sequence) for entry_id, sequence, _emb, _graph, _need_emb, _need_graph in chunk]
        _labels, _strs, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.inference_mode():
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(tokens, repr_layers=[layer], return_contacts=True)
            else:
                outputs = model(tokens, repr_layers=[layer], return_contacts=True)

        reps = outputs["representations"][layer]
        contacts = outputs["contacts"]

        for idx, (entry_id, sequence, emb_path, graph_path, need_emb, need_graph) in enumerate(chunk):
            expected_len = len(sequence)
            valid = tokens[idx] != alphabet.padding_idx
            token_len = tokens.size(1)
            valid_count = int(valid.sum().item())
            bos = 1 if getattr(alphabet, "prepend_bos", False) else 0
            eos = 1 if getattr(alphabet, "append_eos", False) else 0
            seq_len = max(0, valid_count - bos - eos)
            if seq_len != expected_len:
                raise ValueError(
                    f"Unexpected token length for {entry_id}: expected {expected_len}, "
                    f"got {seq_len} after trimming BOS/EOS."
                )

            rep = None
            if need_emb:
                rep = reps[idx][valid]
                if bos and rep.size(0) > 0:
                    rep = rep[1:]
                if eos and rep.size(0) > 0:
                    rep = rep[:-1]
                if rep.size(0) != expected_len:
                    raise ValueError(
                        f"Unexpected embedding length for {entry_id}: expected {expected_len}, "
                        f"got {rep.size(0)}."
                    )

            contact = None
            if need_graph:
                contact = contacts[idx]
                contact_len = contact.size(0)
                if contact_len == token_len:
                    contact = contact[valid][:, valid]
                    if bos and contact.size(0) > 0:
                        contact = contact[1:, 1:]
                    if eos and contact.size(0) > 0:
                        contact = contact[:-1, :-1]
                elif contact_len == valid_count:
                    start = bos
                    end = contact_len - eos
                    contact = contact[start:end, start:end]
                elif contact_len >= seq_len:
                    contact = contact[:seq_len, :seq_len]
                else:
                    raise ValueError(
                        f"Unexpected contact length {contact_len} for {entry_id} "
                        f"(expected at least {seq_len})."
                    )
                if contact.size(0) != expected_len:
                    raise ValueError(
                        f"Unexpected contact length for {entry_id}: expected {expected_len}, "
                        f"got {contact.size(0)}."
                    )

                # Use mean contact probability as a pLDDT-like per-residue signal.
                plddt = contact.float().mean(dim=-1)
                edge_index, edge_weight = _sparsify_contacts(
                    contact,
                    top_k=args.top_k,
                    min_prob=args.min_prob,
                    band=args.band,
                    symmetrize=args.symmetrize,
                    mutual=args.mutual,
                )

                np.savez_compressed(
                    graph_path,
                    edge_index=edge_index.detach().cpu().numpy().astype(np.int64, copy=False),
                    edge_weight=edge_weight.detach().cpu().numpy().astype(np.float16, copy=False),
                    plddt=plddt.detach().cpu().numpy().astype(np.float16, copy=False),
                )

            if need_emb and rep is not None:
                rep_dtype = torch.float16 if args.emb_dtype == "float16" else torch.float32
                rep = rep.to(dtype=rep_dtype)
                np.savez_compressed(
                    emb_path,
                    embeddings=rep.detach().cpu().numpy().astype(emb_dtype, copy=False),
                )

        processed += len(chunk)
        if args.log_every and processed % int(args.log_every) == 0:
            log.info("Processed %d/%d sequences.", processed, len(pending))

    log.info("Finished writing ESM2 caches to %s.", output_root)


if __name__ == "__main__":
    main()
