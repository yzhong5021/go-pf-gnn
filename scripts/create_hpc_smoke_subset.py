"""Create a small CAFA subset for HPC smoke testing.

The script copies the first N protein sequences and associated GO term
entries from the training corpus into an HPC-friendly `smoke` directory.
Paths mirror the expectations baked into `configs/data_config/hpc_smoke.yaml`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Dict, Iterator, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from modules.dataloader import parse_ground_truth_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HPC smoke-test subset files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/orcd/home/002/lerchen/code/cafa_proj/cafa6"),
        help="Directory containing the CAFA training data.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Directory with train_sequences.fasta and train_terms.tsv "
        "(default: <dataset-root>/Train).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for the smoke subset "
        "(default: <dataset-root>/smoke).",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=64,
        help="Number of protein entries to retain.",
    )
    parser.add_argument(
        "--sequences-name",
        type=str,
        default="train_sequences.fasta",
        help="Source FASTA filename under --train-dir.",
    )
    parser.add_argument(
        "--terms-name",
        type=str,
        default="train_terms.tsv",
        help="Source GO term TSV filename under --train-dir.",
    )
    parser.add_argument(
        "--output-sequences-name",
        type=str,
        default="train_sequences_smoke.fasta",
        help="Output FASTA filename under --output-dir.",
    )
    parser.add_argument(
        "--output-terms-name",
        type=str,
        default="train_terms_smoke.tsv",
        help="Output GO term TSV filename under --output-dir.",
    )
    return parser.parse_args()


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, str]]:
    header: str | None = None
    seq_lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines).strip()
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, "".join(seq_lines).strip()


def extract_entry_id(header: str) -> str:
    cleaned = header.lstrip(">").strip()
    if not cleaned:
        raise ValueError("Encountered FASTA header without an identifier.")
    first_token = cleaned.split()[0]
    if "|" in first_token:
        tokens = [token for token in first_token.split("|") if token]
        if len(tokens) >= 2 and tokens[1]:
            return tokens[1]
    return first_token


def write_fasta(records: Sequence[Tuple[str, str]], dest: Path) -> None:
    with dest.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            sequence = sequence.replace("\n", "").strip()
            if not sequence:
                raise ValueError(f"Sequence for header '{header}' is empty.")
            handle.write(f"{header}\n{sequence}\n")


def select_sequences_with_terms(
    fasta_path: Path,
    desired_count: int,
    annotated_ids: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    annotated = set(annotated_ids)
    records: List[Tuple[str, str]] = []
    selected_ids: List[str] = []
    for header, sequence in iter_fasta_records(fasta_path):
        if not sequence:
            continue
        entry_id = extract_entry_id(header)
        if entry_id not in annotated:
            continue
        records.append((header, sequence))
        selected_ids.append(entry_id)
        if len(records) >= desired_count:
            break
    return records, selected_ids


def load_terms_lookup(path: Path) -> Dict[str, List[Dict[str, str]]]:
    df = parse_ground_truth_table(path)
    if df.empty:
        raise RuntimeError(f"No GO term rows found in {path}.")
    lookup: Dict[str, List[Dict[str, str]]] = {}
    for entry_id, group in df.groupby("entry_id"):
        lookup[entry_id] = [
            {"EntryID": entry_id, "term": row.term, "aspect": row.aspect} for row in group.itertuples(index=False)
        ]
    return lookup


def write_terms_subset(
    terms_lookup: Dict[str, List[Dict[str, str]]],
    dest_path: Path,
    keep_ids: Sequence[str],
) -> int:
    fieldnames = ["EntryID", "term", "aspect"]
    written = 0
    with dest_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for entry_id in keep_ids:
            for row in terms_lookup.get(entry_id, []):
                writer.writerow(row)
                written += 1
    if written == 0:
        raise RuntimeError("No GO term rows matched the selected sequences; aborting smoke subset creation.")
    return written


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    dataset_root = args.dataset_root.expanduser().resolve()
    train_dir = (args.train_dir or dataset_root / "Train").expanduser().resolve()
    output_dir = (args.output_dir or dataset_root / "smoke").expanduser().resolve()

    sequences_src = train_dir / args.sequences_name
    terms_src = train_dir / args.terms_name
    sequences_dest = output_dir / args.output_sequences_name
    terms_dest = output_dir / args.output_terms_name

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading GO annotations from %s", terms_src)
    terms_lookup = load_terms_lookup(terms_src)
    annotated_ids = list(terms_lookup.keys())

    logging.info("Reading sequences from %s", sequences_src)
    selected_records, keep_ids = select_sequences_with_terms(
        sequences_src,
        args.num_sequences,
        annotated_ids,
    )
    if not selected_records:
        raise RuntimeError(
            "No annotated sequences were found; ensure the FASTA and term files share EntryIDs."
        )
    if len(selected_records) < args.num_sequences:
        logging.warning(
            "Requested %d annotated sequences but only found %d.",
            args.num_sequences,
            len(selected_records),
        )

    write_fasta(selected_records, sequences_dest)
    logging.info("Wrote %d sequences to %s", len(selected_records), sequences_dest)

    logging.info("Filtering GO terms for %d EntryIDs", len(keep_ids))
    written_terms = write_terms_subset(terms_lookup, terms_dest, keep_ids)
    logging.info("Wrote %d GO term rows to %s", written_terms, terms_dest)


if __name__ == "__main__":
    main()
