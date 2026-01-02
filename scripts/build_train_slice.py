"""Create a representative 15% training slice for PF-AGCN experiments.

The script performs a stratified sampling over GO aspect combinations so that the
subset maintains label diversity observed in the full CAFA training set. Source
files and output paths are fixed to the MIT SuperCloud storage layout described
in project docs.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set

TRAIN_ROOT = Path("/orcd/home/002/lerchen/code/cafa_proj/cafa6/Train")
SEQUENCES_PATH = TRAIN_ROOT / "train_sequences.fasta"
TERMS_PATH = TRAIN_ROOT / "train_terms.tsv"
OUTPUT_ROOT = Path("/orcd/home/002/lerchen/code/cafa_proj/cafa6/Train_slice")
SLICE_NAME = "train"

ASPECT_NORMALISER = {
    "C": "C",
    "CC": "C",
    "CELLULARCOMPONENT": "C",
    "F": "F",
    "MF": "F",
    "MOLECULARFUNCTION": "F",
    "P": "P",
    "BP": "P",
    "BIOLOGICALPROCESS": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.15,
        help="Fraction of proteins to retain (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Destination directory for sliced train files.",
    )
    return parser.parse_args()


def _normalise_aspect(value: str) -> str:
    key = value.strip().upper()
    return ASPECT_NORMALISER.get(key, key[:1])


def load_entry_aspects(terms_path: Path) -> Dict[str, Set[str]]:
    """Read term table once to recover aspect membership per entry."""

    entry_aspects: Dict[str, Set[str]] = {}
    with terms_path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            raise RuntimeError(f"{terms_path} appears empty.")
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                continue
            entry_id, _term, aspect = parts[0], parts[1], parts[2]
            normalised = _normalise_aspect(aspect)
            entry_aspects.setdefault(entry_id, set()).add(normalised)
    if not entry_aspects:
        raise RuntimeError("Failed to parse any entries from terms table.")
    return entry_aspects


def _stratum_key(aspects: Iterable[str]) -> str:
    cleaned = sorted(aspects)
    return "".join(cleaned) if cleaned else "NA"


def stratified_sample(
    entry_aspects: Mapping[str, Set[str]],
    *,
    fraction: float,
    seed: int,
) -> List[str]:
    """Sample entry identifiers while preserving aspect ratios."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be within (0, 1].")
    rng = random.Random(seed)
    entry_ids = list(entry_aspects.keys())
    total_entries = len(entry_ids)
    target = max(1, int(round(total_entries * fraction)))

    strata: MutableMapping[str, List[str]] = defaultdict(list)
    for entry_id, aspects in entry_aspects.items():
        strata[_stratum_key(aspects)].append(entry_id)

    counts: Dict[str, int] = {}
    remainders: List[tuple[float, str]] = []
    for key, members in strata.items():
        expected = (len(members) * target) / float(total_entries)
        base = min(len(members), int(math.floor(expected)))
        counts[key] = base
        remainders.append((expected - base, key))

    assigned = sum(counts.values())
    remaining = target - assigned
    remainders.sort(key=lambda item: item[0], reverse=True)
    for _frac, key in remainders:
        if remaining <= 0:
            break
        if counts[key] >= len(strata[key]):
            continue
        counts[key] += 1
        remaining -= 1

    if remaining > 0:
        for key, members in strata.items():
            if remaining <= 0:
                break
            capacity = len(members) - counts.get(key, 0)
            if capacity <= 0:
                continue
            delta = min(capacity, remaining)
            counts[key] += delta
            remaining -= delta
    if remaining > 0:
        raise RuntimeError("Unable to allocate requested subset size; please lower fraction.")

    selected: Set[str] = set()
    for key, members in strata.items():
        k = min(len(members), counts.get(key, 0))
        if k <= 0:
            continue
        chosen = rng.sample(members, k)
        selected.update(chosen)

    if len(selected) < target:
        remaining_ids = [eid for eid in entry_ids if eid not in selected]
        needed = target - len(selected)
        if needed > len(remaining_ids):
            raise RuntimeError("Not enough proteins to satisfy requested subset.")
        selected.update(rng.sample(remaining_ids, needed))

    return sorted(selected)


def write_subset_sequences(entry_ids: Set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep = False
    current_id = None
    with SEQUENCES_PATH.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].strip()
                parts = header.split("|")
                current_id = parts[1] if len(parts) > 1 else header.split()[0]
                keep = current_id in entry_ids
            if keep:
                dst.write(line)


def write_subset_terms(entry_ids: Set[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TERMS_PATH.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        header = src.readline()
        if header:
            dst.write(header)
        for line in src:
            if not line.strip():
                continue
            entry_id = line.split("\t", 1)[0].split()[0]
            if entry_id in entry_ids:
                dst.write(line)


def summarise_subset(entry_ids: Sequence[str], entry_aspects: Mapping[str, Set[str]]) -> str:
    total = len(entry_aspects)
    counts = defaultdict(int)
    for entry_id in entry_ids:
        key = _stratum_key(entry_aspects.get(entry_id, set()))
        counts[key] += 1
    lines = [
        f"Total proteins: {total}",
        f"Subset proteins: {len(entry_ids)} ({len(entry_ids)/total:.2%})",
        "Aspect combination counts:",
    ]
    for key in sorted(counts):
        lines.append(f"  {key or 'NA'}: {counts[key]}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entry_aspects = load_entry_aspects(TERMS_PATH)
    selected_ids = stratified_sample(
        entry_aspects,
        fraction=args.fraction,
        seed=args.seed,
    )
    selected_set = set(selected_ids)

    output_root = args.output_root
    seq_out = output_root / f"{SLICE_NAME}_sequences.fasta"
    terms_out = output_root / f"{SLICE_NAME}_terms.tsv"
    output_root.mkdir(parents=True, exist_ok=True)

    write_subset_sequences(selected_set, seq_out)
    write_subset_terms(selected_set, terms_out)

    summary = summarise_subset(selected_ids, entry_aspects)
    print(summary)
    print(f"Wrote sequences to {seq_out}")
    print(f"Wrote terms to {terms_out}")


if __name__ == "__main__":
    main()
