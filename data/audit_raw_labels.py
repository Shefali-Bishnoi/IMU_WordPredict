"""Audit raw-file labels against their directory labels.

The audit reports exact, empty, case-mismatched, inconsistent, malformed,
and out-of-range files without modifying sensor data or building the dataset.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LABEL_COL, MAX_RAW_LINES, MIN_RAW_LINES  # noqa: E402
from preprocessing.io import iter_raw_instances, load_raw_file  # noqa: E402


def classify(filepath: Path, expected: str) -> tuple[str, str]:
    """Returns (category, detail_string)."""
    try:
        raw = load_raw_file(filepath)
    except Exception as e:  # noqa: BLE001
        return "BAD_LOAD_ERROR", str(e)

    if not (MIN_RAW_LINES <= raw.shape[0] <= MAX_RAW_LINES):
        return "OUT_OF_LENGTH_BAND", f"rows={raw.shape[0]}"

    labels = np.unique(raw[:, LABEL_COL])
    if len(labels) != 1:
        return "BAD_INCONSISTENT_FILE", f"values={labels.tolist()}"

    label = str(labels[0]).strip()
    if not label:
        return "OK_EMPTY_LABEL", "resolved via directory label"
    if len(label) != 1:
        return "BAD_LOAD_ERROR", f"multi-char label={label!r}"
    if label == expected:
        return "OK_EXACT_MATCH", ""
    if label.lower() == expected.lower():
        return "OK_CASE_MISMATCH", f"file={label!r} dir={expected!r}"
    return "BAD_DIFFERENT_CHAR", f"file={label!r} dir={expected!r}"


def main(raw_root: Path) -> None:
    counts: Counter[str] = Counter()
    mismatch_folders: dict[tuple[str, str], list[str]] = defaultdict(list)
    folder_totals: Counter[tuple[str, str]] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    n = 0
    for inst in iter_raw_instances(raw_root):
        n += 1
        key = (inst.char_label, inst.participant_id)
        folder_totals[key] += 1
        cat, detail = classify(inst.filepath, inst.char_label)
        counts[cat] += 1
        if len(examples[cat]) < 5:
            examples[cat].append(f"{inst.filepath} ({detail})" if detail else str(inst.filepath))
        if cat == "BAD_DIFFERENT_CHAR":
            mismatch_folders[key].append(detail)
        if n % 20000 == 0:
            print(f"[audit] scanned {n} files...")

    print(f"\n[audit] total files scanned: {n}\n")
    print("=" * 70)
    print("CATEGORY BREAKDOWN")
    print("=" * 70)
    for cat, cnt in counts.most_common():
        pct = 100 * cnt / n if n else 0.0
        print(f"{cat:24s} {cnt:8d}   ({pct:.2f}%)")
        for ex in examples[cat]:
            print(f"    e.g. {ex}")

    if mismatch_folders:
        print("\n" + "=" * 70)
        print("BAD_DIFFERENT_CHAR -- folders affected (possible systematic issue)")
        print("=" * 70)
        rows = []
        for (char_label, participant), details in mismatch_folders.items():
            total = folder_totals[(char_label, participant)]
            frac = len(details) / total if total else 0.0
            rows.append((frac, char_label, participant, len(details), total, details[0]))
        rows.sort(reverse=True)
        for frac, char_label, participant, n_bad, total, sample_detail in rows[:40]:
            flag = "  <-- LIKELY SYSTEMATIC (check this folder manually)" if frac >= 0.5 else ""
            print(
                f"  {char_label}/{participant}: {n_bad}/{total} files mismatched "
                f"({frac:.0%}) e.g. {sample_detail}{flag}"
            )
        if len(rows) > 40:
            print(f"  ... and {len(rows) - 40} more folders")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(
        "OK_* categories need no action -- build_dataset.py already resolves them\n"
        "to the directory label.\n\n"
        "BAD_* categories are correctly SKIPPED as corrupt by build_dataset.py --\n"
        "this is the safe default because guessing which side (file vs directory)\n"
        "is correct risks silently training on a wrong label.\n\n"
        "Folders flagged 'LIKELY SYSTEMATIC' above are worth opening 2-3 files to\n"
        "look at manually (it may indicate a whole session/participant recorded\n"
        "under the wrong character), but you do NOT need to resolve those before\n"
        "running the full build -- skip-by-default already handles them safely.\n"
        "This audit exists so you see the full scope up front instead of learning\n"
        "about each category one warning at a time."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    main(args.raw_root)