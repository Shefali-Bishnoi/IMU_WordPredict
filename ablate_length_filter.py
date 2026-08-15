"""
ablate_length_filter.py

Runs data/build_dataset.py + train.py + evaluate.py (--arch tcn) for each
candidate (MIN_RAW_LINES, MAX_RAW_LINES) pair, WITHOUT permanently editing
config.py -- overrides are applied via environment variables that
config.py reads at import time (see the small config.py patch below this
script). Produces a single comparison table (overall + per-class F1 for
the classes hit hardest under 40-80: m, c, E, V, L, U, q) so you decide
from real accuracy, not just retention counts.

Usage:
    python ablate_length_filter.py --raw-root "D:\\...\\Dataset"

WARNING: each candidate = one full build_dataset.py pass (~15-50 min on
your machine, per the output.md timing) + one TCN train (~15-55 min) +
one evaluate. Two candidates ~= 1-2 hours. Don't run this unattended
without checking disk space -- each build overwrites data/processed/*.npz
and each candidate's model overwrites models/artifacts/tcn/tcn.keras, so
this script copies the metrics.json out to a per-candidate name before
moving to the next candidate (the .npz/.keras files themselves are NOT
kept per-candidate -- add that if you want to keep every trained model,
not just the metrics).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    (20, 100),
    (30, 120),
    (40, 80),  # current, included as the control
]

WATCH_CLASSES = ["m", "c", "E", "V", "L", "U", "q", "p", "P", "z", "k"]

RESULTS_DIR = Path("experiments/length_filter_ablation")


def run(cmd: list[str], env: dict) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main(raw_root: str, resample: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for min_len, max_len in CANDIDATES:
        tag = f"{min_len}-{max_len}"
        print(f"\n{'='*70}\nCANDIDATE RANGE: {tag}\n{'='*70}")

        env = os.environ.copy()
        env["WORDPREDICT_MIN_RAW_LINES"] = str(min_len)
        env["WORDPREDICT_MAX_RAW_LINES"] = str(max_len)

        build_cmd = [sys.executable, "-m", "data.build_dataset", "--raw-root", raw_root]
        if resample:
            build_cmd.append("--resample")
        run(build_cmd, env)

        run([sys.executable, "train.py", "--arch", "tcn"], env)
        run([sys.executable, "evaluate.py", "--arch", "tcn"], env)

        # Stash this candidate's metrics before the next candidate overwrites them.
        src = Path("experiments/tcn_metrics.json")
        dst = RESULTS_DIR / f"tcn_metrics_{tag}.json"
        shutil.copy(src, dst)

        with open(dst) as f:
            m = json.load(f)
        summary.append((tag, m))

    # --- Comparison table -------------------------------------------------
    print(f"\n{'='*100}\nLENGTH FILTER ABLATION -- RESULTS\n{'='*100}")
    header = f"{'Range':<10}{'MacroF1':>10}{'Accuracy':>10}" + "".join(f"{c+'_F1':>8}" for c in WATCH_CLASSES)
    print(header)
    print("-" * len(header))
    for tag, m in summary:
        row = f"{tag:<10}{m['macro_f1']*100:>9.2f}%{m['accuracy']*100:>9.2f}%"
        for c in WATCH_CLASSES:
            f1 = m["per_class"].get(c, {}).get("f1", float("nan"))
            row += f"{f1*100:>7.1f}%"
        print(row)

    out_path = RESULTS_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump([{"range": tag, "metrics": m} for tag, m in summary], f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--resample", action="store_true")
    args = parser.parse_args()
    main(args.raw_root, args.resample)