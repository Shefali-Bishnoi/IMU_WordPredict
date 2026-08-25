"""
Build the processed dataset from raw IMU instance files.

Usage:
    python -m data.build_dataset --raw-root data/raw/Dataset --resample
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    DATASET_MANIFEST_PATH,
    MAX_RAW_LINES,
    MIN_RAW_LINES,
    MODEL_SEQ_LEN,
    NORM_STATS_PATH,
    NUM_CLASSES,
    NUM_SENSOR_CHANNELS,
    PAD_TARGET_LEN,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    SPLIT_CONFIG_PATH,
    TEST_NPZ_PATH,
    TRAIN_NPZ_PATH,
    index_to_label,
    label_to_index,
)
from preprocessing.clean import clean_sensor_matrix  # noqa: E402
from preprocessing.io import (  # noqa: E402
    get_char_label,
    get_sensor_matrix,
    get_timestamps,
    iter_raw_instances,
    load_raw_file,
)
from preprocessing.segment import preprocess, resample_to_uniform_grid  # noqa: E402
from preprocessing.split import assert_disjoint_splits, build_or_load_split  # noqa: E402


def _process_raw(raw: np.ndarray, resample: bool) -> np.ndarray | None:
    """Returns a clean (rows, 9) float matrix, or None if the file should
    be skipped (outside the [MIN_RAW_LINES, MAX_RAW_LINES] band, matching
    the original filtering rule, or corrupt)."""
    if not (MIN_RAW_LINES <= raw.shape[0] <= MAX_RAW_LINES):
        return None
    sensor = clean_sensor_matrix(get_sensor_matrix(raw))
    if resample:
        ts = get_timestamps(raw)
        sensor = resample_to_uniform_grid(sensor, ts)
        if sensor.shape[0] < 5:
            return None
    return sensor


def _check_class_coverage(labels: dict[str, list[int]]) -> None:
    """Fail fast if a class has zero TRAIN examples (a 52-unit output
    layer trained on <52 observed classes is a silent architecture
    mismatch, not a warning-level issue). Val/test missing a class is
    logged since it only makes that class's val/test metrics undefined,
    not the model itself wrong."""
    for bucket in ("train", "val", "test"):
        present = set(labels[bucket])
        missing = sorted(set(range(NUM_CLASSES)) - present)
        if missing:
            missing_chars = [index_to_label(i) for i in missing]
            msg = f"[class-check] '{bucket}' split is missing classes: {missing_chars}"
            if bucket == "train":
                raise RuntimeError(
                    msg + " -- cannot train a faithful 52-class model without this. "
                    "Check --raw-root, the length-band filter, or the participant split."
                )
            print(f"[warn] {msg} (per-class metrics for these will be undefined on this split)")
    print(f"[class-check] all {NUM_CLASSES} classes present in train: PASS")


def _report_class_imbalance(labels: dict[str, list[int]]) -> dict:
    """Print per-class counts for the train split; return counts for the manifest."""
    counts_by_bucket: dict[str, dict[str, int]] = {}
    for bucket in ("train", "val", "test"):
        counter = Counter(labels[bucket])
        counts_by_bucket[bucket] = {
            index_to_label(i): int(counter.get(i, 0)) for i in range(NUM_CLASSES)
        }

    train_counts = counts_by_bucket["train"]
    values = list(train_counts.values())
    lo_char, lo_n = min(train_counts.items(), key=lambda kv: kv[1])
    hi_char, hi_n = max(train_counts.items(), key=lambda kv: kv[1])
    ratio = (hi_n / lo_n) if lo_n > 0 else float("inf")

    print("\n[class-imbalance] train-split per-class counts (min -> max):")
    for char, n in sorted(train_counts.items(), key=lambda kv: kv[1]):
        bar = "#" * max(1, round(40 * n / hi_n)) if hi_n else ""
        print(f"    {char:>3s}: {n:6d}  {bar}")
    print(
        f"[class-imbalance] smallest class: '{lo_char}' ({lo_n}), "
        f"largest class: '{hi_char}' ({hi_n}), "
        f"max/min ratio: {ratio:.2f}x, mean: {np.mean(values):.1f}, "
        f"std: {np.std(values):.1f}\n"
    )
    return counts_by_bucket


def build(raw_root: Path, resample: bool, seed: int) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    split = build_or_load_split(raw_root, SPLIT_CONFIG_PATH, seed=seed)
    assert_disjoint_splits(split)
    train_ids, val_ids, test_ids = set(split["train"]), set(split["val"]), set(split["test"])
    print(f"[split] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} participants")

    # Cleaned, UNPADDED sensor matrices (variable length, 40-80 rows).
    # Normalization stats are computed from these directly -- never from
    # the padded/windowed arrays -- so pad_or_trim()'s boundary-repeated
    # rows can't bias mean/std toward whatever a short sequence's edge
    # values happened to be.
    cleaned: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    labels: dict[str, list] = {"train": [], "val": [], "test": []}
    meta: dict[str, list] = {"train": [], "val": [], "test": []}

    skipped, corrupt = 0, 0
    n_total = 0
    n_done = 0
    warns_printed = 0
    t0 = time.perf_counter()
    progress_every = 5000

    for inst in iter_raw_instances(raw_root):
        n_total += 1
        if inst.participant_id in train_ids:
            bucket = "train"
        elif inst.participant_id in val_ids:
            bucket = "val"
        elif inst.participant_id in test_ids:
            bucket = "test"
        else:
            continue  # participant not in the frozen split (new data since split was built)

        try:
            raw = load_raw_file(inst.filepath)
            char_label = get_char_label(raw, expected=inst.char_label)
            sensor = _process_raw(raw, resample=resample)
        except Exception as e:  # noqa: BLE001 - log and skip, don't crash the whole build
            if warns_printed < 20:
                print(f"[warn] failed on {inst.filepath}: {e}")
                warns_printed += 1
            corrupt += 1
            continue

        if sensor is None:
            skipped += 1
            continue

        cleaned[bucket].append(sensor.astype(np.float32))
        labels[bucket].append(label_to_index(char_label))
        meta[bucket].append(f"{inst.participant_id}:{inst.char_label}:{inst.filepath.name}")
        n_done += 1

        if n_done % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            print(
                f"[progress] kept={n_done} scanned={n_total} "
                f"skipped={skipped} corrupt={corrupt} "
                f"{rate:.0f} files/s ({elapsed:.1f}s)"
            )

    print(f"[scan] total_files={n_total} kept={n_done} skipped_len_filter={skipped} corrupt={corrupt}")
    print(f"[timing] {(time.perf_counter() - t0):.1f}s")
    for b in ("train", "val", "test"):
        print(f"[bucket] {b}: {len(cleaned[b])} instances")
        if not cleaned[b]:
            raise RuntimeError(
                f"No usable instances in '{b}' split -- check --raw-root points at the "
                f"real Dataset/ folder."
            )

    _check_class_coverage(labels)
    class_counts = _report_class_imbalance(labels)

    # --- Normalization stats from REAL, UNPADDED train sensor rows -------
    train_real_rows = np.concatenate(cleaned["train"], axis=0)  # (total_real_rows, 9)
    mean = train_real_rows.mean(axis=0)
    std = train_real_rows.std(axis=0)
    std[std < 1e-8] = 1.0  # guard against a dead/constant channel
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "seq_len": MODEL_SEQ_LEN,
                "pad_target_len": PAD_TARGET_LEN,
                "n_channels": NUM_SENSOR_CHANNELS,
                "resample": resample,
                "computed_from": "train real (unpadded) sensor rows only",
                "n_train_real_rows": int(train_real_rows.shape[0]),
            },
            f,
            indent=2,
        )
    print(
        f"[norm] stats computed on {train_real_rows.shape[0]} real train rows "
        f"(padding excluded) -> {NORM_STATS_PATH}"
    )
    n_train_real_rows = int(train_real_rows.shape[0])
    del train_real_rows  # free memory before the windowing pass below

    # --- Pad/window/normalize each instance via the shared preprocess() --
    # This is the exact same function realtime.py calls at inference time,
    # so pad -> window -> normalize happens identically in both places.
    saved_shapes = {}
    for bucket, path in (("train", TRAIN_NPZ_PATH), ("val", PROCESSED_DIR / "val.npz"), ("test", TEST_NPZ_PATH)):
        windowed = [
            preprocess(sensor, mean=mean, std=std, pad_target_len=PAD_TARGET_LEN, window_len=MODEL_SEQ_LEN)
            for sensor in cleaned[bucket]
        ]
        X = np.stack(windowed).astype(np.float32)
        y = np.array(labels[bucket], dtype=np.int64)
        m = np.array(meta[bucket], dtype=object)
        np.savez_compressed(path, X=X, y=y, meta=m)
        saved_shapes[bucket] = list(X.shape)
        print(f"[save] {bucket}: X={X.shape} y={y.shape} -> {path}")

    # --- Dataset manifest (new) -------------------------------------------
    # Records exactly what data configuration produced these .npz files, so
    # any later experiment/report can be traced back to it instead of
    # guessing from file timestamps.
    manifest = {
        "raw_root": str(raw_root),
        "resample": resample,
        "seed": seed,
        "min_raw_lines": MIN_RAW_LINES,
        "max_raw_lines": MAX_RAW_LINES,
        "pad_target_len": PAD_TARGET_LEN,
        "model_seq_len": MODEL_SEQ_LEN,
        "num_sensor_channels": NUM_SENSOR_CHANNELS,
        "num_classes": NUM_CLASSES,
        "scan": {
            "total_files_scanned": n_total,
            "kept": n_done,
            "skipped_length_filter": skipped,
            "corrupt": corrupt,
        },
        "participants": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "instances": {b: len(cleaned[b]) for b in ("train", "val", "test")},
        "saved_shapes": saved_shapes,
        "n_train_real_sensor_rows": n_train_real_rows,
        "class_counts": class_counts,
    }
    DATASET_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[save] dataset manifest -> {DATASET_MANIFEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--resample", action="store_true", help="Resample onto a fixed-rate grid using real timestamps (ActionPlan.md 4.4 option b)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.raw_root, resample=args.resample, seed=args.seed)