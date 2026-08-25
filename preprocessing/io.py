"""
Raw file I/O for the IMU air-writing dataset.

Directory layout (confirmed):
    Dataset/
      capital letters/<CHAR>/<PARTICIPANT_ID>/<CHAR>-<NN>.txt
      small letters/<char>/<PARTICIPANT_ID>/<char>-<NN>.txt

Canonical schema is 12 comma-separated columns, no header (see config.py).
Some collection sessions insert an empty field after the timestamp
(13 columns: label, ts, '', sensors..., flag). That slot is occasionally
corrupted (e.g. ``0î`` from a bad encoding round-trip). Both cases are
normalized back to 12 columns here — pandas.read_csv is intentionally
avoided because ~280k tiny instance files make per-file DataFrame
construction dominate runtime.

Some sessions also omit the label column entirely (leading comma) or
write the wrong case (``p`` under ``capital letters/P/``). The directory
path is treated as the canonical label in those cases — see
``get_char_label(..., expected=...)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from config import (
    CAPITAL_DIR_NAME,
    FLAG_COL,
    LABEL_COL,
    NUM_RAW_COLS,
    SENSOR_COLS,
    SMALL_DIR_NAME,
    TIMESTAMP_COL,
)

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass(frozen=True)
class RawInstance:
    char_label: str
    participant_id: str
    filepath: Path


def iter_raw_instances(raw_root: Path) -> Iterator[RawInstance]:
    """Walk the raw dataset directory and yield one entry per instance file.

    Covers BOTH "capital letters" and "small letters" (the original
    prototype only ever walked "capital letters" -- see config.py docstring
    on the 52-class fix).
    """
    raw_root = Path(raw_root)
    for case_dir_name in (CAPITAL_DIR_NAME, SMALL_DIR_NAME):
        case_dir = raw_root / case_dir_name
        if not case_dir.is_dir():
            continue
        for char_dir in sorted(case_dir.iterdir()):
            if not char_dir.is_dir():
                continue
            char_label = char_dir.name
            if len(char_label) != 1 or not char_label.isalpha():
                continue
            for participant_dir in sorted(char_dir.iterdir()):
                if not participant_dir.is_dir():
                    continue
                participant_id = participant_dir.name
                for f in sorted(participant_dir.iterdir()):
                    if f.suffix == ".txt":
                        yield RawInstance(char_label, participant_id, f)


def _is_spurious_post_timestamp_field(value: str) -> bool:
    """True when column 2 is a corrupted empty slot, not a real sensor reading."""
    value = value.strip()
    if not value:
        return True
    if value.lower() in ("ovf", "nan"):
        return False
    try:
        float(value)
    except ValueError:
        return True
    else:
        return False


def _normalize_row(parts: list[str], filepath: Path, line_no: int) -> list[str]:
    """Coerce a split CSV row to exactly NUM_RAW_COLS fields.

    Known quirk: an empty field right after the timestamp (13 cols).
    Also tolerate a single trailing empty field.
    Some sessions write the label with trailing whitespace (e.g. ``'M '``).
    """
    if parts:
        parts[0] = parts[0].strip()
    n = len(parts)
    if n == NUM_RAW_COLS:
        return parts

    if n == NUM_RAW_COLS + 1:
        # Extra empty after timestamp: label, ts, '', ax, ...
        if _is_spurious_post_timestamp_field(parts[2]):
            return parts[:2] + parts[3:]
        # Trailing delimiter / empty last field
        if parts[-1] == "":
            return parts[:-1]
        # Empty immediately before the flag (sensors still contiguous)
        if parts[FLAG_COL] == "" and parts[-1] != "":
            return parts[:FLAG_COL] + parts[FLAG_COL + 1 :]

    raise ValueError(
        f"{filepath}:{line_no}: expected {NUM_RAW_COLS} columns, got {n}  {','.join(parts)}"
    )


def load_raw_file(filepath: Path) -> np.ndarray:
    """Load a single raw instance file as a (rows, 12) object array.

    Sensor columns may contain literal strings like 'ovf' or 'nan' in
    addition to numeric values, so this is read as dtype=object and cleaned
    downstream (preprocessing.clean). Rows outside [40, 80] should be
    filtered by the caller before this is used for training (see
    preprocessing.filters).
    """
    filepath = Path(filepath)
    rows: list[list[str]] = []
    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            rows.append(_normalize_row(parts, filepath, line_no))

    if not rows:
        raise ValueError(f"{filepath}: empty file")

    return np.asarray(rows, dtype=object)


def get_char_label(raw_matrix: np.ndarray, expected: str | None = None) -> str:
    """Return the (verified-consistent) character label for an instance.

    The label is redundant on every row; inconsistent labels indicate corruption.

    When *expected* is supplied (the label implied by the directory layout,
    e.g. ``capital letters/P/...`` -> ``'P'``), it is used as the canonical
    label when the file column is blank (some sessions, e.g. S120, omit it)
    or differs only by case (some sessions write ``p`` under ``P/``).
    A genuinely different character still raises.
    """
    labels = np.unique(raw_matrix[:, LABEL_COL])
    if len(labels) != 1:
        raise ValueError(f"Inconsistent label column values: {labels}")
    label = str(labels[0]).strip()
    if not label:
        if expected is not None:
            return expected
        raise ValueError("Empty label column")
    if len(label) != 1:
        raise ValueError(f"Unexpected multi-character label: {label!r}")
    if expected is not None:
        if label == expected or label.lower() == expected.lower():
            return expected
        raise ValueError(
            f"Label column {label!r} disagrees with directory label {expected!r}"
        )
    return label


def get_sensor_matrix(raw_matrix: np.ndarray) -> np.ndarray:
    """Extract the 9 sensor channels as an object array (rows, 9)."""
    return raw_matrix[:, SENSOR_COLS]


def get_timestamps(raw_matrix: np.ndarray) -> np.ndarray:
    """Parse the timestamp column to float seconds-since-first-sample.

    Validated (fixed after review) so a corrupt/out-of-order timestamp
    column fails loudly here instead of silently producing garbage in
    segment.resample_to_uniform_grid()'s np.interp() call later -- np.interp
    assumes its x-coordinates are increasing and will not error on a
    violation, it will just interpolate nonsense.
    """
    parsed = [
        datetime.strptime(str(v).strip(), _TS_FMT)
        for v in raw_matrix[:, TIMESTAMP_COL]
    ]
    t0 = parsed[0]
    ts = np.asarray([(t - t0).total_seconds() for t in parsed], dtype=np.float64)

    if not np.all(np.isfinite(ts)):
        raise ValueError("Timestamp column contains non-finite values after parsing")
    if np.any(np.diff(ts) < 0):
        raise ValueError(
            "Timestamp column is not monotonically non-decreasing "
            "(out-of-order rows) -- refusing to resample/trust this file"
        )
    return ts