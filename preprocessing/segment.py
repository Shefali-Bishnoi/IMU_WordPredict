"""
Sequence-length normalization and per-channel scaling.

This is the "one shared preprocess() function" required by ActionPlan.md
Priority 0 / Step 15: it must run identically at training time (on files
loaded from disk) and at real-time inference time (on a live buffer). It
operates purely on numpy arrays, never touching disk itself.
"""
from __future__ import annotations

import numpy as np

from config import MODEL_SEQ_LEN, NOMINAL_DT_SECONDS, NUM_SENSOR_CHANNELS, PAD_TARGET_LEN


def pad_or_trim(sensor_matrix: np.ndarray, target_len: int = PAD_TARGET_LEN) -> np.ndarray:
    """Pad short sequences (by edge-repeating the boundary sample) or trim
    long ones to exactly `target_len` rows, split symmetrically top/bottom.

    Unlike the original prototype (which padded with a malformed 11-column
    row missing the timestamp field, see ActionPlan.md 4.4), this operates
    purely on the already-extracted 9-channel float matrix, so there is no
    schema to get wrong -- padding rows repeat the nearest real sensor
    reading rather than injecting synthetic zeros, which better preserves
    the "at rest" motion signature than a hard zero would.
    """
    n_rows = sensor_matrix.shape[0]
    if n_rows == target_len:
        return sensor_matrix.copy()

    if n_rows > target_len:
        # Center-trim.
        excess = n_rows - target_len
        start = excess // 2
        return sensor_matrix[start:start + target_len]

    # Pad: split the deficit across top and bottom, repeating the nearest
    # boundary row (first row above, last row below).
    deficit = target_len - n_rows
    pad_above = deficit // 2
    pad_below = deficit - pad_above
    top_pad = np.repeat(sensor_matrix[0:1], pad_above, axis=0)
    bottom_pad = np.repeat(sensor_matrix[-1:], pad_below, axis=0)
    return np.concatenate([top_pad, sensor_matrix, bottom_pad], axis=0)


def center_window(sensor_matrix: np.ndarray, window_len: int = MODEL_SEQ_LEN) -> np.ndarray:
    """Crop a (>= window_len, channels) matrix to a centered window.

    NOTE (fixed after review): MODEL_SEQ_LEN now equals PAD_TARGET_LEN (see
    config.py), so in the standard pipeline this is a no-op -- it used to
    crop an already-padded 80-row sequence down to 50, which discarded up
    to ~30 real sensor rows for any instance close to 80 real rows. The
    function is kept (rather than removed) because it's still useful if you
    deliberately want a smaller fixed window for a future architecture
    experiment; just don't set MODEL_SEQ_LEN < PAD_TARGET_LEN without
    re-reading that trade-off.
    """
    n_rows = sensor_matrix.shape[0]
    if n_rows < window_len:
        raise ValueError(f"Cannot window {n_rows} rows to {window_len}")
    start = (n_rows - window_len) // 2
    return sensor_matrix[start:start + window_len]


def resample_to_uniform_grid(
    sensor_matrix: np.ndarray,
    timestamps: np.ndarray,
    dt: float = NOMINAL_DT_SECONDS,
) -> np.ndarray:
    """Linearly interpolate onto a fixed-rate grid using real timestamps.

    This implements option (b) from ActionPlan.md section 4.4: the actual
    per-sample interval is ~18-19ms (~52-56Hz), not a clean 20ms/50Hz, so
    treating raw rows as if they were evenly spaced silently absorbs timing
    jitter into the signal. This function is opt-in (see build_dataset.py
    --resample flag) so both handling strategies can be A/B compared per
    ActionPlan.md 19.5, rather than one being silently assumed correct.

    Validated (fixed after review): np.interp silently produces garbage if
    its x-coordinates (timestamps) aren't sorted/finite -- callers should
    already get monotonic timestamps from preprocessing.io.get_timestamps,
    but this is checked again here defensively since this function can also
    be called directly (e.g. from a live inference buffer) without going
    through that loader.
    """
    if sensor_matrix.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"sensor_matrix has {sensor_matrix.shape[0]} rows but timestamps "
            f"has {timestamps.shape[0]} -- must match 1:1"
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contains non-finite values")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be monotonically non-decreasing")

    if timestamps[-1] <= timestamps[0]:
        return sensor_matrix  # degenerate/duplicate timestamps, skip
    n_channels = sensor_matrix.shape[1]
    grid = np.arange(timestamps[0], timestamps[-1], dt)
    out = np.empty((len(grid), n_channels), dtype=np.float64)
    for c in range(n_channels):
        out[:, c] = np.interp(grid, timestamps, sensor_matrix[:, c])
    return out


def compute_norm_stats(all_sensor_matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a (N, seq_len, channels) array.

    Must be computed on TRAIN data only and persisted -- never recomputed
    at inference time (ActionPlan.md Step 15 / Definition of Done).
    """
    flat = all_sensor_matrices.reshape(-1, all_sensor_matrices.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0  # guard against a dead/constant channel
    return mean, std


def apply_normalization(sensor_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (sensor_matrix - mean) / std


def preprocess(
    sensor_matrix: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    pad_target_len: int = PAD_TARGET_LEN,
    window_len: int = MODEL_SEQ_LEN,
) -> np.ndarray:
    """The single shared preprocessing entry point.

    Input: a clean float64 (rows, NUM_SENSOR_CHANNELS) matrix (already
    passed through preprocessing.clean.clean_sensor_matrix).
    Output: a (window_len, NUM_SENSOR_CHANNELS) float32 matrix ready for
    the model, normalized if mean/std are provided.

    This exact function is called both by data/build_dataset.py (training)
    and inference/realtime.py (serving), so there is zero drift between
    the two by construction.
    """
    if sensor_matrix.shape[1] != NUM_SENSOR_CHANNELS:
        raise ValueError(
            f"Expected {NUM_SENSOR_CHANNELS} sensor channels, got {sensor_matrix.shape[1]}"
        )
    padded = pad_or_trim(sensor_matrix, target_len=pad_target_len)
    windowed = center_window(padded, window_len=window_len)
    if mean is not None and std is not None:
        windowed = apply_normalization(windowed, mean, std)
    return windowed.astype(np.float32)