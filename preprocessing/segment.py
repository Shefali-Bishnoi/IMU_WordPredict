"""Sequence-length normalization and per-channel scaling for training and inference."""
from __future__ import annotations

import numpy as np

from config import MODEL_SEQ_LEN, NOMINAL_DT_SECONDS, NUM_SENSOR_CHANNELS, PAD_TARGET_LEN


def pad_or_trim(sensor_matrix: np.ndarray, target_len: int = PAD_TARGET_LEN) -> np.ndarray:
    """Pad or center-trim to exactly target_len rows; padding repeats boundary rows."""
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
    """Crop to a centered window of window_len rows."""
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
    """Linearly interpolate onto a fixed-rate grid using real timestamps."""
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
    """Per-channel mean/std; compute on training data only."""
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
    """Shared preprocessing: pad/trim, window, normalize."""
    if sensor_matrix.shape[1] != NUM_SENSOR_CHANNELS:
        raise ValueError(
            f"Expected {NUM_SENSOR_CHANNELS} sensor channels, got {sensor_matrix.shape[1]}"
        )
    padded = pad_or_trim(sensor_matrix, target_len=pad_target_len)
    windowed = center_window(padded, window_len=window_len)
    if mean is not None and std is not None:
        windowed = apply_normalization(windowed, mean, std)
    return windowed.astype(np.float32)