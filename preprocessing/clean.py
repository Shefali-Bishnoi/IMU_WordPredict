"""Replace invalid ovf/nan/unparsable sensor values with ffill/bfill per channel."""
from __future__ import annotations

import numpy as np


def _to_float_or_nan(value) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.nan
    try:
        f = float(value)
        if np.isnan(f):
            return np.nan
        return f
    except (TypeError, ValueError):
        return np.nan


def _ffill_bfill_1d(col: np.ndarray) -> np.ndarray:
    """In-place-safe ffill then bfill for a 1-D float array with NaNs."""
    out = col.copy()
    valid = ~np.isnan(out)
    if not valid.any():
        out[:] = 0.0
        return out

    # Forward-fill
    idx = np.where(valid, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]

    # Back-fill remaining leading NaNs
    valid = ~np.isnan(out)
    if not valid.all():
        idx = np.where(valid, np.arange(len(out)), len(out) - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        out = out[idx]

    out[np.isnan(out)] = 0.0
    return out


def clean_sensor_matrix(sensor_matrix: np.ndarray) -> np.ndarray:
    """Convert an (rows, channels) object array to clean float64.

    Invalid cells ('ovf', 'nan', empty, unparsable) are filled per-channel
    using forward-fill then back-fill along the time axis; if an entire
    channel is invalid it is filled with 0.0.
    """
    n_rows, n_channels = sensor_matrix.shape
    float_matrix = np.empty((n_rows, n_channels), dtype=np.float64)
    for c in range(n_channels):
        col = np.fromiter(
            (_to_float_or_nan(v) for v in sensor_matrix[:, c]),
            dtype=np.float64,
            count=n_rows,
        )
        float_matrix[:, c] = _ffill_bfill_1d(col)
    return float_matrix
