"""
Real-time inference path for a single new marker stroke (ActionPlan.md
section 15). Loads the SAME preprocessing function and the SAME persisted
normalization stats used at training time -- nothing is recomputed here.

Output contract (Priority 2, section 10.1): the full probability vector is
always returned, never collapsed to argmax internally. Callers (e.g. a
future beam decoder) decide what to do with it.

Priority 1 change: CharacterRecognizer now loads its model via
config.model_path(arch) instead of a hardcoded BASELINE_MODEL_PATH, and
defaults to arch="tcn" -- the Priority 1 winner (highest macro F1, lowest
latency, smallest model; see experiments/architecture_comparison.md).
Normalization stats (config.NORM_STATS_PATH) are architecture-independent
(computed once from raw sensor rows during data/build_dataset.py) so they
are NOT parameterized by arch -- every architecture was trained on the
exact same normalized inputs, which is what makes the Priority 1
comparison valid in the first place.

If you later re-run Priority 1 (new data, new hyperparameters) and a
different architecture wins, change DEFAULT_SERVING_ARCH below -- that is
the single place this decision lives, so nothing else in this file (or in
app/main.py, if it just does `CharacterRecognizer()` with no arguments)
needs to change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    NORM_STATS_PATH,
    NUM_CLASSES,
    index_to_label,
    model_path,
)
from preprocessing.clean import clean_sensor_matrix
from preprocessing.segment import preprocess

# Priority 1 winner (ActionPlan.md 9.3): TCN had the highest macro F1,
# the lowest latency, AND the smallest model of the three architectures
# compared -- a clean sweep, not a trade-off call. See
# experiments/architecture_comparison.md for the numbers this was decided
# from. Change this constant (not call sites) if a future Priority 1 rerun
# picks a different winner.
DEFAULT_SERVING_ARCH = "tcn"


class CharacterRecognizer:
    """Loads the global model + normalization stats once; call .predict()
    per stroke. Designed to be instantiated once at server startup (see
    app/main.py), not per-request.

    Priority 1 change: takes an optional `arch` argument (default
    DEFAULT_SERVING_ARCH = "tcn"). Existing code that calls
    `CharacterRecognizer()` with no arguments automatically picks up the
    Priority 1 winner with no changes needed. Pass `arch="cnn_lstm"` or
    `arch="cnn_bilstm"` explicitly if you want to serve a different
    architecture (e.g. for a side-by-side demo).
    """

    def __init__(
        self,
        arch: str = DEFAULT_SERVING_ARCH,
        model_file: Path | None = None,
        norm_stats_path: Path = NORM_STATS_PATH,
    ):
        self.arch = arch
        # model_file lets a caller override the path directly (e.g. to
        # load a specific checkpoint by hand); normal use just passes
        # `arch` and gets the right file via config.model_path(arch).
        resolved_model_path = model_file if model_file is not None else model_path(arch)
        self.model = tf.keras.models.load_model(resolved_model_path)
        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float64)
        self.std = np.array(stats["std"], dtype=np.float64)
        self.seq_len = stats["seq_len"]
        print(f"[realtime] serving architecture={arch!r} from {resolved_model_path}")

    def predict(self, raw_sensor_matrix: np.ndarray, top_k: int = 5) -> dict:
        """raw_sensor_matrix: (rows, 9) array, may contain 'ovf'/'nan'
        strings straight off the device buffer -- same as a raw file's
        sensor columns. Returns the full probability contract, not a
        single character."""
        cleaned = clean_sensor_matrix(np.asarray(raw_sensor_matrix, dtype=object))
        x = preprocess(cleaned, mean=self.mean, std=self.std, window_len=self.seq_len)
        x = x[np.newaxis, ...]  # batch dim

        probs = self.model.predict(x, verbose=0)[0]
        top_k_idx = np.argsort(probs)[::-1][:top_k]

        return {
            "architecture": self.arch,
            "probabilities": probs.tolist(),
            "top_k": [
                {"char": index_to_label(int(i)), "p": float(probs[i])} for i in top_k_idx
            ],
        }


if __name__ == "__main__":
    # Smoke test with a synthetic stroke (real usage loads this from disk
    # or a live BLE/serial buffer -- see app/main.py for the HTTP wrapper).
    recognizer = CharacterRecognizer()  # now loads TCN by default
    fake_stroke = np.random.randn(60, 9)
    result = recognizer.predict(fake_stroke)
    print(json.dumps(result, indent=2)[:500])
