"""
Real-time inference path for a single new marker stroke (ActionPlan.md
section 15). Loads the SAME preprocessing function and the SAME persisted
normalization stats used at training time -- nothing is recomputed here.

Output contract (Priority 2, section 10.1): the full probability vector is
always returned, never collapsed to argmax internally. Callers (e.g. a
future beam decoder) decide what to do with it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    BASELINE_MODEL_PATH,
    NORM_STATS_PATH,
    NUM_CLASSES,
    index_to_label,
)
from preprocessing.clean import clean_sensor_matrix
from preprocessing.segment import preprocess


class CharacterRecognizer:
    """Loads the global model + normalization stats once; call .predict()
    per stroke. Designed to be instantiated once at server startup (see
    app/main.py), not per-request."""

    def __init__(self, model_path: Path = BASELINE_MODEL_PATH, norm_stats_path: Path = NORM_STATS_PATH):
        self.model = tf.keras.models.load_model(model_path)
        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float64)
        self.std = np.array(stats["std"], dtype=np.float64)
        self.seq_len = stats["seq_len"]

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
            "probabilities": probs.tolist(),
            "top_k": [
                {"char": index_to_label(int(i)), "p": float(probs[i])} for i in top_k_idx
            ],
        }


if __name__ == "__main__":
    # Smoke test with a synthetic stroke (real usage loads this from disk
    # or a live BLE/serial buffer -- see app/main.py for the HTTP wrapper).
    recognizer = CharacterRecognizer()
    fake_stroke = np.random.randn(60, 9)
    result = recognizer.predict(fake_stroke)
    print(json.dumps(result, indent=2)[:500])
