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

Personalization change (session-level adapter, see personalization/):
CharacterRecognizer now ALSO loads the encoder and classifier as
SEPARATE Keras models (self.encoder / self.classifier), in addition to
the already-existing fused self.model. This is a pure ADDITION -- it
loads the exact same trained weights a second way, via
config.encoder_weights_path(arch) / config.classifier_weights_path(arch)
(already saved by train.py, labeled "Priority 5 reuse" there). self.model
and predict() are completely untouched, so /predict and
/session/{id}/stroke behave exactly as before. self.encoder/self.classifier
exist purely so personalization/adapter.py can build a session-scoped
personalized model without touching self.model at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    NORM_STATS_PATH,
    NUM_CLASSES,
    NUM_SENSOR_CHANNELS,
    classifier_weights_path,
    encoder_weights_path,
    index_to_label,
    model_path,
)
from models import build_model
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

    self.model     -- fused encoder+classifier, used by predict() (unchanged).
    self.encoder   -- same trained weights, as a standalone Model. Used
                       only by personalization/adapter.py.
    self.classifier -- same trained weights, as a standalone Model. Used
                       only by personalization/adapter.py.
    """

    def __init__(
        self,
        arch: str = DEFAULT_SERVING_ARCH,
        model_file: Path | None = None,
        norm_stats_path: Path = NORM_STATS_PATH,
    ):
        self.arch = arch
        resolved_model_path = model_file if model_file is not None else model_path(arch)
        self.model = tf.keras.models.load_model(resolved_model_path)

        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float64)
        self.std = np.array(stats["std"], dtype=np.float64)
        self.seq_len = stats["seq_len"]
        print(f"[realtime] serving architecture={arch!r} from {resolved_model_path}")

        # --- Personalization support: load encoder/classifier separately ---
        # This rebuilds the SAME architecture (fresh/random weights) then
        # immediately overwrites those weights from the .weights.h5 files
        # train.py already saved -- so these carry the exact trained
        # weights, just as two independent Model objects instead of one
        # fused model. Nothing above this point (self.model) is affected.
        _, self.encoder, self.classifier = build_model(
            arch, seq_len=self.seq_len, n_channels=NUM_SENSOR_CHANNELS, num_classes=NUM_CLASSES
        )
        self.encoder.load_weights(encoder_weights_path(arch))
        self.classifier.load_weights(classifier_weights_path(arch))

    def predict(self, raw_sensor_matrix: np.ndarray, top_k: int = 5) -> dict:
        """raw_sensor_matrix: (rows, 9) array, may contain 'ovf'/'nan'
        strings straight off the device buffer -- same as a raw file's
        sensor columns. Returns the full probability contract, not a
        single character. UNCHANGED behavior -- always uses self.model."""
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

    def preprocess_stroke(self, raw_sensor_matrix: np.ndarray) -> np.ndarray:
        """Returns the preprocessed (seq_len, 9) float32 array WITHOUT the
        batch dim -- used by the personalization endpoint, which needs
        the model input in the exact same shape used for training a
        SessionAdapter (personalization.buffer.SessionAdaptationBuffer
        stores exactly this shape)."""
        cleaned = clean_sensor_matrix(np.asarray(raw_sensor_matrix, dtype=object))
        return preprocess(cleaned, mean=self.mean, std=self.std, window_len=self.seq_len)


if __name__ == "__main__":
    # Smoke test with a synthetic stroke (real usage loads this from disk
    # or a live BLE/serial buffer -- see app/main.py for the HTTP wrapper).
    recognizer = CharacterRecognizer()  # now loads TCN by default
    fake_stroke = np.random.randn(60, 9)
    result = recognizer.predict(fake_stroke)
    print(json.dumps(result, indent=2)[:500])