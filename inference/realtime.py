"""Real-time inference for a single IMU character stroke."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    MAX_RAW_LINES,
    MIN_RAW_LINES,
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

# Default serving architecture.
DEFAULT_SERVING_ARCH = "tcn"


class StrokeLengthError(ValueError):
    """Raised when a stroke's row count is outside [MIN_RAW_LINES, MAX_RAW_LINES]."""


def _validate_raw_length(n_rows: int) -> None:
    if not (MIN_RAW_LINES <= n_rows <= MAX_RAW_LINES):
        raise StrokeLengthError(
            f"Stroke has {n_rows} raw sensor rows, but the model was only "
            f"trained on strokes with {MIN_RAW_LINES}-{MAX_RAW_LINES} rows "
            f"(see config.MIN_RAW_LINES / config.MAX_RAW_LINES, enforced "
            f"identically by data/build_dataset.py's length filter). "
            f"Predicting on a length outside this band is out-of-"
            f"distribution input -- see check.md's real before/after "
            f"comparison (a 246-row and a 1-row stroke of the same true "
            f"letter both predicted the WRONG class at <3% confidence, "
            f"while the same letter's 54-row stroke -- inside the band -- "
            f"predicted correctly at 64% confidence). Write the character "
            f"again, aiming for roughly {MIN_RAW_LINES}-{MAX_RAW_LINES} "
            f"sensor rows (about {MIN_RAW_LINES / 50:.1f}-"
            f"{MAX_RAW_LINES / 50:.1f}s of writing at the nominal 50Hz "
            f"sample rate)."
        )


class CharacterRecognizer:
    """Loads model and normalization stats once; call .predict() per stroke.

    self.model is the fused encoder+classifier; self.encoder and
    self.classifier are separate models for personalization.
    """

    def __init__(
        self,
        arch: str = DEFAULT_SERVING_ARCH,
        model_file: Path | None = None,
        norm_stats_path: Path = NORM_STATS_PATH,
        enforce_length_band: bool = True,
    ):
        self.arch = arch
        self.enforce_length_band = enforce_length_band
        resolved_model_path = model_file if model_file is not None else model_path(arch)
        self.model = tf.keras.models.load_model(resolved_model_path)

        with open(norm_stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float64)
        self.std = np.array(stats["std"], dtype=np.float64)
        self.seq_len = stats["seq_len"]
        print(f"[realtime] serving architecture={arch!r} from {resolved_model_path}")
        print(
            f"[realtime] length-band enforcement: "
            f"{'ON' if enforce_length_band else 'OFF (unsafe -- debugging only)'} "
            f"[{MIN_RAW_LINES}, {MAX_RAW_LINES}] rows"
        )

        # Load encoder/classifier separately for session personalization.
        _, self.encoder, self.classifier = build_model(
            arch, seq_len=self.seq_len, n_channels=NUM_SENSOR_CHANNELS, num_classes=NUM_CLASSES
        )
        self.encoder.load_weights(encoder_weights_path(arch))
        self.classifier.load_weights(classifier_weights_path(arch))

    def predict(self, raw_sensor_matrix: np.ndarray, top_k: int = 5) -> dict:
        """Predict character probabilities for one stroke.

        raw_sensor_matrix: (rows, 9), may contain 'ovf'/'nan' strings.
        Returns the full probability vector, not argmax.

        Raises StrokeLengthError when enforce_length_band is True and the
        row count is outside [MIN_RAW_LINES, MAX_RAW_LINES].
        """
        raw_sensor_matrix = np.asarray(raw_sensor_matrix, dtype=object)
        if self.enforce_length_band:
            _validate_raw_length(raw_sensor_matrix.shape[0])

        cleaned = clean_sensor_matrix(raw_sensor_matrix)
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
        """Return preprocessed (seq_len, 9) float32 array without batch dim."""
        raw_sensor_matrix = np.asarray(raw_sensor_matrix, dtype=object)
        if self.enforce_length_band:
            _validate_raw_length(raw_sensor_matrix.shape[0])
        cleaned = clean_sensor_matrix(raw_sensor_matrix)
        return preprocess(cleaned, mean=self.mean, std=self.std, window_len=self.seq_len)


if __name__ == "__main__":
    recognizer = CharacterRecognizer()
    fake_stroke = np.random.randn(60, 9)
    result = recognizer.predict(fake_stroke)
    print(json.dumps(result, indent=2)[:500])

    try:
        recognizer.predict(np.random.randn(5, 9))
    except StrokeLengthError as e:
        print(f"\n[smoke test] correctly rejected an out-of-band stroke:\n{e}")