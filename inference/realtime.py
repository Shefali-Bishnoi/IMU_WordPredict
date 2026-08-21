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

NEW (real-time hardware bridging fix): CharacterRecognizer now enforces
the same [MIN_RAW_LINES, MAX_RAW_LINES] length band that
data/build_dataset.py's _process_raw() enforces before a raw file is ever
included in the training set. Before this change, predict()/
preprocess_stroke() would silently accept a stroke of ANY length (1 row,
246 rows, anything) and run it through pad_or_trim() anyway -- producing
a numeric prediction that LOOKS valid but is actually out-of-distribution
input the model never trained on. This was caught and diagnosed in
check.md: a 246-row and a 1-row raw stroke of the same true character
("X") both predicted the wrong class with <3% confidence, purely because
their padded/trimmed representation never existed in any training
example, while the in-band 54-row stroke of the exact same letter
predicted correctly with 64% confidence. That gap is invisible unless you
go looking for it -- a live marker stroke that runs a few hundred ms too
long or too short will now be REJECTED with a clear, actionable error
instead of silently returning a confident-looking but meaningless
prediction. This matters specifically now that real hardware (not just
pre-filtered dataset files) is about to start feeding this function.

Enforcement is opt-out (enforce_length_band=False), not opt-in, so any
existing caller gets the safety net by default; scripts that
deliberately want to probe out-of-band behavior (e.g. a debugging
script reproducing check.md's investigation) can disable it explicitly.
"""
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

# Priority 1 winner (ActionPlan.md 9.3): TCN had the highest macro F1,
# the lowest latency, AND the smallest model of the three architectures
# compared -- a clean sweep, not a trade-off call. See
# experiments/architecture_comparison.md for the numbers this was decided
# from. Change this constant (not call sites) if a future Priority 1 rerun
# picks a different winner.
DEFAULT_SERVING_ARCH = "tcn"


class StrokeLengthError(ValueError):
    """Raised when a live stroke's raw row count falls outside the
    [MIN_RAW_LINES, MAX_RAW_LINES] band the model was actually trained
    on (config.py / data/build_dataset.py's _process_raw). This is a hard
    rejection, not a warning, because predicting on an out-of-band length
    is not graceful degradation -- it is a confident-looking but
    meaningless number. See check.md for the concrete before/after: a
    246-row stroke and a 1-row stroke of the true letter "X" both
    predicted a wrong class at <3% confidence once pushed through
    pad_or_trim() anyway, purely because that padded/trimmed shape never
    occurred in training. Catch this exception at the API boundary
    (see app/main.py) and surface it as a 400, not a 500 -- it is a
    request-shape problem, not a server error.
    """


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
        single character.

        Raises StrokeLengthError if the stroke's row count falls outside
        [MIN_RAW_LINES, MAX_RAW_LINES] and self.enforce_length_band is
        True (the default) -- see the module docstring / check.md for why
        this matters once real hardware (not just pre-filtered dataset
        files) is the caller."""
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
        """Returns the preprocessed (seq_len, 9) float32 array WITHOUT the
        batch dim -- used by the personalization endpoint, which needs
        the model input in the exact same shape used for training a
        SessionAdapter (personalization.buffer.SessionAdaptationBuffer
        stores exactly this shape).

        Same length-band enforcement as predict() -- a personalization
        correction sample built from an out-of-band stroke would poison
        that session's adapter with a shape the model never learned from,
        which is arguably worse than a bad live prediction since it's
        meant to make *future* predictions better, not just this one."""
        raw_sensor_matrix = np.asarray(raw_sensor_matrix, dtype=object)
        if self.enforce_length_band:
            _validate_raw_length(raw_sensor_matrix.shape[0])
        cleaned = clean_sensor_matrix(raw_sensor_matrix)
        return preprocess(cleaned, mean=self.mean, std=self.std, window_len=self.seq_len)


if __name__ == "__main__":
    # Smoke test with a synthetic stroke (real usage loads this from disk
    # or a live BLE/serial buffer -- see app/main.py for the HTTP wrapper,
    # hardware/marker_bridge.py for the live serial->WebSocket bridge).
    recognizer = CharacterRecognizer()  # now loads TCN by default
    fake_stroke = np.random.randn(60, 9)  # 60 rows -- inside [40, 80], valid
    result = recognizer.predict(fake_stroke)
    print(json.dumps(result, indent=2)[:500])

    try:
        recognizer.predict(np.random.randn(5, 9))  # 5 rows -- outside band
    except StrokeLengthError as e:
        print(f"\n[smoke test] correctly rejected an out-of-band stroke:\n{e}")