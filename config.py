"""
Central configuration for the WordPredict pipeline.

All paths and constants that preprocessing, training, evaluation, and
real-time inference must agree on live here so there is exactly one
source of truth (see ActionPlan.md Priority 0 / Step 4 and Step 15's
requirement for a single shared preprocess() contract).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

SPLIT_CONFIG_PATH = SPLITS_DIR / "participant_split.json"
NORM_STATS_PATH = PROCESSED_DIR / "norm_stats.json"
TRAIN_NPZ_PATH = PROCESSED_DIR / "train.npz"
TEST_NPZ_PATH = PROCESSED_DIR / "test.npz"
DATASET_MANIFEST_PATH = EXPERIMENTS_DIR / "dataset_manifest.json"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_cnn_lstm.keras"
ENCODER_WEIGHTS_PATH = MODELS_DIR / "encoder.weights.h5"
CLASSIFIER_WEIGHTS_PATH = MODELS_DIR / "classifier.weights.h5"

METRICS_PATH = EXPERIMENTS_DIR / "baseline_metrics.json"
CONFUSION_MATRIX_PATH = EXPERIMENTS_DIR / "baseline_confusion_matrix.csv"

# ---------------------------------------------------------------------------
# Raw file schema (confirmed from an actual A-01.txt sample, see
# ActionPlan.md section 4.4). 12 comma-separated columns, no header:
#   0: character label (repeated on every row)
#   1: timestamp "YYYY-MM-DD HH:MM:SS.ffffff"
#   2-4: Accel X, Y, Z
#   5-7: Gyro X, Y, Z
#   8-10: Mag X, Y, Z
#   11: writing flag (1 = actively writing)
# Some sessions insert an empty field after the timestamp (13 cols); that
# quirk is normalized in preprocessing.io.load_raw_file.
# ---------------------------------------------------------------------------
LABEL_COL = 0
TIMESTAMP_COL = 1
SENSOR_COLS = slice(2, 11)  # 9 channels
FLAG_COL = 11
NUM_RAW_COLS = 12
NUM_SENSOR_CHANNELS = 9

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
MIN_RAW_LINES = 40
MAX_RAW_LINES = 80
PAD_TARGET_LEN = 80     # pad/trim every instance to this length
# NOTE (fixed after review): this used to be 50, which forced a second
# center-crop step (80 -> 50) on top of pad_or_trim's own padding/trimming.
# For any real instance close to 80 rows that crop silently discarded up to
# ~30 real sensor rows before the model ever saw them (see ActionPlan review
# notes). The CNN-LSTM (and CNN-BiLSTM/TCN) encoders all reduce the time
# axis to a fixed-size feature vector regardless of input length, so there
# is no architectural reason to crop further -- MODEL_SEQ_LEN now equals
# PAD_TARGET_LEN so every real row that survived the [40, 80]-line filter
# is actually used. segment.center_window() still exists and is still
# exercised (harmlessly, as a no-op) so nothing downstream needs to change.
MODEL_SEQ_LEN = PAD_TARGET_LEN
NOMINAL_SAMPLE_RATE_HZ = 50
NOMINAL_DT_SECONDS = 1.0 / NOMINAL_SAMPLE_RATE_HZ

# ---------------------------------------------------------------------------
# Label space: 52 classes, A-Z -> 0-25, a-z -> 26-51.
# NOTE: the original prototype notebook (GRU_model.ipynb) only ever loaded
# the "capital letters" directory and used ord(c) - ord('A') / ord(c) -
# ord('a') independently, which COLLIDES upper and lower case into the same
# 0-25 index whenever both are present (confirmed by its final
# to_categorical shape being (N, 26), not (N, 52)). This is fixed here.
# ---------------------------------------------------------------------------
NUM_CLASSES = 52
CAPITAL_DIR_NAME = "capital letters"
SMALL_DIR_NAME = "small letters"


def label_to_index(char_label: str) -> int:
    """A-Z -> 0-25, a-z -> 26-51."""
    if "A" <= char_label <= "Z":
        return ord(char_label) - ord("A")
    if "a" <= char_label <= "z":
        return 26 + (ord(char_label) - ord("a"))
    raise ValueError(f"Unrecognized character label: {char_label!r}")


def index_to_label(index: int) -> str:
    if 0 <= index <= 25:
        return chr(ord("A") + index)
    if 26 <= index <= 51:
        return chr(ord("a") + (index - 26))
    raise ValueError(f"Unrecognized class index: {index!r}")


# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.1  # carved out of TRAIN participants, disjoint from TEST

# Fixed test participant IDs used by the original prototype, kept here only
# as a reproducible DEFAULT if no split has been frozen yet. A fresh split
# is generated (and then frozen) if these IDs aren't found in the raw data.
DEFAULT_TEST_PARTICIPANT_IDS = [
    "S05", "S15", "S82", "S101", "S104", "S113", "S30", "S76", "S03", "S08",
    "S26", "S31", "S32", "S37", "S117", "S22", "S47", "S89", "S01", "S109",
    "S97", "S13", "S49",
]

# ---------------------------------------------------------------------------
# Priority 1 -- per-architecture paths (ActionPlan.md section 9)
# ---------------------------------------------------------------------------
# Priority 1 trains and compares THREE architectures (cnn_lstm, cnn_bilstm,
# tcn). Each one needs its own model file, its own encoder/classifier
# weights, its own metrics.json, and its own confusion matrix -- otherwise
# training cnn_bilstm would silently overwrite the cnn_lstm baseline from
# Priority 0.
#
# Backward compatibility (important): arch="cnn_lstm" resolves to the
# EXACT SAME paths as before (BASELINE_MODEL_PATH, METRICS_PATH, etc.) --
# so an already-trained Priority 0 baseline model, its metrics.json, and
# its confusion matrix are untouched and still load correctly. Only the
# two NEW architectures (cnn_bilstm, tcn) get fresh subdirectories under
# models/artifacts/ and fresh experiments/<arch>_*.json files.
# ---------------------------------------------------------------------------
ARCHITECTURES = ["cnn_lstm", "cnn_bilstm", "tcn"]

_LEGACY_ARCH_PATHS = {
    "cnn_lstm": {
        "model": BASELINE_MODEL_PATH,
        "encoder_weights": ENCODER_WEIGHTS_PATH,
        "classifier_weights": CLASSIFIER_WEIGHTS_PATH,
        "metrics": METRICS_PATH,
        "confusion_matrix": CONFUSION_MATRIX_PATH,
        "training_history": EXPERIMENTS_DIR / "training_history.json",
    }
}


def _check_arch(arch: str) -> None:
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}. Choose one of: {ARCHITECTURES}")


def arch_model_dir(arch: str) -> Path:
    _check_arch(arch)
    return MODELS_DIR / arch


def model_path(arch: str) -> Path:
    """Path to the saved full (encoder+classifier) Keras model for `arch`."""
    _check_arch(arch)
    if arch in _LEGACY_ARCH_PATHS:
        return _LEGACY_ARCH_PATHS[arch]["model"]
    return arch_model_dir(arch) / f"{arch}.keras"


def encoder_weights_path(arch: str) -> Path:
    _check_arch(arch)
    if arch in _LEGACY_ARCH_PATHS:
        return _LEGACY_ARCH_PATHS[arch]["encoder_weights"]
    return arch_model_dir(arch) / "encoder.weights.h5"


def classifier_weights_path(arch: str) -> Path:
    _check_arch(arch)
    if arch in _LEGACY_ARCH_PATHS:
        return _LEGACY_ARCH_PATHS[arch]["classifier_weights"]
    return arch_model_dir(arch) / "classifier.weights.h5"


def arch_metrics_path(arch: str) -> Path:
    _check_arch(arch)
    if arch in _LEGACY_ARCH_PATHS:
        return _LEGACY_ARCH_PATHS[arch]["metrics"]
    return EXPERIMENTS_DIR / f"{arch}_metrics.json"


def arch_confusion_matrix_path(arch: str) -> Path:
    _check_arch(arch)
    if arch in _LEGACY_ARCH_PATHS:
        return _LEGACY_ARCH_PATHS[arch]["confusion_matrix"]
    return EXPERIMENTS_DIR / f"{arch}_confusion_matrix.csv"


def arch_training_history_path(arch: str) -> Path:
    _check_arch(arch)
    if arch in _LEGACY_ARCH_PATHS:
        return _LEGACY_ARCH_PATHS[arch]["training_history"]
    return EXPERIMENTS_DIR / f"{arch}_training_history.json"


ARCHITECTURE_COMPARISON_PATH = EXPERIMENTS_DIR / "architecture_comparison.md"
