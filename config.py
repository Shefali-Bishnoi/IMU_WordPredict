"""Central configuration for the WordPredict pipeline."""
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

# Raw file schema: 12 comma-separated columns, no header.
#   0: character label  1: timestamp  2-4: accel  5-7: gyro  8-10: mag  11: writing flag
# Some sessions have 13 cols (empty field after timestamp); normalized in preprocessing.io.
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
# MODEL_SEQ_LEN matches PAD_TARGET_LEN so no extra center-crop discards real rows.
MODEL_SEQ_LEN = PAD_TARGET_LEN
NOMINAL_SAMPLE_RATE_HZ = 50
NOMINAL_DT_SECONDS = 1.0 / NOMINAL_SAMPLE_RATE_HZ

NGRAM_MODEL_PATH = EXPERIMENTS_DIR / "ngram_model.json"


# Label space: 52 classes, A-Z -> 0-25, a-z -> 26-51.
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

# Default test participant IDs if no split has been frozen yet.
DEFAULT_TEST_PARTICIPANT_IDS = [
    "S05", "S15", "S82", "S101", "S104", "S113", "S30", "S76", "S03", "S08",
    "S26", "S31", "S32", "S37", "S117", "S22", "S47", "S89", "S01", "S109",
    "S97", "S13", "S49",
]

# Per-architecture paths. cnn_lstm keeps legacy paths for backward compatibility.
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
import os

# Contextual language-model settings (see language/contextual_scorer.py).
LANGUAGE_MODEL_ENABLED = os.environ.get("LANGUAGE_MODEL_ENABLED", "true").lower() in ("1", "true", "yes")
LANGUAGE_MODEL_NAME = os.environ.get("LANGUAGE_MODEL_NAME", "distilgpt2")
LANGUAGE_MODEL_WEIGHT = float(os.environ.get("LANGUAGE_MODEL_WEIGHT", "0.15"))
LANGUAGE_CONTEXT_WORDS = int(os.environ.get("LANGUAGE_CONTEXT_WORDS", "20"))
LANGUAGE_MODEL_MAX_CONTEXT_TOKENS = int(os.environ.get("LANGUAGE_MODEL_MAX_CONTEXT_TOKENS", "256"))
LANGUAGE_MODEL_TOP_K_CANDIDATES = int(os.environ.get("LANGUAGE_MODEL_TOP_K_CANDIDATES", "5"))
