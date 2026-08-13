"""
Train the baseline CNN-LSTM on the processed dataset (run
data/build_dataset.py first).

Usage:
    python train.py [--epochs 100] [--batch-size 64]

Saves:
    models/artifacts/baseline_cnn_lstm.keras   (full end-to-end model)
    models/artifacts/encoder.weights.h5        (for Priority 5 reuse)
    models/artifacts/classifier.weights.h5
    experiments/training_history.json

Priority 0 audit fix applied here: num_classes is now the architectural
constant NUM_CLASSES (52) from config.py, not inferred from
max(y_train.max(), y_val.max()) + 1. Inferring it from observed labels
would silently shrink the model to fewer than 52 output units if a class
was ever absent from train or val -- see data/build_dataset.py's
class-coverage check, which now prevents that split from even being
written, and the explicit range/coverage checks below as a second
independent guard.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    BATCH_SIZE,
    BASELINE_MODEL_PATH,
    CLASSIFIER_WEIGHTS_PATH,
    EARLY_STOPPING_PATIENCE,
    ENCODER_WEIGHTS_PATH,
    EPOCHS,
    EXPERIMENTS_DIR,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_CLASSES,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_NPZ_PATH,
    TRAIN_NPZ_PATH,
)
from models.cnn_lstm import build_full_model


def load_split(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"]


def _validate_labels(name: str, y: np.ndarray) -> None:
    if y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError(
            f"{name} labels out of range for NUM_CLASSES={NUM_CLASSES}: "
            f"min={int(y.min())} max={int(y.max())}"
        )


def main(epochs: int, batch_size: int, learning_rate: float) -> None:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(TRAIN_NPZ_PATH)
    val_path = PROCESSED_DIR / "val.npz"
    X_val, y_val = load_split(val_path)

    # Hard architectural constant, not inferred from this run's data.
    num_classes = NUM_CLASSES
    _validate_labels("y_train", y_train)
    _validate_labels("y_val", y_val)

    train_classes_present = len(set(y_train.tolist()))
    if train_classes_present < NUM_CLASSES:
        raise ValueError(
            f"Only {train_classes_present}/{NUM_CLASSES} classes present in y_train -- "
            f"rebuild the dataset with data/build_dataset.py (it now fails fast on "
            f"this too, so seeing this error here means the .npz files are stale)."
        )

    seq_len, n_channels = X_train.shape[1], X_train.shape[2]
    print(f"[data] train={X_train.shape} val={X_val.shape} num_classes={num_classes}")

    full_model, encoder, classifier = build_full_model(
        seq_len=seq_len, n_channels=n_channels, num_classes=num_classes
    )
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    full_model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    start = time.time()
    history = full_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start

    full_model.save(BASELINE_MODEL_PATH)
    encoder.save_weights(ENCODER_WEIGHTS_PATH)
    classifier.save_weights(CLASSIFIER_WEIGHTS_PATH)
    print(f"[save] full model -> {BASELINE_MODEL_PATH}")
    print(f"[save] encoder weights -> {ENCODER_WEIGHTS_PATH} (Priority 5 reuse)")
    print(f"[save] classifier weights -> {CLASSIFIER_WEIGHTS_PATH}")

    hist_out = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_out["train_seconds"] = elapsed
    hist_out["epochs_ran"] = len(history.history["loss"])
    with open(EXPERIMENTS_DIR / "training_history.json", "w") as f:
        json.dump(hist_out, f, indent=2)
    print(f"[time] trained {hist_out['epochs_ran']} epochs in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.lr)