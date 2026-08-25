"""
Train a sensor recognizer on the processed dataset (run data/build_dataset.py first).

Usage:
    python train.py --arch cnn_lstm
    python train.py --arch cnn_bilstm
    python train.py --arch tcn
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    ARCHITECTURES,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    EXPERIMENTS_DIR,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_CLASSES,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_NPZ_PATH,
    TRAIN_NPZ_PATH,
    arch_model_dir,
    arch_training_history_path,
    classifier_weights_path,
    encoder_weights_path,
    model_path,
)
from models import build_model


def load_split(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"]


def _validate_labels(name: str, y: np.ndarray) -> None:
    if y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError(
            f"{name} labels out of range for NUM_CLASSES={NUM_CLASSES}: "
            f"min={int(y.min())} max={int(y.max())}"
        )


def main(arch: str, epochs: int, batch_size: int, learning_rate: float) -> None:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    arch_model_dir(arch).mkdir(parents=True, exist_ok=True)
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
    print(f"[arch] {arch}")
    print(f"[data] train={X_train.shape} val={X_val.shape} num_classes={num_classes}")

    full_model, encoder, classifier = build_model(
        arch, seq_len=seq_len, n_channels=n_channels, num_classes=num_classes
    )
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    full_model.summary()

    num_params = int(full_model.count_params())
    feature_dim = int(encoder.output_shape[-1])
    print(f"[model] total_params={num_params:,} encoder_feature_dim={feature_dim}")

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

    out_model_path = model_path(arch)
    out_encoder_path = encoder_weights_path(arch)
    out_classifier_path = classifier_weights_path(arch)

    full_model.save(out_model_path)
    encoder.save_weights(out_encoder_path)
    classifier.save_weights(out_classifier_path)
    print(f"[save] full model -> {out_model_path}")
    print(f"[save] encoder weights -> {out_encoder_path} (Priority 5 reuse)")
    print(f"[save] classifier weights -> {out_classifier_path}")

    hist_out = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_out["train_seconds"] = elapsed
    hist_out["epochs_ran"] = len(history.history["loss"])
    hist_out["architecture"] = arch
    hist_out["num_params"] = num_params
    hist_out["encoder_feature_dim"] = feature_dim

    hist_path = arch_training_history_path(arch)
    with open(hist_path, "w") as f:
        json.dump(hist_out, f, indent=2)
    print(f"[save] training history -> {hist_path}")
    print(f"[time] trained {hist_out['epochs_ran']} epochs in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", type=str, default="cnn_lstm", choices=ARCHITECTURES,
        help="Which Priority-1 sensor architecture to train (default: cnn_lstm, "
             "i.e. identical behavior to Priority 0).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    main(args.arch, args.epochs, args.batch_size, args.lr)
