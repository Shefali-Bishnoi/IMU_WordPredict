"""
Evaluate a trained sensor architecture on the test split.

Usage:
    python evaluate.py --arch cnn_lstm
    python evaluate.py --arch cnn_bilstm
    python evaluate.py --arch tcn
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import (
    ARCHITECTURES,
    NUM_CLASSES,
    TEST_NPZ_PATH,
    arch_confusion_matrix_path,
    arch_metrics_path,
    index_to_label,
    model_path,
)


def _model_size_bytes(path) -> int:
    """Total size on disk. Keras .keras files are a single zip archive;
    handle that directly. Falls back to 0 (recorded, not crashed) if the
    path is missing so a comparison run over partially-trained
    architectures doesn't die on this alone."""
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def main(arch: str) -> None:
    data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    X_test, y_test = data["X"], data["y"]

    model_file = model_path(arch)
    model = tf.keras.models.load_model(model_file)

    # Latency: single-sample inference, median over repeated calls (warm).
    _ = model.predict(X_test[:1], verbose=0)  # warm up
    latencies = []
    for i in range(min(100, len(X_test))):
        t0 = time.perf_counter()
        model.predict(X_test[i : i + 1], verbose=0)
        latencies.append(time.perf_counter() - t0)
    latency_ms = float(np.median(latencies) * 1000)

    probs = model.predict(X_test, batch_size=128, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Fixed 52-class label space -- always the same shape, always the same
    # index -> character mapping, run over run, architecture over
    # architecture (this is what makes cross-architecture comparison valid).
    labels = np.arange(NUM_CLASSES)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    accuracy = float((y_pred == y_test).mean())

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    per_class = {
        index_to_label(int(idx)): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, idx in enumerate(labels)
    }

    n_absent_from_test = int(sum(1 for s in support if s == 0))
    if n_absent_from_test:
        absent_chars = [index_to_label(int(idx)) for idx, s in zip(labels, support) if s == 0]
        print(f"[warn] {n_absent_from_test} classes have zero test support: {absent_chars}")

    # Track worst classes by F1 for robustness analysis.
    worst_classes = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:5]

    metrics = {
        "architecture": arch,
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "n_test_samples": int(len(y_test)),
        "n_classes_evaluated": int(NUM_CLASSES),
        "n_classes_absent_from_test": n_absent_from_test,
        "median_single_sample_latency_ms": latency_ms,
        "num_params": int(model.count_params()),
        "model_size_bytes": _model_size_bytes(model_file),
        "worst_5_classes_by_f1": [
            {"char": char, **stats} for char, stats in worst_classes
        ],
        "per_class": per_class,
    }

    metrics_path = arch_metrics_path(arch)
    confusion_matrix_path = arch_confusion_matrix_path(arch)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    header = ",".join([""] + [index_to_label(int(l)) for l in labels])
    lines = [header]
    for i, l in enumerate(labels):
        lines.append(",".join([index_to_label(int(l))] + [str(v) for v in cm[i].tolist()]))
    with open(confusion_matrix_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Architecture:    {arch}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Median latency:  {latency_ms:.2f} ms/sample")
    print(f"Params:          {metrics['num_params']:,}")
    print(f"Model size:      {metrics['model_size_bytes'] / 1024:.1f} KB")
    print(f"Worst classes:   {[c['char'] for c in metrics['worst_5_classes_by_f1']]}")
    print(f"[save] {metrics_path}")
    print(f"[save] {confusion_matrix_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", type=str, default="cnn_lstm", choices=ARCHITECTURES,
        help="Which Priority-1 sensor architecture to evaluate (default: cnn_lstm, "
             "i.e. identical behavior to Priority 0).",
    )
    args = parser.parse_args()
    main(args.arch)
