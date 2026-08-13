"""
The single canonical baseline evaluation script (ActionPlan.md Priority 0,
"one single, canonical baseline evaluation script that anyone can rerun").

Usage:
    python evaluate.py

Produces:
    experiments/baseline_metrics.json      (macro P/R/F1, per-class acc, latency)
    experiments/baseline_confusion_matrix.csv

Priority 0 audit fix applied here: metrics and the confusion matrix are now
always computed against the fixed label space np.arange(NUM_CLASSES) (52
classes), not against whatever subset of classes happened to appear in
y_test/y_pred this run. Previously the confusion matrix's shape and the
per-class index -> character mapping could silently drift between runs
(e.g. a class missing from a small test slice would shrink the matrix and
shift every index after it), making experiments impossible to compare
apples-to-apples. Any class truly absent from y_test now shows up with
support=0 rather than being omitted.
"""
from __future__ import annotations

import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import (
    BASELINE_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    NUM_CLASSES,
    TEST_NPZ_PATH,
    index_to_label,
)


def main() -> None:
    data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    X_test, y_test = data["X"], data["y"]

    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)

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
    # index -> character mapping, run over run.
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

    metrics = {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "n_test_samples": int(len(y_test)),
        "n_classes_evaluated": int(NUM_CLASSES),
        "n_classes_absent_from_test": n_absent_from_test,
        "median_single_sample_latency_ms": latency_ms,
        "per_class": per_class,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    header = ",".join([""] + [index_to_label(int(l)) for l in labels])
    lines = [header]
    for i, l in enumerate(labels):
        lines.append(",".join([index_to_label(int(l))] + [str(v) for v in cm[i].tolist()]))
    with open(CONFUSION_MATRIX_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Median latency:  {latency_ms:.2f} ms/sample")
    print(f"[save] {METRICS_PATH}")
    print(f"[save] {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()