"""
Adapter-only training + validation gate (ActionPlan.md 13.8's safety
rule, session-scoped instead of persisted): a candidate update is only
kept if it does not regress held-out accuracy on this SAME session's
already-confirmed samples. Rejected updates restore the prior adapter
weights exactly -- the live model can only get better or stay the same
across a session, never silently worse.
"""
from __future__ import annotations

import numpy as np
from tensorflow import keras


def _clone_weights(adapter) -> list:
    return [w.numpy().copy() for w in adapter.weights]


def _restore_weights(adapter, weights: list) -> None:
    for w, val in zip(adapter.weights, weights):
        w.assign(val)


def adapt_session(
    personalized_model: keras.Model,
    adapter,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 5,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    min_val_samples: int = 3,
    min_train_samples: int = 3,
    seed: int = 0,
) -> dict:
    n = len(y)
    if n < (min_val_samples + min_train_samples):
        return {"updated": False, "reason": f"not enough samples yet ({n})"}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(min_val_samples, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) < min_train_samples:
        return {"updated": False, "reason": "not enough train samples after val split"}

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    old_weights = _clone_weights(adapter)

    baseline_acc = float(
        (np.argmax(personalized_model.predict(X_val, verbose=0), axis=1) == y_val).mean()
    )

    personalized_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    personalized_model.fit(X_train, y_train, epochs=epochs, verbose=0)

    new_acc = float(
        (np.argmax(personalized_model.predict(X_val, verbose=0), axis=1) == y_val).mean()
    )

    if new_acc + 1e-9 < baseline_acc:
        _restore_weights(adapter, old_weights)
        return {
            "updated": False,
            "reason": "held-out accuracy regressed -- rolled back",
            "baseline_acc": baseline_acc,
            "candidate_acc": new_acc,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }

    return {
        "updated": True,
        "baseline_acc": baseline_acc,
        "candidate_acc": new_acc,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }