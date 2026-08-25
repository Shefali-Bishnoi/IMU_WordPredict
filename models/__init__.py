"""Architecture registry: maps names to build_full_model() functions."""
from __future__ import annotations

from typing import Callable

from tensorflow import keras

from . import cnn_bilstm, cnn_lstm, tcn

ARCH_BUILDERS: dict[str, Callable[..., tuple[keras.Model, keras.Model, keras.Model]]] = {
    "cnn_lstm": cnn_lstm.build_full_model,
    "cnn_bilstm": cnn_bilstm.build_full_model,
    "tcn": tcn.build_full_model,
}

ARCH_MODULES = {
    "cnn_lstm": cnn_lstm,
    "cnn_bilstm": cnn_bilstm,
    "tcn": tcn,
}


def build_model(arch: str, **kwargs) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Returns (full_model, encoder, classifier) for the named architecture."""
    if arch not in ARCH_BUILDERS:
        raise ValueError(
            f"Unknown architecture {arch!r}. Choose one of: {sorted(ARCH_BUILDERS)}"
        )
    return ARCH_BUILDERS[arch](**kwargs)
