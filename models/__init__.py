"""
Architecture registry (ActionPlan.md Priority 1, section 9).

This is the single place that maps an architecture *name* to its
build_full_model() function. train.py, evaluate.py, and
compare_architectures.py all go through build_model(arch, ...) instead of
importing a specific architecture module directly -- that's what makes
"pick the winner later" a one-line change (config.py's default --arch)
instead of a code change.

Every architecture module (cnn_lstm.py, cnn_bilstm.py, tcn.py) exposes the
exact same three functions with the exact same signatures:

    build_encoder(seq_len, n_channels, dropout, ...)      -> keras.Model
    build_classifier(feature_dim, num_classes)             -> keras.Model
    build_full_model(seq_len, n_channels, num_classes, dropout)
        -> (full_model, encoder, classifier)

This is a hard contract (ActionPlan.md Step 5 / Step 25): Priority 5's
personalization adapter gets inserted between `encoder` and `classifier`
later, for whichever architecture wins Priority 1 -- so every architecture
must keep that split, no exceptions.
"""
from __future__ import annotations

from typing import Callable

from tensorflow import keras

from . import cnn_bilstm, cnn_lstm, tcn

# name -> build_full_model function. Add a new architecture by writing a
# module with the same three-function contract and registering it here --
# nothing else needs to change.
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
    """Returns (full_model, encoder, classifier) for the named architecture.

    Raises a clear error (listing valid names) rather than a KeyError, since
    this is a common typo point on the command line (--arch).
    """
    if arch not in ARCH_BUILDERS:
        raise ValueError(
            f"Unknown architecture {arch!r}. Choose one of: {sorted(ARCH_BUILDERS)}"
        )
    return ARCH_BUILDERS[arch](**kwargs)
