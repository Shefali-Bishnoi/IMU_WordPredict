"""
CNN-BiLSTM sensor model (ActionPlan.md Priority 1, section 9.1 Option B).

Identical CNN front-end to the baseline cnn_lstm.py, but the uni-directional
LSTM(32) is replaced with a Bidirectional(LSTM(32)) (-> 64-dim feature
vector), so the encoder can use information from *both* temporal
directions. Per ActionPlan.md, this is a fair thing to try specifically
because each training instance is a whole, already-segmented character
window -- we are not doing streaming/online recognition, so looking
"backwards" in time within one instance is not a form of cheating/leakage.

Deliberately mirrors cnn_lstm.py's public interface exactly
(build_encoder, build_classifier, build_full_model) so the rest of the
codebase (train.py, evaluate.py, Priority 5's adapter) can select between
architectures purely by name, never by branching on internal structure --
see models/__init__.py's ARCH_BUILDERS registry.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from config import MODEL_SEQ_LEN, NUM_CLASSES, NUM_SENSOR_CHANNELS


def build_encoder(
    seq_len: int = MODEL_SEQ_LEN,
    n_channels: int = NUM_SENSOR_CHANNELS,
    dropout: float = 0.3,
    lstm_units: int = 32,
) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    x = layers.Conv1D(64, kernel_size=3, padding="same")(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)

    # Only architectural change vs. cnn_lstm.py: Bidirectional wrapper.
    # Output feature dim is 2 * lstm_units (forward + backward concatenated).
    x = layers.Bidirectional(layers.LSTM(lstm_units), name="bilstm")(x)
    features = layers.Dropout(dropout, name="features")(x)

    return keras.Model(inputs, features, name="sensor_encoder_cnn_bilstm")


def build_classifier(feature_dim: int, num_classes: int = NUM_CLASSES) -> keras.Model:
    inputs = keras.Input(shape=(feature_dim,), name="features_input")
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="char_probabilities")(x)
    return keras.Model(inputs, outputs, name="char_classifier")


def build_full_model(
    seq_len: int = MODEL_SEQ_LEN,
    n_channels: int = NUM_SENSOR_CHANNELS,
    num_classes: int = NUM_CLASSES,
    dropout: float = 0.3,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Returns (full_model, encoder, classifier) -- same contract as
    cnn_lstm.build_full_model. `feature_dim` handed to build_classifier is
    read off encoder.output_shape[-1] rather than hardcoded, since it's
    64 here (2x lstm_units) instead of cnn_lstm's 32 -- this is exactly
    the kind of thing ActionPlan.md 13.10 warns Priority 5's adapter must
    not hardcode either."""
    encoder = build_encoder(seq_len, n_channels, dropout)
    classifier = build_classifier(encoder.output_shape[-1], num_classes)

    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    features = encoder(inputs)
    outputs = classifier(features)
    full_model = keras.Model(inputs, outputs, name="cnn_bilstm")
    return full_model, encoder, classifier
