"""
Baseline CNN-LSTM sensor model (ActionPlan.md Priority 1, Option A --
"keep as the mandatory baseline, always in the comparison table").
Architecture unchanged from Priority 0 -- this file is copied here
byte-for-byte from the version you already trained/evaluated, so any
model you've already saved under the legacy paths (config.model_path
("cnn_lstm") resolves to the same file as before: BASELINE_MODEL_PATH)
keeps working with zero retraining required.

    Input(seq_len, 9)
    Conv1D 64  (LeakyReLU) -> MaxPool
    Conv1D 128 (LeakyReLU) -> MaxPool
    Conv1D 256 (LeakyReLU) -> MaxPool
    LSTM 32
    Dropout
    -- feature vector (32-dim) --
    Dense 256 -> Dense 128 -> Dense NUM_CLASSES -> Softmax

The encoder and classifier are built and returned as SEPARATE Keras models
(ActionPlan.md Step 5 / Step 25): this is a hard requirement for Priority 5
(the personalization adapter gets inserted between them later as
`h' = h + Adapter(h)` without any change to this file). A combined
end-to-end model is also returned for convenience during training.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from config import MODEL_SEQ_LEN, NUM_CLASSES, NUM_SENSOR_CHANNELS


def build_encoder(
    seq_len: int = MODEL_SEQ_LEN,
    n_channels: int = NUM_SENSOR_CHANNELS,
    dropout: float = 0.3,
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

    x = layers.LSTM(32)(x)
    features = layers.Dropout(dropout, name="features")(x)

    return keras.Model(inputs, features, name="sensor_encoder_cnn_lstm")


def build_classifier(feature_dim: int = 32, num_classes: int = NUM_CLASSES) -> keras.Model:
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
    """Returns (full_model, encoder, classifier). full_model is what you
    train end-to-end; encoder/classifier share its weights and are what
    get saved/loaded separately for Priority 5 and for real-time
    inference's probability-preserving output contract (Priority 2)."""
    encoder = build_encoder(seq_len, n_channels, dropout)
    classifier = build_classifier(encoder.output_shape[-1], num_classes)

    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    features = encoder(inputs)
    outputs = classifier(features)
    full_model = keras.Model(inputs, outputs, name="cnn_lstm_baseline")
    return full_model, encoder, classifier
