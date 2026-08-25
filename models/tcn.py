"""Temporal Convolutional Network (TCN) sensor model."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from config import MODEL_SEQ_LEN, NUM_CLASSES, NUM_SENSOR_CHANNELS


def _residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float, block_idx: int):
    prev = x
    h = layers.Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation_rate,
        name=f"tcn_block{block_idx}_conv1",
    )(x)
    h = layers.LeakyReLU()(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation_rate,
        name=f"tcn_block{block_idx}_conv2",
    )(h)
    h = layers.LeakyReLU()(h)
    h = layers.Dropout(dropout)(h)

    if prev.shape[-1] != filters:
        prev = layers.Conv1D(filters, 1, padding="same", name=f"tcn_block{block_idx}_proj")(prev)
    return layers.Add(name=f"tcn_block{block_idx}_residual")([prev, h])


def build_encoder(
    seq_len: int = MODEL_SEQ_LEN,
    n_channels: int = NUM_SENSOR_CHANNELS,
    dropout: float = 0.3,
    filters: int = 64,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    x = inputs
    for i, d in enumerate(dilations):
        x = _residual_block(
            x, filters=filters, kernel_size=kernel_size, dilation_rate=d,
            dropout=dropout, block_idx=i,
        )

    x = layers.GlobalAveragePooling1D(name="temporal_pool")(x)
    features = layers.Dropout(dropout, name="features")(x)

    return keras.Model(inputs, features, name="sensor_encoder_tcn")


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
    """Returns (full_model, encoder, classifier)."""
    encoder = build_encoder(seq_len, n_channels, dropout)
    classifier = build_classifier(encoder.output_shape[-1], num_classes)

    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    features = encoder(inputs)
    outputs = classifier(features)
    full_model = keras.Model(inputs, outputs, name="tcn")
    return full_model, encoder, classifier
