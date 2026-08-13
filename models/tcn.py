"""
Temporal Convolutional Network (TCN) sensor model (ActionPlan.md Priority
1, section 9.1 Option C).

Implemented directly with dilated causal Conv1D residual blocks rather
than pulling in an external TCN package -- this keeps requirements.txt
unchanged for what is, for now, just one ablation option among three (see
ActionPlan.md's "golden rule": don't add a dependency/component before an
experiment justifies it).

Structure:

    IMU -> ResidualBlock(dilation=1) -> ResidualBlock(dilation=2)
        -> ResidualBlock(dilation=4) -> ResidualBlock(dilation=8)
        -> GlobalAveragePooling1D -> features

Each residual block:
    Conv1D(filters, kernel_size, dilation_rate=d, causal) -> LeakyReLU -> Dropout
    Conv1D(filters, kernel_size, dilation_rate=d, causal) -> LeakyReLU -> Dropout
    + residual connection (1x1 Conv1D to match channels on the first block)

Causal padding + growing dilation is the standard TCN recipe from the
literature this ablation is drawn from (ActionPlan.md 9.1: "TCNs model
long temporal dependencies without recurrence, and are typically easier
to parallelize/faster at inference"). Causal masking isn't strictly
required for correctness here (each instance is a whole, already-segmented
window, same as the BiLSTM case) but keeping it standard makes this a fair,
literature-comparable ablation entry rather than a bespoke variant.

Mirrors cnn_lstm.py's public interface (build_encoder, build_classifier,
build_full_model) so train.py / evaluate.py select architectures by name
only -- see models/__init__.py's ARCH_BUILDERS registry.
"""
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

    # 1x1 conv to match channel counts so the residual add is valid the
    # first time channels change (input has NUM_SENSOR_CHANNELS=9,
    # blocks have `filters`); a no-op-shape passthrough after that.
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
    """Returns (full_model, encoder, classifier) -- same contract as
    cnn_lstm.build_full_model. Feature dim here equals `filters` (64 by
    default, from GlobalAveragePooling1D), read off encoder.output_shape
    rather than hardcoded."""
    encoder = build_encoder(seq_len, n_channels, dropout)
    classifier = build_classifier(encoder.output_shape[-1], num_classes)

    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    features = encoder(inputs)
    outputs = classifier(features)
    full_model = keras.Model(inputs, outputs, name="tcn")
    return full_model, encoder, classifier
