"""Session-scoped residual adapter between frozen encoder and classifier.

    h  = encoder(x)
    h' = h + adapter(h)
    p  = classifier(h')

The adapter output projection is zero-initialized so h' == h at startup.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


class SessionAdapter(keras.layers.Layer):
    def __init__(self, feature_dim: int, bottleneck: int = 16, name: str = "session_adapter", **kw):
        super().__init__(name=name, **kw)
        self.down = layers.Dense(bottleneck, activation="relu", name=f"{name}_down")
        self.up = layers.Dense(
            feature_dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name=f"{name}_up",
        )

    def call(self, h):
        return h + self.up(self.down(h))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"feature_dim": self.up.units, "bottleneck": self.down.units})
        return cfg


def build_personalized_model(
    encoder: keras.Model,
    classifier: keras.Model,
    bottleneck: int = 16,
) -> tuple[keras.Model, SessionAdapter]:
    """Build a session-scoped model with a trainable adapter."""
    feature_dim = int(encoder.output_shape[-1])
    seq_len, n_channels = encoder.input_shape[1], encoder.input_shape[2]

    encoder.trainable = False
    classifier.trainable = False

    inputs = keras.Input(shape=(seq_len, n_channels), name="imu_input")
    h = encoder(inputs, training=False)
    adapter = SessionAdapter(feature_dim=feature_dim, bottleneck=bottleneck)
    h_adapted = adapter(h)
    outputs = classifier(h_adapted, training=False)
    model = keras.Model(inputs, outputs, name="personalized_session_model")
    return model, adapter
