"""
Session-scoped residual adapter (ActionPlan.md Priority 5, session-level
variant). Sits between the frozen encoder and frozen classifier of
whichever Priority-1 architecture won (tcn):

    h  = encoder(x)          # frozen, unchanged weights
    h' = h + adapter(h)      # NEW, trainable, lives only for one session
    p  = classifier(h')      # frozen, unchanged weights

Safety-by-construction: `up` (the adapter's output projection) is
zero-initialized, so adapter(h) == 0 for every h at construction time
-> h' == h exactly. A freshly-created SessionAdapter is mathematically
identical to no adapter at all. This is what guarantees every existing
script (evaluate.py, test_beam_dictionary.py, experiments/*) is
completely unaffected by this module existing -- nothing changes unless
adapt_session() is explicitly called on a session's own adapter.

encoder/classifier are frozen (`.trainable = False`) INSIDE
build_personalized_model only -- this flips the `.trainable` attribute
on the actual Keras Layer objects you pass in. If you reuse the SAME
encoder/classifier objects elsewhere (e.g. CharacterRecognizer.model),
freezing them here does not change CharacterRecognizer.model's own
predictions (that's a separate compiled Model over the same layers;
`.trainable` only affects gradient computation through the model that
is *actually compiled and fit*, i.e. `personalized_model`), but it DOES
mean encoder/classifier won't accidentally be trained if some other
code later calls .fit() on a model containing them. That's intentional
and matches ActionPlan.md 13.4 ("global model frozen during
personalization").
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
    """encoder/classifier should be the SEPARATE encoder/classifier
    objects loaded via inference.realtime.CharacterRecognizer (its
    .encoder / .classifier attributes, NOT .model). Returns a fresh
    (personalized_model, adapter) pair -- call this once per session,
    the first time that session needs personalization."""
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