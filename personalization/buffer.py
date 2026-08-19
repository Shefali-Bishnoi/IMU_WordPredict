"""
Session-scoped adaptation buffer -- Level-1 (explicit user correction)
labels only, per ActionPlan.md 13.6's "safest first" build order. Holds
already-preprocessed model inputs (same preprocess() as everywhere else
-- preprocessing.segment.preprocess), since the adapter needs real
model-shaped inputs to train on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptationSample:
    x: np.ndarray   # preprocessed (seq_len, n_channels) float32
    y: int          # correct class index (config.label_to_index)


@dataclass
class SessionAdaptationBuffer:
    samples: list = field(default_factory=list)

    def add(self, x: np.ndarray, y: int) -> None:
        self.samples.append(AdaptationSample(x=np.asarray(x, dtype=np.float32), y=int(y)))

    def __len__(self) -> int:
        return len(self.samples)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([s.x for s in self.samples]).astype(np.float32)
        y = np.array([s.y for s in self.samples], dtype=np.int64)
        return X, y

    def clear(self) -> None:
        self.samples.clear()