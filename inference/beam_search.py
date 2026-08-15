"""
Beam search over per-position character-probability vectors
(ActionPlan.md Priority 3).

Input shape: (T, 52) -- one 52-class probability vector per character
position, exactly what repeated calls to CharacterRecognizer.predict()
already produce. This module doesn't care how the (T, 52) sequence was
assembled -- app/session.py's WordBuffer already does that.
"""
from __future__ import annotations

import math

from config import NUM_CLASSES, index_to_label

_EPS = 1e-12  # floor for log(0) safety -- never take log of a raw 0.0


class Hypothesis:
    __slots__ = ("chars", "log_prob")

    def __init__(self, chars: tuple[str, ...] = (), log_prob: float = 0.0):
        self.chars = chars
        self.log_prob = log_prob

    @property
    def text(self) -> str:
        return "".join(self.chars)

    def extend(self, char: str, char_log_prob: float) -> "Hypothesis":
        return Hypothesis(self.chars + (char,), self.log_prob + char_log_prob)


def _validate_probabilities(probabilities) -> None:
    if not probabilities:
        raise ValueError("probabilities must contain at least one position")
    for i, row in enumerate(probabilities):
        if len(row) != NUM_CLASSES:
            raise ValueError(
                f"position {i}: expected {NUM_CLASSES} class probabilities, got {len(row)}"
            )


def _top_k_for_position(prob_row, top_k: int) -> list[tuple[str, float]]:
    """Top-k (char, log_prob) pairs for one position, sorted by
    probability descending. Zero/negative/NaN probabilities are floored
    to _EPS before log() so a genuinely zero softmax output never
    raises or produces -inf."""
    indexed = sorted(enumerate(prob_row), key=lambda kv: kv[1], reverse=True)
    out = []
    for idx, p in indexed[:top_k]:
        safe_p = p if (p is not None and p > 0) else _EPS
        out.append((index_to_label(idx), math.log(safe_p)))
    return out


def beam_search(probabilities, beam_width: int = 5, top_k: int = 5) -> list[dict]:
    """
    probabilities: sequence of T rows, each a 52-length probability
    vector (list/tuple/np.ndarray all fine -- only indexing/len used).
    beam_width: hypotheses kept alive at every step. 1 == greedy.
    top_k: candidate characters considered per position when expanding
    each surviving hypothesis (NOT the number of final results
    returned -- that's always `beam_width` many).

    Returns candidates ranked best-first:
        {"text": str, "log_probability": float, "probability": float}
    Log probabilities are summed throughout the search, never
    multiplied; `probability` is only computed once at the end via
    exp(log_probability), purely for display.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    _validate_probabilities(probabilities)

    beams = [Hypothesis()]
    for position_row in probabilities:
        char_options = _top_k_for_position(position_row, top_k)
        expanded = [
            hyp.extend(char, char_log_prob)
            for hyp in beams
            for char, char_log_prob in char_options
        ]
        expanded.sort(key=lambda h: h.log_prob, reverse=True)
        beams = expanded[:beam_width]

    beams.sort(key=lambda h: h.log_prob, reverse=True)
    return [
        {"text": h.text, "log_probability": h.log_prob, "probability": math.exp(h.log_prob)}
        for h in beams
    ]