"""Beam search over per-position character probability vectors."""
from __future__ import annotations

import math
from typing import Callable, Optional

from config import NUM_CLASSES, index_to_label

_EPS = 1e-12  # floor for log(0)

# (prefix_chars_so_far, next_char) -> log P(next_char | prefix)
LmScorer = Callable[[list, str], float]


class Hypothesis:
    __slots__ = ("chars", "sensor_log_prob", "lm_log_prob", "search_score")

    def __init__(
        self,
        chars: tuple = (),
        sensor_log_prob: float = 0.0,
        lm_log_prob: float = 0.0,
        search_score: float = 0.0,
    ):
        self.chars = chars
        self.sensor_log_prob = sensor_log_prob
        self.lm_log_prob = lm_log_prob
        self.search_score = search_score

    @property
    def text(self) -> str:
        return "".join(self.chars)

    @property
    def log_prob(self) -> float:
        """Sensor log-prob only (LM influence is in search_score)."""
        return self.sensor_log_prob

    def extend(
        self,
        char: str,
        char_log_prob: float,
        lambda_sensor: float,
        lambda_lm: float,
        lm_scorer: Optional[LmScorer],
    ) -> "Hypothesis":
        new_sensor = self.sensor_log_prob + char_log_prob
        new_lm = (
            self.lm_log_prob + lm_scorer(list(self.chars), char)
            if lm_scorer is not None else self.lm_log_prob
        )
        new_search = lambda_sensor * new_sensor + lambda_lm * new_lm
        return Hypothesis(self.chars + (char,), new_sensor, new_lm, new_search)


def _validate_probabilities(probabilities) -> None:
    if not probabilities:
        raise ValueError("probabilities must contain at least one position")
    for i, row in enumerate(probabilities):
        if len(row) != NUM_CLASSES:
            raise ValueError(
                f"position {i}: expected {NUM_CLASSES} class probabilities, got {len(row)}"
            )


def _top_k_for_position(prob_row, top_k: int) -> list:
    """Top-k (char, log_prob) pairs for one position."""
    indexed = sorted(enumerate(prob_row), key=lambda kv: kv[1], reverse=True)
    out = []
    for idx, p in indexed[:top_k]:
        safe_p = p if (p is not None and p > 0) else _EPS
        out.append((index_to_label(idx), math.log(safe_p)))
    return out


def beam_search(
    probabilities,
    beam_width: int = 5,
    top_k: int = 5,
    lm_scorer: Optional[LmScorer] = None,
    lambda_sensor: float = 1.0,
    lambda_lm: float = 0.0,
) -> list:
    """Beam search over (T, 52) character probability sequences."""
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    _validate_probabilities(probabilities)

    beams = [Hypothesis()]
    for position_row in probabilities:
        char_options = _top_k_for_position(position_row, top_k)
        expanded = [
            hyp.extend(char, char_log_prob, lambda_sensor, lambda_lm, lm_scorer)
            for hyp in beams
            for char, char_log_prob in char_options
        ]
        expanded.sort(key=lambda h: h.search_score, reverse=True)
        beams = expanded[:beam_width]

    beams.sort(key=lambda h: h.search_score, reverse=True)
    return [
        {
            "text": h.text,
            "log_probability": h.sensor_log_prob,
            "lm_log_probability": h.lm_log_prob,
            "search_score": h.search_score,
            "probability": math.exp(h.sensor_log_prob),
        }
        for h in beams
    ]