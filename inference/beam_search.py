"""
Beam search over per-position character-probability vectors
(ActionPlan.md Priority 3), OPTIONALLY guided by a character n-gram
language model during expansion (ActionPlan.md Priority 4 / FuturePlan.md
Sec.6.3).

Why the LM has to be wired in HERE, not only as post-hoc re-ranking:
inference/word_decoder.py's dictionary/frequency correction only ever
sees the beam's final top-B candidates -- it can reorder them, but it can
never pull in a hypothesis that pruning already discarded during search.
Passing `lm_scorer` lets the language model influence WHICH hypotheses
survive each expansion step, so a wider beam_width can actually recover
sequences that sensor-score-only pruning would have thrown away before
the dictionary/LM ever got a look. Per FuturePlan.md Sec.6.2, without
this the beam's #1 output is mathematically guaranteed to equal greedy's
#1 output, since summing independent per-position sensor log-probs is
maximized by the position-wise argmax -- an LM term only reordering a
FIXED top-B set can never change that fact; it has to affect survival.

Backward compatibility: `lm_scorer=None` and `lambda_lm=0.0` are the
defaults, and with lambda_lm=0.0 `search_score` reduces to exactly
`lambda_sensor * sensor_log_prob` -- identical ranking to the pre-LM
version of this file for any existing caller that doesn't pass the new
arguments. `log_probability` and `probability` in the returned dicts keep
their exact old meaning (sensor-only) so nothing downstream
(inference/word_decoder.py's beam_score normalization) needs to change
just because an LM is attached.

Input shape: (T, 52) -- one 52-class probability vector per character
position, exactly what repeated calls to CharacterRecognizer.predict()
already produce. This module doesn't care how the (T, 52) sequence was
assembled -- app/session.py's WordBuffer already does that.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from config import NUM_CLASSES, index_to_label

_EPS = 1e-12  # floor for log(0) safety -- never take log of a raw 0.0

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
        """Backward-compatible alias -- pure sensor log-prob only. LM
        influence lives in `lm_log_prob` / `search_score`, never mixed
        into this field."""
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


def beam_search(
    probabilities,
    beam_width: int = 5,
    top_k: int = 5,
    lm_scorer: Optional[LmScorer] = None,
    lambda_sensor: float = 1.0,
    lambda_lm: float = 0.0,
) -> list:
    """
    probabilities: sequence of T rows, each a 52-length probability
    vector (list/tuple/np.ndarray all fine -- only indexing/len used).
    beam_width: hypotheses kept alive at every step. 1 == greedy.
    top_k: candidate characters considered per position when expanding
    each surviving hypothesis (NOT the number of final results returned
    -- that's always `beam_width` many).
    lm_scorer: optional (prefix_chars, next_char) -> log P callable, e.g.
    language.ngram.NgramLanguageModel.next_char_logprob. When given,
    PRUNING at every step ranks hypotheses by
    `lambda_sensor*sensor_log_prob + lambda_lm*lm_log_prob`, not sensor
    score alone -- this is what lets the LM change which hypotheses
    survive, not just how survivors get reordered afterward.
    lambda_sensor / lambda_lm: weights on the two terms above, used ONLY
    to steer search-time pruning (separate concern from
    inference/word_decoder.py's ScoreWeights, which does the FINAL
    re-ranking over survivors). lambda_lm=0.0 (default) reproduces the
    exact previous sensor-only behavior even if lm_scorer is passed.

    Returns candidates ranked best-first by search_score:
        {"text": str,
         "log_probability": float,       # sensor-only, unchanged meaning
         "lm_log_probability": float,     # new; 0.0 if no lm_scorer given
         "search_score": float,           # new; what pruning/final sort used
         "probability": float}            # exp(log_probability), unchanged meaning
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