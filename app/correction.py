"""
Word-correction seam. Loads ScoreWeights + tau_word from
experiments/decoder_weights.json when present (produced by
experiments/tune_decoder_weights.py); falls back to untuned defaults
otherwise, so this works before AND after tuning with no code change.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from inference.word_decoder import ScoreWeights, WordDecoder

_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "decoder_weights.json"


def _load_tuned_config() -> tuple[ScoreWeights, float]:
    if _WEIGHTS_PATH.exists():
        with open(_WEIGHTS_PATH) as f:
            cfg = json.load(f)
        weights = ScoreWeights(alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
        tau_word = float(cfg.get("tau_word", 0.6))
        print(f"[correction] loaded tuned weights: alpha={weights.alpha} "
              f"beta={weights.beta} gamma={weights.gamma} tau_word={tau_word}")
        return weights, tau_word
    print(f"[correction] no tuned weights at {_WEIGHTS_PATH} -- using untuned defaults. "
          f"Run `python -m experiments.tune_decoder_weights` to generate one.")
    return ScoreWeights(), 0.6


_weights, _tau_word = _load_tuned_config()
_decoder = WordDecoder(beam_width=5, top_k=5, weights=_weights)


@dataclass
class CorrectionResult:
    raw_word: str
    corrected_word: str
    confidence: float
    is_low_confidence: bool


def correct_word(characters: list[str], probabilities: list[list[float]]) -> CorrectionResult:
    raw_word = "".join(characters)
    if not probabilities:
        return CorrectionResult(raw_word, raw_word, confidence=1.0, is_low_confidence=False)

    result = _decoder.decode(probabilities)
    candidates = result["candidates"]
    best = candidates[0]

    if len(candidates) > 1:
        margin = best["final_score"] - candidates[1]["final_score"]
        confidence = 1.0 / (1.0 + math.exp(-6.0 * margin))
    else:
        confidence = 1.0

    return CorrectionResult(
        raw_word=raw_word,
        corrected_word=result["prediction"],
        confidence=confidence,
        is_low_confidence=confidence < _tau_word,
    )