"""
Word-correction seam. Loads ScoreWeights (incl. delta) + tau_word +
search_lambda_lm from experiments/decoder_weights.json when present
(produced by experiments/tune_decoder_weights.py); falls back to
untuned defaults otherwise, so this works before AND after tuning with
no code change.

If decoder_weights.json has a nonzero "delta" and/or "search_lambda_lm"
(i.e. it was produced by a tuning run that had an n-gram LM available),
the same n-gram model is loaded here too -- otherwise those fields would
be silently inert at inference (delta would multiply a ScoreWeights.
lm_score of 0.0, and search_lambda_lm would have no lm_scorer to steer
with). Older decoder_weights.json files without these keys still load
fine: ScoreWeights.delta and search_lambda_lm both default to 0.0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import config
from inference.word_decoder import ScoreWeights, WordDecoder
from language.ngram import NgramLanguageModel

_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "decoder_weights.json"


def _load_tuned_config() -> tuple[ScoreWeights, float, "NgramLanguageModel | None", float]:
    if _WEIGHTS_PATH.exists():
        with open(_WEIGHTS_PATH) as f:
            cfg = json.load(f)
        weights = ScoreWeights(
            alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
            delta=cfg.get("delta", 0.0),
        )
        tau_word = float(cfg.get("tau_word", 0.6))
        search_lambda_lm = float(cfg.get("search_lambda_lm", 0.0))

        ngram_model = None
        if weights.delta != 0.0 or search_lambda_lm != 0.0:
            ngram_path = Path(cfg["ngram_model_used"]) if cfg.get("ngram_model_used") \
                else config.NGRAM_MODEL_PATH
            if ngram_path.exists():
                ngram_model = NgramLanguageModel.load(ngram_path)
            else:
                print(f"[correction] WARNING: tuned config wants delta={weights.delta} "
                      f"search_lambda_lm={search_lambda_lm} but n-gram model not found at "
                      f"{ngram_path} -- LM terms will have no effect this run.")

        print(f"[correction] loaded tuned weights: alpha={weights.alpha} "
              f"beta={weights.beta} gamma={weights.gamma} delta={weights.delta} "
              f"tau_word={tau_word} search_lambda_lm={search_lambda_lm} "
              f"ngram_loaded={ngram_model is not None}")
        return weights, tau_word, ngram_model, search_lambda_lm
    print(f"[correction] no tuned weights at {_WEIGHTS_PATH} -- using untuned defaults. "
          f"Run `python -m experiments.tune_decoder_weights` to generate one.")
    return ScoreWeights(), 0.6, None, 0.0


_weights, _tau_word, _ngram_model, _search_lambda_lm = _load_tuned_config()
_decoder = WordDecoder(
    beam_width=5, top_k=5, weights=_weights,
    ngram_model=_ngram_model, search_lambda_lm=_search_lambda_lm,
)


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