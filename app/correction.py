"""Word-correction seam: loads tuned weights from experiments/decoder_weights.json."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import config
from inference.word_decoder import ScoreWeights, WordDecoder
from language.ngram import NgramLanguageModel

_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "experiments" / "decoder_weights.json"

_TOP_CANDIDATES_LIMIT = getattr(config, "LANGUAGE_MODEL_TOP_K_CANDIDATES", 5)


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
    beam_score: float | None = None
    edit_similarity: float | None = None
    word_frequency: float | None = None
    lm_score: float | None = None
    final_score: float | None = None
    is_known_word: bool | None = None
    top_candidates: list = field(default_factory=list)


def correct_word(characters: list[str], probabilities: list[list[float]]) -> CorrectionResult:
    raw_word = "".join(characters)
    if not probabilities:
        return CorrectionResult(
            raw_word, raw_word, confidence=1.0, is_low_confidence=False,
            top_candidates=[{"word": raw_word, "final_score": 1.0}],
        )

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
        beam_score=best.get("beam_score"),
        edit_similarity=best.get("edit_similarity"),
        word_frequency=best.get("word_frequency"),
        lm_score=best.get("lm_score"),
        final_score=best.get("final_score"),
        is_known_word=best.get("is_known_word"),
        top_candidates=candidates[:_TOP_CANDIDATES_LIMIT],
    )
