"""
language/contextual_scorer.py

Level 3 of the three-level architecture (ActionPlan.md-style separation,
extended for this feature):

    Level 1: IMU -> character                  (TCN, unchanged)
    Level 2: user IMU -> personalized character (SessionAdapter, unchanged)
    Level 3: words -> contextual plausibility   (THIS MODULE, new)

This module is the ONLY place app/main.py talks to for language scoring
-- it never imports transformers/torch itself; all of that lives behind
language/causal_lm.CausalLanguageModel. That keeps the rest of the app
completely decoupled from which LM library/model is in use.

Contract:

    rerank_candidates(context_words, candidates, lm, weight) -> dict

`candidates` is the EXISTING word-decoder output (a list of dicts with
at least "word" and "final_score" -- see inference/word_decoder.py's
Candidate/score_raw_candidates output). This function does NOT invent
new words -- it only re-scores and re-orders the candidates the
sensor/beam/dictionary pipeline already proposed (Definition of Done:
"Top-K candidate information is preserved", "language model does not
blindly replace IMU predictions", "goal is contextual RERANKING, not
unconstrained text generation").

Score combination: the existing per-candidate `final_score` (already a
combination of beam/edit-distance/frequency/n-gram per
inference/word_decoder.py's ScoreWeights) is treated as the
"sensor_or_beam_score + dictionary_score (+ existing n-gram, if any)"
term. This module does NOT re-derive alpha/beta/gamma/delta -- it adds
ONE new term on top, so nothing already tuned (experiments/decoder_weights.json)
is double-counted:

    combined_score = final_score + weight * normalized_lm_score

`normalized_lm_score` is the LM's log P(candidate | context), min-max
normalized to [0, 1] within this candidate set -- the same normalization
pattern inference/word_decoder.py already uses for beam_score/lm_score,
so the new term lives on a comparable scale to the existing ones instead
of an arbitrary raw log-prob magnitude dominating or being swamped.
"""
from __future__ import annotations

from typing import Optional

from config import LANGUAGE_CONTEXT_WORDS, LANGUAGE_MODEL_TOP_K_CANDIDATES, LANGUAGE_MODEL_WEIGHT
from language.causal_lm import CausalLanguageModel


def build_context_string(committed_words: list[str], max_words: int = LANGUAGE_CONTEXT_WORDS) -> str:
    """Only the most recent `max_words` committed words are used as
    context (Definition of Done: "context-window length is
    configurable"). Deliberately does NOT send the entire session
    history -- keeps latency/token count bounded regardless of how long
    the text buffer grows (ActionPlan-style "don't over-engineer")."""
    if not committed_words:
        return ""
    return " ".join(committed_words[-max_words:]) if max_words > 0 else " ".join(committed_words)


def rerank_candidates(
    context_words: list[str],
    candidates: list[dict],
    lm: Optional[CausalLanguageModel],
    weight: float = LANGUAGE_MODEL_WEIGHT,
    top_k: int = LANGUAGE_MODEL_TOP_K_CANDIDATES,
    context_word_limit: int = LANGUAGE_CONTEXT_WORDS,
) -> dict:
    """
    candidates: list of dicts, each with at least {"word": str,
        "final_score": float} -- typically WordDecoder.decode()'s
        result["candidates"], already ranked by the existing sensor/
        dictionary/personalization pipeline.

    Returns:
        {
          "used_language_model": bool,
          "context": str,
          "candidates": [ {word, final_score, lm_log_prob, lm_score,
                            combined_score}, ... ]   # re-sorted
          "selected_word": str,
          "reranked": bool,   # True iff the #1 word changed vs. input order
        }

    Never raises -- any failure (missing lm, empty candidates, scoring
    error) falls back to the ORIGINAL candidate order/selection, exactly
    matching the existing (pre-language-layer) behavior.
    """
    if not candidates:
        return {
            "used_language_model": False, "context": "", "candidates": [],
            "selected_word": "", "reranked": False,
        }

    original_best = candidates[0]["word"]

    if lm is None:
        # Language layer unavailable/disabled -- fall back exactly to
        # what the existing decoder already decided. This is the primary
        # "graceful degradation" path (config.LANGUAGE_MODEL_ENABLED=false,
        # or the model failed to load at startup).
        return {
            "used_language_model": False,
            "context": build_context_string(context_words, context_word_limit),
            "candidates": [dict(c) for c in candidates],
            "selected_word": original_best,
            "reranked": False,
        }

    context = build_context_string(context_words, context_word_limit)
    subset = candidates[:max(1, top_k)]

    try:
        scored = []
        for c in subset:
            lm_log_prob = lm.score_next_word(context, c["word"])
            scored.append({**c, "lm_log_prob": lm_log_prob})
    except Exception as e:  # noqa: BLE001 - any LM inference failure -> safe fallback
        print(f"[language] scoring failed ({e}) -- falling back to existing ranking")
        return {
            "used_language_model": False, "context": context,
            "candidates": [dict(c) for c in candidates],
            "selected_word": original_best, "reranked": False,
        }

    log_probs = [c["lm_log_prob"] for c in scored]
    lo, hi = min(log_probs), max(log_probs)
    span = (hi - lo) or 1.0
    for c in scored:
        c["lm_score"] = (c["lm_log_prob"] - lo) / span
        c["combined_score"] = c["final_score"] + weight * c["lm_score"]

    # Any candidates beyond top_k keep their original score, unscored by
    # the LM, and are appended after the reranked subset -- they were
    # already ranked lower by the sensor/dictionary pipeline, and we
    # don't spend LM calls on them.
    remainder = [
        {**c, "lm_log_prob": None, "lm_score": None, "combined_score": c["final_score"]}
        for c in candidates[len(subset):]
    ]

    scored.sort(key=lambda c: c["combined_score"], reverse=True)
    final_candidates = scored + remainder
    selected_word = final_candidates[0]["word"]

    print(f"[language] context={context!r} candidates={[c['word'] for c in subset]} "
          f"scores={[round(c['lm_log_prob'], 3) for c in scored]} selected={selected_word!r}")

    return {
        "used_language_model": True,
        "context": context,
        "candidates": final_candidates,
        "selected_word": selected_word,
        "reranked": selected_word != original_best,
    }
