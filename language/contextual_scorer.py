"""Contextual reranking of word-decoder candidates using a causal LM."""
from __future__ import annotations

from typing import Optional

from config import LANGUAGE_CONTEXT_WORDS, LANGUAGE_MODEL_TOP_K_CANDIDATES, LANGUAGE_MODEL_WEIGHT
from language.causal_lm import CausalLanguageModel


def build_context_string(committed_words: list[str], max_words: int = LANGUAGE_CONTEXT_WORDS) -> str:
    """Build context from the most recent committed words."""
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
    """Rerank decoder candidates; falls back to original order on failure."""
    if not candidates:
        return {
            "used_language_model": False, "context": "", "candidates": [],
            "selected_word": "", "reranked": False,
        }

    original_best = candidates[0]["word"]

    if lm is None:
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
