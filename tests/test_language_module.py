"""
tests/test_language_module.py

Unit tests for the NEW Level-3 language layer (language/causal_lm.py,
language/contextual_scorer.py). These tests deliberately do NOT download
a real HuggingFace model -- they use a tiny in-memory FakeLM that
implements the same `score_next_word(context, candidate_word) -> float`
interface CausalLanguageModel exposes, so the test suite is fast,
deterministic, and runnable with no network access or GPU.

Also includes a smoke-level regression check that the EXISTING pipeline
(dictionary correction via app/correction.py) is untouched by this
feature -- see test_existing_dictionary_correction_unaffected.

Run:
    pytest tests/test_language_module.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from language.contextual_scorer import build_context_string, rerank_candidates


class FakeLM:
    """Deterministic stand-in for CausalLanguageModel. `preferences` maps
    (context, candidate_word) -> log-prob; anything not listed gets a
    fixed low score, so tests can assert exactly which candidate wins."""

    def __init__(self, preferences: dict, default: float = -10.0):
        self.preferences = preferences
        self.default = default
        self.calls = []

    def score_next_word(self, context: str, candidate_word: str) -> float:
        self.calls.append((context, candidate_word))
        return self.preferences.get((context, candidate_word), self.default)


class FailingLM:
    def score_next_word(self, context: str, candidate_word: str) -> float:
        raise RuntimeError("simulated inference failure")


# ---------------------------------------------------------------------------
# 1. Context window
# ---------------------------------------------------------------------------
def test_context_window_respects_limit():
    words = [f"w{i}" for i in range(30)]
    ctx = build_context_string(words, max_words=5)
    assert ctx == " ".join(words[-5:])


def test_context_window_empty_history():
    assert build_context_string([], max_words=20) == ""


# ---------------------------------------------------------------------------
# 2. Candidate scoring / reranking
# ---------------------------------------------------------------------------
def test_rerank_prefers_higher_lm_score():
    candidates = [
        {"word": "too", "final_score": 0.9},   # decoder's top pick
        {"word": "to", "final_score": 0.85},
        {"word": "two", "final_score": 0.80},
    ]
    lm = FakeLM({
        ("I am going", "too"): -8.0,
        ("I am going", "to"): -1.0,   # much more plausible in context
        ("I am going", "two"): -9.0,
    })
    result = rerank_candidates(
        context_words=["I", "am", "going"], candidates=candidates, lm=lm,
        weight=1.0,  # large weight so the LM term can actually flip the order in this test
    )
    assert result["used_language_model"] is True
    assert result["selected_word"] == "to"
    assert result["reranked"] is True  # decoder's top was "too", LM flipped it to "to"


def test_rerank_does_not_flip_when_lm_agrees():
    candidates = [
        {"word": "apple", "final_score": 0.95},
        {"word": "apply", "final_score": 0.40},
    ]
    lm = FakeLM({("I ate an", "apple"): -1.0, ("I ate an", "apply"): -9.0})
    result = rerank_candidates(["I", "ate", "an"], candidates, lm=lm, weight=0.15)
    assert result["selected_word"] == "apple"
    assert result["reranked"] is False


# ---------------------------------------------------------------------------
# 3. Language model disabled / unavailable -> exact fallback
# ---------------------------------------------------------------------------
def test_rerank_with_no_lm_falls_back_to_existing_order():
    candidates = [{"word": "hello", "final_score": 0.9}, {"word": "hallo", "final_score": 0.5}]
    result = rerank_candidates(["hi"], candidates, lm=None)
    assert result["used_language_model"] is False
    assert result["selected_word"] == "hello"
    assert result["reranked"] is False
    assert result["candidates"][0]["word"] == "hello"


def test_rerank_survives_lm_failure_and_falls_back():
    candidates = [{"word": "hello", "final_score": 0.9}, {"word": "hallo", "final_score": 0.5}]
    result = rerank_candidates(["hi"], candidates, lm=FailingLM())
    assert result["used_language_model"] is False
    assert result["selected_word"] == "hello"  # unchanged, exactly the decoder's own pick


# ---------------------------------------------------------------------------
# 4. Empty candidate list
# ---------------------------------------------------------------------------
def test_rerank_with_empty_candidates():
    result = rerank_candidates(["some", "context"], [], lm=FakeLM({}))
    assert result["used_language_model"] is False
    assert result["selected_word"] == ""
    assert result["candidates"] == []


# ---------------------------------------------------------------------------
# 5. Empty context (first word of a session)
# ---------------------------------------------------------------------------
def test_rerank_with_empty_context_still_scores():
    candidates = [{"word": "hello", "final_score": 0.5}, {"word": "hallo", "final_score": 0.4}]
    lm = FakeLM({("", "hello"): -1.0, ("", "hallo"): -5.0})
    result = rerank_candidates([], candidates, lm=lm, weight=1.0)
    assert result["context"] == ""
    assert result["selected_word"] == "hello"


# ---------------------------------------------------------------------------
# 6. Top-K truncation -- LM is never asked to score more than top_k
# ---------------------------------------------------------------------------
def test_rerank_only_scores_top_k_candidates():
    candidates = [{"word": f"w{i}", "final_score": 1.0 - i * 0.1} for i in range(10)]
    lm = FakeLM({})
    result = rerank_candidates(["ctx"], candidates, lm=lm, top_k=3)
    assert len(lm.calls) == 3
    # Candidates beyond top_k are still present in the output, unscored,
    # appended after the reranked subset -- never dropped.
    assert len(result["candidates"]) == 10


# ---------------------------------------------------------------------------
# 7. Weight configurability -- a near-zero weight should not flip the order
# ---------------------------------------------------------------------------
def test_low_weight_does_not_override_strong_sensor_evidence():
    candidates = [
        {"word": "apple", "final_score": 0.95},
        {"word": "apply", "final_score": 0.10},
    ]
    lm = FakeLM({("context", "apple"): -9.0, ("context", "apply"): -0.1})
    result = rerank_candidates(["context"], candidates, lm=lm, weight=0.01)
    # LM strongly prefers "apply", but the weight is tiny relative to the
    # 0.85 sensor-score gap -- "apple" should still win. This is the
    # concrete regression test for "language model does not blindly
    # replace IMU predictions" / "default conservatively".
    assert result["selected_word"] == "apple"


# ---------------------------------------------------------------------------
# 8. CausalLanguageModel.load() degrades gracefully without transformers
# ---------------------------------------------------------------------------
def test_causal_lm_load_never_raises_on_missing_dependency(monkeypatch):
    from language import causal_lm

    monkeypatch.setattr(causal_lm, "_HF_IMPORT_ERROR", "simulated: transformers not installed")
    result = causal_lm.CausalLanguageModel.load("distilgpt2")
    assert result is None  # must not raise


# ---------------------------------------------------------------------------
# 9. Existing pipeline (dictionary correction) unaffected -- regression
# guard for "Existing dictionary correction still works".
# ---------------------------------------------------------------------------
def test_existing_dictionary_correction_unaffected():
    from language.wordfreq_scorer import is_known_word

    assert is_known_word("apple") is True
    assert is_known_word("zzqxwnotaword") is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
