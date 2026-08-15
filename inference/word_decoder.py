"""
High-level word decoder: combines beam_search.py's character-level beam
search with language/wordfreq_scorer.py + language/edit_distance.py
(dictionary/frequency correction) and, optionally, language/ngram.py (a
character n-gram language model -- ActionPlan.md Priority 4).

Pipeline:
    TCN probabilities (T, 52)
            |
     beam_search()  <-- optionally LM-guided during expansion
            |
    top `beam_width` candidate strings
            |
    per candidate: wordfreq lookup + edit-distance correction
          + (optional) full-word n-gram LM score
            |
    combine beam score + edit similarity + word frequency + LM score
            |
    ranked final candidates + single best prediction

Two SEPARATE places the LM can matter, and they are controlled by
separate knobs on purpose (FuturePlan.md Sec.6.3):
  1. SEARCH-TIME steering (`search_lambda_lm`, passed into beam_search):
     changes which hypotheses survive pruning at all.
  2. FINAL re-ranking (`ScoreWeights.delta`): changes how the SURVIVING
     candidates get ordered, exactly like alpha/beta/gamma already do.
Setting both to non-zero is expected once you've validated the LM helps;
setting either to 0.0 disables just that half without touching the other.

IMPORTANT (perf, unchanged from before): beam search + dictionary
correction (decode_raw) does NOT depend on ScoreWeights at all -- only
the final weighted sum does. A weight grid-search should call
decode_raw() ONCE per word and then call score_raw_candidates() many
times (once per weight combo) against that same result, instead of
re-running decode() per combo. See experiments/tune_decoder_weights.py
-- if you add `delta`/ngram support there, keep this split: decode_raw()
is still the only place beam search + BK-tree lookups happen, and it now
ALSO includes the one-time full-word LM score per candidate; grid-search
over delta should still be a cheap re-weighting of already-computed
values, not a re-decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inference.beam_search import beam_search
from language import edit_distance
from language.ngram import NgramLanguageModel
from language.wordfreq_scorer import DEFAULT_VOCAB_SIZE, frequency_score, is_known_word


@dataclass
class ScoreWeights:
    """final_score = alpha*beam_score + beta*edit_similarity
                    + gamma*word_frequency + delta*lm_score
    Defaults keep delta=0.0 so any existing decoder_weights.json (which
    has no "delta" key) still loads correctly via
    ScoreWeights(alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    in app/correction.py -- the LM term is opt-in, not silently active."""

    alpha: float = 0.6    # weight on the beam/model (sensor) score
    beta: float = 0.25    # weight on edit-distance similarity to nearest known word
    gamma: float = 0.15   # weight on that word's wordfreq frequency
    delta: float = 0.0    # weight on the n-gram LM's full-word score (NEW)


@dataclass
class RawCandidate:
    """The weight-INDEPENDENT output of decoding one beam hypothesis.
    Computing this is the expensive part (beam search + BK-tree
    edit-distance lookup + one n-gram score_word call). Cache/reuse it
    across many ScoreWeights instead of recomputing it per weight combo."""

    word: str
    raw: str
    beam_score: float
    edit_similarity: float
    word_frequency: float
    is_known_word: bool
    log_probability: float
    lm_score: float = 0.0            # normalized [0,1] within this candidate set; 0.0 if no ngram_model
    lm_log_probability: float = 0.0  # raw log-prob, kept for debugging/inspection


@dataclass
class Candidate:
    word: str
    raw: str
    beam_score: float
    edit_similarity: float
    word_frequency: float
    lm_score: float
    final_score: float
    is_known_word: bool
    log_probability: float


class WordDecoder:
    """
    decoder = WordDecoder(beam_width=5, top_k=5)
    result = decoder.decode(character_probabilities)

    With an n-gram LM (ActionPlan.md Priority 4):
        model = NgramLanguageModel.load(EXPERIMENTS_DIR / "ngram_model.json")
        decoder = WordDecoder(
            beam_width=5, top_k=5, ngram_model=model,
            search_lambda_lm=0.2,               # steers search (0.0 = off)
            weights=ScoreWeights(delta=0.2),    # steers final re-ranking (0.0 = off)
        )

    For weight grid-searches, prefer:
        raw = decoder.decode_raw(character_probabilities)          # once
        result = WordDecoder.score_raw_candidates(raw, weights)     # many times, cheap
    """

    def __init__(
        self,
        beam_width: int = 5,
        top_k: int = 5,
        case_sensitive: bool = False,
        weights: "ScoreWeights" = None,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        max_search_distance: int = 3,
        ngram_model: Optional[NgramLanguageModel] = None,
        search_lambda_sensor: float = 1.0,
        search_lambda_lm: float = 0.0,
    ):
        self.beam_width = beam_width
        self.top_k = top_k
        self.case_sensitive = case_sensitive
        self.weights = weights or ScoreWeights()
        self.vocab_size = vocab_size
        self.max_search_distance = max_search_distance
        self.ngram_model = ngram_model
        # search_lambda_lm=0.0 (default) reproduces the old sensor-only
        # beam search exactly, even with an ngram_model attached -- so
        # attaching a model purely for the FINAL re-ranking term (delta
        # > 0) without opting into search-time steering is safe.
        self.search_lambda_sensor = search_lambda_sensor
        self.search_lambda_lm = search_lambda_lm

    def _normalize_case(self, word: str) -> str:
        return word if self.case_sensitive else word.lower()

    def _correct_candidate(self, raw_text: str) -> tuple:
        lookup_text = self._normalize_case(raw_text)
        if is_known_word(lookup_text, self.vocab_size):
            return lookup_text, 1.0, frequency_score(lookup_text), True
        nearest, similarity = edit_distance.nearest_known_word(
            lookup_text, self.vocab_size, self.max_search_distance
        )
        return nearest, similarity, frequency_score(nearest), False

    def decode_raw(self, character_probabilities) -> list:
        """The expensive, weight-INDEPENDENT half of decode(): runs beam
        search (optionally LM-guided) once, corrects each beam candidate
        against the dictionary once, and (if an ngram_model is attached)
        scores each corrected candidate's full word once. Cache/reuse
        the returned list across many ScoreWeights via
        score_raw_candidates() instead of calling decode() repeatedly."""
        lm_scorer = self.ngram_model.next_char_logprob if self.ngram_model else None
        beam_results = beam_search(
            character_probabilities,
            beam_width=self.beam_width,
            top_k=self.top_k,
            lm_scorer=lm_scorer,
            lambda_sensor=self.search_lambda_sensor,
            lambda_lm=self.search_lambda_lm,
        )
        if not beam_results:
            return []

        log_probs = [r["log_probability"] for r in beam_results]
        lo, hi = min(log_probs), max(log_probs)
        span = (hi - lo) or 1.0  # avoid div-by-zero when all beams tie

        raw_candidates: list = []
        for r in beam_results:
            raw_text = r["text"]
            beam_score_norm = (r["log_probability"] - lo) / span  # -> [0, 1]
            corrected_word, edit_sim, word_freq, known = self._correct_candidate(raw_text)

            lm_log = 0.0
            if self.ngram_model is not None:
                # Score the CORRECTED word (what will actually be shown/
                # scored), not the raw beam text -- consistent with how
                # edit_similarity/word_frequency already work.
                lm_log = self.ngram_model.score_word(corrected_word)

            raw_candidates.append(
                RawCandidate(
                    word=corrected_word, raw=raw_text, beam_score=beam_score_norm,
                    edit_similarity=edit_sim, word_frequency=word_freq,
                    is_known_word=known, log_probability=r["log_probability"],
                    lm_score=0.0, lm_log_probability=lm_log,
                )
            )

        if self.ngram_model is not None and raw_candidates:
            # Normalize lm_log_probability to [0,1] relative to THIS
            # candidate set's own min/max, the same way beam_score is
            # normalized -- raw n-gram log-probs aren't naturally in
            # [0,1] and depend on word length, so a fixed global scale
            # would make `delta` mean something different for every word.
            lm_logs = [c.lm_log_probability for c in raw_candidates]
            lm_lo, lm_hi = min(lm_logs), max(lm_logs)
            lm_span = (lm_hi - lm_lo) or 1.0
            for c in raw_candidates:
                c.lm_score = (c.lm_log_probability - lm_lo) / lm_span

        return raw_candidates

    @staticmethod
    def score_raw_candidates(raw_candidates: list, weights: "ScoreWeights") -> dict:
        """Cheap weighted-sum scoring over an already-decoded candidate
        list -- no beam search, no dictionary lookups, no n-gram calls.
        This is what should run inside a weight-grid-search loop;
        decode_raw() should run only once per word."""
        if not raw_candidates:
            return {"prediction": "", "candidates": []}

        scored: list = []
        for rc in raw_candidates:
            final_score = (
                weights.alpha * rc.beam_score
                + weights.beta * rc.edit_similarity
                + weights.gamma * rc.word_frequency
                + weights.delta * rc.lm_score
            )
            scored.append(
                Candidate(
                    word=rc.word, raw=rc.raw, beam_score=rc.beam_score,
                    edit_similarity=rc.edit_similarity, word_frequency=rc.word_frequency,
                    lm_score=rc.lm_score, final_score=final_score,
                    is_known_word=rc.is_known_word, log_probability=rc.log_probability,
                )
            )

        scored.sort(key=lambda c: c.final_score, reverse=True)
        best = scored[0]
        return {
            "prediction": best.word,
            "candidates": [
                {
                    "word": c.word, "raw": c.raw, "beam_score": c.beam_score,
                    "edit_similarity": c.edit_similarity, "word_frequency": c.word_frequency,
                    "lm_score": c.lm_score, "final_score": c.final_score,
                    "is_known_word": c.is_known_word, "log_probability": c.log_probability,
                }
                for c in scored
            ],
        }

    def decode(self, character_probabilities) -> dict:
        """Convenience wrapper: decode_raw() + score with self.weights in
        one call. If you're sweeping many weight combos against the same
        words, use decode_raw() + score_raw_candidates() directly instead
        (see experiments/tune_decoder_weights.py)."""
        raw_candidates = self.decode_raw(character_probabilities)
        return self.score_raw_candidates(raw_candidates, self.weights)