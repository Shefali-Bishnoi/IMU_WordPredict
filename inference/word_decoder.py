"""
Word decoder: beam search, dictionary correction, and optional n-gram scoring.

Pipeline: character probabilities -> beam_search -> dictionary correction
-> weighted scoring (beam + edit distance + frequency + optional LM).

search_lambda_lm steers beam expansion; ScoreWeights.delta re-ranks survivors.
For weight tuning, call decode_raw() once and score_raw_candidates() many times.
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
    """final_score = alpha*beam + beta*edit + gamma*frequency + delta*lm_score"""

    alpha: float = 0.6
    beta: float = 0.25
    gamma: float = 0.15
    delta: float = 0.0


@dataclass
class RawCandidate:
    """Weight-independent decode output for one (beam, correction) pair."""

    word: str
    raw: str
    beam_score: float
    edit_similarity: float
    word_frequency: float
    is_known_word: bool
    log_probability: float
    lm_score: float = 0.0
    lm_log_probability: float = 0.0


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
    """Beam search + dictionary correction + weighted scoring."""

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
        max_ties_per_beam: int = 3,
    ):
        self.beam_width = beam_width
        self.top_k = top_k
        self.case_sensitive = case_sensitive
        self.weights = weights or ScoreWeights()
        self.vocab_size = vocab_size
        self.max_search_distance = max_search_distance
        self.ngram_model = ngram_model
        self.search_lambda_sensor = search_lambda_sensor
        self.search_lambda_lm = search_lambda_lm
        self.max_ties_per_beam = max_ties_per_beam

    def _normalize_case(self, word: str) -> str:
        return word if self.case_sensitive else word.lower()

    def _correct_candidates(self, raw_text: str) -> list[tuple[str, float, float, bool]]:
        """Return all dictionary corrections tied at minimum edit distance."""
        lookup_text = self._normalize_case(raw_text)
        if is_known_word(lookup_text, self.vocab_size):
            return [(lookup_text, 1.0, frequency_score(lookup_text), True)]
        hits = edit_distance.nearest_known_words(
            lookup_text, self.vocab_size, self.max_search_distance,
            max_candidates=self.max_ties_per_beam,
        )
        return [(word, sim, frequency_score(word), False) for word, sim in hits]

    def decode_raw(self, character_probabilities) -> list:
        """Run beam search and dictionary correction; cache for weight tuning."""
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
        span = (hi - lo) or 1.0

        raw_candidates: list = []
        for r in beam_results:
            raw_text = r["text"]
            beam_score_norm = (r["log_probability"] - lo) / span

            for corrected_word, edit_sim, word_freq, known in self._correct_candidates(raw_text):
                lm_log = 0.0
                if self.ngram_model is not None:
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
            lm_logs = [c.lm_log_probability for c in raw_candidates]
            lm_lo, lm_hi = min(lm_logs), max(lm_logs)
            lm_span = (lm_hi - lm_lo) or 1.0
            for c in raw_candidates:
                c.lm_score = (c.lm_log_probability - lm_lo) / lm_span

        return self._dedupe_by_word(raw_candidates)

    def _dedupe_by_word(self, raw_candidates: list) -> list:
        """Keep one RawCandidate per word (best log_probability)."""
        best_by_word: dict[str, RawCandidate] = {}
        for rc in raw_candidates:
            key = rc.word.lower()
            current = best_by_word.get(key)
            if current is None:
                best_by_word[key] = rc
                continue
            candidate_key = (rc.log_probability, rc.edit_similarity, rc.word_frequency)
            current_key = (current.log_probability, current.edit_similarity, current.word_frequency)
            if candidate_key > current_key:
                best_by_word[key] = rc
        return sorted(
            best_by_word.values(),
            key=lambda c: c.log_probability,
            reverse=True,
        )

    @staticmethod
    def score_raw_candidates(raw_candidates: list, weights: "ScoreWeights") -> dict:
        """Apply ScoreWeights to cached decode_raw() output."""
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
        """decode_raw() + score with self.weights."""
        raw_candidates = self.decode_raw(character_probabilities)
        return self.score_raw_candidates(raw_candidates, self.weights)