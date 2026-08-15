"""
High-level word decoder: combines beam_search.py's character-level beam
search with language/wordfreq_scorer.py + language/edit_distance.py, so
a beam candidate can be corrected toward the nearest real English word
instead of being handed to the user as-is.

Pipeline:
    TCN probabilities (T, 52)
            |
        beam_search()
            |
    top `beam_width` candidate strings
            |
    per candidate: wordfreq lookup + edit-distance correction
            |
    combine beam score + edit similarity + word frequency
            |
    ranked final candidates + single best prediction

IMPORTANT (perf): beam search + dictionary correction (decode_raw) does
NOT depend on ScoreWeights at all -- only the final weighted sum does.
A weight grid-search should call decode_raw() ONCE per word and then
call score_raw_candidates() many times (once per weight combo) against
that same result, instead of re-running decode() per combo. See
experiments/tune_decoder_weights.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from inference.beam_search import beam_search
from language import edit_distance
from language.wordfreq_scorer import DEFAULT_VOCAB_SIZE, frequency_score, is_known_word


@dataclass
class ScoreWeights:
    """final_score = alpha*beam_score + beta*edit_similarity + gamma*word_frequency
    Defaults are a reasonable starting point, NOT tuned on a validation
    set yet -- change these here, or pass your own ScoreWeights(...)
    into WordDecoder, once you have real accuracy numbers to tune
    against."""

    alpha: float = 0.6   # weight on the beam/model score
    beta: float = 0.25   # weight on edit-distance similarity to the nearest known word
    gamma: float = 0.15  # weight on that word's wordfreq frequency


@dataclass
class RawCandidate:
    """The weight-INDEPENDENT output of decoding one beam hypothesis:
    beam search's normalized score plus its dictionary correction.
    Computing this is the expensive part (beam search + BK-tree
    edit-distance lookup). Cache/reuse it across many ScoreWeights
    instead of recomputing it per weight combo."""

    word: str                 # corrected word used for scoring/display
    raw: str                  # original beam-search spelling, untouched
    beam_score: float         # normalized model/beam score, in [0, 1]
    edit_similarity: float
    word_frequency: float
    is_known_word: bool
    log_probability: float


@dataclass
class Candidate:
    word: str
    raw: str
    beam_score: float
    edit_similarity: float
    word_frequency: float
    final_score: float
    is_known_word: bool
    log_probability: float


class WordDecoder:
    """
    decoder = WordDecoder(beam_width=5, top_k=5)
    result = decoder.decode(character_probabilities)

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
    ):
        self.beam_width = beam_width
        self.top_k = top_k
        self.case_sensitive = case_sensitive
        self.weights = weights or ScoreWeights()
        self.vocab_size = vocab_size
        self.max_search_distance = max_search_distance

    def _normalize_case(self, word: str) -> str:
        return word if self.case_sensitive else word.lower()

    def _correct_candidate(self, raw_text: str) -> tuple[str, float, float, bool]:
        lookup_text = self._normalize_case(raw_text)
        if is_known_word(lookup_text, self.vocab_size):
            return lookup_text, 1.0, frequency_score(lookup_text), True
        nearest, similarity = edit_distance.nearest_known_word(
            lookup_text, self.vocab_size, self.max_search_distance
        )
        return nearest, similarity, frequency_score(nearest), False

    def decode_raw(self, character_probabilities) -> list[RawCandidate]:
        """The expensive, weight-INDEPENDENT half of decode(): runs beam
        search once and corrects each beam candidate against the
        dictionary once. Cache/reuse the returned list across many
        ScoreWeights via score_raw_candidates() instead of calling
        decode() repeatedly."""
        beam_results = beam_search(
            character_probabilities, beam_width=self.beam_width, top_k=self.top_k
        )
        if not beam_results:
            return []

        log_probs = [r["log_probability"] for r in beam_results]
        lo, hi = min(log_probs), max(log_probs)
        span = (hi - lo) or 1.0  # avoid div-by-zero when all beams tie

        raw_candidates: list[RawCandidate] = []
        for r in beam_results:
            raw_text = r["text"]
            beam_score_norm = (r["log_probability"] - lo) / span  # -> [0, 1]
            corrected_word, edit_sim, word_freq, known = self._correct_candidate(raw_text)
            raw_candidates.append(
                RawCandidate(
                    word=corrected_word, raw=raw_text, beam_score=beam_score_norm,
                    edit_similarity=edit_sim, word_frequency=word_freq,
                    is_known_word=known, log_probability=r["log_probability"],
                )
            )
        return raw_candidates

    @staticmethod
    def score_raw_candidates(raw_candidates: list[RawCandidate], weights: "ScoreWeights") -> dict:
        """Cheap weighted-sum scoring over an already-decoded candidate
        list -- no beam search, no dictionary lookups. This is what
        should run inside a weight-grid-search loop; decode_raw() should
        run only once per word."""
        if not raw_candidates:
            return {"prediction": "", "candidates": []}

        scored: list[Candidate] = []
        for rc in raw_candidates:
            final_score = (
                weights.alpha * rc.beam_score
                + weights.beta * rc.edit_similarity
                + weights.gamma * rc.word_frequency
            )
            scored.append(
                Candidate(
                    word=rc.word, raw=rc.raw, beam_score=rc.beam_score,
                    edit_similarity=rc.edit_similarity, word_frequency=rc.word_frequency,
                    final_score=final_score, is_known_word=rc.is_known_word,
                    log_probability=rc.log_probability,
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
                    "final_score": c.final_score, "is_known_word": c.is_known_word,
                    "log_probability": c.log_probability,
                }
                for c in scored
            ],
        }

    def decode(self, character_probabilities) -> dict:
        """Convenience wrapper: decode_raw() + score with self.weights in
        one call. Unchanged public contract for existing callers
        (app/correction.py, test_beam_dictionary.py) -- if you're only
        decoding with ONE fixed set of weights, this is fine as-is. If
        you're sweeping many weight combos against the same words, use
        decode_raw() + score_raw_candidates() directly instead (see
        experiments/tune_decoder_weights.py)."""
        raw_candidates = self.decode_raw(character_probabilities)
        return self.score_raw_candidates(raw_candidates, self.weights)