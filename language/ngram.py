"""Character-level n-gram language model trained on external English text."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

START = "^"  # start-of-word boundary symbol
END = "$"    # end-of-word boundary symbol

DEFAULT_ORDER = 4
ALPHA_BACKOFF = 0.4
DEFAULT_MODEL_VOCAB_SIZE = 100_000


@dataclass
class NgramLanguageModel:
    order: int
    counts: dict          # context (len < order) -> {char: count}
    totals: dict          # context -> sum(counts[context].values())
    vocab: list            # characters seen during training (+ START/END)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        order: int = DEFAULT_ORDER,
        vocab_size: int = DEFAULT_MODEL_VOCAB_SIZE,
        weight_by_frequency: bool = True,
    ) -> "NgramLanguageModel":
        """Train from wordfreq's English word list."""
        words = [w for w in top_n_list("en", vocab_size) if w.isalpha()]
        counts: dict = defaultdict(Counter)
        totals: dict = defaultdict(int)
        vocab_chars: set = {START, END}

        for word in words:
            word = word.lower()
            weight = max(1, round(zipf_frequency(word, "en"))) if weight_by_frequency else 1
            padded = START * (order - 1) + word + END
            vocab_chars.update(padded)
            for n in range(1, order + 1):
                for i in range(n - 1, len(padded)):
                    context = padded[i - (n - 1): i]
                    char = padded[i]
                    counts[context][char] += weight
                    totals[context] += weight

        return cls(
            order=order, counts=dict(counts), totals=dict(totals),
            vocab=sorted(vocab_chars),
        )

    # ------------------------------------------------------------------
    # Persistence -- same JSON-file pattern already used by
    # experiments/decoder_weights.json / experiments/dataset_manifest.json.
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        payload = {
            "order": self.order,
            "counts": {ctx: dict(c) for ctx, c in self.counts.items()},
            "totals": self.totals,
            "vocab": self.vocab,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "NgramLanguageModel":
        with open(path) as f:
            payload = json.load(f)
        counts = {ctx: Counter(c) for ctx, c in payload["counts"].items()}
        return cls(
            order=payload["order"], counts=counts,
            totals=payload["totals"], vocab=payload["vocab"],
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def char_logprob(self, context: str, char: str) -> float:
        """log P(char | context), stupid-backoff smoothed. `context` may
        be longer than needed -- only the last (order-1) characters are
        used, so callers can just pass "everything written so far"."""
        context = context[-(self.order - 1):] if self.order > 1 else ""
        discount = 1.0
        while True:
            total = self.totals.get(context, 0)
            if total > 0:
                count = self.counts.get(context, {}).get(char, 0)
                if count > 0:
                    return math.log(discount * count / total)
            if not context:
                # Nothing seen even at the unigram level for this char
                # (shouldn't happen for the fixed 52-letter alphabet once
                # trained, but kept as a safety net rather than crashing).
                return math.log(discount * ALPHA_BACKOFF / max(1, len(self.vocab)))
            context = context[1:]  # drop the OLDEST context char, back off one order
            discount *= ALPHA_BACKOFF

    def next_char_logprob(self, prefix_chars: list, next_char: str) -> float:
        """log P(next_char | prefix_chars); used as beam_search lm_scorer."""
        padded_prefix = [START] * (self.order - 1) + [c.lower() for c in prefix_chars]
        context = "".join(padded_prefix[-(self.order - 1):])
        return self.char_logprob(context, next_char.lower())

    def score_partial(self, chars: list) -> float:
        """Cumulative log-prob of an in-progress lowercase character
        sequence (no END boundary). Mathematically equals the sum of
        next_char_logprob() called incrementally over the same sequence
        -- beam_search.py relies on this equivalence to track score
        incrementally instead of recomputing from scratch every step."""
        text = START * (self.order - 1) + "".join(c.lower() for c in chars)
        total = 0.0
        for i in range(self.order - 1, len(text)):
            context = text[max(0, i - (self.order - 1)):i]
            total += self.char_logprob(context, text[i])
        return total

    def score_word(self, word: str) -> float:
        """Full-sequence log-prob including END boundary."""
        return self.score_partial(list(word) + [END])


@lru_cache(maxsize=1)
def _cached_model(path_str: str) -> NgramLanguageModel:
    return NgramLanguageModel.load(Path(path_str))


def load_default(path: Path) -> NgramLanguageModel:
    """Cached loader; parsed from disk at most once per process."""
    return _cached_model(str(path))