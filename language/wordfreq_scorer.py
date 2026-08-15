"""
Word-frequency / vocabulary lookups using the `wordfreq` package instead
of a hand-maintained dictionary.txt.
"""
from __future__ import annotations

from functools import lru_cache

from wordfreq import zipf_frequency, top_n_list

DEFAULT_VOCAB_SIZE = 50_000
MAX_ZIPF = 8.0  # wordfreq's zipf scale tops out ~7-8 for "the", "a", etc.


@lru_cache(maxsize=4)
def _vocab_set(vocab_size: int) -> frozenset[str]:
    """Cached, built once per process per vocab_size."""
    return frozenset(top_n_list("en", vocab_size))


def is_known_word(word: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> bool:
    return word.lower() in _vocab_set(vocab_size)


def frequency_score(word: str) -> float:
    """zipf_frequency normalized to [0, 1]. Returns 0.0 for OOV words
    (wordfreq already returns 0.0 for those; this is just normalization)."""
    z = zipf_frequency(word.lower(), "en")
    return max(0.0, min(1.0, z / MAX_ZIPF))