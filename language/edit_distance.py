"""
Normalized Levenshtein similarity + a BK-tree index over the wordfreq
vocabulary for fast fuzzy correction.

The earlier brute-force length-windowed scan was fine at 50k words for
occasional lookups, but doesn't scale if vocab_size grows or corrections
happen at high frequency. A BK-tree prunes most of the tree per query
instead of scanning every candidate word -- the standard technique for
"find words within edit distance D of X" at dictionary scale.

CHANGE (tie-preservation fix): the original `nearest_known_word` picked
ONE winner among words tied at the same minimum edit distance (via a
frequency tiebreak) and discarded the rest. This silently ate genuine
ambiguity -- e.g. for raw text "helo", both "help" and "hello" are
edit-distance 1, but only "help" ever reached the decoder's scoring
stage, so "hello" could never win even when it was the actually-intended
word and later context (or a closer word-frequency/LM score) would have
favored it. `nearest_known_words` (plural) now returns ALL tied words;
`nearest_known_word` (singular) is kept for any caller that still only
wants one answer, and is now just a thin wrapper around the plural
version so there's a single source of truth for the search logic.
"""
from __future__ import annotations

from functools import lru_cache

from language.wordfreq_scorer import DEFAULT_VOCAB_SIZE, frequency_score, is_known_word, _vocab_set


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def normalized_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


class _BKTreeNode:
    __slots__ = ("word", "children")

    def __init__(self, word: str):
        self.word = word
        self.children: dict[int, "_BKTreeNode"] = {}


class BKTree:
    """Burkhard-Keller tree over Levenshtein distance. Build once
    (O(n) inserts), then query for all words within a given edit
    distance in roughly O(log n) amortized instead of O(n) per lookup."""

    def __init__(self, words):
        self._root: _BKTreeNode | None = None
        for w in words:
            self.insert(w)

    def insert(self, word: str) -> None:
        if self._root is None:
            self._root = _BKTreeNode(word)
            return
        node = self._root
        while True:
            d = levenshtein(word, node.word)
            if d == 0:
                return  # duplicate
            child = node.children.get(d)
            if child is None:
                node.children[d] = _BKTreeNode(word)
                return
            node = child

    def query(self, word: str, max_distance: int) -> list[tuple[str, int]]:
        if self._root is None:
            return []
        results: list[tuple[str, int]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = levenshtein(word, node.word)
            if d <= max_distance:
                results.append((node.word, d))
            lo, hi = d - max_distance, d + max_distance
            for dist, child in node.children.items():
                if lo <= dist <= hi:
                    stack.append(child)
        return results


@lru_cache(maxsize=4)
def _bk_tree(vocab_size: int = DEFAULT_VOCAB_SIZE) -> BKTree:
    """Built lazily on first out-of-vocabulary correction, then cached
    for the rest of the process -- expect a one-time delay of a few
    seconds at 50k words the first time a correction is needed, not on
    import."""
    return BKTree(_vocab_set(vocab_size))


def nearest_known_words(
    candidate: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_search_distance: int = 3,
    max_candidates: int = 5,
) -> list[tuple[str, float]]:
    """Like nearest_known_word, but returns ALL words tied at the minimum
    edit distance (up to max_candidates), not just one frequency-selected
    winner.

    This matters because a tie (e.g. "help" vs "hello", both distance 1
    from "helo") should be decided by the full downstream scoring
    (beam_score + edit_similarity + word_frequency + LM), not thrown
    away before that scoring ever runs.

    Sorted by (frequency desc, word asc) for determinism, but unlike the
    old nearest_known_word, nothing at the winning distance is dropped
    (only candidates beyond max_candidates, if there are more ties than
    that -- kept small by default since this is meant to catch genuine
    close ties, not flood the candidate pool).

    Returns [(word, normalized_similarity), ...], best-first (by
    frequency), all sharing the same edit distance (or the single exact
    match / single fallback, in the two special-case branches below).
    """
    lowered = candidate.lower()
    if is_known_word(lowered, vocab_size):
        return [(lowered, 1.0)]

    tree = _bk_tree(vocab_size)
    for radius in range(1, max_search_distance + 1):
        hits = tree.query(lowered, radius)
        if not hits:
            continue
        min_dist = min(d for _, d in hits)
        tied = [w for w, d in hits if d == min_dist]
        tied.sort(key=lambda w: (-frequency_score(w), w))
        top = tied[:max_candidates]
        denom = max(len(lowered), max((len(w) for w in top), default=1))
        similarity = 1.0 - min_dist / denom
        return [(w, similarity) for w in top]

    return [(lowered, 0.0)]


def nearest_known_word(
    candidate: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_search_distance: int = 3,
) -> tuple[str, float]:
    """Return (closest_known_word, normalized_similarity) -- the single
    best answer. Kept for any caller that only wants one result (e.g.
    quick debugging/inspection scripts); internally just takes the top
    of nearest_known_words() so there is one shared source of truth for
    the search logic instead of two independently-maintained versions.

    NOTE: inference/word_decoder.py's WordDecoder no longer calls this --
    it calls nearest_known_words() (plural) directly so ties are not
    collapsed before scoring. This singular function remains for
    everything else that only ever wanted one answer.
    """
    return nearest_known_words(candidate, vocab_size, max_search_distance, max_candidates=1)[0]