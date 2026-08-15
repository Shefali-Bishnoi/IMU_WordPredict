"""
Normalized Levenshtein similarity + a BK-tree index over the wordfreq
vocabulary for fast fuzzy correction.

The earlier brute-force length-windowed scan was fine at 50k words for
occasional lookups, but doesn't scale if vocab_size grows or corrections
happen at high frequency. A BK-tree prunes most of the tree per query
instead of scanning every candidate word -- the standard technique for
"find words within edit distance D of X" at dictionary scale.
"""
from __future__ import annotations

from functools import lru_cache

from language.wordfreq_scorer import DEFAULT_VOCAB_SIZE, _vocab_set, is_known_word
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






def nearest_known_word(
    candidate: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_search_distance: int = 3,
) -> tuple[str, float]:
    """Return (closest_known_word, normalized_similarity). Expands the
    search radius until a hit is found. Ties at the same edit distance
    are broken by word frequency (more common word wins) -- e.g. "hero"
    over "melo" at distance 1 from "helo" -- then alphabetically as a
    final deterministic tiebreak, so results never depend on set/dict
    iteration order."""
    lowered = candidate.lower()
    if is_known_word(lowered, vocab_size):
        return lowered, 1.0

    tree = _bk_tree(vocab_size)
    for radius in range(1, max_search_distance + 1):
        hits = tree.query(lowered, radius)
        if not hits:
            continue
        min_dist = min(d for _, d in hits)
        tied = [w for w, d in hits if d == min_dist]
        best_word = max(tied, key=lambda w: (frequency_score(w), w))
        return best_word, 1.0 - min_dist / max(len(lowered), len(best_word))

    return lowered, 0.0
    