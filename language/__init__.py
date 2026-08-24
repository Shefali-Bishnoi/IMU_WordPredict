"""
Language / lexical scoring package (wordfreq-based, no maintained
dictionary.txt per project decision) -- PLUS the new Level-3 contextual
correction layer (causal_lm.py / contextual_scorer.py).

Submodules are imported explicitly here so a missing/broken submodule
fails loudly at `import language` with a traceback pointing at the real
file, instead of surfacing later as a confusing "cannot import name X
from language" error at some unrelated call site.

NOTE: causal_lm / contextual_scorer are intentionally NOT imported here.
They pull in `transformers`/`torch`, which are optional, heavy
dependencies -- importing them eagerly at `import language` time would
mean ANY use of the dictionary/edit-distance/n-gram modules (all of
Priority 2-4's existing, working code) would fail if transformers isn't
installed. Import them explicitly where needed (app/main.py) instead,
exactly the same lazy pattern already used for the n-gram model in
app/correction.py.
"""
from __future__ import annotations

from language import wordfreq_scorer  # noqa: F401
from language import edit_distance  # noqa: F401
from language import ngram  # noqa: F401

__all__ = ["wordfreq_scorer", "edit_distance", "ngram"]
