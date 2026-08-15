"""
Language / lexical scoring package (wordfreq-based, no maintained
dictionary.txt per project decision).

Submodules are imported explicitly here so a missing/broken submodule
fails loudly at `import language` with a traceback pointing at the real
file, instead of surfacing later as a confusing "cannot import name X
from language" error at some unrelated call site.
"""
from __future__ import annotations

from language import wordfreq_scorer  # noqa: F401
from language import edit_distance  # noqa: F401

__all__ = ["wordfreq_scorer", "edit_distance"]