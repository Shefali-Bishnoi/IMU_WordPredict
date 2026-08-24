"""
Session and word-boundary state (ActionPlan.md Sec.15 real-time inference
path; commit-button word-boundary design), carrying:
  1. session-scoped PERSONALIZATION state (personalization/) -- UNCHANGED,
     see the original module docstring reasoning below.
  2. NEW: the Level-3 contextual-correction TEXT BUFFER + history.

Word boundaries are NOT detected from a drawn gesture -- see the original
module docstring reasoning (unchanged, kept below for the commit-button
design) -- v1 (this file): commit-button only, no timing heuristics.
See FuturePlan.md for the planned v2/v3. There is still NO commit-sentence
action anywhere in this file or in app/main.py -- the text buffer simply
grows by one word every time /commit runs, exactly as before this change,
just with an additional (optional) contextual-reranking step applied to
each newly committed word.

PERSONALIZATION (UNCHANGED): each Session still optionally owns a
personalized_model + adapter (personalization.adapter.build_personalized_model)
and a SessionAdaptationBuffer (personalization.buffer). Both are built
LAZILY -- a session that never calls /correct-character never
instantiates either. This is character-model personalization -- it is
NOT touched, reinterpreted, or replaced by the new text-buffer/language
fields below. The language layer never writes to personalized_model,
adapter, or adaptation_buffer, and personalization never reads
text_buffer/contextual_corrections.

NEW (Level-3 / language layer, purely additive):
  - `text_buffer` is a plain property returning the same string
    `text_so_far` already computed from `committed_words` -- kept as an
    explicit name because the language-correction requirements
    (LANGUAGE_CONTEXT_WORDS, contextual_scorer.build_context_string)
    talk about "the text buffer" directly; it is NOT a second, separate
    store of the same data -- `committed_words` remains the single
    source of truth.
  - `contextual_corrections`: an append-only log of what the contextual
    scorer did for each committed word (context used, candidates
    considered, whether it changed the word) -- purely for
    debugging/UI display (see the "CONTEXTUAL CORRECTION" UI panel).
    Never read by any prediction/personalization logic.

STORAGE NOTE (unchanged): this is an IN-MEMORY store. Fine for a single
dev/demo uvicorn worker; swap for Redis before multi-worker production.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from personalization.buffer import SessionAdaptationBuffer


@dataclass
class WordBuffer:
    """Accumulates one in-progress (not-yet-committed) word: one entry
    per recognized character stroke, in writing order."""

    characters: list = field(default_factory=list)
    probabilities: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def append(self, character: str, probability_vector) -> None:
        self.characters.append(character)
        self.probabilities.append(probability_vector)

    @property
    def raw_string(self) -> str:
        return "".join(self.characters)

    def is_empty(self) -> bool:
        return len(self.characters) == 0


@dataclass
class Session:
    session_id: str
    user_id: Optional[str] = None
    current_word: WordBuffer = field(default_factory=WordBuffer)
    committed_words: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)

    # --- Personalization state (UNCHANGED: all lazily created, all
    # session-scoped, all character-model-level -- see module docstring) ---
    personalized_model: object = None   # keras.Model, built on first correction
    adapter: object = None              # personalization.adapter.SessionAdapter
    adaptation_buffer: SessionAdaptationBuffer = field(default_factory=SessionAdaptationBuffer)
    adaptation_history: list = field(default_factory=list)  # log of adapt_session() results

    # --- NEW: Level-3 contextual-correction bookkeeping (purely additive,
    # never read/written by personalization or character recognition) ---
    contextual_corrections: list = field(default_factory=list)

    def touch(self) -> None:
        self.last_activity_at = time.time()

    @property
    def text_so_far(self) -> str:
        return " ".join(self.committed_words)

    @property
    def text_buffer(self) -> str:
        """Explicit alias for text_so_far -- see module docstring. Both
        names are kept so existing callers of text_so_far are unaffected
        while the new language-layer code (and the UI's "TEXT / SENTENCE
        BUFFER" panel) can use the more descriptive name."""
        return self.text_so_far

    @property
    def has_personalization(self) -> bool:
        return self.personalized_model is not None

    def record_contextual_correction(self, entry: dict) -> None:
        """Append-only debug/UI log -- capped so a very long session
        doesn't grow this unboundedly in memory."""
        self.contextual_corrections.append(entry)
        if len(self.contextual_corrections) > 200:
            self.contextual_corrections = self.contextual_corrections[-200:]


class SessionStore:
    """In-process session registry. See module docstring for the
    single-worker / no-persistence caveat."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, user_id: Optional[str] = None) -> Session:
        session = Session(session_id=uuid.uuid4().hex, user_id=user_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        # Dropping the Session drops personalized_model/adapter/
        # adaptation_buffer/contextual_corrections with it (standard
        # Python GC) -- this IS the "forget after session" requirement
        # (unchanged), and it also means the text buffer/contextual log
        # are never persisted to disk, same as before this feature.
        self._sessions.pop(session_id, None)

    def prune_idle(self, max_idle_seconds: float = 3600) -> int:
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_activity_at > max_idle_seconds
        ]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)


# One shared, process-wide store -- imported by app/main.py the same way
# _recognizer is a single process-wide model instance.
store = SessionStore()
