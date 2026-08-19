"""
Session and word-boundary state (ActionPlan.md Sec.15 real-time inference
path; commit-button word-boundary design), NOW ALSO carrying session-scoped
personalization state (personalization/).

Word boundaries are NOT detected from a drawn gesture -- see the original
module docstring reasoning (unchanged, kept below for the commit-button
design) -- v1 (this file): commit-button only, no timing heuristics.
See FuturePlan.md for the planned v2/v3.

PERSONALIZATION ADDITION: each Session now optionally owns a
personalized_model + adapter (personalization.adapter.build_personalized_model)
and a SessionAdaptationBuffer (personalization.buffer). Both are built
LAZILY -- a session that never calls /correct-character never
instantiates either, so sessions that don't use personalization have
zero extra memory/compute cost versus before this file changed.

"Forget after session" requirement: personalized_model/adapter/
adaptation_buffer are plain attributes on the Session dataclass. When
SessionStore.delete() drops a Session, Python garbage-collects
everything hanging off it, including these -- no separate cleanup code
needed, and nothing is ever persisted to disk.

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

    # --- Personalization state (all lazily created, all session-scoped) ---
    personalized_model: object = None   # keras.Model, built on first correction
    adapter: object = None              # personalization.adapter.SessionAdapter
    adaptation_buffer: SessionAdaptationBuffer = field(default_factory=SessionAdaptationBuffer)
    adaptation_history: list = field(default_factory=list)  # log of adapt_session() results

    def touch(self) -> None:
        self.last_activity_at = time.time()

    @property
    def text_so_far(self) -> str:
        return " ".join(self.committed_words)

    @property
    def has_personalization(self) -> bool:
        return self.personalized_model is not None


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
        # adaptation_buffer with it (standard Python GC) -- this IS the
        # "forget after session" requirement, no extra code needed.
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