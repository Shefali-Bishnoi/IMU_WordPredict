"""
Session and word-boundary state (ActionPlan.md Sec.15 real-time inference
path; commit-button word-boundary design).

Word boundaries are NOT detected from a drawn gesture -- a marker can't
draw "space" the way it draws a letter. Instead, "end of word" is an
explicit control signal from the UI/hardware: the user (or a physical
button) triggers POST /session/{id}/commit. This mirrors the existing
writing-flag button already used to segment individual character strokes
(config.FLAG_COL) -- word segmentation is just one more control signal
of the same kind, not a new thing for the recognizer to learn.

v1 (this file): commit-button only, no timing heuristics -- deliberately,
since tuning an inter-character-pause threshold needs real continuous-
writing timing data this project's dataset doesn't have (ActionPlan.md
Sec.4.3). See FuturePlan.md for the planned v2 (pause-based automatic
boundary detection) and v3 (pause detection + commit-button override).

Nothing about this module's public contract (WordBuffer, Session,
SessionStore) needs to change to add v2/v3 later -- a pause detector
would just call the same commit path this already exposes (see
app/main.py's /commit endpoint), triggered by elapsed time between
strokes instead of an explicit user action.

STORAGE NOTE: this is an IN-MEMORY store. Fine for a single dev/demo
uvicorn worker, but it is per-process state -- it will NOT be shared
across multiple uvicorn workers and will NOT survive a restart. Swap for
Redis (or similar) before any multi-worker/production deployment; nothing
above SessionStore's four methods (create/get/delete/prune_idle) needs to
change to do that -- callers only ever go through this interface.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class WordBuffer:
    """Accumulates one in-progress (not-yet-committed) word: one entry
    per recognized character stroke, in writing order."""

    characters: list[str] = field(default_factory=list)
    # Full probability vector per position (Priority 2 contract -- never
    # collapsed to argmax before this point), so that when Priority 3's
    # beam decoder exists it can rescore the *whole* word using every
    # position's full distribution, not just the top-1 characters this
    # buffer displays live to the user.
    probabilities: list[list[float]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def append(self, character: str, probability_vector: list[float]) -> None:
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
    user_id: str | None = None
    current_word: WordBuffer = field(default_factory=WordBuffer)
    committed_words: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity_at = time.time()

    @property
    def text_so_far(self) -> str:
        return " ".join(self.committed_words)


class SessionStore:
    """In-process session registry. See module docstring for the
    single-worker / no-persistence caveat."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, user_id: str | None = None) -> Session:
        session = Session(session_id=uuid.uuid4().hex, user_id=user_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def prune_idle(self, max_idle_seconds: float = 3600) -> int:
        """Drop sessions idle longer than max_idle_seconds. Not wired up
        automatically here (no background scheduler in this minimal
        server) -- call it from a periodic task once this moves past a
        single-user dev setup, so abandoned sessions don't leak memory
        forever."""
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