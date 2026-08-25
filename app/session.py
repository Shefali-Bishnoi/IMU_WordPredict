"""In-memory session state for word-boundary prediction and personalization.

Word boundaries use a commit-button model (no automatic segmentation).
In-memory only; use Redis for multi-worker deployments.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from personalization.buffer import SessionAdaptationBuffer


@dataclass
class WordBuffer:
    """Accumulates one in-progress word: one entry per character stroke."""

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

    personalized_model: object = None
    adapter: object = None
    adaptation_buffer: SessionAdaptationBuffer = field(default_factory=SessionAdaptationBuffer)
    adaptation_history: list = field(default_factory=list)
    contextual_corrections: list = field(default_factory=list)

    def touch(self) -> None:
        self.last_activity_at = time.time()

    @property
    def text_so_far(self) -> str:
        return " ".join(self.committed_words)

    @property
    def text_buffer(self) -> str:
        """Alias for text_so_far."""
        return self.text_so_far

    @property
    def has_personalization(self) -> bool:
        return self.personalized_model is not None

    def record_contextual_correction(self, entry: dict) -> None:
        """Append-only debug/UI log, capped at 200 entries."""
        self.contextual_corrections.append(entry)
        if len(self.contextual_corrections) > 200:
            self.contextual_corrections = self.contextual_corrections[-200:]


class SessionStore:
    """In-process session registry."""

    def __init__(self) -> None:
        self._sessions: dict = {}

    def create(self, user_id: Optional[str] = None) -> Session:
        session = Session(session_id=uuid.uuid4().hex, user_id=user_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
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


store = SessionStore()
