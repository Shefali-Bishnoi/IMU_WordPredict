"""Integration tests for word commit and contextual reranking.

Model and language-model dependencies are replaced with test doubles so the
tests exercise session bookkeeping and response handling locally.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import app.main as main_module
from app.correction import CorrectionResult
from app.session import store as session_store


@pytest.fixture()
def client(monkeypatch):
    # Stub startup before entering TestClient; it loads the real models.
    monkeypatch.setattr(main_module, "_load_model", lambda: None)

    with TestClient(main_module.app) as c:
        monkeypatch.setattr(main_module, "_recognizer", SimpleNamespace())  # not exercised by these tests
        monkeypatch.setattr(main_module, "_language_model", None)  # start disabled; tests override per-case
        yield c


def _start_session(client) -> str:
    res = client.post("/session/start", json={})
    assert res.status_code == 200
    return res.json()["session_id"]


def _stub_correct_word(monkeypatch, corrected_word: str, top_candidates: list[dict] | None = None):
    def _fake_correct_word(characters, probabilities):
        raw = "".join(characters)
        return CorrectionResult(
            raw_word=raw, corrected_word=corrected_word, confidence=0.9,
            is_low_confidence=False, final_score=0.9,
            top_candidates=top_candidates or [{"word": corrected_word, "final_score": 0.9}],
        )
    monkeypatch.setattr(main_module, "correct_word", _fake_correct_word)


class FakeLM:
    def __init__(self, preferences: dict, default: float = -10.0):
        self.preferences = preferences
        self.default = default

    def score_next_word(self, context: str, candidate_word: str) -> str:
        return self.preferences.get((context, candidate_word), self.default)


def test_no_sentence_commit_endpoint_exists(client):
    paths = {route.path for route in main_module.app.routes}
    assert not any("sentence" in p.lower() for p in paths), (
        "Found a route referencing 'sentence' -- there must be no "
        "Commit Sentence action anywhere in the API."
    )
    assert any(p.endswith("/commit") for p in paths)  # Commit Word still exists


def test_commit_word_appends_to_text_buffer(monkeypatch, client):
    _stub_correct_word(monkeypatch, "hello")
    session_id = _start_session(client)

    session = session_store.get(session_id)
    session.current_word.append("h", [0.0] * 52)  # non-empty so /commit is allowed

    res = client.post(f"/session/{session_id}/commit")
    assert res.status_code == 200
    body = res.json()
    assert body["corrected_word"] == "hello"
    assert body["final_word"] == "hello"
    assert body["text_so_far"] == "hello"
    assert body["language_model_used"] is False  # LM disabled in this fixture


def test_text_buffer_grows_word_by_word(monkeypatch, client):
    session_id = _start_session(client)
    session = session_store.get(session_id)

    for word in ("i", "am", "going"):
        _stub_correct_word(monkeypatch, word)
        session.current_word.append("x", [0.0] * 52)
        res = client.post(f"/session/{session_id}/commit")
        assert res.status_code == 200

    final = client.post(f"/session/{session_id}/commit")
    # No more characters buffered -> 400, not a silent no-op or a
    # "commit sentence" fallback of any kind.
    assert final.status_code == 400

    session = session_store.get(session_id)
    assert session.text_so_far == "i am going"
    assert session.text_buffer == "i am going"  # explicit alias, same value


def test_session_reset_via_delete_clears_text_buffer(monkeypatch, client):
    _stub_correct_word(monkeypatch, "hello")
    session_id = _start_session(client)
    session = session_store.get(session_id)
    session.current_word.append("h", [0.0] * 52)
    client.post(f"/session/{session_id}/commit")
    assert session_store.get(session_id).text_so_far == "hello"

    # /session/{id}/end is this project's session-teardown mechanism
    # (there is no separate "reset text buffer" endpoint) -- starting a
    # NEW session must not inherit the old one's text buffer.
    client.post(f"/session/{session_id}/end")
    assert session_store.get(session_id) is None

    new_id = _start_session(client)
    assert session_store.get(new_id).text_so_far == ""


def test_commit_word_with_contextual_reranking(monkeypatch, client):
    candidates = [
        {"word": "too", "final_score": 0.9},
        {"word": "to", "final_score": 0.85},
    ]
    _stub_correct_word(monkeypatch, "too", top_candidates=candidates)
    monkeypatch.setattr(
        main_module, "_language_model",
        FakeLM({("i am going", "too"): -8.0, ("i am going", "to"): -0.5}),
    )

    session_id = _start_session(client)
    session = session_store.get(session_id)
    session.committed_words.extend(["i", "am", "going"])
    session.current_word.append("t", [0.0] * 52)

    res = client.post(f"/session/{session_id}/commit")
    assert res.status_code == 200
    body = res.json()
    assert body["corrected_word"] == "too"        # existing pipeline's own pick, unchanged
    assert body["final_word"] == "to"              # contextual layer's pick, appended to buffer
    assert body["language_model_used"] is True
    assert body["reranked"] is True
    assert body["text_so_far"].endswith("to")


def test_empty_word_buffer_rejected(client):
    session_id = _start_session(client)
    res = client.post(f"/session/{session_id}/commit")
    assert res.status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))