"""
Minimal production-ready serving layer.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

POST /predict
    body: {"sensor": [[a_x, a_y, a_z, g_x, g_y, g_z, m_x, m_y, m_z], ...]}
    (variable length, 9 columns -- same schema as raw file columns 2-10)
    returns: {"probabilities": [...52 floats...], "top_k": [{"char": "a", "p": 0.9}, ...]}
    Stateless single-character prediction -- unchanged from before. Use
    this if the client is doing its own session/word management.

Word-boundary (commit-button) endpoints -- see app/session.py and
app/correction.py for the design rationale.

POST /session/start
    body: {"user_id": "optional-string"}
    returns: {"session_id": "..."}

POST /session/{session_id}/stroke
    body: {"sensor": [[...]], "top_k": 5}
    Recognizes one character stroke AND appends it to the session's
    in-progress word (unlike /predict, which is stateless).
    returns: {"character": "p", "confidence": 0.61, "top_k": [...],
              "current_word_raw": "app"}

POST /session/{session_id}/commit
    No body needed. This is the "commit word" button from the UI design:
    the user has finished writing the current word. Finalizes the
    in-progress word, runs it through app/correction.py (currently a
    pass-through stub -- see that file), appends the corrected word to
    the session's text, and resets the word buffer for the next word.
    returns: {"raw_word": "appel", "corrected_word": "appel",
              "confidence": 1.0, "text_so_far": "i am appel"}

POST /session/{session_id}/end
    Ends the session and returns the final committed text. Any
    not-yet-committed characters in the current word buffer are
    discarded (the client should call /commit first if it wants them
    included -- /end intentionally does not silently auto-commit, since
    a partial word being force-committed without the user's explicit
    signal could otherwise leak an unintended correction step).
    returns: {"text": "i am appel"}

The model is loaded exactly once at process startup (not per-request).
CORS is left open here for local development; lock this down to your real
frontend origin before shipping.
"""
from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.realtime import CharacterRecognizer
from app.correction import correct_word
from app.session import store as session_store

app = FastAPI(title="WordPredict Inference API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your deployed frontend origin
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_recognizer: CharacterRecognizer | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _recognizer
    _recognizer = CharacterRecognizer()


def _require_recognizer() -> CharacterRecognizer:
    if _recognizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _recognizer


def _require_session(session_id: str):
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")
    return session


# ---------------------------------------------------------------------------
# Stateless single-character prediction (unchanged)
# ---------------------------------------------------------------------------
class StrokeRequest(BaseModel):
    sensor: List[List[float]] = Field(
        ..., description="Rows of [ax, ay, az, gx, gy, gz, mx, my, mz]"
    )
    top_k: int = 5


class TopKEntry(BaseModel):
    char: str
    p: float


class PredictResponse(BaseModel):
    probabilities: List[float]
    top_k: List[TopKEntry]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _recognizer is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: StrokeRequest) -> PredictResponse:
    recognizer = _require_recognizer()
    if not req.sensor or any(len(row) != 9 for row in req.sensor):
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")
    result = recognizer.predict(req.sensor, top_k=req.top_k)
    return PredictResponse(**result)


# ---------------------------------------------------------------------------
# Session / word-boundary (commit-button) endpoints
# ---------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    user_id: str | None = None


class StartSessionResponse(BaseModel):
    session_id: str


@app.post("/session/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    session = session_store.create(user_id=req.user_id)
    return StartSessionResponse(session_id=session.session_id)


class StrokeInSessionResponse(BaseModel):
    character: str
    confidence: float
    top_k: List[TopKEntry]
    current_word_raw: str


@app.post("/session/{session_id}/stroke", response_model=StrokeInSessionResponse)
def submit_stroke(session_id: str, req: StrokeRequest) -> StrokeInSessionResponse:
    recognizer = _require_recognizer()
    session = _require_session(session_id)
    if not req.sensor or any(len(row) != 9 for row in req.sensor):
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")

    result = recognizer.predict(req.sensor, top_k=req.top_k)
    top1 = result["top_k"][0]
    session.current_word.append(top1["char"], result["probabilities"])
    session.touch()

    return StrokeInSessionResponse(
        character=top1["char"],
        confidence=top1["p"],
        top_k=[TopKEntry(**e) for e in result["top_k"]],
        current_word_raw=session.current_word.raw_string,
    )


class CommitWordResponse(BaseModel):
    raw_word: str
    corrected_word: str
    confidence: float
    is_low_confidence: bool
    text_so_far: str


@app.post("/session/{session_id}/commit", response_model=CommitWordResponse)
def commit_word(session_id: str) -> CommitWordResponse:
    session = _require_session(session_id)
    if session.current_word.is_empty():
        raise HTTPException(status_code=400, detail="No characters written since the last commit")

    result = correct_word(
        session.current_word.characters, session.current_word.probabilities
    )
    session.committed_words.append(result.corrected_word)
    session.current_word = type(session.current_word)()  # fresh WordBuffer
    session.touch()

    return CommitWordResponse(
        raw_word=result.raw_word,
        corrected_word=result.corrected_word,
        confidence=result.confidence,
        is_low_confidence=result.is_low_confidence,
        text_so_far=session.text_so_far,
    )


class EndSessionResponse(BaseModel):
    text: str


@app.post("/session/{session_id}/end", response_model=EndSessionResponse)
def end_session(session_id: str) -> EndSessionResponse:
    session = _require_session(session_id)
    text = session.text_so_far
    session_store.delete(session_id)
    return EndSessionResponse(text=text)