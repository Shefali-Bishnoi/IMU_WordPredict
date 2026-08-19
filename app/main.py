"""
Minimal production-ready serving layer.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

POST /predict
    Stateless single-character prediction -- unchanged.

Word-boundary (commit-button) endpoints -- see app/session.py and
app/correction.py for the design rationale. UNCHANGED from before:

POST /session/start
POST /session/{session_id}/stroke
POST /session/{session_id}/commit
POST /session/{session_id}/end

PERSONALIZATION ADDITIONS (all new, nothing above changed):

POST /session/{session_id}/correct-character
    body: {"sensor": [[...]], "correct_char": "b"}
    The user is telling the system: "the stroke I just wrote (same raw
    sensor data already sent to /stroke) was actually 'b', not what got
    displayed." This is Level-1 (ActionPlan.md 13.6) -- the highest
    quality personalization label. On first call for a session, this
    lazily builds that session's personalized_model + adapter (starts
    as an exact identity copy of the global model, see
    personalization/adapter.py). The sample is added to the session's
    adaptation buffer, and a session-scoped adapter update is attempted,
    gated by a held-out-accuracy check (personalization/trainer.py) so
    it can only help or no-op, never regress that session's own recent
    accuracy.
    returns: {"updated": bool, "reason"?: str, "baseline_acc"?: float,
              "candidate_acc"?: float, "n_train"?: int, "n_val"?: int}

POST /session/{session_id}/personalized-predict
    body: {"sensor": [[...]], "top_k": 5}
    Same shape as /predict, but runs the session's personalized model
    if one exists yet (falls back to the identical global-model
    prediction if personalization hasn't started for this session --
    see the identity guarantee in personalization/adapter.py). Use this
    instead of /predict once a session has started correcting
    characters, if you want predictions to actually reflect the
    adaptation.
    returns: same shape as /predict, plus "personalized": bool

The model is loaded exactly once at process startup (not per-request).
CORS is left open here for local development; lock this down to your real
frontend origin before shipping.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.correction import correct_word
from app.session import store as session_store
from config import label_to_index
from inference.realtime import CharacterRecognizer
from personalization.adapter import build_personalized_model
from personalization.trainer import adapt_session

app = FastAPI(title="WordPredict Inference API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your deployed frontend origin
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_recognizer: Optional[CharacterRecognizer] = None


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
# Session / word-boundary (commit-button) endpoints (unchanged)
# ---------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    user_id: Optional[str] = None


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


# ---------------------------------------------------------------------------
# Personalization endpoints (new)
# ---------------------------------------------------------------------------
class CorrectCharacterRequest(BaseModel):
    sensor: List[List[float]] = Field(
        ..., description="Same raw stroke shape as /stroke -- the stroke being corrected"
    )
    correct_char: str = Field(..., description="Single character, e.g. 'b'")


class CorrectCharacterResponse(BaseModel):
    updated: bool
    reason: Optional[str] = None
    baseline_acc: Optional[float] = None
    candidate_acc: Optional[float] = None
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    buffer_size: int


@app.post("/session/{session_id}/correct-character", response_model=CorrectCharacterResponse)
def correct_character(session_id: str, req: CorrectCharacterRequest) -> CorrectCharacterResponse:
    recognizer = _require_recognizer()
    session = _require_session(session_id)
    if not req.sensor or any(len(row) != 9 for row in req.sensor):
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")
    if len(req.correct_char) != 1 or not req.correct_char.isalpha():
        raise HTTPException(status_code=400, detail="correct_char must be a single letter")

    try:
        y = label_to_index(req.correct_char)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    x = recognizer.preprocess_stroke(req.sensor)

    if session.personalized_model is None:
        session.personalized_model, session.adapter = build_personalized_model(
            recognizer.encoder, recognizer.classifier
        )

    session.adaptation_buffer.add(x, y)
    X, Y = session.adaptation_buffer.as_arrays()

    result = adapt_session(session.personalized_model, session.adapter, X, Y)
    session.adaptation_history.append(result)
    session.touch()

    return CorrectCharacterResponse(**result, buffer_size=len(session.adaptation_buffer))


class PersonalizedPredictResponse(BaseModel):
    probabilities: List[float]
    top_k: List[TopKEntry]
    personalized: bool


@app.post("/session/{session_id}/personalized-predict", response_model=PersonalizedPredictResponse)
def personalized_predict(session_id: str, req: StrokeRequest) -> PersonalizedPredictResponse:
    recognizer = _require_recognizer()
    session = _require_session(session_id)
    if not req.sensor or any(len(row) != 9 for row in req.sensor):
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")

    x = recognizer.preprocess_stroke(req.sensor)[np.newaxis, ...]

    if session.personalized_model is not None:
        probs = session.personalized_model.predict(x, verbose=0)[0]
        personalized = True
    else:
        probs = recognizer.model.predict(x, verbose=0)[0]
        personalized = False

    top_k_idx = np.argsort(probs)[::-1][: req.top_k]
    from config import index_to_label

    session.touch()
    return PersonalizedPredictResponse(
        probabilities=probs.tolist(),
        top_k=[TopKEntry(char=index_to_label(int(i)), p=float(probs[i])) for i in top_k_idx],
        personalized=personalized,
    )