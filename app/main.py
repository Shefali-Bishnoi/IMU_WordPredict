"""FastAPI serving layer for IMU character and word prediction.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict                         - single-character prediction
    POST /session/start
    POST /session/{id}/stroke             - predict one character
    POST /session/{id}/commit               - decode word + optional LM rerank
    POST /session/{id}/end
    POST /session/{id}/correct-character  - personalization correction
    POST /session/{id}/personalized-predict
    GET  /model/info
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from app.correction import _ngram_model, _search_lambda_lm, _tau_word, _weights, correct_word
from app.session import store as session_store
from config import MAX_RAW_LINES, MIN_RAW_LINES, NUM_CLASSES, label_to_index
from inference.realtime import CharacterRecognizer, StrokeLengthError
from language.contextual_scorer import rerank_candidates
from personalization.adapter import build_personalized_model
from personalization.trainer import adapt_session

app = FastAPI(title="WordPredict Inference API", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to your deployed frontend origin
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_recognizer: Optional[CharacterRecognizer] = None
_language_model = None


@app.on_event("startup")
def _load_model() -> None:
    global _recognizer
    _recognizer = CharacterRecognizer()
    _load_language_model()


def _load_language_model() -> None:
    """Load causal LM at startup; leave _language_model None on failure."""
    global _language_model
    if not config.LANGUAGE_MODEL_ENABLED:
        print("[language] LANGUAGE_MODEL_ENABLED=false -- contextual correction disabled")
        _language_model = None
        return
    try:
        from language.causal_lm import CausalLanguageModel
        _language_model = CausalLanguageModel.load(
            config.LANGUAGE_MODEL_NAME,
            max_context_tokens=config.LANGUAGE_MODEL_MAX_CONTEXT_TOKENS,
        )
    except Exception as e:  # noqa: BLE001 - startup must never crash over this
        print(f"[language] unexpected error initializing language layer: {e} -- disabled")
        _language_model = None


def _require_recognizer() -> CharacterRecognizer:
    if _recognizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _recognizer


def _require_session(session_id: str):
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id!r}")
    return session


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
    architecture: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _recognizer is not None,
        "language_model_loaded": _language_model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: StrokeRequest) -> PredictResponse:
    recognizer = _require_recognizer()
    if not req.sensor or any(len(row) != 9 for row in req.sensor):
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")
    try:
        result = recognizer.predict(req.sensor, top_k=req.top_k)
    except StrokeLengthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PredictResponse(**result)


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
    """Predict one character and append it to the current word buffer."""
    recognizer = _require_recognizer()
    session = _require_session(session_id)
    if not req.sensor or any(len(row) != 9 for row in req.sensor):
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")

    try:
        result = recognizer.predict(req.sensor, top_k=req.top_k)
    except StrokeLengthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    top1 = result["top_k"][0]
    session.current_word.append(top1["char"], result["probabilities"])
    session.touch()

    return StrokeInSessionResponse(
        character=top1["char"],
        confidence=top1["p"],
        top_k=[TopKEntry(**e) for e in result["top_k"]],
        current_word_raw=session.current_word.raw_string,
    )


class ContextualCandidate(BaseModel):
    word: str
    final_score: float
    lm_log_prob: Optional[float] = None
    lm_score: Optional[float] = None
    combined_score: Optional[float] = None


class CommitWordResponse(BaseModel):
    raw_word: str
    corrected_word: str
    confidence: float
    is_low_confidence: bool
    text_so_far: str
    beam_score: Optional[float] = None
    edit_similarity: Optional[float] = None
    word_frequency: Optional[float] = None
    lm_score: Optional[float] = None
    final_word: str = ""
    language_model_used: bool = False
    reranked: bool = False
    context: str = ""
    top_candidates: List[ContextualCandidate] = Field(default_factory=list)


@app.post("/session/{session_id}/commit", response_model=CommitWordResponse)
def commit_word(session_id: str) -> CommitWordResponse:
    """Decode the current word and optionally rerank with the causal LM."""
    session = _require_session(session_id)
    if session.current_word.is_empty():
        raise HTTPException(status_code=400, detail="No characters written since the last commit")

    result = correct_word(
        session.current_word.characters, session.current_word.probabilities
    )

    context_words = session.committed_words
    rerank = rerank_candidates(
        context_words=context_words,
        candidates=result.top_candidates or [{"word": result.corrected_word, "final_score": result.final_score or 1.0}],
        lm=_language_model,
    )
    final_word = rerank["selected_word"] or result.corrected_word

    session.committed_words.append(final_word)
    session.current_word = type(session.current_word)()
    session.record_contextual_correction({
        "raw_word": result.raw_word,
        "decoder_word": result.corrected_word,
        "final_word": final_word,
        "context": rerank["context"],
        "language_model_used": rerank["used_language_model"],
        "reranked": rerank["reranked"],
    })
    session.touch()

    return CommitWordResponse(
        raw_word=result.raw_word,
        corrected_word=result.corrected_word,
        confidence=result.confidence,
        is_low_confidence=result.is_low_confidence,
        text_so_far=session.text_so_far,
        beam_score=result.beam_score,
        edit_similarity=result.edit_similarity,
        word_frequency=result.word_frequency,
        lm_score=result.lm_score,
        final_word=final_word,
        language_model_used=rerank["used_language_model"],
        reranked=rerank["reranked"],
        context=rerank["context"],
        top_candidates=[
            ContextualCandidate(
                word=c["word"], final_score=c.get("final_score", 0.0),
                lm_log_prob=c.get("lm_log_prob"), lm_score=c.get("lm_score"),
                combined_score=c.get("combined_score"),
            )
            for c in rerank["candidates"]
        ],
    )


class EndSessionResponse(BaseModel):
    text: str


@app.post("/session/{session_id}/end", response_model=EndSessionResponse)
def end_session(session_id: str) -> EndSessionResponse:
    session = _require_session(session_id)
    text = session.text_so_far
    session_store.delete(session_id)
    return EndSessionResponse(text=text)


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

    try:
        x = recognizer.preprocess_stroke(req.sensor)
    except StrokeLengthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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

    try:
        x = recognizer.preprocess_stroke(req.sensor)[np.newaxis, ...]
    except StrokeLengthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

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


@app.get("/model/info")
def model_info() -> dict:
    recognizer = _require_recognizer()
    return {
        "architecture": recognizer.arch,
        "seq_len": recognizer.seq_len,
        "num_classes": NUM_CLASSES,
        "beam_search_available": True,
        "dictionary_correction_available": True,
        "ngram_language_model_available": _ngram_model is not None,
        "ngram_order": _ngram_model.order if _ngram_model is not None else None,
        "personalization_available": True,
        "tau_word": _tau_word,
        "search_lambda_lm": _search_lambda_lm,
        "decoder_weights": {
            "alpha": _weights.alpha,
            "beta": _weights.beta,
            "gamma": _weights.gamma,
            "delta": _weights.delta,
        },
        "min_raw_lines": MIN_RAW_LINES,
        "max_raw_lines": MAX_RAW_LINES,
        "language_model_available": _language_model is not None,
        "language_model_enabled_config": config.LANGUAGE_MODEL_ENABLED,
        "language_model_name": config.LANGUAGE_MODEL_NAME if _language_model is not None else None,
        "language_model_weight": config.LANGUAGE_MODEL_WEIGHT,
        "language_context_words": config.LANGUAGE_CONTEXT_WORDS,
    }
