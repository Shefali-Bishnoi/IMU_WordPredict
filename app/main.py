"""
Minimal production-ready serving layer.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

POST /predict
    Stateless single-character prediction -- unchanged, PLUS an additive
    "architecture" field (see PredictResponse) so a frontend can display
    the real serving architecture without hardcoding it.

Word-boundary (commit-button) endpoints -- see app/session.py and
app/correction.py for the design rationale. UNCHANGED contract, with
ADDITIVE optional score-breakdown fields on CommitWordResponse (see
below) -- existing clients that only read raw_word/corrected_word/
confidence/is_low_confidence/text_so_far are unaffected.

POST /session/start
POST /session/{session_id}/stroke      <- this IS "Predict Character":
    everything accumulated in the session's current-word buffer since
    the last /stroke or /commit call is treated as ONE character
    instance and predicted. No new endpoint was needed for the
    Predict-Character UX -- this one already matches it exactly.
POST /session/{session_id}/commit      <- this IS "Predict Word": runs
    beam search + dictionary/LM correction over the characters
    predicted so far, finalizes the word, appends it to the session's
    text buffer, AND (NEW, additive) runs Level-3 contextual reranking
    over the top word candidates using the preceding committed text as
    context. There is still NO "commit sentence" action -- the text
    buffer simply grows by one word per call, exactly as before.
POST /session/{session_id}/end

PERSONALIZATION ADDITIONS (UNCHANGED from before this feature; still
character-model personalization, still completely separate from the
new language layer -- see app/session.py's module docstring):

POST /session/{session_id}/correct-character
POST /session/{session_id}/personalized-predict

GET /model/info  (additive)
    Returns real, non-hardcoded metadata about the currently serving
    pipeline, now including whether the Level-3 language layer is
    available/enabled.

NEW (length-band enforcement, see inference/realtime.py's
StrokeLengthError docstring / check.md): /predict, /session/{id}/stroke,
and /session/{id}/correct-character now catch StrokeLengthError and
return HTTP 400 with a clear, actionable message instead of either
crashing with a 500 or (the previous, worse behavior) silently returning
a confident-looking but meaningless prediction for an out-of-distribution
stroke length. This matters specifically now that real hardware strokes
(via hardware/marker_bridge.py) are a live input source, not just
pre-filtered dataset files.

NEW (Level-3 contextual correction / language layer -- ActionPlan-style
"three distinct learning levels", extended):
    Level 1: IMU -> character              (TCN, unchanged)
    Level 2: user IMU -> personalized char  (SessionAdapter, unchanged)
    Level 3: words -> contextual score      (NEW: language/*, this file)
The language model is a PRETRAINED, CAUSAL (left-to-right) HuggingFace
model, loaded ONCE at startup (mirrors CharacterRecognizer's own
load-once pattern). It NEVER trains on the IMU dataset, NEVER touches
TCN/adapter weights, and NEVER replaces the character/dictionary
pipeline -- it only reranks the top-K word candidates that pipeline
already produced, using config.LANGUAGE_MODEL_WEIGHT as a conservative
combination weight (see language/contextual_scorer.py). Toggle entirely
via config.LANGUAGE_MODEL_ENABLED (env var LANGUAGE_MODEL_ENABLED); if
the model fails to load, `language_model_available=false` and every
existing code path continues to work exactly as it did before this
feature existed -- see _load_language_model() below.

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

# NEW: Level-3 language layer. None until (and unless) startup loads it
# successfully -- every call site below treats None exactly like
# "disabled" (see language/contextual_scorer.rerank_candidates).
_language_model = None


@app.on_event("startup")
def _load_model() -> None:
    global _recognizer
    _recognizer = CharacterRecognizer()
    _load_language_model()


def _load_language_model() -> None:
    """Loads the Level-3 causal LM once, at startup, exactly like the
    character recognizer above. Never raises -- a failed/disabled load
    just leaves _language_model as None, and every downstream caller
    already degrades gracefully (contextual reranking is skipped, the
    existing word-decoder result is used as-is, per
    Definition of Done: 'language model failure gracefully falls back
    to the existing pipeline')."""
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


# ---------------------------------------------------------------------------
# Stateless single-character prediction (unchanged, + additive field)
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
    # ADDITIVE: CharacterRecognizer.predict() already computes this
    # internally (see inference/realtime.py); it just wasn't surfaced
    # over HTTP before. Optional + defaulted so old clients are unaffected.
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


# ---------------------------------------------------------------------------
# Session / word-boundary (commit-button) endpoints (unchanged contract)
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
    """This IS "Predict Character": everything in req.sensor is treated
    as one already-segmented character instance (matching exactly how
    every training file was collected -- one file, one character). The
    caller (Node backend / bridge-fed frontend) is responsible for
    deciding when a character is "done" and calling this -- there is
    still no automatic character-boundary detection anywhere in this
    system (FuturePlan.md Sec.0.1). UNCHANGED by the language-layer
    feature -- character-level prediction never touches the LM."""
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
    # ADDITIVE (from app/correction.py's CorrectionResult, which in turn
    # comes straight from inference/word_decoder.py's already-computed
    # Candidate fields -- nothing new is computed, just surfaced).
    beam_score: Optional[float] = None
    edit_similarity: Optional[float] = None
    word_frequency: Optional[float] = None
    lm_score: Optional[float] = None

    # --- NEW ADDITIVE: Level-3 contextual correction -----------------
    # `word` (the field above named `corrected_word`) is still exactly
    # what the EXISTING sensor/beam/dictionary/personalization pipeline
    # decided -- this block describes what, if anything, the contextual
    # language layer changed ON TOP of that, purely for transparency.
    # `final_word` is what actually got appended to text_so_far/
    # text_buffer: it equals `corrected_word` whenever the language
    # layer is disabled/unavailable/agrees with the existing pick.
    final_word: str = ""
    language_model_used: bool = False
    reranked: bool = False
    context: str = ""
    top_candidates: List[ContextualCandidate] = Field(default_factory=list)


@app.post("/session/{session_id}/commit", response_model=CommitWordResponse)
def commit_word(session_id: str) -> CommitWordResponse:
    """This IS "Predict Word": beam search + dictionary/LM correction
    over every character predicted since the last commit, followed
    (additively) by Level-3 contextual reranking against the session's
    preceding committed text. There is still NO "commit sentence"
    action -- exactly one word is appended to the text buffer per call,
    same as before this feature existed."""
    session = _require_session(session_id)
    if session.current_word.is_empty():
        raise HTTPException(status_code=400, detail="No characters written since the last commit")

    # --- Existing pipeline: sensor + beam + dictionary + n-gram --------
    # (UNCHANGED behavior/weights/tuning -- this call is byte-for-byte
    # the same as before the language layer was added.)
    result = correct_word(
        session.current_word.characters, session.current_word.probabilities
    )

    # --- NEW: Level-3 contextual reranking (additive, on top) -----------
    # Uses the word-decoder's own top-K candidates (already computed by
    # correct_word() above -- no re-decoding) and the words already
    # committed THIS session as left-to-right context. Falls back to
    # result.corrected_word untouched if the LM is unavailable/disabled
    # or scoring fails for any reason -- see rerank_candidates().
    context_words = session.committed_words
    rerank = rerank_candidates(
        context_words=context_words,
        candidates=result.top_candidates or [{"word": result.corrected_word, "final_score": result.final_score or 1.0}],
        lm=_language_model,
    )
    final_word = rerank["selected_word"] or result.corrected_word

    session.committed_words.append(final_word)
    session.current_word = type(session.current_word)()  # fresh WordBuffer
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


# ---------------------------------------------------------------------------
# Personalization endpoints (UNCHANGED, + StrokeLengthError -> 400).
# Character-model personalization only -- never touched by the language
# layer. See app/session.py's module docstring for the boundary.
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


# ---------------------------------------------------------------------------
# Model / pipeline metadata (additive) -- what actually exists, nothing
# invented. Backs the Node backend's /api/model/info and
# /api/pipeline/status so the frontend never hardcodes "TCN" or guesses
# whether beam search/dictionary/LM/personalization are wired in.
# ---------------------------------------------------------------------------
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
        # ADDITIVE: lets a frontend show the valid stroke-length band
        # (e.g. next to the sample counter: "47 / 40-80 rows") instead
        # of hardcoding it or discovering it only via a 400 error.
        "min_raw_lines": MIN_RAW_LINES,
        "max_raw_lines": MAX_RAW_LINES,
        # NEW ADDITIVE: Level-3 contextual-correction layer status.
        "language_model_available": _language_model is not None,
        "language_model_enabled_config": config.LANGUAGE_MODEL_ENABLED,
        "language_model_name": config.LANGUAGE_MODEL_NAME if _language_model is not None else None,
        "language_model_weight": config.LANGUAGE_MODEL_WEIGHT,
        "language_context_words": config.LANGUAGE_CONTEXT_WORDS,
    }
