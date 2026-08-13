"""
Minimal production-ready serving layer.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

POST /predict
    body: {"sensor": [[a_x, a_y, a_z, g_x, g_y, g_z, m_x, m_y, m_z], ...]}
    (variable length, 9 columns -- same schema as raw file columns 2-10)
    returns: {"probabilities": [...52 floats...], "top_k": [{"char": "a", "p": 0.9}, ...]}

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

app = FastAPI(title="WordPredict Inference API", version="0.1.0")

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
    if _recognizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.sensor or len(req.sensor[0]) != 9:
        raise HTTPException(status_code=400, detail="sensor must be rows of 9 values")
    result = _recognizer.predict(req.sensor, top_k=req.top_k)
    return PredictResponse(**result)
