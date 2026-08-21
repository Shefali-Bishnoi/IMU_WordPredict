# WordPredict — Realtime System (Node backend + WebSocket UI)

This document covers the NEW realtime layer only. The ML pipeline itself
(preprocessing, TCN model, beam search, dictionary correction, n-gram LM,
personalization) is unchanged and lives entirely in the existing Python
service — see ActionPlan.md / README.md for that.

## Architecture

```text
Browser (frontend/, black & white console)
        |
        |  WebSocket ws://.../ws   +   REST /api/*
        v
Node.js backend (backend/)
  - session state machine (RUNNING/PREDICTING/WORD_COMMITTING/STOPPED)
  - sensor stroke buffering
  - forwards to Python, never reimplements ML logic
        |
        |  HTTP (internal only)
        v
Python FastAPI inference service (app/main.py, port 8000)
  - preprocessing -> TCN character model -> full probability vector
  - /session/{id}/commit -> beam search -> dictionary -> (n-gram LM if loaded)
        |
        v
Decoder (inference/word_decoder.py, language/*, personalization/*)
```

The browser never talks to the Python service directly.

## Why "Predict Character" / "Commit Word" instead of automatic segmentation

The IMU dataset (and this system) has no reliable signal for "this
character/word has ended" other than an explicit control action — this
mirrors exactly how the training data itself was collected (one file per
character, boundaries set by a physical button, see ActionPlan.md §4.4 and
FuturePlan.md §0.1/§1). So:

- **Predict Character** = "everything I've written since the last predict
  (or session start) is ONE character" — sent as the accumulated sensor
  rows to Python's `/session/{id}/stroke`.
- **Commit Word** = "everything predicted since the last commit is ONE
  word" — triggers Python's `/session/{id}/commit` (beam search +
  dictionary + LM).

No automatic boundary detection is implemented or assumed, per FuturePlan.md.

## Running it

```bash
# 1. Python inference service (existing, unchanged startup)
uvicorn app.main:app --reload --port 8000

# 2. Node backend (serves the API AND the frontend)
cd backend
npm install
npm start          # http://localhost:4000

# Frontend: nothing extra to run — open http://localhost:4000 in a browser.
```

Environment variables (Node backend):

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `4000` | Node backend port |
| `PYTHON_BASE_URL` | `http://localhost:8000` | Python inference service base URL |
| `PYTHON_TIMEOUT_MS` | `15000` | HTTP timeout for Python calls |

## Swagger

`http://localhost:4000/api-docs` (raw spec at `/openapi.json`).

## REST endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Node + Python health |
| POST | `/api/session/start` | Create a session |
| GET | `/api/session/status?sessionId=` | Session status |
| POST | `/api/session/stop` | Stop a session |
| POST | `/api/session/reset` | Reset a session's buffers + Python session |
| POST | `/api/predict/character` | Predict one character from accumulated stroke |
| POST | `/api/word/commit` | Commit accumulated characters as a word |
| GET | `/api/pipeline/status` | Which pipeline stages actually exist |
| GET | `/api/model/info` | Architecture / decoder weights / LM availability |

## WebSocket protocol (`ws://localhost:4000/ws`)

### Client → Server

| type | data | Effect |
|---|---|---|
| `start_session` | — | Creates a session on this connection |
| `sensor` | `[ax,ay,az,gx,gy,gz,mx,my,mz]` | Appends one row to the in-progress stroke |
| `predict_character` | — | Sends buffered stroke to Python `/stroke`, clears stroke buffer |
| `commit_word` | — | Sends accumulated characters to Python `/commit`, clears word buffer |
| `reset_session` | — | Ends + restarts the underlying Python session, clears all buffers |
| `stop_session` | — | Ends the Python session, marks Node session STOPPED |

### Server → Client

Every message: `{ type, timestamp, session_id, request_id, data, error }`.

| type | Meaning |
|---|---|
| `connected` | New WS connection established (`data.connectionId`) |
| `session_started` | Full session status |
| `sensor_ack` | `data.strokeLength`, `data.totalSamples` |
| `prediction_update` | `data.character.{predicted,confidence,top_k}`, `data.pipeline`, `data.currentWordRaw` |
| `word_committed` | `data.rawWord`, `data.correctedWord`, `data.confidence`, `data.pipeline` |
| `session_reset` | Fresh session status |
| `session_stopped` | Final session status |
| `error` | `error` string; `type` stays `"error"` |

### Example sequence

```text
connect
  -> connected
start_session
  -> session_started
sensor x40            (one character's worth of IMU rows)
predict_character
  -> prediction_update
sensor x35             (next character)
predict_character
  -> prediction_update
commit_word
  -> word_committed
stop_session
  -> session_stopped
```

## Session state machine

```text
RUNNING -> PREDICTING -> RUNNING -> WORD_COMMITTING -> RUNNING -> STOPPED
```

Invalid transitions are rejected with an `error` message / 4xx response
(no active session, no current stroke, no characters to commit, session
already stopped).

## Testing

```bash
cd backend
npm test              # REST + WebSocket integration tests (node:test),
                       # run against an in-process stub Python server —
                       # no real model/GPU required.
```

## Important assumptions (things inferred, not already in the repo)

1. **No automatic character/word segmentation exists or was added** —
   `Predict Character` / `Commit Word` are the explicit boundaries, per
   the user's own description and FuturePlan.md.
2. **The canvas is a raw Accel-X debug trace**, not a reconstructed
   handwriting trajectory — the repository has no 2D trajectory
   reconstruction, so nothing was invented here.
3. **Personalization is not wired into this realtime UI/session flow.**
   The Python endpoints (`/correct-character`, `/personalized-predict`)
   exist and are reported as "available" via `/api/pipeline/status`, but
   `active` is always `false` here — no UI control was requested for it.
4. **Two small, additive Python changes** were made purely to make
   already-computed decoder values observable over HTTP (they were never
   invented): `PredictResponse.architecture`, `CommitWordResponse.{beam_score,
   edit_similarity, word_frequency, lm_score}`, and a new `GET /model/info`.
   All existing fields/behavior are unchanged.
5. **In-memory session state only** (both Node and the existing Python
   `SessionStore`) — fine for a single-process BTP prototype, per the
   "don't over-engineer" instruction.
6. **Frontend is plain HTML/CSS/JS, no build step**, served by Express —
   chosen over a React/Vite setup to keep dependencies minimal, per
   instructions; it is a straightforward swap-in for React later since
   the WS/REST contracts are already framework-agnostic.