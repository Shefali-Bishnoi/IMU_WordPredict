# WordPredict — IMU Air-Writing Recognition

Write letters in the air with a 9-axis IMU marker; get corrected English words out the other end.

```
IMU marker (accel + gyro + mag, 9 channels)
        ↓
Preprocessing (clean → pad/trim → normalize)
        ↓
TCN character model  →  52-class probability vector (never argmax'd early)
        ↓
Beam search  →  multiple word hypotheses kept alive at once
        ↓
Dictionary + n-gram language model correction
        ↓
Corrected word  →  optional per-user personalization
```

This repo contains the **full working system**: the ML pipeline (data cleaning → TCN → beam search → dictionary/LM correction → session-scoped personalization), a Python inference API, a Node.js realtime gateway, a browser UI, and a hardware bridge for a real IMU marker device. Every claim below is either something you can run yourself or is explicitly marked as not-yet-implemented — see [Project status](#project-status).

---

## Table of Contents

- [What this actually does](#what-this-actually-does)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [1. Install](#1-install)
- [2. Get a trained model](#2-get-a-trained-model)
- [3. Run the servers](#3-run-the-servers)
- [4. Use the UI](#4-use-the-ui)
- [5. Connect real hardware (optional)](#5-connect-real-hardware-optional)
- [Running the ML pipeline from scratch](#running-the-ml-pipeline-from-scratch)
- [API reference](#api-reference)
- [Results](#results)
- [Project status](#project-status)
- [Further reading](#further-reading)
- [License](#license)

---

## What this actually does

A person writes a letter in the air with a pen-shaped marker containing a 9-axis motion sensor (accelerometer + gyroscope + magnetometer, sampled ~50 Hz). A **Temporal Convolutional Network (TCN)** looks at that motion burst and outputs a confidence score for all 52 possible characters (`A`–`Z`, `a`–`z`) — it never collapses early to a single best guess. A **beam search** decoder explores several plausible letter sequences at once, and a **dictionary + character n-gram language model** nudges each sequence toward the closest real, common English word. The corrected word is shown to the user, who can accept or correct it — and either way that becomes a signal a small **per-user adapter** can learn from, personalizing recognition to that person's handwriting style without ever touching the shared model.

The system works with **three interchangeable input sources**, so you can develop and demo the whole pipeline without hardware:

| Source | What it is |
|---|---|
| **Demo** | Synthetic random sensor data — quick UI/pipeline smoke test |
| **Training Sample** | Paste the raw contents of any dataset `.txt` file straight into the UI — exercises the exact same pipeline a real stroke would, using real recorded motion data |
| **Marker** | A real IMU marker, streamed live over serial through the included hardware bridge |

All three feed into the identical prediction path — nothing about the model, decoder, or backend cares which one is active.

---

## Repository layout

```
config.py                   Every shared constant/path: label mapping, length filters, model paths
preprocessing/               Raw file loading, cleaning, padding/normalization (shared train+inference)
data/                        Dataset build script, raw-label audit tool
models/                      cnn_lstm.py / cnn_bilstm.py / tcn.py — the three compared architectures
train.py / evaluate.py       Train + evaluate any architecture
compare_architectures.py     Side-by-side architecture comparison table

inference/
  realtime.py                 CharacterRecognizer — loads model once, predicts per stroke
  beam_search.py               Beam search over per-position probabilities
  word_decoder.py              Beam search + dictionary + n-gram scoring → final word

language/                     wordfreq-backed dictionary, edit-distance/BK-tree, n-gram LM
personalization/              Session-scoped residual adapter (identity-at-init, safety-gated)
experiments/                  Decoder weight tuning, full A–E ablation, n-gram training

app/                          FastAPI inference server (Python) — /predict, /session/*, /model/info
hardware/
  marker_bridge.py             Serial → local WebSocket adapter for a real marker (no ML inside)

backend/                      Node.js realtime gateway (REST + WebSocket) in front of the Python API
frontend/                     Browser UI — sensor source selector, live capture view, word flow

ActionPlan.md                 Full original design document (why every component exists)
FuturePlan.md                 Word/character-boundary future work + decoder ablation deep-dive
REALTIME_SYSTEM.md            Node/WebSocket protocol reference
CHANGES.md                    Hardware bridge + training-sample mode + UI redesign changelog
```

---

## Prerequisites

- **Python 3.11** (TensorFlow CPU build; a CUDA GPU build works too if you have one)
- **Node.js 18+**
- A real IMU marker is **not required** — Demo and Training Sample modes work with zero hardware.
- If you do have a marker: `pyserial` (see [Section 5](#5-connect-real-hardware-optional))

---

## 1. Install

```bash
git clone <this-repo-url>
cd wordpredict

# --- Python side ---
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# --- Node side ---
cd backend
npm install
cd ..
```

## 2. Get a trained model

You have two options:

**A. Bring your own trained model** — if you've already run the pipeline (see [Running the ML pipeline from scratch](#running-the-ml-pipeline-from-scratch)), you should have `models/artifacts/tcn/tcn.keras` and `data/processed/norm_stats.json`. Skip to Section 3.

**B. Train it yourself** — this repo does not ship a pretrained model or the raw dataset (both are large binary artifacts unsuited to a git repo). Point `data/build_dataset.py` at your copy of the [IMU handwritten-alphabet dataset](https://dx.doi.org/10.21227/av6q-jj17) and follow [Running the ML pipeline from scratch](#running-the-ml-pipeline-from-scratch). Expect roughly an hour end-to-end on a normal laptop CPU (dataset build is the slow part; TCN training itself is under 10 minutes once the `.npz` files exist).

## 3. Run the servers

Two processes, in two terminals, both from the repo root:

```bash
# Terminal 1 — Python inference API (loads the model once at startup)
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

```bash
# Terminal 2 — Node.js realtime gateway (serves the API + the browser UI)
cd backend
npm start
```

Open **http://localhost:4000** in a browser. Swagger/OpenAPI docs for the Node gateway are at `http://localhost:4000/api-docs`; the raw Python API docs are at `http://localhost:8000/docs`.

## 4. Use the UI

1. Click **Start Session**.
2. Pick a **Sensor Source**:
   - **Demo** — click "Simulate 40-Sample Stroke" a couple of times.
   - **Training Sample** — paste the raw contents of a dataset file (e.g. `Dataset/capital letters/A/S01/A-01.txt`) into the textarea and click **Load Sample**. If the file has a consistent label column, it's shown as a "ground truth" comparison after prediction — it is never sent to the model.
   - **Marker** — see [Section 5](#5-connect-real-hardware-optional) first.
3. Watch the sample counter, sparklines, and captured-row table update live.
4. Click **Predict Character**. The predicted letter is appended to the current word.
5. Repeat for each letter of the word.
6. Click **Predict Word** to run beam search + dictionary/language-model correction and see the final corrected word.

## 5. Connect real hardware (optional)

The physical marker firmware streams raw sensor lines over serial (USB) or BLE. A small, ML-free bridge script (`hardware/marker_bridge.py`) reads that stream, parses it into the same 9-value rows the model expects, and broadcasts them over a local WebSocket for the browser to pick up directly — it never talks to the model or the Python/Node servers itself.

```bash
pip install -r hardware/requirements.txt
python -m hardware.marker_bridge --serial-port COM5 --baud 115200
# macOS/Linux: --serial-port /dev/ttyUSB0
```

In the UI, select **Marker** as the sensor source, confirm the bridge URL (`ws://localhost:8765` by default), and click **Connect to Bridge**. Write a character — the sample counter and sparklines should update in real time — then click **Predict Character** exactly as with the other two sources.

> The bridge intentionally does **not** auto-trigger prediction when you release the marker's physical writing button. Word/character boundaries are explicit UI actions in this system (see [`FuturePlan.md`](FuturePlan.md) for why, and what an automatic version would need). The writing-flag button state is still forwarded as optional display metadata.

If your firmware's live serial format differs from the recorded dataset's `.txt` schema, only `RowParser.parse_line()` inside `hardware/marker_bridge.py` needs to change — nothing downstream does.

---

## Running the ML pipeline from scratch

Only needed if you're training your own model rather than using an existing one. Run from the repo root, in order:

```bash
# 0. (Optional) audit raw label quality across the whole dataset, fast
python -m data.audit_raw_labels --raw-root "/path/to/Dataset"

# 1. Build the processed dataset (train/val/test .npz + frozen participant split)
python -m data.build_dataset --raw-root "/path/to/Dataset"

# 2. Train + evaluate each architecture
python train.py --arch cnn_lstm   && python evaluate.py --arch cnn_lstm
python train.py --arch cnn_bilstm && python evaluate.py --arch cnn_bilstm
python train.py --arch tcn        && python evaluate.py --arch tcn

# 3. Compare them side by side
python compare_architectures.py
#   -> experiments/architecture_comparison.md

# 4. Train the character n-gram language model (external text, not the IMU data)
python -m experiments.build_ngram_model

# 5. Tune the decoder's weights (dictionary/frequency/LM blend) on held-out data
python -m experiments.tune_decoder_weights --n-words 2000 --workers 4
#   -> experiments/decoder_weights.json

# 6. Run the full decoder ablation (what beam search vs. dictionary vs. LM each contribute)
python -m experiments.evaluate_decoder --n-words 2000 --workers 4
#   -> experiments/decoder_evaluation.json / .txt

# Sanity checks, runnable any time:
python test_beam_dictionary.py
python smoke_test_personalization.py
```

Every one of these steps writes real, inspectable output (metrics JSON, confusion matrices, a dataset manifest) rather than silently overwriting anything — see [`ActionPlan.md`](ActionPlan.md) for the full design rationale behind each stage, and [`FuturePlan.md`](FuturePlan.md) §6 for the decoder ablation's detailed findings.

---

## API reference

### Python inference API (`localhost:8000`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model-loaded status |
| `GET /model/info` | Serving architecture, decoder weights, valid stroke-length band, LM availability |
| `POST /predict` | Stateless single-character prediction from raw sensor rows |
| `POST /session/start` | Create a session |
| `POST /session/{id}/stroke` | **Predict Character** — treats accumulated rows as one character |
| `POST /session/{id}/commit` | **Predict Word** — beam search + dictionary/LM correction |
| `POST /session/{id}/end` | End session, return accumulated text |
| `POST /session/{id}/correct-character` | Personalization: explicit user correction |
| `POST /session/{id}/personalized-predict` | Predict using that session's personalized adapter |

Full interactive docs: `http://localhost:8000/docs`.

### Node.js gateway (`localhost:4000`)

REST endpoints under `/api/*` mirror the Python API (session lifecycle, character prediction, word commit, pipeline/model status). A WebSocket at `/ws` provides the same functionality for continuous streaming — see [`REALTIME_SYSTEM.md`](REALTIME_SYSTEM.md) for the full message protocol and example sequences. Swagger UI: `http://localhost:4000/api-docs`.

---

## Results

Three sensor architectures were trained and compared under identical conditions (same participant-disjoint split, same preprocessing, same training budget):

| Model | Macro F1 | Accuracy | Latency (ms/sample) | Params |
|---|---:|---:|---:|---:|
| CNN-LSTM (baseline) | 74.37% | 74.61% | 385.6 | 210,100 |
| CNN-BiLSTM | 75.40% | 75.56% | 760.5 | 255,284 |
| **TCN (selected)** | **78.30%** | **78.39%** | **45.2** | **145,140** |

TCN won on every axis at once — highest accuracy, lowest latency, smallest model — and became the model used everywhere downstream.

Full decoder ablation (synthetic test words, isolating what each pipeline stage contributes):

| Configuration | Word Accuracy |
|---|---:|
| Greedy character predictions, no correction | 39.0% |
| + Beam search only | 39.0% *(expected — see explanation below)* |
| + Dictionary correction only | 77.6% |
| + Beam search + dictionary | 85.4% |
| + Beam search + dictionary + n-gram LM | 85.9% |

Beam search alone changing nothing (39.0% → 39.0%) looks like a bug but isn't — it's a proven mathematical consequence of the current scoring design, explained in detail in [`FuturePlan.md`](FuturePlan.md) §6. Dictionary correction is the single largest jump in the whole table.

---

## Project status

**Implemented, trained, and measured:**
- Full preprocessing pipeline, identical at training and inference time
- All three sensor architectures, compared under identical conditions
- Full-probability-vector output (no internal argmax)
- Beam search + dictionary + n-gram language model correction, weights tuned on held-out data
- A real FastAPI inference server and Node.js realtime gateway
- Session-scoped personalization adapter with a mathematically verified identity-at-init guarantee and safety/rollback gate
- Three interchangeable sensor input sources (Demo / Training Sample / Marker) sharing one prediction pipeline
- Stroke length-band validation at the API boundary (rejects out-of-distribution input with a clear error instead of a silently wrong prediction)

**Designed but not yet built** (see [`FuturePlan.md`](FuturePlan.md) for the full writeup):
- Automatic word-boundary detection (currently an explicit "Predict Word" action; pause-based auto-detection is scoped but needs a small real-timing data pilot first)
- Fully continuous, button-free character segmentation
- Persistent (cross-session) per-user adapters — currently forgotten when a session ends
- Decoder-confidence-derived pseudo-labels for personalization (only explicit user correction is wired in)

---

