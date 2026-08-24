# WordPredict — IMU Air-Writing Recognition

Write letters in the air with a 9-axis IMU marker; get corrected English words out the other end — now with contextual, sentence-aware correction on top.

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
        ↓
Commit Word  →  text buffer  →  contextual correction (pretrained causal
                                  language model reranks candidates using
                                  the words already committed this session)
```

This repo contains the **full working system**: the ML pipeline (data cleaning → TCN → beam search → dictionary/LM correction → session-scoped personalization → contextual reranking), a Python inference API, a Node.js realtime gateway, a browser UI, and a hardware bridge for a real IMU marker device. Every claim below is either something you can run yourself or is explicitly marked as not-yet-implemented — see [Project status](#project-status).

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
- [6. Contextual (causal-LM) correction](#6-contextual-causal-lm-correction)
- [Running the ML pipeline from scratch](#running-the-ml-pipeline-from-scratch)
- [API reference](#api-reference)
- [Results](#results)
- [Testing](#testing)
- [Project status](#project-status)
- [Full run guide — everything, in order](#full-run-guide--everything-in-order)
- [Further reading](#further-reading)
- [License](#license)

---

## What this actually does

A person writes a letter in the air with a pen-shaped marker containing a 9-axis motion sensor (accelerometer + gyroscope + magnetometer, sampled ~50 Hz). A **Temporal Convolutional Network (TCN)** looks at that motion burst and outputs a confidence score for all 52 possible characters (`A`–`Z`, `a`–`z`) — it never collapses early to a single best guess. A **beam search** decoder explores several plausible letter sequences at once, and a **dictionary + character n-gram language model** nudges each sequence toward the closest real, common English word. The corrected word is shown to the user, who can accept or correct it — and either way that becomes a signal a small **per-user adapter** can learn from, personalizing recognition to that person's handwriting style without ever touching the shared model.

Once a word is **committed**, a fourth stage runs: a **pretrained causal (left-to-right) language model** reranks the word decoder's own top candidates against the words already committed this session — e.g. preferring `"to"` over the equally dictionary-valid `"too"`/`"two"` when the preceding text is `"I am going"`. This stage never invents a word the decoder didn't already propose, never trains on the IMU dataset, and is fully optional/config-gated — see [Section 6](#6-contextual-causal-lm-correction).

The system works with **three interchangeable input sources**, so you can develop and demo the whole pipeline without hardware:

| Source | What it is |
|---|---|
| **Demo** | Synthetic random sensor data — quick UI/pipeline smoke test |
| **Training Sample** | Paste the raw contents of any dataset `.txt` file straight into the UI — exercises the exact same pipeline a real stroke would, using real recorded motion data |
| **Marker** | A real IMU marker, streamed live over serial through the included hardware bridge |

All three feed into the identical prediction path — nothing about the model, decoder, or backend cares which one is active. Contextual correction, once enabled, runs identically regardless of which sensor source produced the word.

---

## Repository layout

```
config.py                   Every shared constant/path: label mapping, length filters, model
                             paths, PLUS LANGUAGE_MODEL_* / LANGUAGE_CONTEXT_WORDS settings
preprocessing/               Raw file loading, cleaning, padding/normalization (shared train+inference)
data/                        Dataset build script, raw-label audit tool
models/                      cnn_lstm.py / cnn_bilstm.py / tcn.py — the three compared architectures
train.py / evaluate.py       Train + evaluate any architecture
compare_architectures.py     Side-by-side architecture comparison table
inference/
  realtime.py                 CharacterRecognizer — loads model once, predicts per stroke
  beam_search.py               Beam search over per-position probabilities
  word_decoder.py              Beam search + dictionary + n-gram scoring → final word
language/                     wordfreq-backed dictionary, edit-distance/BK-tree, n-gram LM,
                               PLUS causal_lm.py (pretrained causal HF model, loaded once) and
                               contextual_scorer.py (candidate reranking against committed text)
personalization/              Session-scoped residual adapter (identity-at-init, safety-gated)
experiments/                  Decoder weight tuning, full A–E ablation, n-gram training
app/                          FastAPI inference server (Python) — /predict, /session/*, /model/info.
                               session.py also carries the text buffer; correction.py surfaces the
                               decoder's own top-K candidates for the language layer to rerank
hardware/
  marker_bridge.py             Serial → local WebSocket adapter for a real marker (no ML inside)
backend/                      Node.js realtime gateway (REST + WebSocket) in front of the Python API
frontend/                     Browser UI — sensor source selector, live capture view, word flow,
                               Text Buffer panel, Contextual Correction panel
tests/                        test_language_module.py, test_commit_integration.py (contextual
                               correction), alongside the existing pipeline tests below
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
- Contextual correction is **optional**: `transformers` + `torch` (see [Section 6](#6-contextual-causal-lm-correction)). Everything else in this repo works with zero LM dependencies installed.

---

## 1. Install

```bash
git clone <this-repo-url>
cd wordpredict

# --- Python side ---
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: contextual (causal-LM) correction — skip this if you don't
# want it; the rest of the pipeline works fine without it.
pip install transformers torch

# --- Node side ---
cd backend
npm install
cd ..
```

## 2. Get a trained model

You have two options:

**A. Bring your own trained model** — if you've already run the pipeline (see [Running the ML pipeline from scratch](#running-the-ml-pipeline-from-scratch)), you should have `models/artifacts/tcn/tcn.keras` and `data/processed/norm_stats.json`. Skip to Section 3.

**B. Train it yourself** — this repo does not ship a pretrained model or the raw dataset (both are large binary artifacts unsuited to a git repo). Point `data/build_dataset.py` at your copy of the [IMU handwritten-alphabet dataset](https://dx.doi.org/10.21227/av6q-jj17) and follow [Running the ML pipeline from scratch](#running-the-ml-pipeline-from-scratch). Expect roughly an hour end-to-end on a normal laptop CPU (dataset build is the slow part; TCN training itself is under 10 minutes once the `.npz` files exist).

The contextual-correction language model (Section 6) is a **separate, pretrained** model — it downloads itself from the HuggingFace Hub on first server startup and needs no dataset/training of its own.

## 3. Run the servers

Two processes, in two terminals, both from the repo root:

```bash
# Terminal 1 — Python inference API (loads the character model once at
# startup, and — if enabled — the contextual language model right after it)
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
6. Click **Commit Word** (formerly "Predict Word") to run beam search + dictionary/language-model correction and see the final corrected word.
7. Watch the **Text Buffer** panel grow by one word, and the **Contextual Correction** panel show whether the language layer changed anything, given the words already committed. Repeat step 1–6 to build up real multi-word context (e.g. "i", "am", "going") — contextual correction has nothing to work with on the very first word of a session.

There is **no "Commit Sentence" button anywhere** — the text buffer simply keeps growing, one word at a time, for as long as the session runs.

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

## 6. Contextual (causal-LM) correction

This is the newest layer, sitting strictly **after** word commit and strictly **on top of** everything above it — it never replaces the TCN, beam search, dictionary correction, or personalization, and it never trains on the IMU dataset.

**What it is:** a pretrained, causal (left-to-right) HuggingFace language model (`distilgpt2` by default — small, downloads once, runs fine on CPU), loaded exactly once at server startup, the same way the TCN is. It's *causal* rather than masked (BERT-style) on purpose: at inference time the system only ever has the text written **so far** (`"I am going ___"`), never the words that come after — a masked model would need those future words to fill in a blank, which isn't available in a real-time system.

**What it does:** for each newly committed word, it re-scores the word decoder's own top-K candidates (typically ≤5, already computed by beam search + dictionary correction — no re-decoding) against the words already committed this session, and combines that score additively on top of the existing, already-tuned decoder score:

```
combined_score = final_score + LANGUAGE_MODEL_WEIGHT * lm_score
```

`LANGUAGE_MODEL_WEIGHT` defaults conservatively (`0.15`) so a strong, unambiguous sensor read isn't casually overridden by a contextual guess — it's meant to resolve close calls (`"to"` vs. `"too"` vs. `"two"`), not override confident predictions.

**Configuration** (all environment variables, no code changes needed):

```bash
export LANGUAGE_MODEL_ENABLED=true          # default; set false to disable the whole layer
export LANGUAGE_MODEL_NAME=distilgpt2       # any pretrained causal HF model works
export LANGUAGE_MODEL_WEIGHT=0.15           # how much the LM term can move the ranking
export LANGUAGE_CONTEXT_WORDS=20            # how many recent committed words are sent as context
export LANGUAGE_MODEL_TOP_K_CANDIDATES=5    # how many decoder candidates get scored per word
```

**Graceful degradation:** if `transformers`/`torch` aren't installed, the model fails to download, or `LANGUAGE_MODEL_ENABLED=false`, the server logs why and keeps running with `language_model_available: false` (visible at `GET /model/info`) — character prediction, beam search, dictionary correction, and personalization are completely unaffected either way.

**What it explicitly does not do:**
- It does not generate new words — only reranks candidates the decoder already proposed.
- It does not detect sentence or word boundaries — `Commit Word` remains the only word-boundary mechanism; there is still no automatic pause-based detection and no "Commit Sentence" action.
- It does not touch personalization — the per-user adapter (`personalization/`) is character-model-level only and is never read or written by this layer.

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

The character n-gram model trained here (step 4) is a **different, older, smaller** language model than the one in [Section 6](#6-contextual-causal-lm-correction) — it scores raw character sequences *during* beam search itself and is trained from scratch on `wordfreq` text; the new causal LM scores whole *committed words* against *sentence context* and is pretrained, never trained by this repo at all. Both run at the same time when both are enabled; they don't conflict or double-count (see `ProjectExplanation.md` §16A.4).

---

## API reference

### Python inference API (`localhost:8000`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model-loaded status (character model **and** language model) |
| `GET /model/info` | Serving architecture, decoder weights, valid stroke-length band, n-gram/contextual-LM availability and config |
| `POST /predict` | Stateless single-character prediction from raw sensor rows |
| `POST /session/start` | Create a session |
| `POST /session/{id}/stroke` | **Predict Character** — treats accumulated rows as one character |
| `POST /session/{id}/commit` | **Commit Word** — beam search + dictionary/LM correction, then (if enabled) contextual reranking; appends the result to the session's text buffer |
| `POST /session/{id}/end` | End session, return accumulated text |
| `POST /session/{id}/correct-character` | Personalization: explicit user correction |
| `POST /session/{id}/personalized-predict` | Predict using that session's personalized adapter |

`POST /session/{id}/commit`'s response carries the existing fields (`raw_word`, `corrected_word`, `confidence`, `text_so_far`, …) **unchanged**, plus additive fields from the contextual layer: `final_word` (what actually got appended to the text buffer), `language_model_used`, `reranked`, `context`, and `top_candidates`. `corrected_word` and `final_word` are identical whenever the language layer is disabled, unavailable, or agrees with the existing pick.

Full interactive docs: `http://localhost:8000/docs`.

### Node.js gateway (`localhost:4000`)

REST endpoints under `/api/*` mirror the Python API (session lifecycle, character prediction, word commit, pipeline/model status). A WebSocket at `/ws` provides the same functionality for continuous streaming — see [`REALTIME_SYSTEM.md`](REALTIME_SYSTEM.md) for the full message protocol and example sequences. The `word_committed` WebSocket event and the REST `/api/word/commit` response both carry an additive `contextual` block mirroring the Python API's new fields above. Swagger UI: `http://localhost:4000/api-docs`.

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

**Contextual (causal-LM) correction is not yet in this table.** The mechanism is implemented, unit- and integration-tested (see [Testing](#testing)), and safe by construction (conservative default weight, falls back cleanly on failure), but a word/sentence-accuracy comparison against a labeled multi-word text corpus hasn't been run yet — see [Project status](#project-status).

---

## Testing

```bash
# Existing pipeline (unaffected by the contextual-correction layer)
python test_beam_dictionary.py
python smoke_test_personalization.py

# Contextual correction (new) — fast, no model download or GPU required;
# uses an in-memory fake language model stub
pytest tests/test_language_module.py -v
pytest tests/test_commit_integration.py -v
```

`tests/test_commit_integration.py` covers the full `character → word → Commit Word → text buffer → contextual reranking` flow against the real FastAPI app, with the TCN and HF models stubbed out — including an explicit assertion that no `Commit Sentence`-style route exists anywhere in the API.

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
- **Contextual (causal-LM) correction** — pretrained model loaded once, reranks the decoder's own top-K candidates using session context, config-gated and conservatively weighted, falls back cleanly if unavailable; unit- and integration-tested

**Designed but not yet built** (see [`FuturePlan.md`](FuturePlan.md) for the full writeup):

- Automatic word-boundary detection (currently an explicit "Commit Word" action; pause-based auto-detection is scoped but needs a small real-timing data pilot first)
- Fully continuous, button-free character segmentation
- Persistent (cross-session) per-user adapters — currently forgotten when a session ends
- Decoder-confidence-derived pseudo-labels for personalization (only explicit user correction is wired in)
- **Measured word/sentence-accuracy comparison for contextual correction** — the reranking mechanism itself is tested and safe by construction, but hasn't been run through the same kind of accuracy ablation the n-gram model has (that needs a labeled multi-word text corpus, deliberately kept separate from the IMU dataset)

---

## Full run guide — everything, in order

Everything above this point explains each piece individually. This
section is the single, linear checklist: clone → install → train →
test → serve → use — for someone running the whole project for the
first time, start to finish, on one machine.

### Step 0 — One-time environment setup

```bash
git clone <this-repo-url>
cd wordpredict

# Python
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r hardware/requirements.txt   # only if you have real hardware
pip install transformers torch pytest httpx  # contextual correction + tests

# Node
cd backend && npm install && cd ..
```

You do **not** need a GPU, a real marker, or the raw dataset to explore
the UI (Demo/Training-Sample modes) — the dataset and training steps
below are only required if you want to train the character model
yourself rather than reuse an existing `models/artifacts/tcn/tcn.keras`.

### Step 1 — Build the dataset (only if training from scratch)

```bash
python -m data.audit_raw_labels --raw-root "/path/to/Dataset"   # optional, fast sanity pass
python -m data.build_dataset --raw-root "/path/to/Dataset"
```

Produces `data/processed/{train,val,test}.npz`, `data/processed/norm_stats.json`,
`data/splits/participant_split.json`, and `experiments/dataset_manifest.json`.
Expect this to be the slowest step (up to ~1–1.5 hours depending on disk
speed and dataset size) — everything after this is much faster.

### Step 2 — Train and evaluate the sensor models

```bash
python train.py --arch cnn_lstm    && python evaluate.py --arch cnn_lstm
python train.py --arch cnn_bilstm  && python evaluate.py --arch cnn_bilstm
python train.py --arch tcn         && python evaluate.py --arch tcn
python compare_architectures.py    # -> experiments/architecture_comparison.md
```

You only strictly need `--arch tcn` (the architecture actually served —
see [Results](#results)); the other two are there so the comparison
table has something to compare against. Confirm `models/artifacts/tcn/tcn.keras`
exists before moving on.

### Step 3 — Train the character n-gram model and tune the decoder

```bash
python -m experiments.build_ngram_model
python -m experiments.tune_decoder_weights --n-words 2000 --workers 4
python -m experiments.evaluate_decoder --n-words 2000 --workers 4
```

Produces `experiments/ngram_model.json`, `experiments/decoder_weights.json`
(the tuned `alpha`/`beta`/`gamma`/`delta`/`tau_word` the server loads at
startup), and the full A–E ablation in `experiments/decoder_evaluation.json/.txt`.

### Step 4 — Run every test suite

```bash
# ML pipeline sanity checks (no server needed)
python test_beam_dictionary.py
python smoke_test_personalization.py

# Contextual-correction unit + integration tests (no model download needed)
pytest tests/test_language_module.py -v
pytest tests/test_commit_integration.py -v

# Node backend REST + WebSocket tests
cd backend && node --test test/ && cd ..
```

All five should pass before you move on — they're what confirm the
character model, decoder, personalization, contextual-correction layer,
and Node gateway are each independently working before you wire them
together live.

### Step 5 — Start the full system

Three terminals, all from the repo root:

```bash
# Terminal 1 — Python inference API
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

```bash
# Terminal 2 — Node.js realtime gateway + frontend
cd backend
npm start
```

```bash
# Terminal 3 — only if using real hardware
python -m hardware.marker_bridge --serial-port COM5 --baud 115200
```

Confirm both servers are healthy before opening the UI:

```bash
curl http://localhost:8000/model/info   # architecture, decoder weights, LM availability
curl http://localhost:4000/health       # Node + Python reachability
```

### Step 6 — Use it

Open **http://localhost:4000**:

1. **Start Session.**
2. Pick a sensor source — **Training Sample** is the fastest way to
   exercise the full pipeline with zero hardware: paste a raw `.txt`
   file's contents (e.g. `Dataset/capital letters/A/S01/A-01.txt`),
   click **Load Sample** → **Predict Character**.
3. Repeat for every letter of a word, then click **Commit Word**.
4. Watch **Text Buffer** grow and **Contextual Correction** report
   whether the language layer changed anything.
5. Repeat for more words in the same session — contextual correction
   gets more useful as there's more preceding context to work with.

That's the whole loop: **train once (Steps 1–3) → test (Step 4) → serve
(Step 5) → write (Step 6)**. Steps 1–3 don't need to be repeated unless
you're retraining; on every later run you only need Step 5 and Step 6.

---

