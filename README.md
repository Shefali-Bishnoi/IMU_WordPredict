# WordPredict — Priority 0–4 (dictionary slice) implementation

This is a real, working codebase — not another planning doc. It implements,
end to end, from `ActionPlan.md`:

- **Priority 0** — clean, reproducible preprocessing + a baseline CNN-LSTM.
- **Priority 1** — comparing three sensor architectures (CNN-LSTM, CNN-BiLSTM, TCN).
- **Priority 2** — full probability-vector output (no internal argmax), done from day one.
- **Priority 3** — beam search over per-position character probabilities.
- **Priority 4 (dictionary/wordfreq slice)** — beam candidates corrected against
  a real English vocabulary (edit-distance + word frequency), with the
  three score weights and the confidence threshold tuned experimentally
  (never hand-picked) against a held-out split.

Structured so **Priority 5** (the personalization adapter) slots in without
rewriting anything already here — see "Deliberately not done yet" below for
exactly what's left.

**Project status:** Priorities 0–3 and the dictionary slice of Priority 4
are implemented, tuned, and evaluated. See
`experiments/architecture_comparison.md` for the sensor-model results and
`experiments/decoder_evaluation.md` / `FuturePlan.md` §6 for the decoder
ablation results. Not yet started: an n-gram/Transformer language model
(Priority 4's stretch options), the personalization adapter (Priority 5),
and automatic word-boundary detection (`FuturePlan.md` v2/v3/v4).

---

## Priority 0 — what was actually true before this

**What was actually true before this:** nothing in the ActionPlan was
implemented. `GRU_model.ipynb` was the only code that existed, and on close
inspection it has real bugs worth knowing about before you trust its
numbers:

1. It only ever walks the `"capital letters"` directory
   (`if filename == 'capital letters'`) — `"small letters"` is never
   loaded, so it was never actually training on all 52 classes.
2. Its label encoding (`ord(c) - ord('A')` and `ord(c) - ord('a')`
   independently) maps both cases to the **same** 0–25 index, so even if
   lowercase had been included, `'A'` and `'a'` would collide. Its own
   `to_categorical` output confirms this: shape `(N, 26)`, not `(N, 52)`.
3. Its padding rows are missing the timestamp column (11 fields instead of
   12), which doesn't match the file schema at all.
4. Preprocessing mutates the original `.txt` files in place and hardcodes
   the test participant IDs inline in a notebook cell — not reproducible,
   not reviewable.

All four are fixed here.

### What's implemented (Priority 0)

- **`preprocessing/io.py`** — walks `capital letters/` **and**
  `small letters/`, parses the real 12-column schema (label, timestamp,
  9 sensor channels, writing flag) confirmed from the sample data.
- **`preprocessing/clean.py`** — vectorized `'ovf'`/`'nan'` handling
  (forward-fill → back-fill → 0.0 fallback per channel), replacing the old
  per-cell nested-loop scan.
- **`preprocessing/segment.py`** — the single shared `preprocess()`
  function used identically at training and inference time (pad/trim →
  center-window → normalize with *persisted* train-only stats). Also
  implements the timestamp-aware resampling option from ActionPlan.md
  §4.4 (`--resample` flag) so you can A/B it against raw sequential rows.
- **`preprocessing/split.py`** — frozen, participant-disjoint train/val/test
  split saved to `data/splits/participant_split.json` (not hardcoded).
- **`data/build_dataset.py`** — turns raw `.txt` files into normalized
  `.npz` arrays. Correct 52-class labels: A–Z → 0–25, a–z → 26–51.
- **`inference/realtime.py`** + **`app/main.py`** — a `CharacterRecognizer`
  class and a FastAPI server (`/predict`) that return the **full 52-class
  probability vector**, not argmax (Priority 2, done from day one) — this
  is what the future beam decoder (Priority 3) will consume, and what a
  website frontend can already call today. Always serves the `cnn_lstm`
  architecture's saved model (see Priority 1 below for how to point it at
  a different winner once you've picked one).

---

## Priority 1 — comparing sensor architectures

ActionPlan.md §9 calls for training and comparing **three** sensor
architectures under identical conditions (same split, same preprocessing,
same training budget) before picking a winner — never assuming the
fanciest one wins.

### What's implemented (Priority 1)

- **`models/cnn_lstm.py`** — Option A, the Priority 0 baseline,
  unchanged. Conv1D(64→128→256) → LSTM(32) → Dense(256→128→52).
- **`models/cnn_bilstm.py`** — Option B. Identical conv front-end, but the
  uni-directional `LSTM(32)` is replaced with `Bidirectional(LSTM(32))`
  (64-dim features instead of 32). Valid here specifically because each
  training instance is a whole, already-segmented character window — this
  is not streaming/online recognition, so looking "backwards" in time
  within one instance isn't leakage.
- **`models/tcn.py`** — Option C. Four dilated causal Conv1D residual
  blocks (dilations 1/2/4/8) → `GlobalAveragePooling1D`, implemented
  directly (no new dependency) rather than pulling in an external TCN
  package, per the ActionPlan's "don't add a component before an
  experiment justifies it" rule.
- **`models/__init__.py`** — the `ARCH_BUILDERS` registry. `train.py` /
  `evaluate.py` select an architecture **by name only** (`--arch`); none
  of them branch on architecture internals. This is also the seam
  Priority 5's adapter will plug into later, whichever architecture wins.
- **`config.py`** — new per-architecture path helpers
  (`model_path(arch)`, `arch_metrics_path(arch)`, etc.). **Backward
  compatible by construction:** `arch="cnn_lstm"` resolves to the exact
  same paths Priority 0 already used (`BASELINE_MODEL_PATH`,
  `METRICS_PATH`, ...), so an already-trained baseline is untouched and
  does not need to be retrained. `cnn_bilstm` and `tcn` each get their own
  clean subdirectory under `models/artifacts/<arch>/` and their own
  `experiments/<arch>_*.json` files — training one architecture can never
  overwrite another.
- **`train.py`** (updated) — now takes `--arch {cnn_lstm,cnn_bilstm,tcn}`
  (default `cnn_lstm`, i.e. identical behavior to Priority 0 if you don't
  pass `--arch`). Also now records parameter count and encoder feature
  dimension in the saved training history, since both matter for the
  Priority 1 selection rule.
- **`evaluate.py`** (updated) — same `--arch` flag. Also now records model
  size on disk, parameter count, and the worst-5-classes-by-F1 in
  `experiments/<arch>_metrics.json` — ActionPlan.md 9.3's selection rule
  needs macro F1 *and* robustness *and* latency *and* model size in one
  place, not scattered across separate runs.
- **`compare_architectures.py`** (new) — reads whichever
  `experiments/<arch>_metrics.json` files already exist and renders the
  ActionPlan.md §9.2 comparison table (also saved to
  `experiments/architecture_comparison.md`). Doesn't train or evaluate
  anything itself; safe to re-run any time to check progress, even with
  only 1 or 2 of the 3 architectures done so far.

### How to pick the winner and move on to Priority 2

`compare_architectures.py` prints a "highest Macro F1" line but
deliberately does **not** auto-select for you — per ActionPlan.md 9.3, use
the table to walk the actual rule in order (Macro F1 → robustness on weak
classes → latency → model size → implementation complexity), pick the
architecture, and from then on just always pass `--arch <winner>` to every
later script (this repo does not currently auto-detect "the chosen
architecture" — that plumbing is a natural first step of Priority 2/3 once
you've decided).

---

## Priority 3 + 4 (dictionary slice) — beam decoding, dictionary correction, weight tuning

ActionPlan.md §11–12 calls for a beam decoder over per-position character
probabilities (never argmax'd away, per Priority 2), corrected against a
real dictionary/frequency signal rather than the raw beam text — and for
the score weights and confidence threshold to be tuned on data, not
hand-picked. All of that is implemented and evaluated here.

### What's implemented (Priority 3 + 4)

- **`inference/beam_search.py`** — pure per-position beam search over the
  52-class probability vectors Priority 2 already returns. Log-domain
  scoring, `beam_width` configurable, `beam_width=1` reproduces greedy
  decoding exactly (used as the sanity-check baseline in the ablation
  below).
- **`inference/word_decoder.py`** — `WordDecoder`, which runs beam search
  and then corrects each beam candidate against a real vocabulary
  (`language/wordfreq_scorer.py`'s `is_known_word` / `frequency_score`,
  falling back to `language/edit_distance.py`'s BK-tree fuzzy match).
  Deliberately split into `decode_raw()` (the expensive, weight-independent
  half: beam search + BK-tree lookup) and `score_raw_candidates()` (the
  cheap, weight-dependent half: a weighted sum + sort) — this is what lets
  the weight-tuning grid search below run in seconds instead of hours, since
  beam search and BK-tree lookups happen exactly once per word, never once
  per weight combination.
- **`language/wordfreq_scorer.py`** + **`language/edit_distance.py`** —
  the dictionary/frequency signal, sourced from the `wordfreq` package (no
  hand-maintained `dictionary.txt`), plus a BK-tree index over a 50k-word
  vocabulary for fast fuzzy correction at any edit distance.
- **`app/session.py`** + **`app/correction.py`** + the `/session/*`
  endpoints in **`app/main.py`** — the v1 commit-button word-boundary
  design (see `FuturePlan.md` §0–§1 for why a button, not an automatic
  timing heuristic, was the correct first version): each character stroke
  is displayed live via `POST /session/{id}/stroke`, and an explicit
  `POST /session/{id}/commit` finalizes the in-progress word through the
  decoder above.
- **`experiments/tune_decoder_weights.py`** — grid-searches `ScoreWeights`
  (`alpha`/`beta`/`gamma`, weighting sensor/beam score vs. edit-distance
  similarity vs. word frequency) and a confidence threshold `tau_word`,
  tuned on the VAL split only, with a single confirmatory run on TEST at
  the end. Automatically sweeps a range of `alpha_min` floors and selects
  whichever floor gives the best pooled VAL accuracy — no hand-picked
  constraint (see `FuturePlan.md` §6.5/§6.7 for why this mattered: an
  earlier hand-picked `alpha >= 0.5` floor was silently costing ~7.5
  percentage points of word accuracy). `tau_word` is derived from the same
  confidence value `app/correction.py` actually compares it against, with
  a fallback ladder (90% → 70% precision targets) instead of a threshold
  that can silently degenerate to "never confident" or "always confident."
- **`experiments/evaluate_decoder.py`** — the canonical A/B/C/D ablation
  (greedy vs. beam search, with vs. without dictionary correction) on a
  fixed synthetic TEST word set, isolating exactly what beam search
  contributes versus what dictionary correction contributes.
- **`test_beam_dictionary.py`** — standalone smoke tests for beam search +
  dictionary correction, including one test that runs the real trained TCN
  end to end on synthetic words.

### Current tuned results (n=2000 words/seed, `--workers 4`)

Tuned weights (`experiments/decoder_weights.json`):
`alpha=0.05, beta=0.85, gamma=0.10`, `tau_word=0.562` (achieves 80.1%
precision on the confidence-gated subset of VAL words — a real, checked
threshold, not a default).

| Config | Beam search | Dictionary correction | TEST word accuracy | 95% CI |
|---|---|---|---:|---|
| A. Greedy, no dictionary | No | No | 28.30% | [26.37%, 30.31%] |
| B. Beam search only | Yes | No | 28.30% | [26.37%, 30.31%] |
| C. Dictionary correction only | No | Yes | 69.15% | [67.09%, 71.14%] |
| D. Beam search + dictionary correction | Yes | Yes | 76.85% | [74.95%, 78.65%] |

`B == A` exactly is expected, not a bug — see `FuturePlan.md` §6.2 for the
proof (given the current position-independent beam scoring, beam search's
#1 output is mathematically guaranteed to equal greedy's #1 output; beam
search's real contribution is generating alternate candidates #2–#5 for
the dictionary stage to consider). See `FuturePlan.md` §6 for the full
error-category breakdown and what it implies about where to invest next
(short version: the sensor recognizer, not the decoder, is now the
bottleneck).

**Reminder (carried through everywhere this is discussed):** these numbers
come from SYNTHETIC words built by concatenating isolated-character test
samples (`ActionPlan.md` §4.3), not real continuous air-writing — treat
them as a pipeline validation, not a claim about real usage.

---

## Running it on your real data

Run these from the repo root, in order. Priority 1 commands are additive —
if you already ran the Priority 0 commands and have a trained baseline,
you do **not** need to redo steps 1–3.

```bash
# 0. Setup (once)
cd wordpredict
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux, use this instead
pip install -r requirements.txt

# 1. (Optional but recommended) audit raw label quality once, fast
python -m data.audit_raw_labels --raw-root "D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"

# 2. Build the processed dataset (train/val/test .npz + frozen split)
python -m data.build_dataset --raw-root "D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"
# add --resample to try the timestamp-aware resampling ablation (§4.4/§19.5)

# --- Priority 0: baseline CNN-LSTM (same as before, --arch defaults to cnn_lstm) ---
python train.py --arch cnn_lstm
python evaluate.py --arch cnn_lstm
#   -> experiments/baseline_metrics.json
#   -> experiments/baseline_confusion_matrix.csv

# --- Priority 1: the other two architectures ---
python train.py --arch cnn_bilstm
python evaluate.py --arch cnn_bilstm
#   -> experiments/cnn_bilstm_metrics.json
#   -> experiments/cnn_bilstm_confusion_matrix.csv

python train.py --arch tcn
python evaluate.py --arch tcn
#   -> experiments/tcn_metrics.json
#   -> experiments/tcn_confusion_matrix.csv

# --- Priority 1: side-by-side comparison table ---
python compare_architectures.py
#   -> prints the table, also saves experiments/architecture_comparison.md
#   -> safe to re-run any time, even with only 1-2 architectures done

# Optional: tune epochs/batch size/LR for any architecture, same flags as before
python train.py --arch cnn_bilstm --epochs 60 --batch-size 64 --lr 0.001

# --- Priority 3 + 4: smoke-test beam search + dictionary correction ---
python test_beam_dictionary.py

# --- Priority 3 + 4: tune decoder weights (VAL only; sweeps alpha_min ---
# --- floors automatically and picks the best; one confirmatory TEST run) ---
python -m experiments.tune_decoder_weights --n-words 2000 --workers 4
#   -> experiments/decoder_weights.json
python test_beam_dictionary.py    # rerun -- now picks up the tuned weights automatically

# --- Priority 3 + 4: canonical A/B/C/D decoder ablation on TEST ---
python -m experiments.evaluate_decoder --n-words 2000 --workers 4 --n-errors 30
#   -> experiments/decoder_evaluation.json / decoder_evaluation.txt

# --- Serve predictions ---
uvicorn app.main:app --reload
# Stateless single-character:
#   POST /predict {"sensor": [[ax,ay,az,gx,gy,gz,mx,my,mz], ...]}
# Full commit-button word flow (see app/session.py, FuturePlan.md §0-1):
#   POST /session/start
#   POST /session/{session_id}/stroke   {"sensor": [[...]]}   # per character
#   POST /session/{session_id}/commit                          # finalize word
#   POST /session/{session_id}/end                              # end session
```

## Deliberately not done yet

Per your own ActionPlan, later priorities depend on this one being solid
first, and you said not to do everything in one go — so Priority 2's
argmax-removal work is already done (the model has always returned the
full probability vector, never argmax, see `inference/realtime.py`), but
the beam decoder (Priority 3), n-gram/dictionary language scoring
(Priority 4), and the personalization adapter (Priority 5) are **not** in
this codebase yet. The `models/__init__.py` registry and the
encoder/classifier split (present in all three Priority 1 architectures)
exist specifically so those can be added incrementally without touching
what's here — pick your Priority 1 winner, wire it into
`inference/realtime.py`, and Priority 3 is the natural next step.

## For GitHub — suggested top-level README additions once this is public

If/when this goes on GitHub, consider adding (not included here since they
depend on things outside this codebase, e.g. your actual results and repo
URL):

- A results table pasted from `experiments/architecture_comparison.md`
  once you've actually run all three architectures on the real dataset,
  so a visitor sees real numbers without cloning and running anything.
- A `LICENSE` file and a one-line badge row (build status / license) at
  the very top.
- A short "Project status" line stating which Priority (0–6) is currently
  active, since ActionPlan.md's own philosophy is incremental,
  one-priority-at-a-time delivery — a visitor shouldn't have to read the
  whole ActionPlan to know where things stand.
- A `CONTRIBUTING.md` only if you expect others besides you to submit
  changes; skip it otherwise, no need to add process overhead for a
  single-author BTP.

python test_beam_dictionary.py
python -m experiments.tune_decoder_weights --n-words 800
python test_beam_dictionary.py    # rerun -- now picks up tuned weights automatically
python -m experiments.evaluate_decoder --n-words 800 --workers 4 --n-errors 30
python -m experiments.build_ngram_model
python -m experiments.tune_decoder_weights --n-words 800 --workers 4
python test_beam_dictionary.py