# WordPredict — Priority 0 + Priority 1 implementation

This is a real, working codebase — not another planning doc. It implements
**Priority 0** (clean, reproducible preprocessing + a baseline CNN-LSTM)
and **Priority 1** (comparing three sensor architectures) from
`ActionPlan.md` end to end, structured so that Priority 2 (probability
preserving output — already done, see below), Priority 3 (beam decoding),
and Priority 5 (the personalization adapter) slot in without rewriting
anything.

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

# --- Serve predictions (still serves the cnn_lstm architecture for now) ---
uvicorn app.main:app --reload
# POST to /predict with {"sensor": [[ax,ay,az,gx,gy,gz,mx,my,mz], ...]}
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
