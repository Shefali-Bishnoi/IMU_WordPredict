# WordPredict — Priority 0 implementation

This is a real, working codebase — not another planning doc. It implements
**Priority 0** from `ActionPlan.md` end to end: a clean, tested,
reproducible preprocessing pipeline and baseline CNN-LSTM, structured so
that Priority 1 (better encoders), Priority 2 (probability-preserving
output — already done, see below), and Priority 5 (the personalization
adapter) slot in without rewriting anything.

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

## What's implemented

- **`preprocessing/io.py`** — walks `capital letters/` **and**
  `small letters/`, parses the real 12-column schema (label, timestamp,
  9 sensor channels, writing flag) confirmed from your sample data.
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
- **`models/cnn_lstm.py`** — the baseline architecture from
  ActionPlan.md §2.2, but with **encoder and classifier returned as
  separate Keras models** (required for Priority 5 — the adapter gets
  inserted as `h' = h + Adapter(h)` between them later, with zero changes
  to this file).
- **`train.py`** — trains the full model, saves the combined model *and*
  encoder/classifier weights separately, logs training history.
- **`evaluate.py`** — the one canonical evaluation script: macro P/R/F1,
  per-class breakdown, confusion matrix, and single-sample latency.
- **`inference/realtime.py`** + **`app/main.py`** — a `CharacterRecognizer`
  class and a FastAPI server (`/predict`) that return the **full 52-class
  probability vector**, not argmax (Priority 2, done from day one) — this
  is what your future beam decoder (Priority 3) will consume, and what a
  website frontend can already call today.

All of the above was tested end-to-end against a synthetic dataset built
to match your exact confirmed file schema (12 columns, ~18.5ms timestamp
spacing, injected `ovf`/`nan` cells, both letter cases, multiple
participants) — build → train → evaluate → serve → predict all ran
successfully before this was handed to you.

## Running it on your real data

1. Unzip your real `Dataset.zip` into `data/raw/Dataset/` (so you have
   `data/raw/Dataset/capital letters/...` and
   `data/raw/Dataset/small letters/...`).
2. `pip install -r requirements.txt`
3. `python -m data.build_dataset` (add `--resample` to try the
   timestamp-aware resampling ablation from §4.4/§19.5).
4. `python train.py` (defaults: 100 epochs, early stopping patience 10 —
   tune with `--epochs`/`--batch-size`/`--lr`).
5. `python evaluate.py` → writes
   `experiments/baseline_metrics.json` and
   `experiments/baseline_confusion_matrix.csv`. This is your real,
   reproducible baseline number to compare every later change against.
6. `uvicorn app.main:app --reload` to serve predictions locally; POST to
   `/predict` with `{"sensor": [[ax,ay,az,gx,gy,gz,mx,my,mz], ...]}`.

## Deliberately not done yet

Per your own ActionPlan, later priorities depend on this one being solid
first, and you said not to do everything in one go — so CNN-BiLSTM/TCN
comparison (Priority 1), the beam decoder (Priority 3), n-gram/dictionary
language scoring (Priority 4), and the personalization adapter
(Priority 5) are **not** in this codebase yet. The encoder/classifier
split and the probability-preserving inference contract exist specifically
so those can be added incrementally without touching what's here — that's
the natural next step once you've run this on real data and have a real
baseline F1 to beat.

# run
cd wordpredict
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m data.audit_raw_labels --raw-root "D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"
python -m data.build_dataset --raw-root "/path/to/Project_folder/Dataset"
python -m data.build_dataset --raw-root "D:\BTP_Marker_Project\IMU_WordPredict_BTP\Dataset"
python train.py
python evaluate.py
