# WordPredict — FuturePlan.md

**Word-boundary detection: v2 and v3. Character-boundary detection: v4
(unscoped). Decoder ablation findings.**

> Companion to `ActionPlan.md`. That document covers Priorities 0–6 of
> the core pipeline. This document covers pieces of scope that came up
> during design/evaluation discussions but are deliberately **not**
> part of the current build: automatic (non-button) word-boundary
> detection (v2/v3), fully continuous character-boundary detection
> (v4), and the findings from the first real A/B/C/D decoder ablation
> (`experiments/evaluate_decoder.py`). It exists so these ideas don't
> get lost, and so nobody re-litigates any of this from scratch later
> without reading why things shipped the way they did.

> **Changelog:** added §0.1 (clarifying that character segmentation and
> word segmentation are two different, already-partially-solved-vs-not
> problems — this had gotten conflated in discussion), §3a (v4: fully
> continuous, button-free character segmentation — new, unscoped, harder
> than v1–v3), and §6 (decoder ablation results: what beam search
> actually contributes vs. dictionary correction, and why).

---

## 0. Where this picks up

`app/session.py` and `app/main.py` (already implemented) give you **v1**:

```text
User writes characters
        ↓
Displayed live, stroke by stroke (POST /session/{id}/stroke)
        ↓
User presses COMMIT (POST /session/{id}/commit)
        ↓
Word finalized, sent to correction (app/correction.py)
```

Word boundaries are 100% explicit and user-driven. No gesture is
"drawn" for space, and no timing heuristic is involved. This was a
deliberate choice, not a placeholder skipped for lack of time — see
§1 for why it's the right *first* version.

v2 and v3 below are strict additions on top of this. **Neither requires
changing `SessionStore`, `WordBuffer`, or the `/commit` endpoint's
contract.** They only change *what calls* `/commit` (or an internal
equivalent) and *when*.

### 0.1 Two separate segmentation problems — don't conflate them

This has come up in discussion enough times that it's worth being
explicit, in writing, once: **"segmentation" means two different things
in this project, and only one of them is currently unsolved.**

**Character segmentation** — deciding where one character's IMU
stroke ends and the next begins (e.g. within a continuous motion,
knowing "the H is finished, the E has started") — **is already solved
in the current system**, and solved the same way the training data
itself was collected: an explicit hardware signal. `config.FLAG_COL`
is the writing-flag column that marks "actively writing" in every raw
`.txt` instance file (§4.4 of `ActionPlan.md`), and the real-time path
uses the exact same mechanism — every call to
`POST /session/{id}/stroke` already carries one pre-segmented single
character (button pressed, character written, button released, stroke
submitted). There is no ML problem here today, because there is no
character-boundary *detection* happening at all — the user (or the
marker's physical button) tells the system directly, per character,
exactly like every training sample was collected. Beam search
(`inference/beam_search.py`) and the word decoder
(`inference/word_decoder.py`) both start *after* this — they take a
sequence of already-segmented character probability vectors as given
and never see a continuous, unsegmented stream.

**Word segmentation** — deciding where one *word* ends and the next
begins, given a sequence of already-character-segmented strokes — is
what the rest of *this* document (v1/v2/v3) is about. v1 solves it
with an explicit COMMIT button; v2/v3 (below) explore replacing that
with automatic pause detection, once real timing data exists to tune
against.

**A third, harder problem — fully continuous writing with no
per-character button at all** (natural, cursive-style motion, the way
a person actually writes on paper, where the system itself has to spot
character boundaries from a raw unsegmented IMU stream) is **not**
addressed by v1, v2, or v3, and is not addressed anywhere in the
current codebase. This is new scope, formalized here as **v4** — see
§3a.

```text
                    Continuous IMU motion
                            │
        ┌───────────────────┴────────────────────┐
        │                                          │
        ▼                                          ▼
  CHARACTER boundaries               WORD boundaries
  ("where does H end,                ("where does one
   E begin?")                         word end, the next
        │                             begin?")
        ▼                                   │
  SOLVED TODAY via the                       ▼
  same writing-flag button          v1 (SOLVED): explicit
  used during data collection       COMMIT button
  (config.FLAG_COL) — one                   │
  button press per character.               ▼
                                     v2/v3 (FUTURE, §2/§3):
  UNSOLVED, button-free version      automatic pause detection,
  of THIS problem = v4 (§3a),        needs a small continuous-
  a separate and harder future       writing timing pilot first
  work item, NOT the same
  thing as v2/v3.
```

Keep this table in your head whenever "segmentation" comes up in a
review or in the BTP writeup — conflating the two makes both sound
either more solved or less solved than they actually are.

---

## 1. Why v1 (commit button only) shipped first

Recap of the reasoning from the design discussion, for anyone revisiting
this later:

- A marker can't draw "space" the way it draws a letter — there's no
  well-defined stroke shape for it, so a 53rd "SPACE" class was rejected
  (would also worsen the existing 1.79x class imbalance from
  `output.md`, and would need an entirely new category of collected
  data).
- Inter-character pause timing is a plausible automatic signal, but
  **tuning a threshold (`τ_gap`) needs real continuous-writing timing
  data**. The current dataset is isolated single-character files only
  (`ActionPlan.md §4.3`) — there is no recorded multi-character
  continuous-writing timing to tune or validate against.
- Shipping a timing heuristic *without* real data to tune or validate it
  against would mean guessing a threshold and hoping — exactly the kind
  of unjustified component `ActionPlan.md`'s golden rule warns against
  ("never add a component because it sounds advanced; every component
  needs a measurable reason and an experiment that proves it").
- The commit button, by contrast, is deterministic, needs no tuning, and
  is required *anyway* for Level-1 personalization labels (explicit user
  correction, `ActionPlan.md §13.6`) — confirming/correcting a word
  already implies a "this word is done" action in the UI. v1 doesn't
  cost anything extra to have.

**v1 is not a stopgap to be embarrassed about — it is the correct
foundation.** v2/v3 are pure UX polish once real timing data exists.

---

## 2. v2 — Pause-based automatic boundary detection

### 2.1 Goal

Detect "the user has stopped writing this word" from motion/timing
alone, so the user never has to press anything for the common case —
the commit button becomes a fallback/override rather than the only
mechanism.

### 2.2 What's needed that doesn't exist yet

**A small continuous-writing timing dataset.** Per `ActionPlan.md §21`
("Explicitly Out of Scope" for the *main* build, but flagged there as
worth a **targeted** pilot): a handful of users (even 5–10) writing
10–20 known common words each, with real stroke-to-stroke timestamps
recorded exactly as `preprocessing/io.get_timestamps` already parses
them. This does **not** need to be the full 20–30 user / 50–100 word
collection `ActionPlan.md §21` describes for other purposes — a much
smaller pilot is enough to get a first working `τ_gap` estimate.

This pilot should happen *before* implementing the rest of v2, not
after — without it there's nothing to tune the threshold against, and
no way to measure whether v2 actually helps.

### 2.3 Mechanism

Every stroke already has a start/stop time via the same writing-flag
button used for character segmentation (`config.FLAG_COL`). Track the
gap between consecutive stroke releases within a session:

```text
gap = stroke[i].release_time - stroke[i-1].release_time

gap < τ_gap   → same word, keep accumulating (no action)
gap ≥ τ_gap   → word boundary: auto-commit the current WordBuffer
                 (call the exact same internal path /commit already
                 uses today)
```

### 2.4 Implementation sketch

- Extend `StrokeRequest` (or a new lightweight event) to optionally
  carry a client-side timestamp per stroke, OR timestamp server-side at
  receipt if client clocks aren't trustworthy — decide based on measured
  network jitter, don't assume either is fine.
- Add `last_stroke_release_at: float` to `WordBuffer` (or `Session`),
  updated on every `/session/{id}/stroke` call.
- On each incoming stroke, before appending it to the *current* word,
  check the gap against the *previous* stroke's release time. If it
  exceeds `τ_gap`, auto-commit the buffer that's already there (same
  code path `commit_word()` already calls) **before** starting the new
  buffer with the just-arrived stroke.
- `τ_gap` is a tunable config value (`config.py`, alongside the other
  experimentally-tuned constants like `EARLY_STOPPING_PATIENCE`), not
  hand-picked. Expect real per-user variance — this is also a natural
  candidate to fold into personalization later (a user's typical
  inter-word gap becomes one more thing the per-user adapter/profile can
  learn), but that's out of scope for v2 itself.

### 2.5 What must NOT change

- The `/commit` endpoint's response contract (`CommitWordResponse`)
  stays identical — v2 just changes what *triggers* the call, not what
  it returns.
- `app/correction.py`'s `correct_word()` seam stays identical — it still
  doesn't care whether it was invoked by a button press or a timeout.

### 2.6 Required ablation (per ActionPlan.md §7's philosophy — experiment before adopting)

Compare, on the pilot dataset from §2.2:

```text
A. Commit-button only (v1, current)
B. Pause-only (v2, no button)
C. Pause + button override (v3, see §3)
```

Measure: correct-boundary rate (did the detected boundary match where a
human annotator would put word breaks), false-boundary rate (word split
mid-word), missed-boundary rate (two words merged), and subjective
usability if you can get even informal user feedback. **Do not adopt v2
over v1 without this comparison** — a worse automatic system that
*feels* more advanced is still worse.

---

## 3. v3 — Pause detection + commit-button override (recommended target state)

### 3.1 Goal

Combine v1 and v2: pause detection handles the common case
automatically, but the explicit commit button always remains available
and always wins — for when the user pauses mid-word (hesitation,
thinking about the next letter) or writes two short words back-to-back
faster than `τ_gap`.

### 3.2 Mechanism

```text
Stroke arrives
      │
      ▼
Append to current WordBuffer (as v1 already does)
      │
      ▼
Was previous gap ≥ τ_gap?  ──yes──▶  Auto-commit the PREVIOUS word
      │no                              (not this stroke — this stroke
      ▼                                 starts the new word)
Continue writing
      │
      ▼
Explicit COMMIT button pressed at any time?  ──yes──▶  Commit current
      │no                                                word immediately,
      ▼                                                   reset gap timer
Continue writing / waiting for next stroke or button
```

The override is why this is a strict superset of v2, not a separate
design: the button always short-circuits the timer, so a user who finds
the timing uncomfortable (or is writing two genuinely fast short words)
is never stuck waiting on it.

### 3.3 Implementation sketch

- Everything from v2 (§2.4), plus: the existing `/session/{id}/commit`
  endpoint is left completely unchanged and still callable at any time —
  it already resets the buffer, so calling it manually mid-pause-timer
  just short-circuits the timer with no special-casing needed on the
  server side.
- One added piece of client-side (or server-side, if timers are tracked
  server-side) bookkeeping: when a manual commit happens, reset whatever
  "time since last stroke" tracking would otherwise fire an auto-commit
  for a word that no longer exists.

### 3.4 UX notes

- Surface *some* visual indicator of the pause timer approaching
  `τ_gap` (e.g. a subtle fade/pulse) so users aren't surprised by an
  auto-commit — a silent automatic action that occasionally guesses
  wrong is more frustrating than a visible one, even at identical
  accuracy.
- Once `τ_gap` is tuned on the pilot dataset, still expect to expose it
  as a per-user-adjustable setting eventually — natural writing pace
  varies a lot person to person, and this is cheap to make configurable
  once it exists as a config value at all.

---

## 3a. v4 (new, unscoped) — Fully continuous character segmentation (no button)

### 3a.1 Why this is a separate, harder problem than v1–v3

Everything in §0–§3 assumes character boundaries are already known —
that assumption holds today because of the button-per-character
mechanism described in §0.1. v4 removes that assumption entirely: the
user writes naturally and continuously, the way people actually write
on paper or in the air when not thinking about hardware, with **no**
explicit signal marking where one character ends and the next begins.

```text
continuous IMU stream
──────────────────────────────────────────────→ time

        H          E          L          L       O
      /────\      /───\      /───\      /───\   /──\
_____/      \____/     \____/     \____/     \_/    \____
```

The system has to answer "the H is finished, the E has started" purely
from the sensor signal — no button, no gap in a per-character flag
column to key off of. This is a real, separate ML problem (character
boundary detection over a raw continuous stream), not a relabeling of
anything already built. Beam search and the word decoder are
unaffected either way — they still only ever consume a sequence of
per-position character probability vectors; what changes is *how that
sequence gets constructed* upstream of them in the first place.

### 3a.2 What's needed that doesn't exist yet

Same category of prerequisite as v2's §2.2, but for *characters* rather
than *words*, and this pilot would need to be run **before** v2's word-
boundary pilot even makes sense in a button-free mode, since the
word-boundary pilot in §2.2 still presupposes each stroke is already a
segmented character (it just times the gaps *between* already-known
strokes). A genuinely button-free system needs its own dataset: a
handful of users writing continuous short words *with the button held
down or removed entirely*, so the raw signal itself has to carry the
boundary information end to end.

### 3a.3 Candidate approaches (unscoped — pick one only after the pilot exists)

**Approach 1 — Pause-based segmentation on raw motion magnitude.**
Cheapest to try. When a character finishes, motion energy
(accelerometer/gyroscope magnitude) may dip briefly before the next
character starts:

```text
H movement          pause       E movement          pause       L movement
████████████████    ░░░░        ██████████████      ░░░         ██████████
```

Threshold the magnitude signal directly; no new model needed, just a
tunable energy threshold (same "don't hand-pick it, tune it against
real pilot data" discipline as `τ_gap` in §2).

**Approach 2 — Sliding window over the continuous stream.**
Run the existing TCN continuously over overlapping windows and try to
infer which windows correspond to "one complete character." More
complex than Approach 1 because the TCN was trained on complete,
pre-segmented character samples (`ActionPlan.md §4`) — feeding it
arbitrary partial windows is out-of-distribution input the model was
never trained to handle, so this would need either retraining on
partial windows or a separate detection step layered on top; it is not
a drop-in reuse of the existing recognizer.

**Approach 3 — A dedicated segmentation model.**
Train a model (could be small — a binary per-timestep classifier) to
label each raw timestep as `CHARACTER` vs. `PAUSE/BOUNDARY`, or to
predict `start-of-character` / `end-of-character` events directly.
Then:

```text
Continuous IMU
      ↓
Segmentation model  (NEW — doesn't exist yet)
      ↓
[H segment] [E segment] [L segment] [L segment] [O segment]
      ↓
TCN (existing, unchanged)
      ↓
Beam search (existing, unchanged)
      ↓
Dictionary correction (existing, unchanged)
```

This is the most ML-heavy option and the most likely to generalize
well, but needs the most new labeled data (per-timestep boundary
labels, not just per-instance character labels) and is real net-new
model-training scope, not a config change.

### 3a.4 Required ablation, if this is ever picked up

Same philosophy as §2.6: compare the three approaches above (plus "no
automatic segmentation, still require the button" as the control) on
the same pilot dataset, using boundary-detection precision/recall
against human-annotated ground truth, and downstream word accuracy
through the existing (unchanged) TCN → beam search → dictionary
pipeline. **Do not adopt any of these without that comparison** — same
reasoning as every other component in this project.

### 3a.5 Explicitly not started

Nothing in §3a is implemented, scheduled, or has a pilot dataset yet.
It's recorded here specifically so that "how would we ever remove the
button" has a documented starting point instead of getting
re-litigated from zero, and so it doesn't get conflated with v1/v2/v3
(§0.1) or presented as more solved than it is in the BTP writeup.

---

## 4. Explicit non-goals (still, even in v2/v3/v4)

Carried over from `ActionPlan.md §21` and the earlier design discussion,
restated here so it isn't accidentally reopened while building this:

- **No "SPACE" character class.** Word boundaries are a control signal
  external to the 52-class recognizer, in both v2 and v3, exactly as in
  v1 — the recognizer's job and output contract never change.
- **No full continuous-sentence dataset collection.** The §2.2 pilot is
  intentionally small and targeted at *timing* only, not a replacement
  for the larger collection `ActionPlan.md §21` already scopes as
  separate, larger future work.
- **No timing heuristic without a tuned-and-validated `τ_gap`.** If the
  §2.2 pilot can't happen, v1 stays the shipped version indefinitely —
  that's a perfectly good place to stay, not a compromise.
- **No fully continuous (button-free) character segmentation without
  its own tuned-and-validated pilot (§3a.2).** v4 is unscoped scope,
  not an implicit promise this project will eventually remove the
  button — see §3a.5.

---

## 5. Suggested order if/when this is picked up

1. Run the small continuous-writing timing pilot (§2.2) — the one
   prerequisite that gates everything else here.
2. Compute real inter-character vs. inter-word gap distributions from
   the pilot; pick a candidate `τ_gap` (e.g. a percentile split between
   the two distributions, not a guess).
3. Implement v2 behind a feature flag / separate endpoint path so v1
   keeps working unmodified while v2 is being validated.
4. Run the §2.6 ablation (A vs. B vs. C).
5. If v3 (pause + override) wins the ablation, make it the default;
   keep v1's pure-button behavior available as a fallback mode (useful
   for noisy environments, accessibility needs, or users who simply
   prefer explicit control).
6. Only then consider exposing `τ_gap` as a per-user tunable, and only
   if real usage data shows enough person-to-person variance to justify
   it (same "experiment before adding a knob" discipline as everything
   else in this project).
7. v4 (§3a) is a separate track, not a next step after v3 — it has its
   own prerequisite pilot (§3a.2) and shouldn't be picked up just
   because v1–v3 are done. Revisit only if "no button at all" becomes
   an actual product requirement.

---

## 6. Decoder ablation findings (`experiments/evaluate_decoder.py`, run against TEST)

Recorded here because it changes what "future work" should prioritize
next, and because one result in particular (`B == A` exactly) looks
like a bug on first read and isn't — it's worth writing down clearly so
nobody "fixes" it later.

### 6.1 Results (SYNTHETIC concatenated-character words, TEST split, n=800, seed=1234 — NOT real continuous handwriting, ActionPlan.md §4.3)

| Config | Beam search | Dictionary correction | Accuracy | 95% CI |
|---|---|---|---:|---|
| A. Greedy, no dictionary | No | No | 28.62% | [25.60%, 31.85%] |
| B. Beam search only | Yes | No | 28.62% | [25.60%, 31.85%] |
| C. Dictionary correction only (greedy) | No | Yes | 68.62% | [65.33%, 71.74%] |
| D. Beam search + dictionary correction | Yes | Yes | 69.38% | [66.09%, 72.47%] |

TCN character-level accuracy on the full TEST split (n=45,149): **73.30%**.

Improvements: B−A = **+0.00pp**, C−A = **+40.00pp**, D−A = **+40.75pp**,
D−B = **+40.75pp**, D−C = **+0.75pp**.

### 6.2 Why `B == A` exactly, and why that's correct, not a bug

The current beam search (`inference/beam_search.py`) scores a hypothesis
purely as the sum of independent per-position character log-probabilities
— there is no cross-position coupling term (no bigram/n-gram/language
model inside the beam-search step itself). Given that, the position-wise
argmax sequence is *provably* the single highest-scoring hypothesis among
all possible character strings, because maximizing a sum of independent
terms is achieved by maximizing each term independently. More precisely:
at every intermediate step, the running-argmax-so-far partial hypothesis
has the highest possible cumulative log-probability among *all* partial
hypotheses of that length — so it can never be pruned out of the beam, at
any beam width ≥ 1 or top_k ≥ 1. That means **beam search's #1 output is
mathematically guaranteed to equal greedy's #1 output**, for this
scoring scheme, regardless of dataset, seed, or beam width. `B == A`
isn't a coincidence of this run — rerunning with any beam width will
reproduce it.

What beam width *does* currently control is which candidates rank
**#2–#5**, which is the only reason `D − C = +0.75pp` is nonzero: the
dictionary/frequency scorer downstream occasionally prefers one of those
alternates over the raw #1 candidate. Beam search today is valuable only
as an alternate-candidate generator for the dictionary stage — it does
not, by itself, improve the top guess.

### 6.3 Implication: where real beam-search value would come from

`ActionPlan.md §12.2` already scopes a character/word n-gram language
model for Priority 4, with a combined score
`Score(W) = λ_sensor·SensorScore + λ_lm·LanguageScore + λ_lexical·LexicalScore`
(§11.2). The finding in §6.2 sharpens exactly *where* that needs to be
wired in: **the LM/dictionary term needs to influence scoring during
beam expansion itself, not only after the beam is finalized (as
`inference/word_decoder.py` currently does it)**. As implemented today,
dictionary/frequency scoring is a strict post-hoc rescoring pass over an
already-fixed top-B set of candidates — it can reorder that fixed set,
but it can never cause a candidate outside the top-B (by pure sensor
score) to be considered at all. Coupling an n-gram/dictionary term into
the per-step beam-search scoring (so the beam itself is no longer
built from sensor-score-alone) is the concrete next step that would make
`beam_width` matter for more than just "how many alternates does the
dictionary get to see" — worth flagging explicitly the next time
Priority 4 is picked up, since it changes where in the pipeline the LM
term needs to be plumbed in, not just whether one exists.

### 6.4 Implication: the decoder is not the current bottleneck — the recognizer is

Dictionary correction alone (C) already captures 68.62 of D's 69.38
points; beam search adds 0.75pp on top. Meanwhile TCN character-level
accuracy on TEST is 73.30% — well below the sensor-model targets in
`ActionPlan.md §9` (Priority 1's ballpark goal was 80–85%, "not
guaranteed"). The 30 sampled config-D errors back this up directly: of
30 errors, **15 (50%) were tagged "ambiguous characters / incorrect
character classification"** by `evaluate_decoder.py`'s error
classifier, vs. 8 attributable to beam+dictionary scoring trade-offs, 5
to dictionary/frequency prior bias, and 2 to genuine beam-search
candidate limitation (true spelling too far from every beam candidate).
Half the word-level errors trace back to the character recognizer
itself getting one or more characters wrong by a small edit distance
(1–2), not to any weakness in the decoder. **This is direct evidence
that further sensor-recognizer work (Priority 1 — architecture,
training data, per-class weak spots like `b`/`h`/`n` from
`ActionPlan.md §2.3`) is currently higher-leverage than further decoder
tuning**, consistent with `ActionPlan.md`'s own stated priority
ordering, now backed by a real ablation rather than just the a priori
argument.

### 6.5 Implication: revisit the `alpha >= 0.5` weight constraint, deliberately

`experiments/tune_decoder_weights.py`'s unconstrained grid already
showed accuracy falling monotonically as `alpha` (sensor/beam weight)
increases, from 70.0% at `alpha=0.05` down to 62.5% at `alpha=0.85`.
This ablation confirms *why*, independent of the grid search: beam
search's own contribution to the #1 candidate is exactly zero (§6.2),
so weighting it heavily in the final score is, by construction, mostly
just diluting the dictionary/frequency signal that's actually doing the
work. The `alpha >= 0.5` design constraint currently costs ~7
percentage points of measured word accuracy (63.04% constrained vs.
70.0% unconstrained-best on VAL). Whether to keep that constraint is a
project-requirements decision, not a modeling one — but it should now
be made with this evidence in hand, not left as an unexamined default.
If/when Priority 4's n-gram LM gets wired into beam-search scoring
itself (§6.3), revisit this again — a beam search that's actually doing
sequence-level work is a very different case for keeping `alpha` high
than one that's provably equivalent to greedy.

### 6.6 What this section does *not* claim

Per the same discipline as the rest of this document: none of the
numbers in §6.1 say anything about real continuous handwriting. They
describe the decoder pipeline's behavior on synthetic
concatenated-character words built from isolated TEST-split character
samples (`ActionPlan.md §4.3`). The character-segmentation discussion
in §0.1/§3a is orthogonal to this section — §6 assumes segmentation is
already solved (it is, for the button-per-character case) and evaluates
only what happens after that point in the pipeline.