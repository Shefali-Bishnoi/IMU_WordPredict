"""
Grid-searches ScoreWeights (alpha/beta/gamma) and derives a tau_word
confidence threshold against SYNTHETIC words built by concatenating
isolated-character samples. Per ActionPlan.md Sec.4.3, this is NOT real
continuous handwriting -- it's the best available signal until
continuous-writing data exists (FuturePlan.md Sec.2.2's pilot).
Treat the output as a defensible starting point, not a final answer --
retune once real word-level data exists.

PERFORMANCE FIX (this is why the script used to hang for hours):
Beam search + dictionary correction (WordDecoder.decode_raw) does NOT
depend on ScoreWeights -- only the final weighted sum does. The old
version called decoder.decode(...), which reruns BOTH steps, once per
(seed x weight-combo x word) -- with ~45 combos x 3 seeds x 800 words
that's up to ~108,000 full beam-search + BK-tree edit-distance runs,
when only ~2,400 (seeds x words) were actually necessary.

This version:
  1. Computes decode_raw() exactly ONCE per (seed, word) -- the expensive
     part -- optionally in parallel across a process pool (this is
     pure-Python CPU-bound work, so THREADS would not help: the GIL
     serializes them anyway. A process pool bypasses the GIL.).
  2. Grid-searches weights by calling only the cheap
     WordDecoder.score_raw_candidates() (a weighted sum + sort) against
     that cached result -- no beam search, no BK-tree lookups, per combo.

ALPHA-FLOOR SWEEP (this revision): the old version required you to
hand-pick a single --alpha-min floor and rerun the whole script to try
another one. That's wasteful, since run_grid_from_raw() already computes
EVERY (alpha, beta, gamma) combo per seed regardless of any floor -- the
floor only affects which combo gets selected afterward. This version
instead sweeps a whole list of candidate floors (default: every alpha
value that actually appears in the grid) against data that's already
been decoded once, and automatically selects whichever floor's winning
combo gives the best pooled VAL accuracy across seeds. This replaces
hand-picking --alpha-min with the same "tune on validation data, don't
hand-pick constants" discipline the project already applies everywhere
else (see ActionPlan.md's golden rule). If you still want to force a
floor (e.g. for a specific ablation write-up), pass a single value via
--alpha-min-candidates 0.5.

Leakage / stability fixes (unchanged from before):
1. Weights/tau are tuned on VAL only; TEST is touched once, at the end,
   purely for a confirmatory number.
2. The full unconstrained (alpha_min=0.05) grid is always run for the
   diagnostic tradeoff curve; each candidate floor's constrained best is
   reported separately, so every floor's cost/benefit is visible at once
   instead of requiring separate reruns.
3. The grid is repeated across --n-seeds different synthetic-word
   samples and the winning combo (per floor) must be checked for
   agreement across seeds (majority vote), with Wilson 95% CI reported.

Usage:
    python -m experiments.tune_decoder_weights --n-words 800
    python -m experiments.tune_decoder_weights --n-words 800 --workers 4
    python -m experiments.tune_decoder_weights --n-words 800 --alpha-min-candidates 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
from wordfreq import top_n_list

from config import EXPERIMENTS_DIR, PROCESSED_DIR, TEST_NPZ_PATH, label_to_index, model_path
from inference.word_decoder import DEFAULT_VOCAB_SIZE, RawCandidate, ScoreWeights, WordDecoder

OUT_PATH = EXPERIMENTS_DIR / "decoder_weights.json"
VAL_NPZ_PATH = PROCESSED_DIR / "val.npz"

# Matches the alpha values run_grid_from_raw() actually produces at the
# default step=0.1 (np.arange(0.05, 0.91, 0.1) rounded to 2dp). Sweeping
# exactly these means every floor corresponds to a real grid boundary --
# no floor is silently a no-op because it falls between two grid points.
DEFAULT_ALPHA_MIN_CANDIDATES = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]


# ---------------------------------------------------------------------------
# Synthetic word construction (unchanged logic)
# ---------------------------------------------------------------------------
def build_synthetic_words(y, n_words, seed, min_len=3, max_len=8):
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(y):
        by_class.setdefault(int(label), []).append(idx)

    vocab = [w for w in top_n_list("en", 20_000) if w.isalpha() and min_len <= len(w) <= max_len]
    rng.shuffle(vocab)

    words_out = []
    for word in vocab:
        if len(words_out) >= n_words:
            break
        try:
            class_indices = [label_to_index(c) for c in word]
        except ValueError:
            continue
        if any(ci not in by_class for ci in class_indices):
            continue
        row_indices = [rng.choice(by_class[ci]) for ci in class_indices]
        words_out.append((word, row_indices))
    return words_out


# ---------------------------------------------------------------------------
# Expensive, weight-independent step: beam search + dictionary correction.
# Run this ONCE per (seed, word) -- never inside the weight grid loop.
# ---------------------------------------------------------------------------
_worker_decoder: WordDecoder | None = None


def _init_worker(beam_width: int, top_k: int, vocab_size: int, max_search_distance: int) -> None:
    """Runs once per worker PROCESS (not per task), so the BK-tree gets
    built at most once per worker instead of once per word."""
    global _worker_decoder
    _worker_decoder = WordDecoder(
        beam_width=beam_width, top_k=top_k,
        vocab_size=vocab_size, max_search_distance=max_search_distance,
    )


def _decode_raw_worker(seq: list[list[float]]) -> list[RawCandidate]:
    assert _worker_decoder is not None, "worker not initialized -- pass initializer to the pool"
    return _worker_decoder.decode_raw(seq)


def precompute_raw_candidates(
    words: list[tuple[str, list[int]]],
    all_probs: np.ndarray,
    beam_width: int = 5,
    top_k: int = 5,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_search_distance: int = 3,
    n_workers: int = 1,
) -> list[tuple[str, list[RawCandidate]]]:
    """Returns [(true_word, raw_candidates), ...] -- decode_raw() called
    exactly once per word. This is the only place beam search / the
    BK-tree get exercised in the whole tuning run.

    n_workers=1 runs single-process (simplest, best for small n_words or
    debugging). n_workers>1 uses a process pool -- NOT a thread pool,
    since this is pure-Python CPU-bound work and threads would be
    serialized by the GIL anyway; only separate processes actually run
    in parallel here.
    """
    tasks = [
        (true_word, [all_probs[i].tolist() for i in row_indices])
        for true_word, row_indices in words
    ]

    if n_workers <= 1:
        decoder = WordDecoder(
            beam_width=beam_width, top_k=top_k,
            vocab_size=vocab_size, max_search_distance=max_search_distance,
        )
        return [(tw, decoder.decode_raw(seq)) for tw, seq in tasks]

    results: list[tuple[str, list[RawCandidate]]] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(beam_width, top_k, vocab_size, max_search_distance),
    ) as ex:
        # map() preserves task order, so results line up with `words`
        # without needing to track futures individually.
        for (tw, _seq), raw in zip(tasks, ex.map(_decode_raw_worker, (seq for _, seq in tasks))):
            results.append((tw, raw))
    return results


# ---------------------------------------------------------------------------
# Cheap, weight-dependent step: run this many times per (seed, word) --
# once per grid point. No beam search, no BK-tree lookups happen here.
# ---------------------------------------------------------------------------
def evaluate_from_raw(
    raw_results: list[tuple[str, list[RawCandidate]]], weights: ScoreWeights
) -> tuple[float, list[tuple[float, bool]]]:
    correct = 0
    margins = []  # (margin, was_correct) -- feeds tau_word suggestion
    for true_word, raw_candidates in raw_results:
        result = WordDecoder.score_raw_candidates(raw_candidates, weights)
        is_correct = result["prediction"].lower() == true_word.lower()
        correct += int(is_correct)
        cands = result["candidates"]
        margin = (cands[0]["final_score"] - cands[1]["final_score"]) if len(cands) > 1 else 1.0
        margins.append((margin, is_correct))
    return (correct / len(raw_results) if raw_results else 0.0), margins


# Must match app/correction.py's sigmoid steepness exactly -- that file
# computes confidence = sigmoid(CONFIDENCE_SIGMOID_STEEPNESS * margin) and
# compares tau_word against THAT, never against the raw margin. tau_word
# has to be selected on the same scale it will be compared against at
# inference, or the threshold saved here is meaningless once deployed.
CONFIDENCE_SIGMOID_STEEPNESS = 6.0


def _margin_to_confidence(margin: float, steepness: float = CONFIDENCE_SIGMOID_STEEPNESS) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * margin))


DEFAULT_TAU_TARGET_PRECISIONS = (0.90, 0.85, 0.80, 0.75, 0.70)


def suggest_tau(margins, target_precisions=DEFAULT_TAU_TARGET_PRECISIONS):
    """Smallest CONFIDENCE threshold (sigmoid(steepness*margin), the same
    transform app/correction.py applies before comparing to tau_word --
    NOT the raw margin) such that predictions at/above it are correct at
    least `target` of the time, for the HIGHEST target in
    target_precisions that is actually reachable for this margin
    distribution.

    Why a fallback ladder instead of a fixed 90% bar: the original
    single-target version silently returned its initial default (1.0)
    whenever no cutoff reached 90% precision -- which looks identical in
    the printed/saved output to a genuinely well-calibrated tau_word of
    1.0, but actually means "the search found nothing usable," not "every
    prediction is maximally confident." At low alpha (beam/sensor score
    barely weighted), the decoder's margin can legitimately fail to carry
    enough correctness signal to hit 90% precision even on its most
    confident-looking subset -- that's a real calibration weakness of
    that weight combo, and returning a degenerate 1.0 for it just hides
    the weakness instead of surfacing it. Retrying at looser targets
    finds the best threshold this combo's margin CAN actually support.

    Returns (tau_word, achieved_target_precision, achieved_precision, n_kept).
    If not even the loosest target is reachable, returns
    (0.0, None, None, n_margins) -- 0.0 means "never gate anything as
    low-confidence," which is the safer failure direction than 1.0
    ("always gate everything"): a confidence field that's silently
    disabled is a smaller problem than one that's silently inverted."""
    scored = [(_margin_to_confidence(m), c) for m, c in margins]
    unique_confidences = sorted({round(conf, 3) for conf, _ in scored})

    for target in target_precisions:
        best_tau, best_count, best_precision = None, 0, None
        for cut in unique_confidences:
            kept = [c for conf, c in scored if conf >= cut]
            if not kept:
                continue
            precision = sum(kept) / len(kept)
            if precision >= target and len(kept) > best_count:
                best_tau, best_count, best_precision = cut, len(kept), precision
        if best_tau is not None:
            return best_tau, target, best_precision, best_count

    return 0.0, None, None, len(scored)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lo = (center - spread) / denom
    hi = (center + spread) / denom
    return (max(0.0, lo), min(1.0, hi))


def run_grid_from_raw(
    raw_results: list[tuple[str, list[RawCandidate]]], step: float = 0.1
):
    """Full unconstrained grid (alpha_min=0.05 floor to avoid a
    degenerate 0-weight). Each grid point only does the cheap weighted
    sum + sort (evaluate_from_raw) -- no re-decoding."""
    grid = [round(x, 2) for x in np.arange(0.05, 0.91, step)]
    results = []
    for alpha in grid:
        remainder = round(1.0 - alpha, 2)
        for beta in [round(x, 2) for x in np.arange(0.05, remainder, step)]:
            gamma = round(remainder - beta, 2)
            if gamma < 0.05:
                continue
            weights = ScoreWeights(alpha=alpha, beta=beta, gamma=gamma)
            acc, margins = evaluate_from_raw(raw_results, weights)
            results.append((acc, weights, margins))
    return results


def print_alpha_tradeoff_curve(results):
    best_by_alpha: dict[float, float] = {}
    for acc, w, _ in results:
        if w.alpha not in best_by_alpha or acc > best_by_alpha[w.alpha]:
            best_by_alpha[w.alpha] = acc
    print("\n[diagnostic] best word_acc achievable AT EACH alpha (unconstrained beta/gamma):")
    for alpha in sorted(best_by_alpha):
        bar = "#" * round(60 * best_by_alpha[alpha])
        print(f"    alpha={alpha:.2f}  acc={best_by_alpha[alpha]:.4f}  {bar}")
    print(
        "    -> this is exactly the shape the floor sweep below uses to pick a floor "
        "automatically -- a floor only costs accuracy if the curve is still rising "
        "when the floor cuts it off.\n"
    )


# ---------------------------------------------------------------------------
# Alpha-floor sweep (new): try every candidate floor against the SAME
# already-computed per_seed_results, no re-decoding, and pick whichever
# floor's winning combo has the best pooled VAL accuracy.
# ---------------------------------------------------------------------------
def select_weights_across_floors(
    per_seed_results: list[list[tuple[float, ScoreWeights, list]]],
    alpha_min_candidates: list[float],
    n_seeds: int,
) -> list[dict]:
    """For each candidate floor: pick the best (alpha,beta,gamma) combo
    per seed among combos with alpha >= floor, take the majority-vote
    winner across seeds, and compute that winner's POOLED accuracy across
    all seeds' words (not just the seeds that voted for it -- this is the
    same "final reported number" semantics the original single-floor
    version used). Floors that eliminate every combo for some seed
    (shouldn't normally happen since 0.05 is always in the grid, but
    guarded anyway) are skipped.

    Returns one summary dict per floor that produced a usable combo.
    """
    summaries = []
    for floor in alpha_min_candidates:
        constrained_best_per_seed = []
        for results in per_seed_results:
            constrained = [(acc, w, m) for acc, w, m in results if w.alpha >= floor]
            if not constrained:
                continue
            constrained_best_per_seed.append(max(constrained, key=lambda r: r[0]))

        if len(constrained_best_per_seed) < n_seeds:
            print(f"[warn] alpha_min={floor}: no combo satisfies this floor for at least "
                  f"one seed -- skipping this floor")
            continue

        combo_votes = Counter((w.alpha, w.beta, w.gamma) for _, w, _ in constrained_best_per_seed)
        winning_combo, n_votes = combo_votes.most_common(1)[0]
        best_weights = ScoreWeights(
            alpha=winning_combo[0], beta=winning_combo[1], gamma=winning_combo[2]
        )

        total_correct, total_n = 0, 0
        pooled_margins = []
        for results in per_seed_results:
            _, _, margins = next(
                r for r in results
                if (r[1].alpha, r[1].beta, r[1].gamma) == winning_combo
            )
            total_correct += sum(c for _, c in margins)
            total_n += len(margins)
            pooled_margins.extend(margins)
        pooled_acc = total_correct / total_n if total_n else 0.0
        lo, hi = wilson_ci(total_correct, total_n)

        summaries.append({
            "alpha_min": floor,
            "weights": best_weights,
            "seed_agreement": f"{n_votes}/{n_seeds}",
            "pooled_val_accuracy": pooled_acc,
            "pooled_val_ci": (lo, hi),
            "pooled_margins": pooled_margins,
        })
    return summaries


def main(n_words: int, alpha_min_candidates: list[float], n_seeds: int, workers: int) -> None:
    val_data = np.load(VAL_NPZ_PATH, allow_pickle=True)
    X_val, y_val = val_data["X"], val_data["y"]
    model = tf.keras.models.load_model(model_path("tcn"))
    val_probs = model.predict(X_val, batch_size=256, verbose=0)

    print(f"[tune] tuning on VAL split ({VAL_NPZ_PATH}) -- TEST is held out until the end")
    print(f"[tune] sweeping alpha_min floors: {alpha_min_candidates} "
          f"(picking whichever gives the best pooled VAL accuracy -- no hand-picked floor)")
    print(f"[tune] workers={workers} for the beam-search/dictionary decode step")

    # --- Expensive step: decode_raw() once per (seed, word). Computed
    # exactly once regardless of how many floors we sweep afterward. -----
    per_seed_raw = []
    for seed in range(n_seeds):
        words = build_synthetic_words(y_val, n_words, seed=seed)
        t0 = time.perf_counter()
        raw_results = precompute_raw_candidates(words, val_probs, n_workers=workers)
        elapsed = time.perf_counter() - t0
        per_seed_raw.append(raw_results)
        print(f"[tune] seed={seed}: decoded {len(raw_results)} synthetic val words "
              f"in {elapsed:.1f}s")

    # --- Cheap step: full grid-search against the cached decodes, once. -
    t0 = time.perf_counter()
    per_seed_results = [run_grid_from_raw(raw_results) for raw_results in per_seed_raw]
    print(f"[tune] weight grid-search over {len(per_seed_results[0])} combos x "
          f"{n_seeds} seeds took {time.perf_counter() - t0:.2f}s (this is the part "
          f"that used to dominate runtime)")

    print_alpha_tradeoff_curve(per_seed_results[0])

    # --- Floor sweep: reuses per_seed_results, no re-decoding. ----------
    summaries = select_weights_across_floors(per_seed_results, alpha_min_candidates, n_seeds)
    if not summaries:
        raise RuntimeError(
            "No alpha_min floor in --alpha-min-candidates produced a usable combo for "
            "every seed -- widen the candidate list (e.g. include 0.05)."
        )

    print("[tune] alpha_min floor sweep (pooled VAL accuracy per floor, all computed "
          "from the SAME decoded words -- no extra decoding per floor):")
    header = f"{'alpha_min':>10}{'alpha':>8}{'beta':>8}{'gamma':>8}{'seed_agree':>12}{'val_acc':>10}"
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda s: s["alpha_min"]):
        w = s["weights"]
        print(f"{s['alpha_min']:>10.2f}{w.alpha:>8.2f}{w.beta:>8.2f}{w.gamma:>8.2f}"
              f"{s['seed_agreement']:>12}{s['pooled_val_accuracy']:>10.4f}")

    # Pick the floor with the best pooled VAL accuracy -- the same
    # evidence-based selection rule used everywhere else in this project,
    # now applied to the floor itself instead of leaving it hand-picked.
    best = max(summaries, key=lambda s: s["pooled_val_accuracy"])
    best_weights = best["weights"]
    print(
        f"\n[tune] SELECTED: alpha_min={best['alpha_min']} -> "
        f"alpha={best_weights.alpha} beta={best_weights.beta} gamma={best_weights.gamma} "
        f"val_word_acc={best['pooled_val_accuracy']:.4f} "
        f"(95% CI [{best['pooled_val_ci'][0]:.4f}, {best['pooled_val_ci'][1]:.4f}]), "
        f"seed_agreement={best['seed_agreement']}"
    )
    if best["seed_agreement"] != f"{n_seeds}/{n_seeds}":
        print(
            f"[warn] the selected floor's winning combo only won {best['seed_agreement']} "
            f"seeds -- the ranking is NOT fully stable at n_words={n_words}. Consider "
            f"raising --n-words.\n"
        )

    tau, tau_target, tau_precision, tau_n_kept = suggest_tau(best["pooled_margins"])
    if tau_target is not None:
        print(
            f"[tune] suggested_tau_word={tau:.3f} -- achieves {tau_precision:.1%} precision "
            f"(target was {tau_target:.0%}) on {tau_n_kept}/{len(best['pooled_margins'])} "
            f"pooled val words at/above this confidence"
        )
    else:
        print(
            f"[tune] suggested_tau_word=0.0 -- WARNING: no confidence cutoff reached even "
            f"the loosest target ({min(DEFAULT_TAU_TARGET_PRECISIONS):.0%} precision) for "
            f"this weight combo. The decoder's margin carries very little correctness signal "
            f"at alpha={best_weights.alpha} -- confidence-gated features (§11.5, §13.6) will "
            f"not work well until this is revisited (e.g. a higher alpha, more val words, or "
            f"a finer weight grid)."
        )

    # --- ONE-SHOT confirmatory evaluation on TEST. ------------------------
    test_data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    X_test, y_test = test_data["X"], test_data["y"]
    test_probs = model.predict(X_test, batch_size=256, verbose=0)
    test_words = build_synthetic_words(y_test, n_words, seed=1234)  # fixed seed, used once
    test_raw = precompute_raw_candidates(test_words, test_probs, n_workers=workers)
    test_acc, test_margins = evaluate_from_raw(test_raw, best_weights)
    test_lo, test_hi = wilson_ci(sum(c for _, c in test_margins), len(test_margins))
    print(
        f"[tune] CONFIRMATORY test_word_acc={test_acc:.4f} "
        f"(95% CI [{test_lo:.4f}, {test_hi:.4f}], n={len(test_margins)}) "
        f"-- weights were never fit against this data\n"
    )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "alpha": best_weights.alpha, "beta": best_weights.beta,
            "gamma": best_weights.gamma, "tau_word": tau,
            "tau_word_achieved_precision_target": tau_target,
            "tau_word_achieved_precision": tau_precision,
            "tau_word_n_kept": tau_n_kept,
            "tau_word_n_total": len(best["pooled_margins"]),
            "alpha_min_selected": best["alpha_min"],
            "alpha_min_candidates_swept": alpha_min_candidates,
            "floor_sweep_summary": [
                {
                    "alpha_min": s["alpha_min"],
                    "alpha": s["weights"].alpha, "beta": s["weights"].beta,
                    "gamma": s["weights"].gamma,
                    "seed_agreement": s["seed_agreement"],
                    "pooled_val_accuracy": s["pooled_val_accuracy"],
                    "pooled_val_ci": list(s["pooled_val_ci"]),
                }
                for s in sorted(summaries, key=lambda s: s["alpha_min"])
            ],
            "tuned_on_split": "val",
            "val_word_accuracy": best["pooled_val_accuracy"],
            "val_word_accuracy_95ci": list(best["pooled_val_ci"]),
            "test_word_accuracy_confirmatory": test_acc,
            "test_word_accuracy_95ci": [test_lo, test_hi],
            "n_words_per_seed": n_words,
            "n_seeds": n_seeds,
            "seed_agreement": best["seed_agreement"],
            "note": "Tuned on SYNTHETIC concatenated-character words from the VAL split "
                    "(ActionPlan.md 4.3), not real continuous writing -- retune once real "
                    "word-level data exists. TEST accuracy above is a single confirmatory "
                    "run, not part of the search. alpha_min was itself selected by sweeping "
                    "alpha_min_candidates_swept and picking the floor with the best pooled "
                    "VAL accuracy (see floor_sweep_summary) -- not hand-picked.",
        }, f, indent=2)
    print(f"[save] {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-words", type=int, default=800)
    parser.add_argument("--n-seeds", type=int, default=3,
                         help="How many independent synthetic-word samples to grid-search "
                              "over, to check the winning combo is stable and not grid noise.")
    parser.add_argument(
        "--alpha-min-candidates", type=str, default=None,
        help="Comma-separated list of alpha_min floors to sweep, e.g. '0.05,0.3,0.5'. "
             "The script picks whichever floor gives the best pooled VAL accuracy -- "
             "pass a single value to force one floor (e.g. for an ablation write-up). "
             f"Default sweeps every alpha value in the grid: {DEFAULT_ALPHA_MIN_CANDIDATES}.",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                         help="Process-pool workers for the beam-search/dictionary decode "
                              "step (the one-time expensive part). Set to 1 to disable "
                              "multiprocessing. Threads are intentionally not used here -- "
                              "this is CPU-bound pure-Python work, so threads would be "
                              "serialized by the GIL and give no speedup; only separate "
                              "processes actually parallelize it.")
    args = parser.parse_args()

    if args.alpha_min_candidates:
        candidates = [float(x) for x in args.alpha_min_candidates.split(",")]
    else:
        candidates = DEFAULT_ALPHA_MIN_CANDIDATES

    main(args.n_words, candidates, args.n_seeds, args.workers)