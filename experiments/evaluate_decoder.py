"""
experiments/evaluate_decoder.py

Ablation: does beam search help? Does dictionary correction help? Does
combining them help more than either alone? All four configurations
(A/B/C/D) are evaluated on the exact SAME fixed synthetic-word TEST set,
so the comparison is apples-to-apples.

    A. Greedy character decoding, NO dictionary correction
    B. Beam search (beam_width=5), NO dictionary correction
    C. Greedy character decoding + dictionary/wordfreq correction
    D. Beam search (beam_width=5) + dictionary correction, using the
       weights/tau_word tuned by experiments/tune_decoder_weights.py

TEST is used exactly once here, for final reporting only. Nothing is
tuned against it -- alpha/beta/gamma/tau_word were already tuned on VAL
by tune_decoder_weights.py and are just loaded here unchanged.

    SYNTHETIC CONCATENATED-CHARACTER WORD EVALUATION -- NOT REAL
    CONTINUOUS HANDWRITING (ActionPlan.md Sec.4.3). Word accuracy here
    measures the decoder pipeline on isolated-character samples stitched
    together, not on real continuous air-writing.

Reused, not duplicated:
    - inference.beam_search.beam_search            (via WordDecoder)
    - inference.word_decoder.WordDecoder            (.decode_raw / .score_raw_candidates)
    - inference.word_decoder.RawCandidate / ScoreWeights / DEFAULT_VOCAB_SIZE
    - language.edit_distance.levenshtein             (error-analysis distance check)
    - experiments.tune_decoder_weights.build_synthetic_words
    - experiments.tune_decoder_weights.wilson_ci
    - experiments.tune_decoder_weights.OUT_PATH       (tuned-weights JSON path)

Efficiency: sensor probabilities for the whole TEST split are computed
ONCE (single batched model.predict). The expensive step -- beam search +
dictionary correction (decode_raw) -- is run at most TWICE per synthetic
word: once at beam_width=5 (feeds B and D) and once at beam_width=1/greedy
(feeds A and C). Configurations never each re-run their own full decode;
they share these two decode_raw() outputs. This is parallelized across a
process pool the same way experiments/tune_decoder_weights.py does it
(threads would not help -- this is CPU-bound pure-Python work, serialized
by the GIL; only separate processes actually parallelize it).

Usage:
    python -m experiments.evaluate_decoder --n-words 800 --n-errors 30
    python -m experiments.evaluate_decoder --n-words 800 --workers 4
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from config import EXPERIMENTS_DIR, TEST_NPZ_PATH, model_path
from inference.word_decoder import DEFAULT_VOCAB_SIZE, RawCandidate, ScoreWeights, WordDecoder
from language import edit_distance

# Reuse the tuning script's word-building / stats helpers and the exact
# path it wrote decoder_weights.json to -- no duplicated logic, no
# duplicated path constant.
from experiments.tune_decoder_weights import OUT_PATH as TUNED_WEIGHTS_PATH
from experiments.tune_decoder_weights import build_synthetic_words, wilson_ci

RESULTS_JSON_PATH = EXPERIMENTS_DIR / "decoder_evaluation.json"
RESULTS_TXT_PATH = EXPERIMENTS_DIR / "decoder_evaluation.txt"

BANNER = (
    "SYNTHETIC CONCATENATED-CHARACTER WORD EVALUATION -- "
    "NOT REAL CONTINUOUS HANDWRITING (ActionPlan.md Sec.4.3)."
)

CONFIG_DESCRIPTIONS = {
    "A": {"name": "Greedy, no dictionary", "beam_search": False, "dictionary_correction": False},
    "B": {"name": "Beam search only", "beam_search": True, "dictionary_correction": False},
    "C": {"name": "Dictionary correction only (greedy)", "beam_search": False, "dictionary_correction": True},
    "D": {"name": "Beam search + dictionary correction", "beam_search": True, "dictionary_correction": True},
}


# ---------------------------------------------------------------------------
# Tuned-weights loading (mirrors app/correction.py's _load_tuned_config,
# done standalone here so this script doesn't need to import the FastAPI
# app just to read a JSON file).
# ---------------------------------------------------------------------------
def load_tuned_weights(path: Path = TUNED_WEIGHTS_PATH) -> tuple[ScoreWeights, float, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No tuned weights at {path} -- run "
            f"`python -m experiments.tune_decoder_weights` first."
        )
    with open(path) as f:
        cfg = json.load(f)
    weights = ScoreWeights(alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"])
    tau_word = float(cfg.get("tau_word", 0.6))
    return weights, tau_word, cfg


# ---------------------------------------------------------------------------
# Expensive, weight-independent step: beam search (both widths) +
# dictionary correction. Run this ONCE per word, never per configuration.
# ---------------------------------------------------------------------------
_worker_beam_decoder: WordDecoder | None = None
_worker_greedy_decoder: WordDecoder | None = None


def _init_worker(
    beam_width: int, top_k: int, vocab_size: int, max_search_distance: int
) -> None:
    """Runs once per worker PROCESS, so the BK-tree (built lazily inside
    language.edit_distance on first correction) gets built at most once
    per worker, not once per word."""
    global _worker_beam_decoder, _worker_greedy_decoder
    _worker_beam_decoder = WordDecoder(
        beam_width=beam_width, top_k=top_k,
        vocab_size=vocab_size, max_search_distance=max_search_distance,
    )
    _worker_greedy_decoder = WordDecoder(
        beam_width=1, top_k=top_k,
        vocab_size=vocab_size, max_search_distance=max_search_distance,
    )


def _decode_word_worker(seq: list[list[float]]) -> tuple[list[RawCandidate], list[RawCandidate]]:
    assert _worker_beam_decoder is not None and _worker_greedy_decoder is not None
    raw_beam = _worker_beam_decoder.decode_raw(seq)
    raw_greedy = _worker_greedy_decoder.decode_raw(seq)
    return raw_beam, raw_greedy


def decode_all_configs(
    words: list[tuple[str, list[int]]],
    all_probs: np.ndarray,
    beam_width: int = 5,
    top_k: int = 5,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_search_distance: int = 3,
    n_workers: int = 1,
) -> list[tuple[str, list[RawCandidate], list[RawCandidate]]]:
    """Returns [(true_word, raw_candidates_beam5, raw_candidates_greedy), ...]
    -- exactly one beam-width-5 decode and one beam-width-1 (greedy) decode
    per word, shared across all four A/B/C/D configurations below."""
    tasks = [
        (true_word, [all_probs[i].tolist() for i in row_indices])
        for true_word, row_indices in words
    ]

    if n_workers <= 1:
        beam_decoder = WordDecoder(
            beam_width=beam_width, top_k=top_k,
            vocab_size=vocab_size, max_search_distance=max_search_distance,
        )
        greedy_decoder = WordDecoder(
            beam_width=1, top_k=top_k,
            vocab_size=vocab_size, max_search_distance=max_search_distance,
        )
        return [
            (tw, beam_decoder.decode_raw(seq), greedy_decoder.decode_raw(seq))
            for tw, seq in tasks
        ]

    out: list[tuple[str, list[RawCandidate], list[RawCandidate]]] = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(beam_width, top_k, vocab_size, max_search_distance),
    ) as ex:
        for (tw, _seq), (raw_beam, raw_greedy) in zip(
            tasks, ex.map(_decode_word_worker, (seq for _, seq in tasks))
        ):
            out.append((tw, raw_beam, raw_greedy))
    return out


# ---------------------------------------------------------------------------
# Error analysis heuristic for config D misses.
# ---------------------------------------------------------------------------
def classify_error(
    true_word: str,
    predicted: str,
    raw_beam: list[RawCandidate],
    scored_candidates: list[dict],
) -> str:
    """Best-effort classification of why config D got a word wrong, using
    only information already computed by decode_raw/score_raw_candidates
    (no extra beam-search or dictionary calls)."""
    true_lower = true_word.lower()
    beam_raws_lower = [c.raw.lower() for c in raw_beam]
    corrected_words_lower = [c.word.lower() for c in raw_beam]

    if true_lower in beam_raws_lower:
        return (
            "dictionary prior / word frequency bias -- the correct raw "
            "spelling WAS in the beam, but scoring (edit-distance or "
            "frequency weight) promoted a different candidate above it"
        )
    if true_lower in corrected_words_lower:
        return (
            "beam+dictionary scoring -- the true word was reachable via "
            "edit-distance correction from some beam candidate, but was "
            "outranked by another candidate's final_score"
        )

    best_raw = raw_beam[0].raw if raw_beam else ""
    dist = edit_distance.levenshtein(best_raw.lower(), true_lower)
    if dist == 0:
        return "other (case/whitespace mismatch despite matching text -- inspect manually)"
    if dist <= 2:
        return (
            f"ambiguous characters / incorrect character classification "
            f"(top beam candidate {best_raw!r} is only edit-distance {dist} from truth)"
        )
    return (
        "beam search candidate limitation -- true spelling is far "
        "(edit-distance > 2) from every candidate in the beam; the "
        "underlying TCN likely misclassified multiple characters and the "
        "beam/top_k width didn't recover the true sequence"
    )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main(n_words: int, seed: int, beam_width: int, top_k: int, workers: int, n_errors: int) -> None:
    # Imported here, not at module level -- so worker processes spawned by
    # decode_all_configs() never pay TensorFlow's import cost (they only
    # ever call decode_raw(), which touches beam search + the dictionary,
    # never the TCN model itself).
    import tensorflow as tf

    print(BANNER)
    print(f"[eval] beam_width={beam_width} top_k={top_k} workers={workers} seed={seed}\n")

    weights, tau_word, tuned_meta = load_tuned_weights()
    print(
        f"[eval] loaded tuned weights from {TUNED_WEIGHTS_PATH}: "
        f"alpha={weights.alpha} beta={weights.beta} gamma={weights.gamma} tau_word={tau_word}"
    )

    data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    X_test, y_test = data["X"], data["y"]
    model = tf.keras.models.load_model(model_path("tcn"))
    all_probs = model.predict(X_test, batch_size=256, verbose=0)

    y_pred_argmax = np.argmax(all_probs, axis=1)
    char_accuracy = float((y_pred_argmax == y_test).mean())
    print(f"[eval] TCN character-level accuracy on full TEST split: {char_accuracy:.4%} "
          f"(n={len(y_test)})\n")

    words = build_synthetic_words(y_test, n_words, seed=seed)
    print(f"[eval] built {len(words)} synthetic TEST words (seed={seed}) -- {BANNER}\n")

    t0 = time.perf_counter()
    decoded = decode_all_configs(
        words, all_probs, beam_width=beam_width, top_k=top_k, n_workers=workers
    )
    print(f"[eval] decoded {len(decoded)} words (beam5 + greedy each) in "
          f"{time.perf_counter() - t0:.1f}s\n")

    dropped = sum(1 for _, rb, rg in decoded if not rb or not rg)
    if dropped:
        print(f"[warn] {dropped} words produced empty candidate lists and are excluded below")
    usable = [(tw, rb, rg) for tw, rb, rg in decoded if rb and rg]

    results = {cfg: {"n_correct": 0, "n_words": len(usable)} for cfg in CONFIG_DESCRIPTIONS}
    errors_D: list[dict] = []

    for true_word, raw_beam, raw_greedy in usable:
        pred_A = raw_greedy[0].raw           # greedy, raw beam text, no dictionary
        pred_B = raw_beam[0].raw             # beam-search-best raw text, no dictionary
        pred_C = raw_greedy[0].word          # greedy, dictionary-corrected
        scored_D = WordDecoder.score_raw_candidates(raw_beam, weights)
        pred_D = scored_D["prediction"]

        results["A"]["n_correct"] += int(pred_A.lower() == true_word.lower())
        results["B"]["n_correct"] += int(pred_B.lower() == true_word.lower())
        results["C"]["n_correct"] += int(pred_C.lower() == true_word.lower())
        is_correct_D = pred_D.lower() == true_word.lower()
        results["D"]["n_correct"] += int(is_correct_D)

        if not is_correct_D:
            reason = classify_error(true_word, pred_D, raw_beam, scored_D["candidates"])
            errors_D.append(
                {
                    "true": true_word,
                    "predicted": pred_D,
                    "raw_or_beam_candidate": raw_beam[0].raw,
                    "reason": reason,
                }
            )

    for cfg, r in results.items():
        n, k = r["n_words"], r["n_correct"]
        r["accuracy"] = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        r["ci_95"] = [lo, hi]

    improvements = {
        "B_minus_A": results["B"]["accuracy"] - results["A"]["accuracy"],
        "C_minus_A": results["C"]["accuracy"] - results["A"]["accuracy"],
        "D_minus_A": results["D"]["accuracy"] - results["A"]["accuracy"],
        "D_minus_B": results["D"]["accuracy"] - results["B"]["accuracy"],
        "D_minus_C": results["D"]["accuracy"] - results["C"]["accuracy"],
    }

    rng = random.Random(seed)
    sample_errors = (
        rng.sample(errors_D, n_errors) if len(errors_D) > n_errors else list(errors_D)
    )

    # --- Printed report ----------------------------------------------------
    table_lines = _format_table(results)
    print("\n".join(table_lines))
    print()
    print(_format_improvements(improvements))
    print()
    print(_format_conclusions(improvements))
    print()
    print(f"[eval] {len(errors_D)}/{results['D']['n_words']} config-D words were wrong; "
          f"showing {len(sample_errors)} sampled errors below\n")
    print(_format_error_table(sample_errors))

    # --- Persist -------------------------------------------------------------
    report = {
        "banner": BANNER,
        "meta": {
            "split": "test",
            "seed": seed,
            "n_words_requested": n_words,
            "n_words_evaluated": results["D"]["n_words"],
            "n_words_dropped": dropped,
            "beam_width": beam_width,
            "top_k": top_k,
            "tuned_weights": {"alpha": weights.alpha, "beta": weights.beta, "gamma": weights.gamma},
            "tau_word": tau_word,
            "tuned_weights_source": str(TUNED_WEIGHTS_PATH),
            "tuned_weights_meta": tuned_meta,
        },
        "tcn_character_accuracy": {
            "accuracy": char_accuracy,
            "n_samples": int(len(y_test)),
        },
        "results": {
            cfg: {**CONFIG_DESCRIPTIONS[cfg], **results[cfg]} for cfg in CONFIG_DESCRIPTIONS
        },
        "improvements": improvements,
        "sample_errors_config_D": sample_errors,
        "n_errors_total_config_D": len(errors_D),
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save] {RESULTS_JSON_PATH}")

    with open(RESULTS_TXT_PATH, "w") as f:
        f.write(BANNER + "\n\n")
        f.write(f"TCN character-level accuracy (full TEST split, n={len(y_test)}): "
                f"{char_accuracy:.4%}\n\n")
        f.write("\n".join(table_lines) + "\n\n")
        f.write(_format_improvements(improvements) + "\n\n")
        f.write(_format_conclusions(improvements) + "\n\n")
        f.write(f"{len(errors_D)}/{results['D']['n_words']} config-D words wrong; "
                f"{len(sample_errors)} sampled below\n\n")
        f.write(_format_error_table(sample_errors) + "\n")
    print(f"[save] {RESULTS_TXT_PATH}")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_table(results: dict) -> list[str]:
    header = f"{'Configuration':<45}{'Beam Search':>13}{'Dictionary':>12}{'Accuracy':>11}{'95% CI':>20}"
    lines = [header, "-" * len(header)]
    for cfg in "ABCD":
        d = CONFIG_DESCRIPTIONS[cfg]
        r = results[cfg]
        ci = f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]"
        lines.append(
            f"{cfg + '. ' + d['name']:<45}"
            f"{'Yes' if d['beam_search'] else 'No':>13}"
            f"{'Yes' if d['dictionary_correction'] else 'No':>12}"
            f"{r['accuracy']:>10.2%}"
            f"{ci:>20}"
        )
    return lines


def _format_improvements(improvements: dict) -> str:
    lines = ["Absolute improvement (percentage points):"]
    labels = {
        "B_minus_A": "B - A  (beam search alone)",
        "C_minus_A": "C - A  (dictionary correction alone)",
        "D_minus_A": "D - A  (combined vs. plain greedy)",
        "D_minus_B": "D - B  (dictionary's added value on top of beam search)",
        "D_minus_C": "D - C  (beam search's added value on top of dictionary)",
    }
    for key, label in labels.items():
        lines.append(f"  {label:<55} {improvements[key] * 100:+.2f} pp")
    return "\n".join(lines)


def _format_conclusions(improvements: dict) -> str:
    return (
        f"Beam search improves word accuracy by {improvements['B_minus_A'] * 100:.2f} "
        f"percentage points over greedy decoding.\n"
        f"Dictionary correction improves word accuracy by {improvements['C_minus_A'] * 100:.2f} "
        f"percentage points over greedy decoding.\n"
        f"Combined beam search + dictionary correction improves word accuracy by "
        f"{improvements['D_minus_A'] * 100:.2f} percentage points over the greedy/no-correction "
        f"baseline (vs. beam-alone: {improvements['D_minus_B'] * 100:+.2f} pp; "
        f"vs. dictionary-alone: {improvements['D_minus_C'] * 100:+.2f} pp)."
    )


def _format_error_table(sample_errors: list[dict]) -> str:
    if not sample_errors:
        return "(no errors to show)"
    header = f"{'TRUE':<15}{'PREDICTED':<15}{'RAW/BEAM CANDIDATE':<22}{'REASON'}"
    lines = [header, "-" * len(header)]
    for e in sample_errors:
        lines.append(
            f"{e['true']:<15}{e['predicted']:<15}{e['raw_or_beam_candidate']:<22}{e['reason']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-words", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1234,
                         help="Fixed seed for the synthetic TEST word set. Default (1234) "
                              "matches tune_decoder_weights.py's confirmatory test run, so "
                              "config D's accuracy here should reproduce that run's "
                              "test_word_accuracy_confirmatory number as a sanity check.")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1,
                         help="Process-pool workers for the beam-search/dictionary decode "
                              "step. Threads are intentionally not used (CPU-bound pure "
                              "Python, serialized by the GIL) -- only separate processes "
                              "parallelize this.")
    parser.add_argument("--n-errors", type=int, default=30,
                         help="How many config-D errors to sample (reproducibly, via --seed) "
                              "for the printed/saved error-analysis table.")
    args = parser.parse_args()
    main(args.n_words, args.seed, args.beam_width, args.top_k, args.workers, args.n_errors)