"""
Decoder evaluation suite on synthetic concatenated-character words.

Usage:
    python -m experiments.evaluate_decoder --n-words 800 --workers 4
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from config import EXPERIMENTS_DIR, NGRAM_MODEL_PATH, TEST_NPZ_PATH, model_path
from inference.word_decoder import RawCandidate, ScoreWeights, WordDecoder
from language import edit_distance
from language.ngram import NgramLanguageModel

# Reuse tuning script helpers and paths.
from experiments.tune_decoder_weights import (
    CONFIDENCE_SIGMOID_STEEPNESS,
    OUT_PATH as TUNED_WEIGHTS_PATH,
    _margin_to_confidence,
    build_synthetic_words,
    precompute_raw_candidates,
    wilson_ci,
)

RESULTS_JSON_PATH = EXPERIMENTS_DIR / "decoder_evaluation.json"
RESULTS_TXT_PATH = EXPERIMENTS_DIR / "decoder_evaluation.txt"

BANNER = (
    "SYNTHETIC CONCATENATED-CHARACTER WORD EVALUATION -- "
    "NOT REAL CONTINUOUS HANDWRITING (ActionPlan.md Sec.4.3)."
)

CONFIG_DESCRIPTIONS = {
    "A": {"name": "Greedy, no dictionary", "beam_search": False, "dictionary_correction": False, "ngram": False},
    "B": {"name": "Beam search only", "beam_search": True, "dictionary_correction": False, "ngram": False},
    "C": {"name": "Dictionary correction only (greedy)", "beam_search": False, "dictionary_correction": True, "ngram": False},
    "D": {"name": "Beam search + dictionary correction", "beam_search": True, "dictionary_correction": True, "ngram": False},
    "E": {"name": "Beam search + dictionary + n-gram LM", "beam_search": True, "dictionary_correction": True, "ngram": True},
}

DEFAULT_ORDER_SWEEP = [2, 3, 4, 5]
DEFAULT_LAMBDA_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_BEAM_WIDTH_SWEEP = [1, 3, 5, 10]


def _float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


# ---------------------------------------------------------------------------
# Tuned-weights loading (mirrors app/correction.py's _load_tuned_config).
# FIXED in this revision: delta and search_lambda_lm are now actually
# read and returned, instead of being silently dropped.
# ---------------------------------------------------------------------------
def load_tuned_weights(path: Path = TUNED_WEIGHTS_PATH) -> tuple[ScoreWeights, float, float, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No tuned weights at {path} -- run "
            f"`python -m experiments.tune_decoder_weights` first."
        )
    with open(path) as f:
        cfg = json.load(f)
    weights = ScoreWeights(
        alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"],
        delta=float(cfg.get("delta", 0.0)),
    )
    tau_word = float(cfg.get("tau_word", 0.6))
    search_lambda_lm = float(cfg.get("search_lambda_lm", 0.0))
    return weights, tau_word, search_lambda_lm, cfg


def load_ngram_model(path: Path = NGRAM_MODEL_PATH) -> NgramLanguageModel | None:
    if not path.exists():
        return None
    return NgramLanguageModel.load(path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _unique_top_words(candidates: list[dict], k: int) -> list[str]:
    """First k DISTINCT corrected words, in ranked order (several beam
    candidates can correct to the same dictionary word; that should
    count once toward top-k, not pad it with duplicates)."""
    seen: list[str] = []
    for c in candidates:
        w = c["word"].lower()
        if w not in seen:
            seen.append(w)
        if len(seen) >= k:
            break
    return seen


def _acc_ci(n_correct: int, n_total: int) -> dict:
    acc = n_correct / n_total if n_total else 0.0
    lo, hi = wilson_ci(n_correct, n_total)
    return {"n_correct": n_correct, "n_words": n_total, "accuracy": acc, "ci_95": [lo, hi]}


def classify_error(
    true_word: str,
    predicted: str,
    raw_beam: list[RawCandidate],
    scored_candidates: list[dict],
) -> str:
    """Best-effort classification of why a config got a word wrong,
    using only information already computed by decode_raw/
    score_raw_candidates (no extra beam-search or dictionary calls)."""
    true_lower = true_word.lower()
    beam_raws_lower = [c.raw.lower() for c in raw_beam]
    corrected_words_lower = [c.word.lower() for c in raw_beam]

    if true_lower in beam_raws_lower:
        return (
            "dictionary/LM prior -- the correct raw spelling WAS in the "
            "beam, but scoring (edit-distance, frequency, or LM weight) "
            "promoted a different candidate above it"
        )
    if true_lower in corrected_words_lower:
        return (
            "beam+dictionary/LM scoring -- the true word was reachable "
            "via edit-distance correction from some beam candidate, but "
            "was outranked by another candidate's final_score"
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


def classify_error_e(
    true_word: str,
    d_correct: bool,
    e_predicted: str,
    e_raw_beam: list[RawCandidate],
    e_scored_candidates: list[dict],
) -> str:
    """Same as classify_error, but with an extra category up front: if
    config D (no LM) already got this word right and E (with LM) got it
    wrong, that is specifically an n-gram regression, not a generic beam/
    dictionary failure -- worth distinguishing so the LM's downside is
    visible on its own, per the requested error-category breakdown."""
    if d_correct:
        return (
            "n-gram language-model regression -- beam+dictionary alone "
            "(config D) already predicted this word correctly; attaching "
            "the n-gram LM (search-time steering and/or final re-ranking) "
            "changed the outcome to a wrong prediction"
        )
    return classify_error(true_word, e_predicted, e_raw_beam, e_scored_candidates)


# ---------------------------------------------------------------------------
# Section 2: n-gram ORDER comparison
# ---------------------------------------------------------------------------
def run_order_sweep(
    words: list[tuple[str, list[int]]],
    all_probs: np.ndarray,
    orders: list[int],
    weights: ScoreWeights,
    search_lambda_lm: float,
    beam_width: int,
    top_k: int,
    n_workers: int,
    cache_dir: Path,
    save_models: bool,
) -> list[dict]:
    print(f"\n[order-sweep] evaluating orders {orders} on {len(words)} words "
          f"(search_lambda_lm={search_lambda_lm}, delta={weights.delta} held fixed)")
    rows = []
    for order in orders:
        cache_path = cache_dir / f"ngram_model_order{order}.json"
        if cache_path.exists():
            model = NgramLanguageModel.load(cache_path)
            print(f"[order-sweep] order={order}: loaded cached model from {cache_path}")
        else:
            t0 = time.perf_counter()
            model = NgramLanguageModel.train(order=order)
            print(f"[order-sweep] order={order}: trained in {time.perf_counter() - t0:.1f}s "
                  f"(contexts={len(model.totals)})")
            if save_models:
                model.save(cache_path)

        raw = precompute_raw_candidates(
            words, all_probs, beam_width=beam_width, top_k=top_k,
            n_workers=n_workers, ngram_model=model,
            search_lambda_lm=search_lambda_lm,
        )
        n_correct = 0
        n_usable = 0
        for true_word, raw_candidates in raw:
            if not raw_candidates:
                continue
            n_usable += 1
            pred = WordDecoder.score_raw_candidates(raw_candidates, weights)["prediction"]
            n_correct += int(pred.lower() == true_word.lower())
        row = {"order": order, **_acc_ci(n_correct, n_usable)}
        rows.append(row)
        print(f"[order-sweep] order={order}: accuracy={row['accuracy']:.2%} "
              f"(n={n_usable}, CI=[{row['ci_95'][0]:.4f}, {row['ci_95'][1]:.4f}])")
    return rows


# ---------------------------------------------------------------------------
# Section 3: n-gram WEIGHT (search_lambda_lm) sweep
# ---------------------------------------------------------------------------
def run_lambda_sweep(
    words: list[tuple[str, list[int]]],
    all_probs: np.ndarray,
    lambdas: list[float],
    ngram_model: NgramLanguageModel,
    weights: ScoreWeights,
    beam_width: int,
    top_k: int,
    n_workers: int,
) -> list[dict]:
    print(f"\n[lambda-sweep] evaluating search_lambda_lm in {lambdas} on {len(words)} words "
          f"(alpha/beta/gamma/delta held fixed at tuned values; lambda=0.0 reproduces "
          f"config D's search behavior exactly since it disables search-time LM steering)")
    rows = []
    for lam in lambdas:
        raw = precompute_raw_candidates(
            words, all_probs, beam_width=beam_width, top_k=top_k,
            n_workers=n_workers, ngram_model=ngram_model, search_lambda_lm=lam,
        )
        n_correct = 0
        n_usable = 0
        for true_word, raw_candidates in raw:
            if not raw_candidates:
                continue
            n_usable += 1
            pred = WordDecoder.score_raw_candidates(raw_candidates, weights)["prediction"]
            n_correct += int(pred.lower() == true_word.lower())
        row = {"search_lambda_lm": lam, **_acc_ci(n_correct, n_usable)}
        rows.append(row)
        print(f"[lambda-sweep] search_lambda_lm={lam:.2f}: accuracy={row['accuracy']:.2%} "
              f"(n={n_usable}, CI=[{row['ci_95'][0]:.4f}, {row['ci_95'][1]:.4f}])")
    return rows


# ---------------------------------------------------------------------------
# Section 8: beam-width sweep, WITH the n-gram LM attached
# ---------------------------------------------------------------------------
def run_beam_width_sweep(
    words: list[tuple[str, list[int]]],
    all_probs: np.ndarray,
    widths: list[int],
    ngram_model: NgramLanguageModel,
    search_lambda_lm: float,
    weights: ScoreWeights,
    top_k: int,
    n_workers: int,
) -> list[dict]:
    print(f"\n[beam-width-sweep] evaluating beam_width in {widths} on {len(words)} words "
          f"(with n-gram LM attached: search_lambda_lm={search_lambda_lm}, delta={weights.delta})")
    rows = []
    for width in widths:
        raw = precompute_raw_candidates(
            words, all_probs, beam_width=width, top_k=max(top_k, width),
            n_workers=n_workers, ngram_model=ngram_model,
            search_lambda_lm=search_lambda_lm,
        )
        n_correct = 0
        n_usable = 0
        for true_word, raw_candidates in raw:
            if not raw_candidates:
                continue
            n_usable += 1
            pred = WordDecoder.score_raw_candidates(raw_candidates, weights)["prediction"]
            n_correct += int(pred.lower() == true_word.lower())
        row = {"beam_width": width, **_acc_ci(n_correct, n_usable)}
        rows.append(row)
        print(f"[beam-width-sweep] beam_width={width:>2}: accuracy={row['accuracy']:.2%} "
              f"(n={n_usable}, CI=[{row['ci_95'][0]:.4f}, {row['ci_95'][1]:.4f}])")
    return rows


# ---------------------------------------------------------------------------
# Section 9: latency, no-LM vs. with-LM, single-process (a process pool's
# startup/IPC overhead would swamp a per-word timing measurement).
# ---------------------------------------------------------------------------
def run_latency_check(
    words: list[tuple[str, list[int]]],
    all_probs: np.ndarray,
    ngram_model: NgramLanguageModel | None,
    search_lambda_lm: float,
    beam_width: int,
    top_k: int,
    n_words: int,
) -> dict:
    """Single-process timing, no-LM vs. with-LM.

    IMPORTANT: language.edit_distance._bk_tree() is built lazily on the
    FIRST out-of-vocabulary correction (@lru_cache-wrapped, so it is
    built at most once per process, not once per decoder). Whichever
    decoder runs first inside this function pays that one-time
    construction cost; timing them back-to-back without a warm-up would
    make the LM decoder look artificially faster than the no-LM decoder
    purely because it ran second. Both decoders are warmed with one
    throwaway decode_raw() call (on a real word from the subset, not a
    synthetic dummy sequence) before EITHER is timed, so the BK-tree
    (and any other first-call caches) are already hot for both passes.
    """
    subset = words[:n_words]
    sequences = [[all_probs[i].tolist() for i in row_indices] for _, row_indices in subset]
    if not sequences:
        return {"n_words": 0, "no_lm_total_seconds": 0.0, "no_lm_ms_per_word": 0.0}
    warmup_seq = sequences[0]

    no_lm_decoder = WordDecoder(beam_width=beam_width, top_k=top_k)
    lm_decoder = (
        WordDecoder(
            beam_width=beam_width, top_k=top_k,
            ngram_model=ngram_model, search_lambda_lm=search_lambda_lm,
        ) if ngram_model is not None else None
    )

    # Warm-up pass (not timed): forces the BK-tree and any other
    # lazily-built, process-wide caches to be constructed before either
    # decoder's timed loop starts, so neither one unfairly "pays" for it.
    no_lm_decoder.decode_raw(warmup_seq)
    if lm_decoder is not None:
        lm_decoder.decode_raw(warmup_seq)

    t0 = time.perf_counter()
    for seq in sequences:
        no_lm_decoder.decode_raw(seq)
    no_lm_total = time.perf_counter() - t0

    result = {
        "n_words": len(sequences),
        "no_lm_total_seconds": no_lm_total,
        "no_lm_ms_per_word": 1000.0 * no_lm_total / len(sequences) if sequences else 0.0,
    }

    if lm_decoder is not None:
        t0 = time.perf_counter()
        for seq in sequences:
            lm_decoder.decode_raw(seq)
        lm_total = time.perf_counter() - t0
        result["with_lm_total_seconds"] = lm_total
        result["with_lm_ms_per_word"] = 1000.0 * lm_total / len(sequences) if sequences else 0.0
        result["overhead_ms_per_word"] = result["with_lm_ms_per_word"] - result["no_lm_ms_per_word"]

    print(f"\n[latency] {len(sequences)} words, single-process, beam_width={beam_width} "
          f"(both decoders warmed with 1 throwaway decode before timing):")
    print(f"    without LM: {result['no_lm_ms_per_word']:.2f} ms/word "
          f"({result['no_lm_total_seconds']:.2f}s total)")
    if ngram_model is not None:
        print(f"    with LM:    {result['with_lm_ms_per_word']:.2f} ms/word "
              f"({result['with_lm_total_seconds']:.2f}s total)")
        print(f"    overhead:   {result['overhead_ms_per_word']:+.2f} ms/word")
    return result


# ---------------------------------------------------------------------------
# Section 10: confidence / coverage table at several tau cutoffs
# ---------------------------------------------------------------------------
def run_tau_coverage_table(margins_correct: list[tuple[float, bool]], tau_candidates: list[float]) -> list[dict]:
    rows = []
    for tau in sorted(set(tau_candidates)):
        kept = [
            correct for margin, correct in margins_correct
            if _margin_to_confidence(margin, CONFIDENCE_SIGMOID_STEEPNESS) >= tau
        ]
        coverage = len(kept) / len(margins_correct) if margins_correct else 0.0
        acc_among_kept = (sum(kept) / len(kept)) if kept else None
        rows.append({
            "tau": tau,
            "coverage": coverage,
            "n_kept": len(kept),
            "accuracy_among_kept": acc_among_kept,
        })
    return rows


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    # Imported here, not at module level -- so worker processes spawned by
    # precompute_raw_candidates() never pay TensorFlow's import cost (they
    # only ever call decode_raw(), which touches beam search + the
    # dictionary + the n-gram model, never the TCN model itself).
    import tensorflow as tf

    print(BANNER)
    print(f"[eval] beam_width={args.beam_width} top_k={args.top_k} "
          f"workers={args.workers} seed={args.seed}\n")

    weights, tau_word, search_lambda_lm, tuned_meta = load_tuned_weights()
    print(
        f"[eval] loaded tuned weights from {TUNED_WEIGHTS_PATH}: "
        f"alpha={weights.alpha} beta={weights.beta} gamma={weights.gamma} "
        f"delta={weights.delta} tau_word={tau_word} search_lambda_lm={search_lambda_lm}"
    )

    ngram_model = None if args.no_ngram else load_ngram_model()
    if args.no_ngram:
        print("[eval] --no-ngram passed: running the pre-n-gram A-D suite only, LM fully disabled")
    elif ngram_model is None:
        print(f"[eval] WARNING: no n-gram model found at {NGRAM_MODEL_PATH} -- "
              f"run `python -m experiments.build_ngram_model` first. Falling back to "
              f"the A-D suite only (config E and every LM-dependent section are skipped).")
    else:
        print(f"[eval] loaded n-gram model from {NGRAM_MODEL_PATH} "
              f"(order={ngram_model.order}, contexts={len(ngram_model.totals)})")
    ngram_available = ngram_model is not None

    # --- TCN probabilities on TEST, computed once -----------------------
    data = np.load(TEST_NPZ_PATH, allow_pickle=True)
    X_test, y_test = data["X"], data["y"]
    model = tf.keras.models.load_model(model_path("tcn"))
    all_probs = model.predict(X_test, batch_size=256, verbose=0)

    y_pred_argmax = np.argmax(all_probs, axis=1)
    char_accuracy = float((y_pred_argmax == y_test).mean())
    print(f"\n[eval] TCN character-level accuracy on full TEST split: {char_accuracy:.4%} "
          f"(n={len(y_test)})")

    words = build_synthetic_words(y_test, args.n_words, seed=args.seed)
    print(f"[eval] built {len(words)} synthetic TEST words (seed={args.seed}) -- {BANNER}\n")

    # =====================================================================
    # SECTION 1: main ablation (A-E) + top-1/3/5
    # =====================================================================
    weights_no_lm = ScoreWeights(alpha=weights.alpha, beta=weights.beta, gamma=weights.gamma, delta=0.0)

    t0 = time.perf_counter()
    raw_greedy_nolm = precompute_raw_candidates(
        words, all_probs, beam_width=1, top_k=args.top_k, n_workers=args.workers,
    )
    raw_beam_nolm = precompute_raw_candidates(
        words, all_probs, beam_width=args.beam_width, top_k=args.top_k, n_workers=args.workers,
    )
    raw_beam_lm = (
        precompute_raw_candidates(
            words, all_probs, beam_width=args.beam_width, top_k=args.top_k,
            n_workers=args.workers, ngram_model=ngram_model, search_lambda_lm=search_lambda_lm,
        ) if ngram_available else None
    )
    decode_seconds = time.perf_counter() - t0
    n_decodes = "3" if ngram_available else "2"
    print(f"[eval] decoded {len(words)} words x {n_decodes} configurations "
          f"(greedy + beam{'+ beam-with-LM' if ngram_available else ''}) in {decode_seconds:.1f}s\n")

    results = {cfg: {"n_correct": 0, "n_words": 0} for cfg in ("A", "B", "C", "D")}
    if ngram_available:
        results["E"] = {"n_correct": 0, "n_words": 0}
    topk_results = {"B": {1: 0, 3: 0, 5: 0}, "D": {1: 0, 3: 0, 5: 0}}
    if ngram_available:
        topk_results["E"] = {1: 0, 3: 0, 5: 0}
    errors_D: list[dict] = []
    errors_E: list[dict] = []
    d_vs_e_matrix = {"fixed_by_lm": 0, "regressed_by_lm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    edit_buckets = {"0": {"D": [0, 0], "E": [0, 0]}, "1": {"D": [0, 0], "E": [0, 0]},
                     "2": {"D": [0, 0], "E": [0, 0]}, ">2": {"D": [0, 0], "E": [0, 0]}}
    tau_margins_e: list[tuple[float, bool]] = []
    dropped = 0

    lm_iter = raw_beam_lm if ngram_available else [None] * len(words)
    for (true_word, rg), (_, rb), lm_entry in zip(raw_greedy_nolm, raw_beam_nolm, lm_iter):
        if not rg or not rb or (ngram_available and (lm_entry is None or not lm_entry[1])):
            dropped += 1
            continue

        pred_A = rg[0].raw
        pred_B = rb[0].raw
        pred_C = rg[0].word
        scored_D = WordDecoder.score_raw_candidates(rb, weights_no_lm)
        pred_D = scored_D["prediction"]
        is_correct_D = pred_D.lower() == true_word.lower()

        for cfg, pred in (("A", pred_A), ("B", pred_B), ("C", pred_C)):
            results[cfg]["n_words"] += 1
            results[cfg]["n_correct"] += int(pred.lower() == true_word.lower())
        results["D"]["n_words"] += 1
        results["D"]["n_correct"] += int(is_correct_D)

        topk_results["B"][1] += int(pred_B.lower() == true_word.lower())
        # "Top-3/5" for a config with no re-ranking (B) is just whether the
        # true raw text appears among the first k unique beam entries.
        raw_words_b = []
        for c in rb:
            if c.raw.lower() not in raw_words_b:
                raw_words_b.append(c.raw.lower())
        topk_results["B"][3] += int(true_word.lower() in raw_words_b[:3])
        topk_results["B"][5] += int(true_word.lower() in raw_words_b[:5])

        top_words_d = _unique_top_words(scored_D["candidates"], 5)
        topk_results["D"][1] += int(is_correct_D)
        topk_results["D"][3] += int(true_word.lower() in top_words_d[:3])
        topk_results["D"][5] += int(true_word.lower() in top_words_d[:5])

        if not is_correct_D:
            reason = classify_error(true_word, pred_D, rb, scored_D["candidates"])
            errors_D.append({"true": true_word, "predicted": pred_D, "raw_or_beam_candidate": rb[0].raw, "reason": reason})

        best_raw_no_lm = rb[0].raw.lower()
        dist = edit_distance.levenshtein(best_raw_no_lm, true_word.lower())
        bucket = "0" if dist == 0 else ("1" if dist == 1 else ("2" if dist == 2 else ">2"))
        edit_buckets[bucket]["D"][1] += 1
        edit_buckets[bucket]["D"][0] += int(is_correct_D)

        if not ngram_available:
            continue

        _, raw_lm = lm_entry
        scored_E = WordDecoder.score_raw_candidates(raw_lm, weights)
        pred_E = scored_E["prediction"]
        is_correct_E = pred_E.lower() == true_word.lower()

        results["E"]["n_words"] += 1
        results["E"]["n_correct"] += int(is_correct_E)
        top_words_e = _unique_top_words(scored_E["candidates"], 5)
        topk_results["E"][1] += int(is_correct_E)
        topk_results["E"][3] += int(true_word.lower() in top_words_e[:3])
        topk_results["E"][5] += int(true_word.lower() in top_words_e[:5])

        edit_buckets[bucket]["E"][1] += 1
        edit_buckets[bucket]["E"][0] += int(is_correct_E)

        if is_correct_D and not is_correct_E:
            d_vs_e_matrix["regressed_by_lm"] += 1
        elif not is_correct_D and is_correct_E:
            d_vs_e_matrix["fixed_by_lm"] += 1
        elif is_correct_D and is_correct_E:
            d_vs_e_matrix["unchanged_correct"] += 1
        else:
            d_vs_e_matrix["unchanged_wrong"] += 1

        if not is_correct_E:
            reason = classify_error_e(true_word, is_correct_D, pred_E, raw_lm, scored_E["candidates"])
            errors_E.append({"true": true_word, "predicted": pred_E, "raw_or_beam_candidate": raw_lm[0].raw, "reason": reason})

        cands = scored_E["candidates"]
        margin = (cands[0]["final_score"] - cands[1]["final_score"]) if len(cands) > 1 else 1.0
        tau_margins_e.append((margin, is_correct_E))

    if dropped:
        print(f"[warn] {dropped} words produced an empty candidate list in some configuration and were excluded\n")

    for cfg in results:
        results[cfg] = {**CONFIG_DESCRIPTIONS[cfg], **_acc_ci(results[cfg]["n_correct"], results[cfg]["n_words"])}

    active_cfgs = "ABCDE" if ngram_available else "ABCD"
    improvements = {
        "B_minus_A": results["B"]["accuracy"] - results["A"]["accuracy"],
        "C_minus_A": results["C"]["accuracy"] - results["A"]["accuracy"],
        "D_minus_A": results["D"]["accuracy"] - results["A"]["accuracy"],
        "D_minus_B": results["D"]["accuracy"] - results["B"]["accuracy"],
        "D_minus_C": results["D"]["accuracy"] - results["C"]["accuracy"],
    }
    if ngram_available:
        improvements["E_minus_D"] = results["E"]["accuracy"] - results["D"]["accuracy"]

    rng = random.Random(args.seed)
    sample_errors_D = rng.sample(errors_D, args.n_errors) if len(errors_D) > args.n_errors else list(errors_D)
    sample_errors_E = rng.sample(errors_E, args.n_errors) if len(errors_E) > args.n_errors else list(errors_E)

    # =====================================================================
    # SECTIONS 2/3/8: optional sweeps (n-gram order, LM weight, beam width)
    # =====================================================================
    order_sweep_rows: list[dict] = []
    lambda_sweep_rows: list[dict] = []
    beam_width_sweep_rows: list[dict] = []

    if ngram_available and not args.no_order_sweep:
        sweep_words = words[: args.order_sweep_words]
        order_sweep_rows = run_order_sweep(
            sweep_words, all_probs, args.order_sweep_orders, weights, search_lambda_lm,
            args.beam_width, args.top_k, args.workers, EXPERIMENTS_DIR, args.save_order_models,
        )

    if ngram_available and not args.no_lambda_sweep:
        sweep_words = words[: args.lambda_sweep_words]
        lambda_sweep_rows = run_lambda_sweep(
            sweep_words, all_probs, args.lambda_sweep_candidates, ngram_model, weights,
            args.beam_width, args.top_k, args.workers,
        )

    if ngram_available and not args.no_beam_width_sweep:
        sweep_words = words[: args.beam_width_sweep_words]
        beam_width_sweep_rows = run_beam_width_sweep(
            sweep_words, all_probs, args.beam_width_sweep_widths, ngram_model,
            search_lambda_lm, weights, args.top_k, args.workers,
        )

    # =====================================================================
    # SECTION 9: latency
    # =====================================================================
    latency = run_latency_check(
        words, all_probs, ngram_model, search_lambda_lm,
        args.beam_width, args.top_k, args.latency_words,
    )

    # =====================================================================
    # SECTION 10: confidence / coverage table (config E, on TEST)
    # =====================================================================
    tau_table: list[dict] = []
    if ngram_available and tau_margins_e:
        tau_candidates = sorted(set(args.tau_candidates + [round(tau_word, 3)]))
        tau_table = run_tau_coverage_table(tau_margins_e, tau_candidates)

    # =====================================================================
    # Printed report
    # =====================================================================
    print("\n" + "=" * 60)
    print("CHARACTER N-GRAM DECODER EVALUATION")
    print("=" * 60)
    topk_denom = results["D"]["n_words"]  # A/B/C/D/E all share the same non-dropped word set
    print(_format_main_table(results, active_cfgs))
    print()
    print(_format_topk_table(topk_results, ngram_available, topk_denom))
    print()
    print(_format_improvements(improvements, ngram_available))
    print()
    print(_format_conclusions(improvements, ngram_available))

    if ngram_available:
        print("\n" + "-" * 60)
        print("N-GRAM EFFECT (config D vs. config E)")
        print("-" * 60)
        print(f"Fixed by LM:       {d_vs_e_matrix['fixed_by_lm']}")
        print(f"Regressed by LM:   {d_vs_e_matrix['regressed_by_lm']}")
        print(f"Unchanged correct: {d_vs_e_matrix['unchanged_correct']}")
        print(f"Unchanged wrong:   {d_vs_e_matrix['unchanged_wrong']}")
        net = d_vs_e_matrix["fixed_by_lm"] - d_vs_e_matrix["regressed_by_lm"]
        print(f"Net effect:        {net:+d} words ({'net positive' if net > 0 else 'net negative' if net < 0 else 'no net change'})")

    if order_sweep_rows:
        print("\n" + "-" * 60)
        print(f"N-GRAM ORDER  (on {len(words[: args.order_sweep_words])} words)")
        print("-" * 60)
        for r in order_sweep_rows:
            print(f"  order={r['order']}  accuracy={r['accuracy']:.2%}  "
                  f"(n={r['n_words']}, CI=[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}])")

    if lambda_sweep_rows:
        print("\n" + "-" * 60)
        print(f"N-GRAM WEIGHT (search_lambda_lm)  (on {len(words[: args.lambda_sweep_words])} words)")
        print("-" * 60)
        for r in lambda_sweep_rows:
            print(f"  lambda={r['search_lambda_lm']:.2f}  accuracy={r['accuracy']:.2%}  "
                  f"(n={r['n_words']}, CI=[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}])")

    if beam_width_sweep_rows:
        print("\n" + "-" * 60)
        print(f"BEAM WIDTH (with n-gram LM)  (on {len(words[: args.beam_width_sweep_words])} words)")
        print("-" * 60)
        for r in beam_width_sweep_rows:
            print(f"  beam_width={r['beam_width']:>2}  accuracy={r['accuracy']:.2%}  "
                  f"(n={r['n_words']}, CI=[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}])")

    print("\n" + "-" * 60)
    print("ACCURACY BY EDIT-DISTANCE BUCKET (raw beam candidate, no LM, vs. true word)")
    print("-" * 60)
    print(f"{'Bucket':<16}{'Dictionary (D)':>18}{'+ N-gram (E)':>18}")
    for bucket in ("0", "1", "2", ">2"):
        d_correct, d_total = edit_buckets[bucket]["D"]
        e_correct, e_total = edit_buckets[bucket]["E"]
        d_acc = f"{d_correct / d_total:.2%} (n={d_total})" if d_total else "n/a"
        e_acc = f"{e_correct / e_total:.2%} (n={e_total})" if (ngram_available and e_total) else "n/a"
        label = "Exact" if bucket == "0" else f"Edit distance {bucket}"
        print(f"{label:<16}{d_acc:>18}{e_acc:>18}")

    print("\n" + "-" * 60)
    print("ERROR CATEGORIES (config D)")
    print("-" * 60)
    print(_format_error_table(sample_errors_D))
    print(f"({len(errors_D)}/{results['D']['n_words']} config-D words wrong; "
          f"{len(sample_errors_D)} sampled above)")

    if ngram_available:
        print("\n" + "-" * 60)
        print("ERROR CATEGORIES (config E)")
        print("-" * 60)
        print(_format_error_table(sample_errors_E))
        print(f"({len(errors_E)}/{results['E']['n_words']} config-E words wrong; "
              f"{len(sample_errors_E)} sampled above)")

    if tau_table:
        print("\n" + "-" * 60)
        print("CONFIDENCE / COVERAGE (config E, TEST split)")
        print("-" * 60)
        print(f"{'tau':>8}{'coverage':>12}{'n_kept':>10}{'accuracy_among_kept':>22}")
        for r in tau_table:
            acc_str = f"{r['accuracy_among_kept']:.2%}" if r["accuracy_among_kept"] is not None else "n/a"
            marker = "  <- tuned tau_word" if abs(r["tau"] - round(tau_word, 3)) < 1e-9 else ""
            print(f"{r['tau']:>8.3f}{r['coverage']:>11.2%}{r['n_kept']:>10}{acc_str:>22}{marker}")

    print()

    # =====================================================================
    # Persist
    # =====================================================================
    report = {
        "banner": BANNER,
        "meta": {
            "split": "test",
            "seed": args.seed,
            "n_words_requested": args.n_words,
            "n_words_evaluated": results["D"]["n_words"],
            "n_words_dropped": dropped,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "ngram_available": ngram_available,
            "tuned_weights": {
                "alpha": weights.alpha, "beta": weights.beta, "gamma": weights.gamma,
                "delta": weights.delta,
            },
            "tau_word": tau_word,
            "search_lambda_lm": search_lambda_lm,
            "tuned_weights_source": str(TUNED_WEIGHTS_PATH),
            "tuned_weights_meta": tuned_meta,
            "ngram_model_path": str(NGRAM_MODEL_PATH) if ngram_available else None,
            "ngram_model_order": ngram_model.order if ngram_available else None,
        },
        "tcn_character_accuracy": {"accuracy": char_accuracy, "n_samples": int(len(y_test))},
        "results": results,
        "topk_accuracy": {
            cfg: {k: (v / topk_denom if topk_denom else 0.0) for k, v in ks.items()}
            for cfg, ks in topk_results.items()
        },
        "improvements": improvements,
        "ngram_effect_matrix": d_vs_e_matrix if ngram_available else None,
        "edit_distance_breakdown": edit_buckets,
        "order_sweep": order_sweep_rows,
        "lambda_sweep": lambda_sweep_rows,
        "beam_width_sweep": beam_width_sweep_rows,
        "latency": latency,
        "confidence_coverage_table": tau_table,
        "sample_errors_config_D": sample_errors_D,
        "n_errors_total_config_D": len(errors_D),
        "sample_errors_config_E": sample_errors_E if ngram_available else [],
        "n_errors_total_config_E": len(errors_E) if ngram_available else None,
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] {RESULTS_JSON_PATH}")

    with open(RESULTS_TXT_PATH, "w") as f:
        f.write(BANNER + "\n\n")
        f.write(f"TCN character-level accuracy (full TEST split, n={len(y_test)}): {char_accuracy:.4%}\n\n")
        f.write(_format_main_table(results, active_cfgs) + "\n\n")
        f.write(_format_topk_table(topk_results, ngram_available, topk_denom) + "\n\n")
        f.write(_format_improvements(improvements, ngram_available) + "\n\n")
        f.write(_format_conclusions(improvements, ngram_available) + "\n\n")
        f.write(f"{len(errors_D)}/{results['D']['n_words']} config-D words wrong; "
                f"{len(sample_errors_D)} sampled below\n\n")
        f.write(_format_error_table(sample_errors_D) + "\n")
        if ngram_available:
            f.write(f"\n{len(errors_E)}/{results['E']['n_words']} config-E words wrong; "
                    f"{len(sample_errors_E)} sampled below\n\n")
            f.write(_format_error_table(sample_errors_E) + "\n")
    print(f"[save] {RESULTS_TXT_PATH}")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_main_table(results: dict, active_cfgs: str) -> str:
    header = f"{'Configuration':<45}{'Beam':>7}{'Dict':>7}{'N-gram':>8}{'Accuracy':>11}{'95% CI':>20}"
    lines = [header, "-" * len(header)]
    for cfg in active_cfgs:
        d = CONFIG_DESCRIPTIONS[cfg]
        r = results[cfg]
        ci = f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]"
        lines.append(
            f"{cfg + '. ' + d['name']:<45}"
            f"{'Yes' if d['beam_search'] else 'No':>7}"
            f"{'Yes' if d['dictionary_correction'] else 'No':>7}"
            f"{'Yes' if d['ngram'] else 'No':>8}"
            f"{r['accuracy']:>10.2%}"
            f"{ci:>20}"
        )
    return "\n".join(lines)


def _format_topk_table(topk_results: dict, ngram_available: bool, denom: int) -> str:
    header = f"{'Configuration':<20}{'Top-1':>10}{'Top-3':>10}{'Top-5':>10}"
    lines = [header, "-" * len(header)]
    labels = {"B": "Beam", "D": "Beam+Dictionary", "E": "+ N-gram"}
    cfgs = ("B", "D", "E") if ngram_available else ("B", "D")
    for cfg in cfgs:
        counts = topk_results[cfg]
        row = f"{labels[cfg]:<20}"
        for k in (1, 3, 5):
            pct = counts[k] / denom if denom else 0.0
            row += f"{pct:>10.2%}"
        lines.append(row)
    return "\n".join(lines)


def _format_improvements(improvements: dict, ngram_available: bool) -> str:
    lines = ["Absolute improvement (percentage points):"]
    labels = {
        "B_minus_A": "B - A  (beam search alone)",
        "C_minus_A": "C - A  (dictionary correction alone)",
        "D_minus_A": "D - A  (combined vs. plain greedy)",
        "D_minus_B": "D - B  (dictionary's added value on top of beam search)",
        "D_minus_C": "D - C  (beam search's added value on top of dictionary)",
    }
    if ngram_available:
        labels["E_minus_D"] = "E - D  (n-gram LM's added value on top of beam+dictionary)"
    for key, label in labels.items():
        lines.append(f"  {label:<58} {improvements[key] * 100:+.2f} pp")
    return "\n".join(lines)


def _format_conclusions(improvements: dict, ngram_available: bool) -> str:
    base = (
        f"Beam search improves word accuracy by {improvements['B_minus_A'] * 100:.2f} "
        f"percentage points over greedy decoding.\n"
        f"Dictionary correction improves word accuracy by {improvements['C_minus_A'] * 100:.2f} "
        f"percentage points over greedy decoding.\n"
        f"Combined beam search + dictionary correction improves word accuracy by "
        f"{improvements['D_minus_A'] * 100:.2f} percentage points over the greedy/no-correction "
        f"baseline (vs. beam-alone: {improvements['D_minus_B'] * 100:+.2f} pp; "
        f"vs. dictionary-alone: {improvements['D_minus_C'] * 100:+.2f} pp)."
    )
    if ngram_available:
        base += (
            f"\nAdding the character n-gram LM changes word accuracy by "
            f"{improvements['E_minus_D'] * 100:+.2f} percentage points relative to "
            f"beam+dictionary alone."
        )
    return base


def _format_error_table(sample_errors: list[dict]) -> str:
    if not sample_errors:
        return "(no errors to show)"
    header = f"{'TRUE':<15}{'PREDICTED':<15}{'RAW/BEAM CANDIDATE':<22}{'REASON'}"
    lines = [header, "-" * len(header)]
    for e in sample_errors:
        lines.append(f"{e['true']:<15}{e['predicted']:<15}{e['raw_or_beam_candidate']:<22}{e['reason']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-words", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1234,
                         help="Fixed seed for the synthetic TEST word set. Default (1234) "
                              "matches tune_decoder_weights.py's confirmatory test run.")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5,
                         help="Beam-search per-position expansion width (NOT the Top-k "
                              "accuracy reported in section 4, which is fixed at 1/3/5).")
    parser.add_argument("--workers", type=int, default=1,
                         help="Process-pool workers for beam-search/dictionary/n-gram "
                              "decoding. Threads are intentionally not used (CPU-bound "
                              "pure Python, serialized by the GIL).")
    parser.add_argument("--n-errors", type=int, default=30,
                         help="How many errors to sample (reproducibly, via --seed) for "
                              "each of the config-D and config-E error tables.")

    parser.add_argument("--no-ngram", action="store_true",
                         help="Disable the n-gram LM entirely -- runs only the pre-n-gram "
                              "A-D suite, no config E, no sweeps.")

    parser.add_argument("--no-order-sweep", action="store_true")
    parser.add_argument("--order-sweep-orders", type=_int_list, default=DEFAULT_ORDER_SWEEP)
    parser.add_argument("--order-sweep-words", type=int, default=400)
    parser.add_argument("--save-order-models", action="store_true", default=True)

    parser.add_argument("--no-lambda-sweep", action="store_true")
    parser.add_argument("--lambda-sweep-candidates", type=_float_list, default=DEFAULT_LAMBDA_SWEEP)
    parser.add_argument("--lambda-sweep-words", type=int, default=400)

    parser.add_argument("--no-beam-width-sweep", action="store_true")
    parser.add_argument("--beam-width-sweep-widths", type=_int_list, default=DEFAULT_BEAM_WIDTH_SWEEP)
    parser.add_argument("--beam-width-sweep-words", type=int, default=400)

    parser.add_argument("--latency-words", type=int, default=100)
    parser.add_argument("--tau-candidates", type=_float_list,
                         default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    main(parser.parse_args())