"""Generate evaluation reports from the trained recognizer and decoder.

Run from the project root. The report requires the processed test set and
trained model artifacts. Personalization evaluation uses fresh adapters and
disjoint held-out samples for each checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config
from inference.realtime import CharacterRecognizer
from inference.beam_search import beam_search
from inference.word_decoder import WordDecoder, ScoreWeights
from language.ngram import NgramLanguageModel
from personalization.adapter import build_personalized_model
from wordfreq import top_n_list

try:
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    raise SystemExit("TensorFlow is required to run this script (same as train.py/evaluate.py).") from e


# ===========================================================================
# Output location
# ===========================================================================
OUT_DIR = config.EXPERIMENTS_DIR / "full_report"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}")


# ===========================================================================
# Loading real data / the real trained pipeline
# ===========================================================================
def load_test_set():
    data = np.load(config.TEST_NPZ_PATH, allow_pickle=True)
    X, y, meta = data["X"], data["y"], data["meta"]
    log(f"Loaded real test set: X={X.shape} y={y.shape} from {config.TEST_NPZ_PATH}")
    return X, y, meta


def load_decoder_pipeline():
    """Mirrors app/correction.py's own loading logic (tuned weights +
    n-gram model if present, untuned defaults otherwise) so this script's
    decoder behaves EXACTLY like the live server's, not a reimplementation
    that could silently drift from it."""
    weights_path = config.EXPERIMENTS_DIR / "decoder_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            cfg = json.load(f)
        weights = ScoreWeights(alpha=cfg["alpha"], beta=cfg["beta"], gamma=cfg["gamma"], delta=cfg.get("delta", 0.0))
        tau_word = float(cfg.get("tau_word", 0.6))
        search_lambda_lm = float(cfg.get("search_lambda_lm", 0.0))
        log(f"Loaded tuned decoder weights from {weights_path}: {cfg.get('alpha')=} {cfg.get('beta')=} "
            f"{cfg.get('gamma')=} {cfg.get('delta', 0.0)=} tau_word={tau_word} search_lambda_lm={search_lambda_lm}")
    else:
        weights, tau_word, search_lambda_lm = ScoreWeights(), 0.6, 0.0
        log(f"No tuned weights at {weights_path} -- using UNTUNED defaults "
            f"(run experiments/tune_decoder_weights.py for real tuned numbers).")

    ngram_model = None
    if config.NGRAM_MODEL_PATH.exists() and (weights.delta != 0.0 or search_lambda_lm != 0.0):
        ngram_model = NgramLanguageModel.load(config.NGRAM_MODEL_PATH)
        log(f"Loaded n-gram model from {config.NGRAM_MODEL_PATH} (order={ngram_model.order})")
    else:
        log("No n-gram model attached (missing file, or delta/search_lambda_lm are 0).")

    decoder = WordDecoder(beam_width=5, top_k=5, weights=weights, ngram_model=ngram_model,
                           search_lambda_lm=search_lambda_lm)
    return decoder, weights, tau_word, ngram_model, search_lambda_lm


# ===========================================================================
# Section 1-3: character-level results (confusion matrix, per-class F1,
# real confidence histogram) -- straight from recognizer.model.predict()
# over the WHOLE real test set.
# ===========================================================================
def run_character_level_sections(recognizer: CharacterRecognizer, X_test, y_test, out_dir: Path):
    log("Section 1-3: running the REAL TCN model over the full real test set...")
    probs = recognizer.model.predict(X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    labels_idx = np.arange(config.NUM_CLASSES)
    label_names = [config.index_to_label(i) for i in labels_idx]

    accuracy = float((y_pred == y_test).mean())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    macro_f1 = float(f1.mean())
    log(f"  REAL accuracy={accuracy:.4f}  REAL macro_f1={macro_f1:.4f}  (n={len(y_test)})")

    # --- 1. Confusion matrix ---------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(np.log1p(cm), cmap="viridis")
    ax.set_xticks(range(len(label_names))); ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, fontsize=6, rotation=90)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Confusion Matrix -- LIVE model run (arch={recognizer.arch}, n={len(y_test)}, "
                 f"acc={accuracy:.2%})\ncolor = log(1+count)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(out_dir / "01_confusion_matrix.png", dpi=180)
    plt.close(fig)
    np.savetxt(out_dir / "01_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(label_names), comments="")

    # --- 2. Per-class F1 distribution -------------------------------------
    per_class_rows = list(zip(label_names, precision, recall, f1, support))
    per_class_rows.sort(key=lambda r: r[3])
    with open(out_dir / "02_per_class_f1.csv", "w") as fh:
        fh.write("char,precision,recall,f1,support\n")
        for char, p, r, f, s in per_class_rows:
            fh.write(f"{char},{p:.4f},{r:.4f},{f:.4f},{int(s)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(f1, bins=15, color="#4263eb", edgecolor="white")
    axes[0].axvline(macro_f1, color="#e8830c", linestyle="--", linewidth=2, label=f"macro F1 = {macro_f1:.3f}")
    axes[0].set_xlabel("Per-class F1"); axes[0].set_ylabel("Number of classes")
    axes[0].set_title(f"Per-Class F1 Distribution -- LIVE ({recognizer.arch}, {config.NUM_CLASSES} classes)")
    axes[0].legend()

    chars_sorted = [r[0] for r in per_class_rows]
    f1_sorted = [r[3] for r in per_class_rows]
    colors = ["#e03131" if v < 0.6 else ("#e8830c" if v < 0.75 else "#2f9e44") for v in f1_sorted]
    axes[1].barh(chars_sorted, f1_sorted, color=colors)
    axes[1].set_xlabel("F1"); axes[1].set_title("Per-Class F1, worst -> best")
    axes[1].tick_params(axis="y", labelsize=6)
    axes[1].axvline(macro_f1, color="black", linestyle=":", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_dir / "02_per_class_f1.png", dpi=180)
    plt.close(fig)

    # --- 3. REAL confidence distribution (per-prediction, not a proxy) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=40, color="#2f9e44", edgecolor="white")
    ax.axvline(float(np.median(confidences)), color="#e03131", linestyle="--",
               label=f"median = {np.median(confidences):.3f}")
    ax.set_xlabel("Prediction confidence (max softmax probability)")
    ax.set_ylabel("Number of test predictions")
    ax.set_title(f"Confidence Distribution -- LIVE, every real test prediction (n={len(confidences)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "03_confidence_distribution.png", dpi=180)
    plt.close(fig)
    np.savetxt(out_dir / "03_confidence_values.csv", confidences, fmt="%.6f",
               header="confidence", comments="")

    summary = {
        "architecture": recognizer.arch, "n_test_samples": int(len(y_test)),
        "accuracy": accuracy, "macro_f1": macro_f1,
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "confidence_mean": float(confidences.mean()), "confidence_median": float(np.median(confidences)),
        "confidence_p10": float(np.percentile(confidences, 10)),
    }
    log(f"  Section 1-3 done -> {out_dir}")
    return summary, probs, y_pred, confidences


# ===========================================================================
# Section 4: decoder ablation (A-E), built LIVE from real held-out test
# characters -- greedy vs beam vs dictionary vs n-gram LM.
# ===========================================================================
def build_synthetic_words(y_test: np.ndarray, n_words: int, seed: int, vocab_size: int = 20_000):
    """Builds n_words real synthetic words (ActionPlan.md Sec.4.3-style:
    concatenating independently recorded REAL isolated-character test
    samples -- not real continuous handwriting, but real recorded
    strokes). For each letter needed, picks a REAL test-set index of
    that class uniformly at random. Returns
    [(word, [row_index, ...]), ...] where row_index indexes into y_test
    (and the caller's matching probability array)."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx, cls in enumerate(y_test):
        by_class[int(cls)].append(idx)

    candidate_words = [w for w in top_n_list("en", vocab_size)
                       if w.isascii() and w.isalpha() and 3 <= len(w) <= 10]
    rng.shuffle(candidate_words)

    words = []
    for w in candidate_words:
        if len(words) >= n_words:
            break
        try:
            row_indices = [rng.choice(by_class[config.label_to_index(c)]) for c in w.lower()]
        except (KeyError, IndexError):
            continue  # a needed class has zero real test samples -- skip this word
        words.append((w.lower(), row_indices))
    return words


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_decoder_ablation(probs_full: np.ndarray, y_test: np.ndarray, meta: np.ndarray,
                          decoder: WordDecoder, n_words: int, seed: int, out_dir: Path):
    log(f"Section 4: building {n_words} REAL synthetic words from held-out test characters "
        f"and decoding them live (greedy / beam / dictionary / n-gram)...")
    words = build_synthetic_words(y_test, n_words, seed)
    log(f"  built {len(words)} usable words")

    weights_no_lm = ScoreWeights(alpha=decoder.weights.alpha, beta=decoder.weights.beta,
                                  gamma=decoder.weights.gamma, delta=0.0)

    results = {k: {"n_correct": 0, "n_words": 0} for k in "ABCD"}
    if decoder.ngram_model is not None:
        results["E"] = {"n_correct": 0, "n_words": 0}

    for true_word, row_indices in words:
        char_probs = [probs_full[i].tolist() for i in row_indices]

        greedy = beam_search(char_probs, beam_width=1, top_k=5)
        beam = beam_search(char_probs, beam_width=5, top_k=5)

        pred_A = greedy[0]["text"]
        pred_B = beam[0]["text"]

        raw_no_lm = WordDecoder(beam_width=1, top_k=5).decode_raw(char_probs)
        pred_C = WordDecoder.score_raw_candidates(raw_no_lm, weights_no_lm)["prediction"] if raw_no_lm else pred_A

        raw_beam_no_lm = WordDecoder(beam_width=5, top_k=5).decode_raw(char_probs)
        pred_D = WordDecoder.score_raw_candidates(raw_beam_no_lm, weights_no_lm)["prediction"] if raw_beam_no_lm else pred_B

        for cfg_key, pred in (("A", pred_A), ("B", pred_B), ("C", pred_C), ("D", pred_D)):
            results[cfg_key]["n_words"] += 1
            results[cfg_key]["n_correct"] += int(pred.lower() == true_word.lower())

        if decoder.ngram_model is not None:
            raw_lm = decoder.decode_raw(char_probs)
            pred_E = WordDecoder.score_raw_candidates(raw_lm, decoder.weights)["prediction"] if raw_lm else pred_D
            results["E"]["n_words"] += 1
            results["E"]["n_correct"] += int(pred_E.lower() == true_word.lower())

    names = {
        "A": "Greedy, no dictionary", "B": "Beam search only",
        "C": "Dictionary correction only (greedy)", "D": "Beam search + dictionary",
        "E": "Beam search + dictionary + n-gram LM",
    }
    for k, r in results.items():
        r["accuracy"] = r["n_correct"] / r["n_words"] if r["n_words"] else 0.0
        r["ci_95"] = wilson_ci(r["n_correct"], r["n_words"])
        r["name"] = names[k]
        log(f"  [{k}] {names[k]}: {r['accuracy']:.2%} (n={r['n_words']})")

    with open(out_dir / "04_decoder_ablation.json", "w") as f:
        json.dump({"n_words_requested": n_words, "n_words_built": len(words), "results": results}, f, indent=2)

    cfg_order = [k for k in "ABCDE" if k in results]
    acc = [results[k]["accuracy"] * 100 for k in cfg_order]
    err_lo = [acc[i] - results[k]["ci_95"][0] * 100 for i, k in enumerate(cfg_order)]
    err_hi = [results[k]["ci_95"][1] * 100 - acc[i] for i, k in enumerate(cfg_order)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = {"A": "#adb5bd", "B": "#adb5bd", "C": "#4263eb", "D": "#4263eb", "E": "#2f9e44"}
    bars = ax.bar(cfg_order, acc, yerr=[err_lo, err_hi], capsize=5, color=[bar_colors[k] for k in cfg_order])
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Word accuracy (%)")
    ax.set_title(f"Decoder Ablation -- LIVE run on {len(words)} real synthetic words")
    fig.tight_layout()
    fig.savefig(out_dir / "04_decoder_ablation.png", dpi=180)
    plt.close(fig)
    log(f"  Section 4 done -> {out_dir}")
    return results


# ===========================================================================
# Sections 5-6: personalization learning curves, built LIVE per real
# held-out participant.
# ===========================================================================
CHECKPOINTS = [0, 10, 25, 50, 100, 200]


def _parse_participant(meta_entry) -> str:
    # meta format written by data/build_dataset.py: "PARTICIPANT:CHAR:FILENAME"
    return str(meta_entry).split(":")[0]


def group_indices_by_participant(meta: np.ndarray) -> dict:
    groups = defaultdict(list)
    for idx, m in enumerate(meta):
        groups[_parse_participant(m)].append(idx)
    return groups


def fit_fresh_adapter(recognizer: CharacterRecognizer, X_train, y_train, epochs: int = 8, lr: float = 1e-3):
    """Builds a BRAND NEW SessionAdapter (identity at init -- see
    personalization/adapter.py) and fits ONLY it (encoder/classifier
    stay frozen) on X_train/y_train. A fresh adapter per checkpoint
    gives a clean, comparable learning curve; it does not simulate
    continual online use (that's what personalization/trainer.py's
    adapt_session is for, used by the live server)."""
    model, adapter = build_personalized_model(recognizer.encoder, recognizer.classifier)
    if len(y_train) > 0:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=min(32, max(1, len(y_train))))
    return model


def run_personalization_sections(recognizer: CharacterRecognizer, X_test, y_test, meta,
                                  decoder: WordDecoder, n_participants: int, n_words_per_checkpoint: int,
                                  seed: int, out_dir: Path):
    log("Section 5-6: LIVE personalization learning curves over real held-out participants "
        "(ActionPlan.md Sec.13.12) -- this is the slow part, be patient...")
    rng = random.Random(seed)
    groups = group_indices_by_participant(meta)

    min_needed = max(CHECKPOINTS) + 30  # leave a real held-out eval slice after the largest checkpoint
    eligible = [pid for pid, idxs in groups.items() if len(idxs) >= min_needed]
    if not eligible:
        log(f"  WARNING: no participant in this test split has >= {min_needed} samples "
            f"(needed for the largest checkpoint {max(CHECKPOINTS)} + a held-out eval slice). "
            f"Falling back to the smaller checkpoints only that DO fit the largest available participant.")
        eligible = sorted(groups, key=lambda p: -len(groups[p]))[:n_participants]
    else:
        rng.shuffle(eligible)
        eligible = eligible[:n_participants]

    log(f"  using {len(eligible)} real held-out participants: {eligible}")

    per_participant_char_acc = defaultdict(dict)   # participant -> checkpoint -> accuracy
    per_participant_word_acc = defaultdict(dict)

    for pid in eligible:
        idxs = groups[pid][:]
        rng.shuffle(idxs)

        usable_checkpoints = [c for c in CHECKPOINTS if c + 30 <= len(idxs)] or [0]
        max_train = max(usable_checkpoints)
        train_pool = idxs[:max_train]
        eval_pool = idxs[max_train: max_train + 30] if max_train + 30 <= len(idxs) else idxs[max_train:]
        if len(eval_pool) < 5:
            log(f"  [{pid}] skipped -- not enough samples left for a real held-out eval slice")
            continue

        X_eval, y_eval = X_test[eval_pool], y_test[eval_pool]
        log(f"  [{pid}] {len(idxs)} total real samples -> checkpoints={usable_checkpoints}, "
            f"eval_pool={len(eval_pool)} (never used for training)")

        for checkpoint in usable_checkpoints:
            train_idxs = train_pool[:checkpoint]
            X_train, y_train = X_test[train_idxs], y_test[train_idxs]

            model = fit_fresh_adapter(recognizer, X_train, y_train)

            # --- character-level accuracy on the REAL held-out eval slice
            eval_probs = model.predict(X_eval, verbose=0)
            char_acc = float((np.argmax(eval_probs, axis=1) == y_eval).mean())
            per_participant_char_acc[pid][checkpoint] = char_acc

            # --- sentence-level (word) correction rate, decoded from the
            # SAME personalized probabilities, on words built ONLY from
            # this participant's held-out (never-trained-on) characters
            words = build_synthetic_words(y_eval, n_words_per_checkpoint, seed + checkpoint)
            if words:
                n_correct = 0
                for true_word, local_row_indices in words:
                    # local_row_indices index into y_eval/eval_probs (0..len(eval_pool)-1)
                    char_prob_seq = [eval_probs[i].tolist() for i in local_row_indices]
                    raw = decoder.decode_raw(char_prob_seq)
                    pred_word = WordDecoder.score_raw_candidates(raw, decoder.weights)["prediction"] if raw else ""
                    n_correct += int(pred_word.lower() == true_word.lower())
                word_acc = n_correct / len(words)
            else:
                word_acc = float("nan")
            per_participant_word_acc[pid][checkpoint] = word_acc

            log(f"    [{pid}] checkpoint={checkpoint:>3} samples -> "
                f"char_acc={char_acc:.4f}  word_acc={word_acc:.4f}")

    # --- aggregate across participants -------------------------------------
    def aggregate(per_participant: dict) -> dict:
        by_checkpoint = defaultdict(list)
        for pid, ck_map in per_participant.items():
            for ck, v in ck_map.items():
                if v == v:  # not NaN
                    by_checkpoint[ck].append(v)
        return {ck: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n": len(vs)}
                for ck, vs in sorted(by_checkpoint.items())}

    char_agg = aggregate(per_participant_char_acc)
    word_agg = aggregate(per_participant_word_acc)

    with open(out_dir / "05_personalization_char_accuracy.json", "w") as f:
        json.dump({"per_participant": per_participant_char_acc, "aggregate": char_agg,
                    "participants": eligible}, f, indent=2)
    with open(out_dir / "06_personalization_word_accuracy.json", "w") as f:
        json.dump({"per_participant": per_participant_word_acc, "aggregate": word_agg,
                    "participants": eligible}, f, indent=2)

    # --- chart 5: character accuracy vs personalization samples ------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pid, ck_map in per_participant_char_acc.items():
        cks = sorted(ck_map)
        ax.plot(cks, [ck_map[c] for c in cks], color="#adb5bd", linewidth=1, alpha=0.6, marker="o", markersize=3)
    cks = sorted(char_agg)
    means = [char_agg[c]["mean"] for c in cks]
    stds = [char_agg[c]["std"] for c in cks]
    ax.errorbar(cks, means, yerr=stds, color="#4263eb", linewidth=2.5, marker="o", markersize=6,
                capsize=4, label=f"mean across {len(eligible)} real held-out participants")
    ax.set_xlabel("Number of personalization samples / confirmed corrections")
    ax.set_ylabel("Character recognition accuracy")
    ax.set_title("Progressive Personalization -- LIVE Character Accuracy vs. Adaptation Samples\n"
                  "(fresh adapter per checkpoint, held-out eval slice never used for training)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_personalization_char_accuracy.png", dpi=180)
    plt.close(fig)

    # --- chart 6: sentence-level correction rate vs adaptation progress ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for pid, ck_map in per_participant_word_acc.items():
        cks = sorted(k for k in ck_map if ck_map[k] == ck_map[k])
        if cks:
            ax.plot(cks, [ck_map[c] for c in cks], color="#adb5bd", linewidth=1, alpha=0.6, marker="o", markersize=3)
    cks = sorted(word_agg)
    means = [word_agg[c]["mean"] for c in cks]
    stds = [word_agg[c]["std"] for c in cks]
    ax.errorbar(cks, means, yerr=stds, color="#2f9e44", linewidth=2.5, marker="o", markersize=6,
                capsize=4, label=f"mean across {len(eligible)} real held-out participants")
    ax.set_xlabel("Adaptation progress (personalization samples)")
    ax.set_ylabel("Sentence-level (word) correction rate")
    ax.set_title(f"Sentence-Level Correction Rate vs. Adaptation Progress -- LIVE\n"
                 f"(decoded words built only from held-out characters, ~{n_words_per_checkpoint} words/checkpoint)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_personalization_word_accuracy.png", dpi=180)
    plt.close(fig)

    log(f"  Section 5-6 done -> {out_dir}")
    return char_agg, word_agg


# ===========================================================================
# Main
# ===========================================================================
def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {out_dir}")

    log(f"Loading REAL trained recognizer (arch={args.arch})...")
    recognizer = CharacterRecognizer(arch=args.arch)

    X_test, y_test, meta = load_test_set()
    decoder, weights, tau_word, ngram_model, search_lambda_lm = load_decoder_pipeline()

    summary, probs_full, y_pred, confidences = run_character_level_sections(recognizer, X_test, y_test, out_dir)

    if not args.skip_decoder_ablation:
        ablation_results = run_decoder_ablation(
            probs_full, y_test, meta, decoder, n_words=args.n_words, seed=args.seed, out_dir=out_dir,
        )
        summary["decoder_ablation"] = {k: v["accuracy"] for k, v in ablation_results.items()}

    if not args.skip_personalization:
        char_agg, word_agg = run_personalization_sections(
            recognizer, X_test, y_test, meta, decoder,
            n_participants=args.n_participants, n_words_per_checkpoint=args.n_words_personalization,
            seed=args.seed, out_dir=out_dir,
        )
        summary["personalization_char_accuracy_by_checkpoint"] = char_agg
        summary["personalization_word_accuracy_by_checkpoint"] = word_agg
    else:
        log("Skipping personalization sections (--skip-personalization).")

    with open(out_dir / "00_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"\nALL DONE. Every chart/CSV/JSON above was computed from a live run of your "
        f"real model in this process -- rerun this script after retraining or collecting "
        f"more data and every number will change accordingly.\nOutputs: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", type=str, default="tcn", choices=config.ARCHITECTURES)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--n-words", type=int, default=500,
                         help="Synthetic words for the decoder ablation (section 4).")
    parser.add_argument("--skip-decoder-ablation", action="store_true")

    parser.add_argument("--n-participants", type=int, default=6,
                         help="How many real held-out test participants to run the "
                              "personalization learning curve over (sections 5-6). "
                              "More participants = a smoother/more trustworthy curve "
                              "but longer runtime (one adapter fit per checkpoint per participant).")
    parser.add_argument("--n-words-personalization", type=int, default=60,
                         help="Synthetic words decoded per checkpoint per participant for "
                              "the sentence-level correction-rate curve (section 6).")
    parser.add_argument("--skip-personalization", action="store_true",
                         help="Skip sections 5-6 (the slow part) -- useful for a quick "
                              "character-level-only run.")

    main(parser.parse_args())