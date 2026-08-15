"""
Standalone + integrated test for beam search + wordfreq correction.
Run: python test_beam_dictionary.py
"""
from __future__ import annotations

import math

from config import label_to_index, NUM_CLASSES
from inference.beam_search import beam_search
from inference.word_decoder import ScoreWeights, WordDecoder


def make_probs(char: str, confidence: float, runner_up: str | None = None,
                runner_up_p: float = 0.0) -> list[float]:
    """One 52-class probability row: mostly `char`, optional runner-up
    carrying probability mass -- lets us build controlled test
    sequences without touching the real model."""
    probs = [0.0] * NUM_CLASSES
    probs[label_to_index(char)] = confidence
    if runner_up:
        probs[label_to_index(runner_up)] = runner_up_p
    remaining = max(0.0, 1.0 - confidence - runner_up_p)
    fill_slots = NUM_CLASSES - (2 if runner_up else 1)
    fill = remaining / fill_slots if fill_slots else 0.0
    for i in range(NUM_CLASSES):
        if probs[i] == 0.0 and i != label_to_index(char) and (
            not runner_up or i != label_to_index(runner_up)
        ):
            probs[i] = fill
    return probs


def header(title: str) -> None:
    print("=" * 60); print(title); print("=" * 60)


def test_greedy_vs_beam_disagreement():
    header("TEST: beam search finds a better word than greedy")
    # Target: "hello". Position 3 ('l') is deliberately ambiguous
    # (model slightly prefers 'i') so greedy alone gives "heilo" --
    # not a word -- while a wider beam recovers "hello".
    sequence = [
        make_probs("h", 0.9),
        make_probs("e", 0.9),
        make_probs("i", 0.55, runner_up="l", runner_up_p=0.45),
        make_probs("l", 0.9),
        make_probs("o", 0.9),
    ]

    greedy = beam_search(sequence, beam_width=1, top_k=5)
    print(f"Greedy (beam_width=1): {greedy[0]['text']!r} (p={greedy[0]['probability']:.4f})")

    for width in (1, 3, 5, 10):
        results = beam_search(sequence, beam_width=width, top_k=5)
        print(f"\nBeam width={width}")
        for i, r in enumerate(results, start=1):
            print(f"  {i}. {r['text']!r:10s} logP={r['log_probability']:.3f} p={r['probability']:.4f}")

    decoder = WordDecoder(beam_width=10, top_k=5)
    result = decoder.decode(sequence)
    header("DICTIONARY / WORDFREQ CORRECTION")
    for i, c in enumerate(result["candidates"][:5], start=1):
        print(f"  {i}. word={c['word']!r:10s} raw={c['raw']!r:10s} "
              f"beam={c['beam_score']:.3f} edit_sim={c['edit_similarity']:.3f} "
              f"freq={c['word_frequency']:.3f} final={c['final_score']:.3f} known={c['is_known_word']}")

    header("FINAL RESULT")
    print(f"Prediction: {result['prediction']!r}")
    assert result["prediction"] == "hello", f"expected 'hello', got {result['prediction']!r}"
    print("PASS: beam search + correction recovered the intended word.\n")


def test_exact_dictionary_match():
    header("TEST: exact dictionary match (no correction needed)")
    sequence = [make_probs(c, 0.95) for c in "world"]
    result = WordDecoder(beam_width=5, top_k=5).decode(sequence)
    print(f"Prediction: {result['prediction']!r}")
    assert result["prediction"] == "world"
    assert result["candidates"][0]["is_known_word"] is True
    print("PASS\n")


def test_misspelled_candidate_gets_corrected():
    header("TEST: misspelled candidate corrected via edit distance")
    sequence = [make_probs(c, 0.9) for c in "helo"]
    result = WordDecoder(beam_width=5, top_k=5).decode(sequence)
    top = result["candidates"][0]
    print(f"raw={top['raw']!r} -> corrected={top['word']!r} edit_sim={top['edit_similarity']:.3f}")
    assert top["raw"] == "helo"
    assert top["word"] != "helo"
    print("PASS\n")


def test_zero_probabilities_are_safe():
    header("TEST: zero probabilities don't crash log()")
    row = [0.0] * NUM_CLASSES
    row[label_to_index("z")] = 1.0
    results = beam_search([row, row, row], beam_width=3, top_k=5)
    print(f"Top result: {results[0]}")
    assert results[0]["text"] == "zzz"
    print("PASS\n")


def test_beam_width_equivalences():
    header("TEST: beam_width=1 matches per-position argmax (greedy)")
    sequence = [make_probs(c, 0.7, runner_up="x", runner_up_p=0.2) for c in "cat"]
    greedy = beam_search(sequence, beam_width=1, top_k=5)[0]
    assert greedy["text"] == "cat"
    print(f"greedy result: {greedy['text']!r} -- PASS\n")


def test_top_k_variants():
    header("TEST: top_k=3 vs top_k=5 character expansion")
    sequence = [make_probs(c, 0.5, runner_up="q", runner_up_p=0.3) for c in "dog"]
    for k in (3, 5):
        results = beam_search(sequence, beam_width=5, top_k=k)
        print(f"top_k={k}: {[r['text'] for r in results]}")
    print("PASS (no crash, both widths return valid candidate sets)\n")


def test_ngram_lm_integration():
    header("TEST: n-gram LM wiring (delta re-ranking + search_lambda_lm steering)")
    try:
        from config import NGRAM_MODEL_PATH
        from language.ngram import NgramLanguageModel
    except Exception as e:  # noqa: BLE001
        print(f"[skip] missing dependency: {e}\n")
        return

    if NGRAM_MODEL_PATH.exists():
        model = NgramLanguageModel.load(NGRAM_MODEL_PATH)
        print(f"loaded {NGRAM_MODEL_PATH} (order={model.order}, "
              f"contexts={len(model.totals)})")
    else:
        print(f"[info] {NGRAM_MODEL_PATH} not found -- training a tiny throwaway model "
              f"for THIS TEST ONLY (run `python -m experiments.build_ngram_model` once "
              f"for the real one; that's what tune_decoder_weights.py will use).")
        model = NgramLanguageModel.train(order=3, vocab_size=2000)

    # Reuse the deliberately-misspelled sequence from
    # test_misspelled_candidate_gets_corrected() so beam search has more
    # than one distinct corrected candidate to work with.
    sequence = [make_probs(c, 0.9) for c in "helo"]

    # 1) Attaching ngram_model with search_lambda_lm=0.0 (the default)
    # must reproduce the EXACT no-LM beam search -- word_decoder.py's
    # docstring promises this so that decode_raw() callers can add an
    # LM purely for the `delta` re-ranking term without opting into
    # search-time steering. This is the one guarantee worth hard-
    # asserting; everything else below is data-dependent.
    plain = WordDecoder(beam_width=5, top_k=5).decode_raw(sequence)
    with_lm = WordDecoder(beam_width=5, top_k=5, ngram_model=model,
                           search_lambda_lm=0.0).decode_raw(sequence)
    plain_raws = [c.raw for c in plain]
    with_lm_raws = [c.raw for c in with_lm]
    assert plain_raws == with_lm_raws, (
        f"attaching ngram_model with search_lambda_lm=0.0 changed the beam search "
        f"(it shouldn't): {plain_raws} vs {with_lm_raws}"
    )
    print("PASS: ngram_model attached + search_lambda_lm=0.0 -> identical beam search "
          "to the no-LM case")

    # 2) lm_score is populated and normalized to [0, 1] once ngram_model
    # is attached (RawCandidate.lm_score defaults to 0.0 without one).
    for c in with_lm:
        assert 0.0 <= c.lm_score <= 1.0, f"lm_score out of [0,1]: {c}"
    lm_by_word = {c.word: round(c.lm_score, 3) for c in with_lm}
    print(f"lm_score by corrected word: {lm_by_word}")

    # 3) delta is a cheap re-weighting of the SAME cached raw candidates
    # (no re-decode) -- score_raw_candidates() must accept delta>0 and
    # return a well-formed result.
    scored_delta0 = WordDecoder.score_raw_candidates(
        with_lm, ScoreWeights(alpha=0.6, beta=0.25, gamma=0.15, delta=0.0))
    scored_delta_hi = WordDecoder.score_raw_candidates(
        with_lm, ScoreWeights(alpha=0.3, beta=0.2, gamma=0.1, delta=0.4))
    print(f"delta=0.0  -> prediction={scored_delta0['prediction']!r} "
          f"top_final_score={scored_delta0['candidates'][0]['final_score']:.3f}")
    print(f"delta=0.4  -> prediction={scored_delta_hi['prediction']!r} "
          f"top_final_score={scored_delta_hi['candidates'][0]['final_score']:.3f}")
    assert scored_delta0["prediction"] and scored_delta_hi["prediction"]

    # 4) search_lambda_lm > 0 actually engages the LM during search --
    # just confirm it runs cleanly and returns a valid, non-empty beam
    # (the exact hypotheses it favors are data/model-dependent, so this
    # isn't asserted beyond "didn't crash, still returns candidates").
    steered = WordDecoder(beam_width=5, top_k=5, ngram_model=model,
                           search_lambda_lm=0.3).decode_raw(sequence)
    assert len(steered) > 0
    print(f"search_lambda_lm=0.3 raw candidates: {[c.raw for c in steered]}")

    print("PASS: n-gram LM wiring (delta + search_lambda_lm) behaves as documented\n")


def test_real_model_integration():
    header("TEST: real TCN -> beam search -> wordfreq correction (SYNTHETIC words)")
    print("NOTE: words below are built by concatenating ISOLATED single-character")
    print("test samples -- this is NOT real continuous air-writing (ActionPlan.md Sec.4.3).")
    print("Treat this as a pipeline sanity check, not a real accuracy number.\n")
    try:
        import random
        import numpy as np
        import tensorflow as tf
        from wordfreq import top_n_list
        from config import TEST_NPZ_PATH, label_to_index, model_path
    except Exception as e:  # noqa: BLE001
        print(f"[skip] missing dependency: {e}\n")
        return

    try:
        data = np.load(TEST_NPZ_PATH, allow_pickle=True)
        X_test, y_test = data["X"], data["y"]
        model = tf.keras.models.load_model(model_path("tcn"))
    except Exception as e:  # noqa: BLE001
        print(f"[skip] could not load test data/model: {e}\n")
        return

    rng = random.Random(7)
    by_class = {}
    for idx, label in enumerate(y_test):
        by_class.setdefault(int(label), []).append(idx)

    vocab = [w for w in top_n_list("en", 20_000) if w.isalpha() and 3 <= len(w) <= 7]
    rng.shuffle(vocab)

    words = []
    for word in vocab:
        if len(words) >= 20:
            break
        try:
            class_idx = [label_to_index(c) for c in word]
        except ValueError:
            continue
        if any(ci not in by_class for ci in class_idx):
            continue
        rows = [rng.choice(by_class[ci]) for ci in class_idx]
        words.append((word, rows))

    if not words:
        print("[skip] could not build any synthetic words from this test split\n")
        return

    all_probs = model.predict(X_test, batch_size=256, verbose=0)
    decoder = WordDecoder(beam_width=5, top_k=5)

    correct = 0
    for true_word, row_indices in words:
        seq = [all_probs[i].tolist() for i in row_indices]
        result = decoder.decode(seq)
        predicted = result["prediction"]
        ok = predicted.lower() == true_word.lower()
        correct += int(ok)
        print(f"  true={true_word!r:10s} predicted={predicted!r:10s} {'OK' if ok else 'MISS'}")

    print(f"\nSynthetic word accuracy: {correct}/{len(words)} = {correct/len(words):.2%}")
    print("(Again: synthetic concatenation, not real continuous writing -- see ActionPlan.md 4.3.)\n")

if __name__ == "__main__":
    test_exact_dictionary_match()
    test_misspelled_candidate_gets_corrected()
    test_zero_probabilities_are_safe()
    test_beam_width_equivalences()
    test_top_k_variants()
    test_greedy_vs_beam_disagreement()
    test_ngram_lm_integration()
    test_real_model_integration()
    header("ALL TESTS COMPLETE")