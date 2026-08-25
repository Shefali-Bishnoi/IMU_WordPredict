"""
Train the character n-gram language model and save to experiments/ngram_model.json.

Usage:
    python -m experiments.build_ngram_model
"""
from __future__ import annotations

import argparse

from config import EXPERIMENTS_DIR
from language.ngram import DEFAULT_MODEL_VOCAB_SIZE, DEFAULT_ORDER, NgramLanguageModel

# Default output path (also available as config.NGRAM_MODEL_PATH).
OUT_PATH = EXPERIMENTS_DIR / "ngram_model.json"


def main(order: int, vocab_size: int) -> None:
    print(
        f"[ngram] training order={order} char n-gram on top {vocab_size} English "
        f"words (external text via wordfreq, per ActionPlan.md 12.2)"
    )
    model = NgramLanguageModel.train(order=order, vocab_size=vocab_size)
    model.save(OUT_PATH)
    print(f"[ngram] vocab (chars) = {len(model.vocab)}, contexts learned = {len(model.totals)}")
    print(f"[save] {OUT_PATH}")

    # Sanity: logP('e'|'appl') should exceed logP('z'|'appl').
    good = model.next_char_logprob(list("appl"), "e")
    bad = model.next_char_logprob(list("appl"), "z")
    status = "OK" if good > bad else "UNEXPECTED -- check training data/order before using this model"
    print(f"[sanity] logP('e'|'appl')={good:.3f}  logP('z'|'appl')={bad:.3f}  ({status})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_MODEL_VOCAB_SIZE)
    args = parser.parse_args()
    main(args.order, args.vocab_size)