"""
Train the character n-gram language model (ActionPlan.md Priority 4,
Step 19: "Implement character n-gram language scoring (trained on
external text, not the IMU dataset)") and save it to
experiments/ngram_model.json.

Trains on wordfreq's English word list -- entirely external text,
decoupled from the IMU dataset per ActionPlan.md Sec.4.3/12.2. No IMU
data, model files, or GPU needed for this step; it runs in seconds.

Usage:
    python -m experiments.build_ngram_model
    python -m experiments.build_ngram_model --order 4 --vocab-size 100000
"""
from __future__ import annotations

import argparse

from config import EXPERIMENTS_DIR
from language.ngram import DEFAULT_MODEL_VOCAB_SIZE, DEFAULT_ORDER, NgramLanguageModel

# Add this to config.py alongside NORM_STATS_PATH / decoder weights path:
#     NGRAM_MODEL_PATH = EXPERIMENTS_DIR / "ngram_model.json"
# Using EXPERIMENTS_DIR directly here so this script works even before
# that one-line config.py addition is made.
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

    # Sanity check: a plausible English continuation should outscore an
    # implausible one -- if this fails, don't proceed to wiring the model
    # into the decoder.
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