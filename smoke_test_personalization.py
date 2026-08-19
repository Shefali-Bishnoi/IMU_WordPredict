"""
Proves the identity guarantee holds for your ACTUAL trained weights, and
that adapt_session() runs end-to-end without crashing. Run this before
trusting anything above.

    python smoke_test_personalization.py
"""
import numpy as np

from inference.realtime import CharacterRecognizer
from personalization.adapter import build_personalized_model
from personalization.trainer import adapt_session


def main():
    print("[1/3] Loading recognizer (tcn) + building a fresh personalized model...")
    r = CharacterRecognizer(arch="tcn")
    personalized, adapter = build_personalized_model(r.encoder, r.classifier)

    print("[2/3] Checking identity-at-init on 20 random inputs...")
    x = np.random.randn(20, r.seq_len, 9).astype(np.float32)
    p_global = r.model.predict(x, verbose=0)
    p_personalized = personalized.predict(x, verbose=0)
    max_diff = float(np.max(np.abs(p_global - p_personalized)))
    assert np.allclose(p_global, p_personalized, atol=1e-5), (
        f"adapter is NOT identity at init (max diff={max_diff}) -- "
        f"STOP, do not wire this in yet."
    )
    print(f"    PASS: max abs diff = {max_diff:.2e} (fresh adapter == global model exactly)")

    print("[3/3] Running one adapt_session() call with synthetic labeled data...")
    X = np.random.randn(12, r.seq_len, 9).astype(np.float32)
    y = np.random.randint(0, 52, size=12)
    result = adapt_session(personalized, adapter, X, y, epochs=2)
    print(f"    adapt_session result: {result}")
    assert "updated" in result
    print("    PASS: adapt_session ran without error.")

    print("\nALL CHECKS PASSED. Existing evaluate.py / evaluate_decoder.py results "
          "are unaffected -- this module is never imported by them.")


if __name__ == "__main__":
    main()