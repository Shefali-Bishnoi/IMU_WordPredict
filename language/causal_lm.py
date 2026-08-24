"""
language/causal_lm.py

Thin wrapper around a PRETRAINED, CAUSAL (left-to-right) HuggingFace
language model. This is the ONLY module in the codebase that imports
`transformers`/`torch` directly -- everything else (contextual_scorer.py,
app/main.py) talks to this module's clean interface only, per the
"don't expose HF internals throughout the application" requirement.

Why causal, not masked (BERT-style):
    At inference time we only ever have text WRITTEN SO FAR:
        "I am going ___"
    We never have future words to fill a [MASK] with -- the whole point
    is scoring a candidate NEXT word given only the past. A causal LM
    (GPT-2 family) models exactly P(token | previous tokens), which is
    the right factorization for this. A masked LM would require
    already knowing what comes after the blank, which we don't.

Why pretrained, not trained from the IMU dataset:
    The IMU dataset (ActionPlan.md Sec.4.3) contains only ISOLATED
    CHARACTER samples -- there is no sentence/word-level signal to
    train a language model from. This module never touches the IMU
    dataset at all; it loads ordinary pretrained weights from the HF
    Hub (or a local cache) once at process startup.

Loaded ONCE at application startup (see app/main.py's startup event),
never per-request -- mirrors exactly how CharacterRecognizer is loaded
once in inference/realtime.py.

Graceful degradation: if transformers/torch aren't installed, or the
model fails to download/load, `CausalLanguageModel.load()` returns None
and logs why, instead of raising and taking down the whole server. Every
caller (contextual_scorer.py) already treats "no model" as "skip
contextual correction, fall back to the existing pipeline" -- this is
required by config.LANGUAGE_MODEL_ENABLED being independently toggleable
too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

_HF_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # noqa: BLE001 - genuinely want to catch anything here
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _HF_IMPORT_ERROR = str(e)


@dataclass
class CausalLanguageModel:
    """Holds one loaded tokenizer + model. Construct via .load(), not
    directly -- .load() is what handles the "missing dependency" /
    "download failed" cases safely."""

    model_name: str
    tokenizer: object
    model: object
    max_context_tokens: int = 256

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_name: str, max_context_tokens: int = 256) -> Optional["CausalLanguageModel"]:
        """Returns a ready CausalLanguageModel, or None (never raises) if
        the model/tokenizer could not be loaded for any reason. Callers
        MUST treat None as "language layer unavailable this run" and
        continue operating without it (Definition of Done: "Language
        model failure gracefully falls back to the existing pipeline")."""
        if _HF_IMPORT_ERROR is not None:
            print(
                f"[language] transformers/torch not available ({_HF_IMPORT_ERROR}) -- "
                f"contextual correction disabled. Run `pip install transformers torch` "
                f"to enable it."
            )
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            print(f"[language] model loaded: {model_name!r} (causal LM, eval mode)")
            return cls(
                model_name=model_name, tokenizer=tokenizer, model=model,
                max_context_tokens=max_context_tokens,
            )
        except Exception as e:  # noqa: BLE001 - loading a HF model can fail in many ways
            print(f"[language] failed to load model {model_name!r}: {e} -- "
                  f"contextual correction disabled for this run.")
            return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def sequence_logprob(self, text: str) -> float:
        """Sum of log P(token_i | token_<i) over `text`, i.e. the total
        causal log-likelihood HF assigns this exact left-to-right
        sequence. Used to score "context + candidate_word" as a whole,
        which is the standard way to get a causal LM's opinion of one
        more word given everything before it -- no [MASK], no future
        tokens ever touched."""
        ids = self.tokenizer.encode(text, return_tensors="pt")
        if ids.shape[1] < 2:
            return 0.0
        ids = ids[:, -self.max_context_tokens:]  # respect the context cap
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        # HF's `loss` is the mean per-token cross-entropy over the whole
        # sequence (teacher-forced, i.e. purely causal -- position i only
        # ever attends to positions < i, exactly the left-to-right
        # factorization this module exists to use). Convert back to a
        # summed log-prob so scores are comparable across different
        # candidate lengths in the same way beam_search.py's cumulative
        # log-probabilities already are.
        n_scored_tokens = ids.shape[1] - 1  # first token has no preceding context
        return float(-out.loss.item() * n_scored_tokens)

    def score_next_word(self, context: str, candidate_word: str) -> float:
        """log P(candidate_word | context), i.e. the causal LM's opinion
        of `candidate_word` as the very next word after `context`. This
        is exactly the "next token | previous tokens" factorization
        required -- context never includes anything after the blank."""
        context = context.strip()
        candidate_word = candidate_word.strip()
        if not candidate_word:
            return -math.inf

        full_with_candidate = f"{context} {candidate_word}".strip()
        if not context:
            # No prior context yet (first word of the session) -- score
            # the candidate word's own unconditional log-prob instead of
            # dividing by a context-only score of "".
            return self.sequence_logprob(full_with_candidate)

        logprob_with_candidate = self.sequence_logprob(full_with_candidate)
        logprob_context_only = self.sequence_logprob(context)
        # P(candidate | context) = P(context, candidate) / P(context)
        # In log space: logP(candidate|context) = logP(context,candidate) - logP(context)
        return logprob_with_candidate - logprob_context_only
