"""Wrapper around a pretrained causal HuggingFace language model."""
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
    """Holds one loaded tokenizer + model. Construct via .load()."""

    model_name: str
    tokenizer: object
    model: object
    max_context_tokens: int = 256

    @classmethod
    def load(cls, model_name: str, max_context_tokens: int = 256) -> Optional["CausalLanguageModel"]:
        """Return a loaded model, or None if loading failed."""
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

    def sequence_logprob(self, text: str) -> float:
        """Sum of causal log-probabilities over the token sequence."""
        ids = self.tokenizer.encode(text, return_tensors="pt")
        if ids.shape[1] < 2:
            return 0.0
        ids = ids[:, -self.max_context_tokens:]
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        n_scored_tokens = ids.shape[1] - 1
        return float(-out.loss.item() * n_scored_tokens)

    def score_next_word(self, context: str, candidate_word: str) -> float:
        """log P(candidate_word | context)."""
        context = context.strip()
        candidate_word = candidate_word.strip()
        if not candidate_word:
            return -math.inf

        full_with_candidate = f"{context} {candidate_word}".strip()
        if not context:
            return self.sequence_logprob(full_with_candidate)

        logprob_with_candidate = self.sequence_logprob(full_with_candidate)
        logprob_context_only = self.sequence_logprob(context)
        return logprob_with_candidate - logprob_context_only
