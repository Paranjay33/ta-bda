"""Model-specific tokenization, kept separate from the shared dataset-level
normalization contract (see normalization.py).

E3 and E4 must use this SAME wrapper (only the checkpoint name differs) so
that max-length policy, truncation/padding strategy, and special-token
handling are identical between the two Transformer experiments, as required
by the build spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class TokenizerBundle:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    checkpoint: str

    def encode_batch(self, texts: list[str]) -> dict[str, Any]:
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # dynamic padding handled by the data collator at train time
        )


def load_tokenizer(checkpoint: str, max_length: int) -> TokenizerBundle:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    return TokenizerBundle(tokenizer=tokenizer, max_length=max_length, checkpoint=checkpoint)
