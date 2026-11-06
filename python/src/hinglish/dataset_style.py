"""Serving-time "dataset-style" transform (train/serve skew guard).

Why this exists
---------------
The supplied SentiMix release was preprocessed *upstream* (before it reached
this project): lowercased, NLTK English stopwords removed, every token
Porter-stemmed. This was verified statistically during the 2026-11-06 audit:

* 0 occurrences of any of the 198 NLTK English stopwords in ~230k training
  tokens (so ``not``/``no``/``but`` never appear in training text);
* 98.6% of training token mass is a fixed point of the Porter stemmer,
  versus ~42% for ordinary English text.

All four models (E1-E4) are therefore trained on text in that space. A live
user typing ``"movie acchi thi but ending bakwas thi"`` would otherwise give
the models tokens they never saw (``movie`` vs ``movi``). This module maps
live/batch input into the training text space.

It is a *reconstruction* of an undocumented upstream step: the exact
tokenizer used upstream is unknown. ``NLTKWordTokenizer`` (what
``nltk.word_tokenize`` uses per sentence) reproduces observed artefacts such
as the stand-alone ``n't`` / ``'s`` tokens in the training data. It is NOT
applied to the already-processed training/eval splits (that would double-
process them) — only to text that arrives raw.

Masked placeholders produced by the shared normalization contract
(``<URL>``, ``<USER>``) are protected from tokenization/stemming.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"<(URL|USER)>")
_PROTECT = {"<URL>": "zzurlzz", "<USER>": "zzuserzz"}
_RESTORE = {v: k for k, v in _PROTECT.items()}


@dataclass(frozen=True)
class DatasetStyleConfig:
    enabled: bool
    version: str
    stopwords_file: Path

    @classmethod
    def from_project_config(cls, cfg) -> "DatasetStyleConfig":
        raw = (cfg.get("preprocessing", "dataset_style_transform", default={}) or {})
        return cls(
            enabled=bool(raw.get("enabled", True)),
            version=str(raw.get("version", "dataset-style-v1")),
            stopwords_file=(cfg.root / raw.get(
                "stopwords_file", "configs/dataset_compat/nltk_english_stopwords.txt"
            )).resolve(),
        )


@lru_cache(maxsize=4)
def _load_stopwords(path: str) -> frozenset[str]:
    with open(path, "r", encoding="utf-8") as f:
        return frozenset(w.strip().lower() for w in f if w.strip())


@lru_cache(maxsize=1)
def _tools():
    from nltk.stem import PorterStemmer
    from nltk.tokenize import NLTKWordTokenizer

    return NLTKWordTokenizer(), PorterStemmer()


def to_dataset_style(text: str, cfg: DatasetStyleConfig) -> str:
    """Map shared-normalized text into the supplied dataset's text space."""
    if not cfg.enabled:
        return text
    tokenizer, stemmer = _tools()
    stop = _load_stopwords(str(cfg.stopwords_file))

    protected = _PLACEHOLDER_RE.sub(lambda m: f" {_PROTECT[m.group(0)]} ", text)
    out: list[str] = []
    for tok in tokenizer.tokenize(protected.lower()):
        if tok in _RESTORE:
            out.append(_RESTORE[tok])
            continue
        if tok in stop:
            continue
        out.append(stemmer.stem(tok))
    return " ".join(out)
