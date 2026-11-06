"""Shared dataset-level normalization contract.

This module implements ONLY the normalization that both the Spark/classical
branch and the Transformer branch should agree on before any model-specific
tokenization happens. It is intentionally conservative: sentiment-bearing
signal (emoji, punctuation, repeated characters, negation words, slang) is
preserved by default, per the build spec.

Every rule below is documented inline. Toggle rules via configs/config.yaml
(``preprocessing`` section) rather than editing this module, so both branches
stay consistent with a single config.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_MENTION_RE = re.compile(r"(?<![\w@])@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationConfig:
    lowercase: bool = False
    strip_html_entities: bool = True
    mask_urls: bool = True
    mask_user_mentions: bool = True
    collapse_whitespace: bool = True
    preserve_emoji: bool = True  # documented no-op flag: we never strip emoji
    preserve_repeated_characters: bool = True  # documented no-op: no char-collapsing applied
    preserve_punctuation: bool = True  # documented no-op: no punctuation stripping applied

    @classmethod
    def from_project_config(cls, raw_preprocessing: dict) -> "NormalizationConfig":
        return cls(
            lowercase=bool(raw_preprocessing.get("lowercase", False)),
            strip_html_entities=bool(raw_preprocessing.get("strip_html_entities", True)),
            mask_urls=bool(raw_preprocessing.get("mask_urls", True)),
            mask_user_mentions=bool(raw_preprocessing.get("mask_user_mentions", True)),
            collapse_whitespace=bool(raw_preprocessing.get("collapse_whitespace", True)),
            preserve_emoji=bool(raw_preprocessing.get("preserve_emoji", True)),
            preserve_repeated_characters=bool(
                raw_preprocessing.get("preserve_repeated_characters", True)
            ),
            preserve_punctuation=bool(raw_preprocessing.get("preserve_punctuation", True)),
        )


def normalize_text(text: str, cfg: NormalizationConfig) -> str:
    """Apply the shared, model-agnostic normalization contract.

    Rule-by-rule documentation:
      1. HTML entity unescape (``&amp;`` -> ``&``): raw scraped tweet text
         commonly contains escaped entities that carry no sentiment signal
         themselves; unescaping does not remove information.
      2. URL masking -> literal ``<URL>`` token: URLs are almost never
         sentiment-bearing themselves and are highly sparse/unique, which
         hurts TF-IDF vocabulary and adds noise to subword tokenizers.
      3. User-mention masking -> literal ``<USER>``: same rationale as URLs,
         and additionally avoids the model keying sentiment off of specific
         usernames (a privacy + generalization concern).
      4. Whitespace collapsing: multiple spaces/tabs/newlines -> single space,
         then strip leading/trailing whitespace. Purely cosmetic, no
         information loss.
      5. Optional lowercasing (OFF by default): disabled by default because
         casing can carry emphasis/sentiment (e.g. "AMAZING" vs "amazing").
         Left as a config toggle for ablation experiments.

    Explicitly NOT done here (by design, preserved for downstream models):
      - Emoji are never stripped or converted to text.
      - Repeated characters (e.g. "sooooo good") are never collapsed.
      - Punctuation (including "!!!", "?!") is never stripped.
      - Negation words, slang, and code-switched tokens are never removed
        or translated.
    """
    if cfg.strip_html_entities:
        text = html.unescape(text)

    if cfg.mask_urls:
        text = _URL_RE.sub("<URL>", text)

    if cfg.mask_user_mentions:
        text = _USER_MENTION_RE.sub("<USER>", text)

    if cfg.collapse_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()

    if cfg.lowercase:
        text = text.lower()

    return text
