"""Shared error-analysis pipeline for E1-E4.

This module does NOT claim to automatically produce ground-truth error
categories. Linguistic phenomena like sarcasm or implied sentiment cannot be
reliably auto-detected; per the build spec, categories are a structured
annotation format for manual or assisted analysis. Where a cheap heuristic
hint is possible (e.g. presence of an explicit negation word, mixed Latin
script density suggesting code-switching), it is written to a clearly
separate ``heuristic_hint_*`` column, never mixed into the annotation columns
a human/annotator is expected to fill in.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ERROR_CATEGORIES = [
    "code_switching",
    "romanized_spelling_variation",
    "abbreviation_slang",
    "negation",
    "contrast",
    "sarcasm_implied_sentiment",
    "mixed_sentiment",
    "emoji_punctuation_informal",
]

_NEGATION_HINT_RE = re.compile(
    r"\b(nahi|nahin|nhi|na|mat|kabhi nahi|never|not|no|don'?t|didn'?t|can'?t)\b",
    re.IGNORECASE,
)
_CONTRAST_HINT_RE = re.compile(r"\b(but|par|lekin|magar|phir bhi|however)\b", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
)


def _romanized_mixed_script_hint(text: str) -> bool:
    """Very rough code-mixing hint: presence of common Hindi function words
    written in Roman script alongside English words. This is a heuristic
    signal only, NOT a code-switching label."""
    hindi_markers = {
        "hai", "hain", "tha", "thi", "kya", "nahi", "nahin", "acha", "accha",
        "bahut", "bhai", "yaar", "kar", "karo", "kyun", "kaise", "mera",
        "tera", "uska", "sab", "kuch", "bilkul",
    }
    tokens = set(w.lower().strip(".,!?") for w in text.split())
    return len(tokens & hindi_markers) > 0


def build_error_analysis_scaffold(
    texts: list[str],
    gold_labels: list[str],
    predictions: dict[str, list[str]],
    ids: list[str] | None = None,
) -> pd.DataFrame:
    """Build the structured annotation template.

    Args:
        texts: raw (or normalized) text per example.
        gold_labels: ground-truth label string per example.
        predictions: mapping experiment_id -> list of predicted label strings,
            same order/length as texts.
        ids: optional example ids.
    """
    n = len(texts)
    ids = ids or [str(i) for i in range(n)]

    any_wrong = pd.Series([False] * n)
    for exp_id, preds in predictions.items():
        assert len(preds) == n, f"prediction length mismatch for {exp_id}"
        any_wrong = any_wrong | (pd.Series(preds) != pd.Series(gold_labels))

    df = pd.DataFrame({"id": ids, "text": texts, "gold_label": gold_labels})
    for exp_id, preds in predictions.items():
        df[f"pred_{exp_id}"] = preds

    # Heuristic hints (clearly namespaced, not ground truth).
    df["heuristic_hint_negation"] = [bool(_NEGATION_HINT_RE.search(t)) for t in texts]
    df["heuristic_hint_contrast"] = [bool(_CONTRAST_HINT_RE.search(t)) for t in texts]
    df["heuristic_hint_emoji_present"] = [bool(_EMOJI_RE.search(t)) for t in texts]
    df["heuristic_hint_code_mixing"] = [_romanized_mixed_script_hint(t) for t in texts]

    # Manual/assisted annotation columns: left empty for a human annotator.
    for cat in ERROR_CATEGORIES:
        df[f"annotation_{cat}"] = ""
    df["annotator_notes"] = ""

    df["any_experiment_wrong"] = any_wrong.values
    # Only rows where at least one experiment is wrong are worth annotating.
    return df[df["any_experiment_wrong"]].reset_index(drop=True)


def save_error_analysis_scaffold(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
