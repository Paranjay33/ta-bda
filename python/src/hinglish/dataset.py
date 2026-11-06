"""SentiMix Hinglish dataset loading, schema detection and validation.

IMPORTANT — read docs/DATASET.md before trusting this module against a new
file. The SemEval-2020 Task 9 SentiMix Hinglish release has historically been
distributed as CoNLL-style token-tagged blocks:

    meta\t<uid>\t<overall_sentiment_label>
    <token_1>\t<lang_tag_1>
    <token_2>\t<lang_tag_2>
    ...
    <blank line>

    meta\t<uid>\t<overall_sentiment_label>
    ...

Some mirrors instead redistribute a flattened CSV/TSV with columns such as
``text``/``sentence`` and ``label``/``sentiment``. This module does NOT
blindly assume either format: ``detect_schema`` inspects the actual file
first, and ``load_dataset`` raises a loud, descriptive error (never a silent
fallback) if the file matches neither known shape. If you have a genuinely
different schema, extend ``detect_schema``/``load_dataset`` explicitly rather
than coercing data through the wrong parser.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import LabelMapping

SchemaKind = Literal["conll_meta", "csv_flat", "tsv_uid_text_label", "tsv_uid_text", "unknown"]

_CSV_TEXT_COL_CANDIDATES = ["text", "tweet", "sentence", "review", "content"]
_CSV_LABEL_COL_CANDIDATES = ["label", "sentiment", "sentiment_label", "class"]


@dataclass
class SchemaInfo:
    kind: SchemaKind
    delimiter: str | None = None
    text_col: str | None = None
    label_col: str | None = None
    id_col: str | None = None
    sample_lines: list[str] = field(default_factory=list)


def detect_schema(path: Path, n_probe_lines: int = 20) -> SchemaInfo:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [next(f, "") for _ in range(n_probe_lines)]
    sample = [ln.rstrip("\n") for ln in lines if ln.strip() != ""]

    if sample and sample[0].split("\t")[0].strip().lower() == "meta":
        return SchemaInfo(kind="conll_meta", delimiter="\t", sample_lines=sample)

    # Try CSV/TSV with header.
    for delimiter in (",", "\t"):
        try:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
        except Exception:
            header = None
        if not header or len(header) < 2:
            continue
        lowered = {c.strip().lower(): c for c in header}
        text_col = next((lowered[c] for c in _CSV_TEXT_COL_CANDIDATES if c in lowered), None)
        label_col = next((lowered[c] for c in _CSV_LABEL_COL_CANDIDATES if c in lowered), None)
        id_col = next((lowered[c] for c in ("id", "uid", "tweet_id") if c in lowered), None)
        if text_col and label_col:
            return SchemaInfo(
                kind="csv_flat",
                delimiter=delimiter,
                text_col=text_col,
                label_col=label_col,
                id_col=id_col,
                sample_lines=sample,
            )

    # Headerless TSV as supplied in this workspace (audited 2026-11-06):
    #   <uid> TAB <text> TAB <raw_label>  (labelled)  or  <uid> TAB <text>  (unlabelled)
    # Accepted only if EVERY probed line has the same column count, a numeric uid
    # and (for 3 columns) a short label-like last field.
    if sample:
        parts = [ln.rstrip("\r").split("\t") for ln in sample]
        widths = {len(p) for p in parts}
        if widths == {3} and all(p[0].strip().isdigit() and len(p[2].strip()) <= 12 for p in parts):
            return SchemaInfo(kind="tsv_uid_text_label", delimiter="\t", sample_lines=sample)
        if widths == {2} and all(p[0].strip().isdigit() for p in parts):
            return SchemaInfo(kind="tsv_uid_text", delimiter="\t", sample_lines=sample)

    return SchemaInfo(kind="unknown", sample_lines=sample)


def _parse_headerless_tsv(path: Path, labelled: bool) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            expected = 3 if labelled else 2
            if len(parts) != expected:
                raise ValueError(
                    f"{path}:{line_no}: expected {expected} tab-separated fields, got {len(parts)}"
                )
            rows.append(
                {
                    "id": parts[0].strip(),
                    "raw_label": parts[2].strip() if labelled else "",
                    "text": parts[1],
                    "token_count": len(parts[1].split()),
                    "source_line": line_no,
                }
            )
    return pd.DataFrame(rows)


def _parse_conll_meta(path: Path) -> pd.DataFrame:
    rows = []
    uid, label, tokens = None, None, []

    def flush():
        if uid is not None:
            rows.append(
                {
                    "id": uid,
                    "raw_label": label,
                    "text": " ".join(tokens),
                    "token_count": len(tokens),
                }
            )

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.strip() == "":
                flush()
                uid, label, tokens = None, None, []
                continue
            parts = line.split("\t")
            if parts[0].strip().lower() == "meta":
                flush()
                tokens = []
                uid = parts[1].strip() if len(parts) > 1 else None
                label = parts[2].strip() if len(parts) > 2 else None
            else:
                token = parts[0]
                tokens.append(token)
    flush()  # last block if file doesn't end with a blank line
    return pd.DataFrame(rows)


def _parse_csv_flat(path: Path, schema: SchemaInfo) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=schema.delimiter, dtype=str, keep_default_na=False)
    out = pd.DataFrame()
    out["id"] = df[schema.id_col] if schema.id_col else [str(i) for i in range(len(df))]
    out["raw_label"] = df[schema.label_col]
    out["text"] = df[schema.text_col]
    out["token_count"] = out["text"].fillna("").str.split().str.len()
    return out


def load_raw_dataframe(path: str | Path) -> tuple[pd.DataFrame, SchemaInfo]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    schema = detect_schema(path)
    if schema.kind == "conll_meta":
        df = _parse_conll_meta(path)
    elif schema.kind == "csv_flat":
        df = _parse_csv_flat(path, schema)
    elif schema.kind == "tsv_uid_text_label":
        df = _parse_headerless_tsv(path, labelled=True)
    elif schema.kind == "tsv_uid_text":
        df = _parse_headerless_tsv(path, labelled=False)
    else:
        raise ValueError(
            f"Could not detect a known schema for {path}. First non-empty lines were:\n"
            + "\n".join(schema.sample_lines)
            + "\n\nExtend hinglish.dataset.detect_schema()/load_dataset() explicitly for "
            "this format rather than guessing — do not silently coerce it."
        )
    return df, schema


@dataclass
class ValidationReport:
    n_rows: int
    n_missing_text: int
    n_missing_label: int
    n_invalid_label: int
    n_empty_after_normalization: int
    n_exact_duplicates: int
    n_duplicate_ids: int
    invalid_label_values: list[str]

    def is_clean(self) -> bool:
        return (
            self.n_missing_text == 0
            and self.n_missing_label == 0
            and self.n_invalid_label == 0
        )

    def summary(self) -> str:
        return (
            f"rows={self.n_rows} missing_text={self.n_missing_text} "
            f"missing_label={self.n_missing_label} invalid_label={self.n_invalid_label} "
            f"empty_after_norm={self.n_empty_after_normalization} "
            f"exact_dup_rows={self.n_exact_duplicates} duplicate_ids={self.n_duplicate_ids}"
        )


def validate_dataset(df: pd.DataFrame, labels: LabelMapping) -> ValidationReport:
    n_rows = len(df)
    missing_text_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    missing_label_mask = df["raw_label"].isna() | (df["raw_label"].astype(str).str.strip() == "")

    normalized_labels = df["raw_label"].astype(str).str.strip().str.lower()
    valid_mask = normalized_labels.map(lambda v: labels.canonical_name(v) is not None)
    invalid_label_mask = (~missing_label_mask) & (~valid_mask)

    empty_after_norm_mask = df["text"].astype(str).str.strip() == ""
    exact_dup_mask = df.duplicated(subset=["text", "raw_label"], keep=False)
    duplicate_ids = 0
    if "id" in df.columns:
        duplicate_ids = int(df["id"].duplicated(keep=False).sum())

    return ValidationReport(
        n_rows=n_rows,
        n_missing_text=int(missing_text_mask.sum()),
        n_missing_label=int(missing_label_mask.sum()),
        n_invalid_label=int(invalid_label_mask.sum()),
        n_empty_after_normalization=int(empty_after_norm_mask.sum()),
        n_exact_duplicates=int(exact_dup_mask.sum()),
        n_duplicate_ids=duplicate_ids,
        invalid_label_values=sorted(set(normalized_labels[invalid_label_mask].tolist())),
    )


def check_split_overlap(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, int]:
    """Count exact-text overlaps between splits. Any non-zero overlap with the
    test split is a serious issue (potential leakage) and must be surfaced,
    never silently ignored."""

    def texts(df: pd.DataFrame) -> set[str]:
        return set(df["text"].astype(str).str.strip().tolist())

    train_t, val_t, test_t = texts(train_df), texts(val_df), texts(test_df)
    return {
        "train_val_overlap": len(train_t & val_t),
        "train_test_overlap": len(train_t & test_t),
        "val_test_overlap": len(val_t & test_t),
    }


def compute_data_hash(df: pd.DataFrame) -> str:
    """SHA-256 over sorted (id, text, raw_label) tuples, for manifest
    reproducibility. Order-independent so row shuffling doesn't change the
    hash, but any content change does."""
    cols = [c for c in ("id", "text", "raw_label") if c in df.columns]
    tuples = sorted(tuple(str(v) for v in row) for row in df[cols].itertuples(index=False))
    h = hashlib.sha256()
    for t in tuples:
        h.update("\u241f".join(t).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def clean_and_map_labels(df: pd.DataFrame, labels: LabelMapping) -> pd.DataFrame:
    """Drop rows with missing/invalid text or label (loudly logged, never
    silent), and attach the canonical integer label column."""
    before = len(df)
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["raw_label_norm"] = df["raw_label"].astype(str).map(labels.canonical_name)

    valid_mask = df["text"].str.strip().ne("") & df["raw_label_norm"].notna()
    dropped = int((~valid_mask).sum())
    if dropped:
        print(
            f"[hinglish.dataset] WARNING: dropping {dropped}/{before} rows "
            "with missing/invalid text or label (see ValidationReport for detail)."
        )
    df = df[valid_mask].reset_index(drop=True)
    df["label_id"] = df["raw_label_norm"].map(labels.label_to_id).astype(int)
    return df
