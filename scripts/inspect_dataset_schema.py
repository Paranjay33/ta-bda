#!/usr/bin/env python3
"""Run this FIRST against any real dataset file before trusting the ingestion
pipeline. Prints the detected schema kind and a few parsed sample rows so you
can confirm hinglish.dataset's assumptions actually match your file, per the
build spec's "do not assume column names blindly" rule.

Usage:
    python scripts/inspect_dataset_schema.py data/raw/train.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

from hinglish.config import load_label_mapping
from hinglish.dataset import detect_schema, load_raw_dataframe, validate_dataset


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-dataset-file>")
        return 1
    path = Path(sys.argv[1])
    schema = detect_schema(path)
    print(f"Detected schema kind: {schema.kind}")
    print(f"Delimiter: {schema.delimiter}  text_col: {schema.text_col}  "
          f"label_col: {schema.label_col}  id_col: {schema.id_col}")
    print("\nFirst probed lines:")
    for ln in schema.sample_lines[:10]:
        print(f"  {ln!r}")

    if schema.kind == "unknown":
        print("\nSchema NOT recognized. Extend hinglish.dataset before proceeding.")
        return 2

    df, _ = load_raw_dataframe(path)
    print(f"\nParsed {len(df)} rows. Sample:")
    print(df.head(5).to_string())

    labels = load_label_mapping()
    report = validate_dataset(df, labels)
    print(f"\nValidation report: {report.summary()}")
    if report.invalid_label_values:
        print(f"Invalid label values seen: {report.invalid_label_values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
