#!/usr/bin/env python3
"""Build the machine-generated E1-E4 comparison (+ E3/E4 significance and error
analysis) from saved artifacts only. Thin wrapper over `python -m hinglish.report`.

Usage:  python scripts/build_comparison_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python" / "src"))

from hinglish.report import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
