"""Lightweight, dependency-free sanity check that the required project
structure and config files exist and are internally consistent. This is
meant to run in seconds with zero heavy dependencies (no torch/pyspark
needed) as a first check after cloning, or in CI before the real test
suites (python/tests, spark/src/test) run.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "spark/src/main/scala/hinglish",
    "spark/src/test/scala/hinglish",
    "python/src/hinglish",
    "python/tests",
    "configs",
    "scripts",
    "docs",
    "data/raw",
    "data/processed",
    "artifacts/models",
    "artifacts/metrics",
    "artifacts/manifests",
    "artifacts/predictions",
    "artifacts/error_analysis",
    "artifacts/scalability",
]

REQUIRED_FILES = [
    "configs/config.yaml",
    "configs/label_mapping.json",
    "spark/build.sbt",
    "python/requirements.txt",
    "README.md",
]


def check_dirs():
    missing = [d for d in REQUIRED_DIRS if not (ROOT / d).is_dir()]
    assert not missing, f"Missing required directories: {missing}"


def check_files():
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    assert not missing, f"Missing required files: {missing}"


def check_label_mapping_contiguous():
    with open(ROOT / "configs" / "label_mapping.json") as f:
        mapping = json.load(f)
    ids = sorted(mapping.values())
    assert ids == list(range(len(ids))), f"Label ids must be contiguous 0..N-1, got {ids}"


if __name__ == "__main__":
    check_dirs()
    check_files()
    check_label_mapping_contiguous()
    print("Repo structure OK.")
