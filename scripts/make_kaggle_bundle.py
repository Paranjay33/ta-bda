#!/usr/bin/env python3
"""Build dist/hinglishsense_kaggle_bundle.zip to upload as a PRIVATE Kaggle dataset.

Contains only what training needs: configs/, python/src/, the frozen split
(data/processed/*.tsv + split_manifest.json) and kaggle/README.md.
The TSVs contain SentiMix tweets -> keep the Kaggle dataset PRIVATE.

    python scripts/make_kaggle_bundle.py
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hinglishsense_kaggle_bundle.zip"


def files():
    for base in ["configs", "python/src/hinglish"]:
        for p in sorted((ROOT / base).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                yield p
    yield ROOT / "python" / "pyproject.toml"
    yield ROOT / "kaggle" / "README.md"
    man = json.loads((ROOT / "data/processed/split_manifest.json").read_text(encoding="utf-8"))
    yield ROOT / "data/processed/split_manifest.json"
    for k in ("train", "validation", "test"):
        f = ROOT / man["splits"][k]["file"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != man["splits"][k]["file_sha256"]:
            sys.exit(f"{f} does not match the frozen manifest - refusing to bundle")
        yield f


def main():
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            z.write(p, p.relative_to(ROOT).as_posix())
            n += 1
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size / 2**20:.1f} MB)")
    print("Upload it to Kaggle as a PRIVATE dataset (it contains SentiMix tweets).")


if __name__ == "__main__":
    main()
