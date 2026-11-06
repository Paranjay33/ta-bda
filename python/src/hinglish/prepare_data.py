"""Build the shared processed-data contract consumed by BOTH branches.

    python -m hinglish.prepare_data            # create (or verify) the split
    python -m hinglish.prepare_data --verify   # verify only; never writes

Policy (see docs/DATASET.md, docs/EVALUATION_PROTOCOL.md):

1. Load every labelled source listed in ``dataset.labelled_sources``.
2. Strip the upstream trailing `` nan`` artefact (configurable), then apply the
   shared normalization contract -> ``normalized_text``.
3. Drop rows whose normalized text is empty or whose label is invalid.
4. Group rows by exact ``normalized_text``. Groups carrying conflicting gold
   labels are dropped entirely (ambiguous gold). Remaining duplicate groups
   are collapsed to ONE representative so a text can never sit in two splits.
5. One deterministic stratified 80/10/10 split (seed from config).
6. Write ``data/processed/{train,validation,test}.tsv`` + ``split_manifest.json``.

Freezing: if a split manifest already exists, the script refuses to write a
different split (the test partition is locked). Pass ``--force-new-split`` only
if you deliberately want to invalidate every downstream test result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .config import ProjectConfig, load_config
from .dataset import load_raw_dataframe, validate_dataset
from .normalization import NormalizationConfig, normalize_text

PROCESSED_COLUMNS = [
    "sample_id",
    "source_file",
    "source_row_id",
    "raw_text",
    "normalized_text",
    "label",
    "label_id",
    "split",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preprocessing_fingerprint(cfg: ProjectConfig) -> dict:
    pre = cfg.get("preprocessing", default={}) or {}
    blob = json.dumps(
        {
            "preprocessing": pre,
            "strip_trailing_nan_artifact": cfg.get("dataset", "strip_trailing_nan_artifact"),
        },
        sort_keys=True,
    )
    return {"version": pre.get("version", "unversioned"), "sha256": sha256_text(blob)}


def strip_nan_artifact(text: str) -> str:
    t = text.rstrip()
    if t == "nan":
        return ""
    if t.endswith(" nan"):
        return t[: -len(" nan")]
    return t


def _ids_hash(ids: list[str]) -> str:
    return sha256_text("\n".join(sorted(ids)))


def build_split(cfg: ProjectConfig) -> tuple[dict[str, pd.DataFrame], dict]:
    from sklearn.model_selection import train_test_split

    labels = cfg.labels
    seed = int(cfg.get("seed", default=42))
    norm_cfg = NormalizationConfig.from_project_config(cfg.get("preprocessing", default={}))
    strip_nan = bool(cfg.get("dataset", "strip_trailing_nan_artifact", default=True))
    ratios = cfg.get("dataset", "split_ratios")

    frames, source_reports = [], []
    for rel in cfg.get("dataset", "labelled_sources"):
        path = (cfg.root / rel).resolve()
        df, schema = load_raw_dataframe(path)
        report = validate_dataset(df, labels)
        source_reports.append(
            {
                "file": rel,
                "sha256": sha256_file(path),
                "schema": schema.kind,
                "rows": int(len(df)),
                "validation": report.__dict__,
            }
        )
        df = df.assign(source_file=Path(rel).name)
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    n_pooled = len(pooled)

    pooled["raw_text"] = pooled["text"].astype(str)
    cleaned = pooled["raw_text"].map(strip_nan_artifact) if strip_nan else pooled["raw_text"]
    pooled["normalized_text"] = cleaned.map(lambda t: normalize_text(t, norm_cfg))
    pooled["label"] = pooled["raw_label"].map(labels.canonical_name)

    empty_mask = pooled["normalized_text"].str.strip().eq("")
    invalid_label_mask = pooled["label"].isna()
    dropped_empty = int(empty_mask.sum())
    dropped_invalid = int((invalid_label_mask & ~empty_mask).sum())
    pooled = pooled[~empty_mask & ~invalid_label_mask].reset_index(drop=True)

    # Conflicting-label groups -> drop entirely; consistent duplicates -> collapse.
    label_sets = pooled.groupby("normalized_text")["label"].nunique()
    conflicting = set(label_sets[label_sets > 1].index)
    n_conflict_rows = int(pooled["normalized_text"].isin(conflicting).sum())
    pooled = pooled[~pooled["normalized_text"].isin(conflicting)]
    before_dedup = len(pooled)
    pooled = pooled.drop_duplicates(subset=["normalized_text"], keep="first").reset_index(drop=True)
    n_dup_collapsed = before_dedup - len(pooled)

    pooled["label_id"] = pooled["label"].map(labels.label_to_id).astype(int)
    pooled["sample_id"] = pooled["normalized_text"].map(lambda t: "hs-" + sha256_text(t)[:16])
    pooled["source_row_id"] = pooled["id"].astype(str)
    assert pooled["sample_id"].is_unique, "sample_id collision"

    test_frac = float(ratios["test"])
    val_frac_of_rest = float(ratios["validation"]) / (1.0 - test_frac)
    rest, test = train_test_split(
        pooled, test_size=test_frac, random_state=seed, stratify=pooled["label_id"]
    )
    train, val = train_test_split(
        rest, test_size=val_frac_of_rest, random_state=seed, stratify=rest["label_id"]
    )
    splits = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        part = part.sort_values("sample_id").reset_index(drop=True).copy()
        part["split"] = name
        for col in ("raw_text", "normalized_text"):
            bad = part[col].str.contains(r"[\t\r\n]", regex=True)
            assert not bad.any(), f"{name}.{col} contains tab/newline; contract violated"
        splits[name] = part[PROCESSED_COLUMNS]

    texts = {k: set(v["normalized_text"]) for k, v in splits.items()}
    overlap = {
        "train_validation": len(texts["train"] & texts["validation"]),
        "train_test": len(texts["train"] & texts["test"]),
        "validation_test": len(texts["validation"] & texts["test"]),
    }
    assert all(v == 0 for v in overlap.values()), f"split overlap detected: {overlap}"

    with open(cfg.root / "configs" / "label_mapping.json", "rb") as f:
        label_map_sha = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "schema_version": 1,
        "dataset_name": cfg.get("dataset", "name"),
        "split_policy": cfg.get("dataset", "split_policy"),
        "seed": seed,
        "ratios": ratios,
        "sources": source_reports,
        "label_mapping_sha256": label_map_sha,
        "raw_label_aliases": labels.aliases,
        "preprocessing": preprocessing_fingerprint(cfg),
        "counts": {
            "pooled_rows": n_pooled,
            "dropped_empty_after_normalization": dropped_empty,
            "dropped_invalid_label": dropped_invalid,
            "dropped_conflicting_label_rows": n_conflict_rows,
            "conflicting_label_texts": len(conflicting),
            "collapsed_exact_duplicates": int(n_dup_collapsed),
            "final_unique_samples": int(len(pooled)),
        },
        "splits": {
            name: {
                "rows": int(len(df)),
                "label_distribution": {
                    k: int(v) for k, v in df["label"].value_counts().sort_index().items()
                },
                "sample_ids_sha256": _ids_hash(df["sample_id"].tolist()),
            }
            for name, df in splits.items()
        },
        "overlap_checks_normalized_text": overlap,
        "sample_id_note": "sample_id is a GENERATED identifier: 'hs-' + sha256(normalized_text)[:16]. "
        "Supplied uids are kept as source_row_id but are NOT unique across the supplied files.",
        "created_at_unix": time.time(),
    }
    return splits, manifest


def write_split(cfg: ProjectConfig, splits: dict[str, pd.DataFrame], manifest: dict) -> None:
    paths = {
        "train": cfg.root / cfg.get("dataset", "processed_train_file"),
        "validation": cfg.root / cfg.get("dataset", "processed_validation_file"),
        "test": cfg.root / cfg.get("dataset", "processed_test_file"),
    }
    for name, df in splits.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(paths[name], sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")
        manifest["splits"][name]["file"] = str(paths[name].relative_to(cfg.root)).replace("\\", "/")
        manifest["splits"][name]["file_sha256"] = sha256_file(paths[name])
    out = cfg.root / cfg.get("dataset", "split_manifest_file")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_processed_split(cfg: ProjectConfig, name: str) -> pd.DataFrame:
    key = {"train": "processed_train_file", "validation": "processed_validation_file",
           "test": "processed_test_file"}[name]
    path = cfg.root / cfg.get("dataset", key)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m hinglish.prepare_data` first.")
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def load_split_manifest(cfg: ProjectConfig) -> dict:
    with open(cfg.root / cfg.get("dataset", "split_manifest_file"), "r", encoding="utf-8") as f:
        return json.load(f)


def split_identity(cfg: ProjectConfig) -> dict:
    """Compact identity block embedded in every model/metrics manifest."""
    path = cfg.root / cfg.get("dataset", "split_manifest_file")
    m = load_split_manifest(cfg)
    return {
        "split_manifest_sha256": sha256_file(path),
        "test_sample_ids_sha256": m["splits"]["test"]["sample_ids_sha256"],
        "train_rows": m["splits"]["train"]["rows"],
        "validation_rows": m["splits"]["validation"]["rows"],
        "test_rows": m["splits"]["test"]["rows"],
        "raw_source_sha256": {s["file"]: s["sha256"] for s in m["sources"]},
        "preprocessing": m["preprocessing"],
        "label_mapping_sha256": m["label_mapping_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify", action="store_true", help="verify existing split only; never write")
    p.add_argument("--force-new-split", action="store_true")
    p.add_argument("--verify-files", action="store_true",
                   help="only check data/processed/*.tsv against the frozen manifest hashes "
                        "(no raw files needed, e.g. on Kaggle)")
    args = p.parse_args(argv)

    cfg = load_config()
    if args.verify_files:
        frozen = load_split_manifest(cfg)
        for k in ("train", "validation", "test"):
            f = cfg.root / frozen["splits"][k]["file"]
            if not f.exists() or sha256_file(f) != frozen["splits"][k]["file_sha256"]:
                print(f"[prepare_data] ERROR: {f} missing or differs from the frozen split.", file=sys.stderr)
                return 3
        print("[prepare_data] Frozen split files verified against split_manifest.json.")
        return 0
    splits, manifest = build_split(cfg)
    manifest_path = cfg.root / cfg.get("dataset", "split_manifest_file")

    if manifest_path.exists():
        frozen = load_split_manifest(cfg)
        same = all(
            frozen["splits"][k]["sample_ids_sha256"] == manifest["splits"][k]["sample_ids_sha256"]
            for k in ("train", "validation", "test")
        )
        if same:
            print("[prepare_data] Existing frozen split verified: rebuilt split is identical.")
            for k in ("train", "validation", "test"):
                f = cfg.root / frozen["splits"][k]["file"]
                if sha256_file(f) != frozen["splits"][k]["file_sha256"]:
                    print(f"[prepare_data] ERROR: {f} was modified after freezing.", file=sys.stderr)
                    return 3
            return 0
        if not args.force_new_split:
            print(
                "[prepare_data] REFUSING: a frozen split exists and the rebuilt split differs. "
                "The locked test partition will not be silently regenerated. "
                "Use --force-new-split only if you accept invalidating every test result.",
                file=sys.stderr,
            )
            return 3
    if args.verify:
        print("[prepare_data] --verify: no frozen split found.", file=sys.stderr)
        return 3

    write_split(cfg, splits, manifest)
    print(json.dumps({"counts": manifest["counts"],
                      "splits": {k: {kk: vv for kk, vv in v.items() if kk != "file_sha256"}
                                 for k, v in manifest["splits"].items()}}, indent=2))
    print(f"[prepare_data] Wrote frozen split manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
