"""Local (single-machine, non-Spark) classical baseline used ONLY to give the
Spark scalability benchmark a comparison point, per the build spec's BDA
scalability requirement.

This intentionally mirrors E1's feature/model choice (word TF-IDF + logistic
regression) using scikit-learn, run at the same synthetic replication scales
(1x/5x/10x) as the Spark benchmark. It is explicitly a single-machine
reference point, not a claim about multi-node performance.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import load_config
from .dataset import clean_and_map_labels, load_raw_dataframe, validate_dataset
from .normalization import NormalizationConfig, normalize_text


def replicate(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    if factor == 1:
        return df.copy()
    return pd.concat([df] * factor, ignore_index=True)


def run_scale(df: pd.DataFrame, seed: int) -> dict:
    """Mirror the Spark benchmark exactly: fit TF-IDF + multinomial LR on the
    (replicated) training rows, then transform/predict all of them.
    Timing only -- replicated rows are not new observations, so no accuracy
    is reported (the midterm version scored a split of replicated data, which
    put copies of training rows in its "test" set)."""
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=20000, token_pattern=r"\S+", lowercase=False)
    X = vectorizer.fit_transform(df["text"])
    clf = LogisticRegression(max_iter=100, random_state=seed)
    clf.fit(X, df["label_id"])
    fit_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    clf.predict(X)
    predict_time = time.perf_counter() - t1
    total = fit_time + predict_time
    return {
        "n_rows": len(df),
        "fit_seconds": fit_time,
        "predict_seconds": predict_time,
        "total_seconds": total,
        "rows_per_second": len(df) / total,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--scales", nargs="+", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config) if args.config else None)
    labels = cfg.labels
    seed = int(cfg.get("seed", default=42))
    norm_cfg = NormalizationConfig.from_project_config(cfg.get("preprocessing", default={}))
    scales = args.scales or cfg.get("scalability", "scale_factors", default=[1, 5, 10])

    from .prepare_data import load_processed_split

    try:
        df = load_processed_split(cfg, "train")  # same rows/text the Spark benchmark uses
    except FileNotFoundError as e:
        print(f"[scalability_compare] BLOCKED: {e}")
        return 2
    df = df.assign(text=df["normalized_text"], label_id=df["label_id"].astype(int))

    import os

    repeats = int(os.environ.get("SCALABILITY_REPEATS", "2"))
    run_scale(df, seed)  # untimed warm-up, mirrors the Spark benchmark
    results = []
    for factor in scales:
        print(
            f"[scalability_compare] Running scale factor {factor}x x{repeats} "
            "(REPLICATED DATA - synthetic scalability workload, not real "
            "additional observations)"
        )
        scaled_df = replicate(df, factor)
        for rep in range(1, repeats + 1):
            r = run_scale(scaled_df, seed)
            r.update(scale_factor=factor, repeat=rep, is_replicated_synthetic_workload=factor != 1)
            results.append(r)
            print(f"[scalability_compare]   -> {r}")

    out_path = cfg.path("paths", "scalability_dir") / "python_local_baseline_scalability.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[scalability_compare] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
