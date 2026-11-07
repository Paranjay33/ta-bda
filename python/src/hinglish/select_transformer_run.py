"""Choose which installed E3/E4 run is served — by VALIDATION Macro-F1 only.

    python -m hinglish.select_transformer_run            # show the comparison (changes nothing)
    python -m hinglish.select_transformer_run --apply    # switch serving to the best run

A "run" is a protocol (e.g. v2-clean-split, v3-kaggle, v3b-kaggle) for which BOTH E3 and E4
are installed. E3 and E4 always come from the same run, so the research comparison stays
matched. Criterion: mean of E3 and E4 validation Macro-F1 (ties keep the current run).
Test metrics are never read here.

--apply:
  * archives the current E3/E4 locked-test results (artifacts/metrics/runs/<version>/) and
    the current reports (artifacts/metrics/runs/report-<run_tag>/);
  * sets transformer.run_tag in configs/config.yaml;
  * writes e3/e4 validation metrics for the selected versions.
Then run `bash scripts/evaluate_v3_locked.sh` (one locked test evaluation of the new run).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys

from .config import load_config


def candidates(cfg) -> dict[str, dict]:
    models = cfg.path("paths", "models_dir")
    out = {}
    for d in sorted((models / "e3").iterdir()) if (models / "e3").exists() else []:
        tag = d.name
        if tag.endswith("-smoke") or "-run-" in tag:
            continue
        ms = {}
        for exp in ("e3", "e4"):
            mf = models / exp / tag / "manifest.json"
            if not mf.exists():
                break
            m = json.loads(mf.read_text(encoding="utf-8"))
            if (m.get("status") or {}).get("training") != "completed" or not (models / exp / tag / "best").exists():
                break
            ms[exp] = m
        if len(ms) == 2:
            out[tag] = ms
    return out


def switch_run_tag(cfg, new_tag: str) -> None:
    path = cfg.root / "configs" / "config.yaml"
    text = path.read_text(encoding="utf-8")
    start = text.index("\ntransformer:\n")
    nxt = re.search(r"\n[A-Za-z_]+:\s*\n", text[start + 1:])
    end = start + 1 + nxt.start() if nxt else len(text)
    block, n = re.subn(r'(\n  run_tag: )"[^"]*"', f'\\1"{new_tag}"', text[start:end], count=1)
    if n != 1:
        raise SystemExit("could not find transformer.run_tag in configs/config.yaml")
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")
    after = load_config()
    if after.get("transformer", "run_tag") != new_tag or after.get("spark", "run_tag") != cfg.get("spark", "run_tag"):
        path.write_text(text, encoding="utf-8")
        raise SystemExit("config switch verification failed; config restored")


def archive_current(cfg) -> None:
    mdir, pdir = cfg.path("paths", "metrics_dir"), cfg.path("paths", "predictions_dir")
    for exp in ("e3", "e4"):
        cur = mdir / f"{exp}_test_metrics.json"
        if not cur.exists():
            continue
        version = json.loads(cur.read_text(encoding="utf-8")).get("model_version", "unknown")
        arch = mdir / "runs" / version
        arch.mkdir(parents=True, exist_ok=True)
        for f in [cur, mdir / f"{exp}_validation_metrics.json", mdir / f"{exp}_test_confusion_matrix.png",
                  pdir / f"{exp}_test_predictions.csv"]:
            if f.exists():
                shutil.copy2(f, arch / f.name)
        print(f"[select] archived {version} -> {arch.relative_to(cfg.root)}")
    rep = mdir / "runs" / f"report-{cfg.get('transformer', 'run_tag')}"
    rep.mkdir(parents=True, exist_ok=True)
    for f in ["comparison_table.csv", "comparison_table.json", "comparison_report.md", "e3_e4_significance.json"]:
        if (mdir / f).exists():
            shutil.copy2(mdir / f, rep / f)
    ea = cfg.path("paths", "error_analysis_dir")
    for f in ["error_summary.json", "ERROR_ANALYSIS_REPORT.md"]:
        if (ea / f).exists():
            shutil.copy2(ea / f, rep / f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true", help="switch serving to the best run by validation")
    args = p.parse_args(argv)
    cfg = load_config()
    current = cfg.get("transformer", "run_tag")
    runs = candidates(cfg)
    if not runs:
        raise SystemExit("no run with both E3 and E4 installed")

    rows = []
    for tag, ms in runs.items():
        e3, e4 = (ms[e]["validation_metrics"]["macro_f1"] for e in ("e3", "e4"))
        rows.append((tag, e3, e4, (e3 + e4) / 2))
    print(f"{'run':<18}{'E3 val':>9}{'E4 val':>9}{'mean':>9}")
    for tag, e3, e4, mean in rows:
        print(f"{tag:<18}{100 * e3:>8.2f}%{100 * e4:>8.2f}%{100 * mean:>8.2f}%{'   <- serving' if tag == current else ''}")
    best = max(rows, key=lambda r: (round(r[3], 6), r[0] == current))[0]
    print(f"\nBest by validation (mean of E3 and E4): {best}")
    if best == current:
        print("Already serving the best run; nothing to change.")
        return 0
    if not args.apply:
        print("Re-run with --apply to switch serving to it.")
        return 0

    archive_current(cfg)
    switch_run_tag(cfg, best)
    mdir = cfg.path("paths", "metrics_dir")
    for exp, m in runs[best].items():
        (mdir / f"{exp}_validation_metrics.json").write_text(json.dumps(
            {"experiment_id": m["experiment_id"], "split": "validation", "model_version": m["version"],
             "metrics": m["validation_metrics"]}, indent=2), encoding="utf-8")
    print(f"[select] transformer.run_tag: {current} -> {best}")
    print("Next: bash scripts/evaluate_v3_locked.sh   (one locked test evaluation of the selected run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
