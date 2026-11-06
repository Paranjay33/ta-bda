"""Import the Kaggle v3 export (hinglishsense_v3_models.zip) into this project.

    python -m hinglish.import_kaggle_v3 path/to/hinglishsense_v3_models.zip

Checks (refuses on any failure):
  * the runs were trained on THIS frozen split (split-manifest + test-id hashes match);
  * each model's weight files match the SHA-256 recorded in its manifest;
  * the checkpoint's label order equals configs/label_mapping.json;
  * the target artifacts/models/<exp>/<run_tag>/ does not exist yet (never overwrites).

Then:
  * archives the current E3/E4 results (metrics, predictions, reports) under
    artifacts/metrics/runs/<previous version>/ so the v2 numbers stay available;
  * installs the models under artifacts/models/e3|e4/<run_tag>/;
  * writes e3/e4 validation metrics for the new version and the v3 search summary;
  * switches `transformer.run_tag` in configs/config.yaml to the imported tag.

Next step (printed at the end): `bash scripts/evaluate_v3_locked.sh` - the ONE locked test
evaluation of the two imported models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

from .config import load_config
from .prepare_data import sha256_file, split_identity


def _sha(p: Path) -> str:
    return sha256_file(p)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip_path")
    p.add_argument("--no-switch", action="store_true", help="install but keep serving the current run_tag")
    args = p.parse_args(argv)
    cfg = load_config()
    zp = Path(args.zip_path)
    if not zp.exists():
        raise SystemExit(f"{zp} not found")

    stage = cfg.root / ".tmp" / "v3_import"
    shutil.rmtree(stage, ignore_errors=True)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.startswith("/") or ".." in Path(n).parts:
                raise SystemExit(f"unsafe path in zip: {n}")
        z.extractall(stage)

    summary = json.loads((stage / "v3_search_summary.json").read_text(encoding="utf-8"))
    local = split_identity(cfg)
    remote = summary["dataset"]
    for k in ("split_manifest_sha256", "test_sample_ids_sha256"):
        if remote.get(k) != local.get(k):
            raise SystemExit(f"REFUSING: export was trained on a different split ({k} differs).")
    print("[import] split identity matches the local frozen split")

    model_root = stage / "models"
    installs = []
    for exp_dir in sorted(model_root.iterdir()):
        for run in sorted(exp_dir.iterdir()):
            m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if m["dataset"]["split_manifest_sha256"] != local["split_manifest_sha256"]:
                raise SystemExit(f"REFUSING: {run} manifest split hash differs")
            for name, digest in m["artifact_sha256"].items():
                if _sha(run / "best" / name) != digest:
                    raise SystemExit(f"REFUSING: {run}/best/{name} hash mismatch (corrupt download?)")
            conf = json.loads((run / "best" / "config.json").read_text(encoding="utf-8"))
            if {int(k): v for k, v in conf["id2label"].items()} != cfg.labels.id_to_label:
                raise SystemExit(f"REFUSING: {run} label order differs from label_mapping.json")
            target = cfg.path("paths", "models_dir") / exp_dir.name / run.name
            if target.exists():
                raise SystemExit(f"REFUSING: {target} already exists (runs are never overwritten)")
            installs.append((exp_dir.name, run, target, m))
            print(f"[import] {m['experiment_id']} {m['version']}: hashes and label order verified")
    if not installs:
        raise SystemExit("no models found in the export")

    # Archive current transformer results before anything is replaced.
    mdir = cfg.path("paths", "metrics_dir")
    pdir = cfg.path("paths", "predictions_dir")
    for exp, _, _, _ in installs:
        cur = mdir / f"{exp}_test_metrics.json"
        if cur.exists():
            version = json.loads(cur.read_text(encoding="utf-8")).get("model_version", "unknown")
            arch = mdir / "runs" / version
            arch.mkdir(parents=True, exist_ok=True)
            for f in [cur, mdir / f"{exp}_validation_metrics.json", mdir / f"{exp}_test_confusion_matrix.png",
                      pdir / f"{exp}_test_predictions.csv"]:
                if f.exists():
                    shutil.copy2(f, arch / f.name)
            print(f"[import] archived current {exp.upper()} results -> {arch.relative_to(cfg.root)}")
    # Run-level reports (E1-E4 table, significance, error analysis) of the previous state.
    prev_tag = cfg.get("transformer", "run_tag")
    arch = mdir / "runs" / f"report-{prev_tag}"
    arch.mkdir(parents=True, exist_ok=True)
    for f in ["comparison_table.csv", "comparison_table.json", "comparison_report.md", "e3_e4_significance.json"]:
        if (mdir / f).exists():
            shutil.copy2(mdir / f, arch / f)
    ea = cfg.path("paths", "error_analysis_dir")
    for f in ["error_summary.json", "ERROR_ANALYSIS_REPORT.md"]:
        if (ea / f).exists():
            shutil.copy2(ea / f, arch / f)

    # Install.
    for exp, run, target, m in installs:
        shutil.copytree(run, target)
        (mdir / f"{exp}_validation_metrics.json").write_text(json.dumps(
            {"experiment_id": m["experiment_id"], "split": "validation", "model_version": m["version"],
             "metrics": m["validation_metrics"]}, indent=2), encoding="utf-8")
        print(f"[import] installed {target.relative_to(cfg.root)}")
    shutil.copy2(stage / "v3_search_summary.json", mdir / "v3_search_summary.json")

    tags = {t.name for _, _, t, _ in installs}
    if len(tags) != 1:
        raise SystemExit(f"expected one run tag, got {tags}")
    new_tag = tags.pop()
    if not args.no_switch:
        cfg_path = cfg.root / "configs" / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        start = text.index("\ntransformer:\n")
        nxt = re.search(r"\n[A-Za-z_]+:\s*\n", text[start + 1:])
        end = start + 1 + nxt.start() if nxt else len(text)
        block, n = re.subn(r'(\n  run_tag: )"[^"]*"', f'\\1"{new_tag}"', text[start:end], count=1)
        if n != 1:
            raise SystemExit("could not find transformer.run_tag in configs/config.yaml")
        cfg_path.write_text(text[:start] + block + text[end:], encoding="utf-8")
        if load_config().get("transformer", "run_tag") != new_tag or \
                load_config().get("spark", "run_tag") != cfg.get("spark", "run_tag"):
            cfg_path.write_text(text, encoding="utf-8")
            raise SystemExit("config switch verification failed; config restored")
        print(f"[import] configs/config.yaml: transformer.run_tag -> {new_tag} (was {prev_tag})")
    shutil.rmtree(stage, ignore_errors=True)
    print("\nNext: bash scripts/evaluate_v3_locked.sh   (one locked test evaluation of the imported models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
