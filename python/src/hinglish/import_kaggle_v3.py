"""Install a Kaggle export (hinglishsense_<run_tag>_models.zip) into this project.

    python -m hinglish.import_kaggle_v3 path/to/hinglishsense_v3-kaggle_models.zip

Install ONLY — serving, metrics and configs are not changed. Choosing which installed
run is served is a separate, validation-only step: `python -m hinglish.select_transformer_run`.

Refuses (exit 1) if:
  * the runs were trained on a different split (split-manifest / test-id hashes differ);
  * any weight file's SHA-256 differs from its manifest (corrupt or partial download);
  * the checkpoint label order differs from configs/label_mapping.json;
  * artifacts/models/<exp>/<run_tag>/ already exists (runs are never overwritten).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from .config import load_config
from .prepare_data import sha256_file, split_identity


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip_path")
    args = p.parse_args(argv)
    cfg = load_config()
    zp = Path(args.zip_path)
    if not zp.exists():
        raise SystemExit(f"{zp} not found")

    stage = cfg.root / ".tmp" / "kaggle_import"
    shutil.rmtree(stage, ignore_errors=True)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.startswith("/") or ".." in Path(n).parts:
                raise SystemExit(f"unsafe path in zip: {n}")
        z.extractall(stage)

    summary = json.loads((stage / "v3_search_summary.json").read_text(encoding="utf-8"))
    local = split_identity(cfg)
    for k in ("split_manifest_sha256", "test_sample_ids_sha256"):
        if summary["dataset"].get(k) != local.get(k):
            raise SystemExit(f"REFUSING: export was trained on a different split ({k} differs).")
    print("[import] split identity matches the local frozen split")

    installs = []
    for exp_dir in sorted((stage / "models").iterdir()):
        for run in sorted(exp_dir.iterdir()):
            m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if m["dataset"]["split_manifest_sha256"] != local["split_manifest_sha256"]:
                raise SystemExit(f"REFUSING: {run} manifest split hash differs")
            for name, digest in m["artifact_sha256"].items():
                if sha256_file(run / "best" / name) != digest:
                    raise SystemExit(f"REFUSING: {run.name}/best/{name} hash mismatch (corrupt download?)")
            conf = json.loads((run / "best" / "config.json").read_text(encoding="utf-8"))
            if {int(k): v for k, v in conf["id2label"].items()} != cfg.labels.id_to_label:
                raise SystemExit(f"REFUSING: {run} label order differs from label_mapping.json")
            target = cfg.path("paths", "models_dir") / exp_dir.name / run.name
            if target.exists():
                raise SystemExit(f"REFUSING: {target} already exists (runs are never overwritten)")
            installs.append((run, target, m))
            print(f"[import] {m['experiment_id']} {m['version']}: hashes and label order verified "
                  f"(val Macro-F1 {m['validation_metrics']['macro_f1']:.4f})")
    if not installs:
        raise SystemExit("no models found in the export")
    tags = {t.name for _, t, _ in installs}
    if len(tags) != 1:
        raise SystemExit(f"expected one run tag, got {tags}")
    tag = tags.pop()

    for run, target, _ in installs:
        shutil.copytree(run, target)
        print(f"[import] installed {target.relative_to(cfg.root)}")
    dst = cfg.path("paths", "metrics_dir") / f"search_summary_{tag}.json"
    shutil.copy2(stage / "v3_search_summary.json", dst)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"\nInstalled run '{tag}'. Nothing is served or evaluated yet.\n"
          "Next: python -m hinglish.select_transformer_run   (compares installed runs on VALIDATION only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
