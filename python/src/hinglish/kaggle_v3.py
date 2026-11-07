"""v3 training protocol for E3 and E4 (designed for a Kaggle 16 GB GPU).

    HINGLISH_CONFIG_OVERLAY=configs/kaggle_v3.yaml python -m hinglish.kaggle_v3 --out /kaggle/working/export

For EACH of E3 and E4 (identical procedure = matched protocol):
  1. learning-rate search: one run per rate in v3_search.learning_rates (seed 42);
  2. choose the rate with the best VALIDATION Macro-F1 (ties -> smaller rate);
  3. re-run the chosen rate with the other confirmation seeds (43, 44);
  4. served model = the confirmation run with the best VALIDATION Macro-F1;
  5. export the served model (weights, tokenizer, calibration, manifest) + a summary
     with every run's validation curve and the seed mean/sd.

The TEST split is never read here: the locked test evaluation of the two exported
models is done once, locally, after import (python -m hinglish.import_kaggle_v3).

Each training run is a separate subprocess (clean GPU memory). Re-running the command
skips runs whose manifest already exists, so a restarted Kaggle session can resume.
Weights of runs that can no longer be selected are deleted as soon as possible to stay
under Kaggle's 20 GB /kaggle/working limit.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from .config import load_config

CHECKPOINTS = {"E3": "FacebookAI/xlm-roberta-base", "E4": "l3cube-pune/hing-roberta"}
WEIGHT_GLOBS = ("*.bin", "*.safetensors", "tokenizer.json", "sentencepiece.bpe.model")
SMOKE = False  # --smoke: tiny local dry run of the whole orchestration (train.py --smoke-test)


PREFIX = "v3"  # set in main() from transformer.run_tag ("v3-kaggle" -> "v3", "v3b-kaggle" -> "v3b")


def _tag(lr: float, seed: int) -> str:
    return f"{PREFIX}-run-lr{lr:g}-s{seed}"


def _run_dir(cfg, exp: str, tag: str) -> Path:
    # train.py appends "-smoke" to the run tag in smoke mode
    return cfg.path("paths", "models_dir") / exp.lower() / (tag + ("-smoke" if SMOKE else ""))


def _val_f1(cfg, exp: str, tag: str) -> float | None:
    m = _run_dir(cfg, exp, tag) / "manifest.json"
    if not m.exists():
        return None
    return json.loads(m.read_text(encoding="utf-8"))["validation_metrics"]["macro_f1"]


def _train(exp: str, tag: str, lr: float, seed: int, log_dir: Path) -> None:
    cmd = [sys.executable, "-m", "hinglish.train", "--experiment-id", exp,
           "--model-name-or-path", CHECKPOINTS[exp], "--run-tag", tag, "--discard-checkpoints",
           "--set", f"transformer.learning_rate={lr}", "--set", f"seed={seed}"] + (["--smoke-test"] if SMOKE else [])
    log = log_dir / f"{exp.lower()}_{tag}.log"
    print(f"[kaggle_v3] {time.strftime('%H:%M:%S')} START {exp} {tag}  (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy()).returncode
    if rc != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise SystemExit(f"[kaggle_v3] {exp} {tag} FAILED (exit {rc}). Last log lines:\n{tail}")
    print(f"[kaggle_v3] {time.strftime('%H:%M:%S')} END   {exp} {tag} in {(time.time() - t0) / 60:.1f} min", flush=True)


def _drop_weights(cfg, exp: str, tag: str) -> None:
    best = _run_dir(cfg, exp, tag) / "best"
    for pattern in WEIGHT_GLOBS:
        for p in best.glob(pattern):
            p.unlink()


def run_experiment(cfg, exp: str, log_dir: Path) -> dict:
    search = cfg.get("v3_search")
    lrs = [float(x) for x in search["learning_rates"]]
    s0 = int(search["search_seed"])
    seeds = [int(s) for s in search["confirm_seeds"]]

    # 1-2. learning-rate search (validation only)
    results = {}
    keep = None
    for lr in lrs:
        tag = _tag(lr, s0)
        if _val_f1(cfg, exp, tag) is None:
            _train(exp, tag, lr, s0, log_dir)
        results[tag] = _val_f1(cfg, exp, tag)
        print(f"[kaggle_v3] {exp} lr={lr:g} seed={s0}: val Macro-F1 = {results[tag]:.4f}", flush=True)
        if keep is None or results[tag] > results[keep]:
            if keep is not None:
                _drop_weights(cfg, exp, keep)
            keep = tag
        else:
            _drop_weights(cfg, exp, tag)
    # lrs are ascending and only a strictly better score replaces `keep`, so `keep` is the
    # smallest rate among the best -- the same tie-break as below.
    best_lr = max(lrs, key=lambda lr: (results[_tag(lr, s0)], -lr))
    assert keep == _tag(best_lr, s0), (keep, best_lr)
    if s0 not in seeds:
        raise SystemExit("v3_search.search_seed must be one of confirm_seeds")

    # 3. confirmation seeds at the chosen rate (the search-seed run is reused)
    seed_scores = {s0: results[keep]}
    for seed in seeds:
        if seed == s0:
            continue
        tag = _tag(best_lr, seed)
        if _val_f1(cfg, exp, tag) is None:
            _train(exp, tag, best_lr, seed, log_dir)
        seed_scores[seed] = _val_f1(cfg, exp, tag)
        print(f"[kaggle_v3] {exp} lr={best_lr:g} seed={seed}: val Macro-F1 = {seed_scores[seed]:.4f}", flush=True)
        if seed_scores[seed] > _val_f1(cfg, exp, keep):
            _drop_weights(cfg, exp, keep)
            keep = tag
        else:
            _drop_weights(cfg, exp, tag)

    # 4. served model = best validation run among the confirmation seeds (ties -> smaller seed)
    sel_seed = max(sorted(seeds), key=lambda s: (seed_scores[s], -s))
    sel_tag = _tag(best_lr, sel_seed)
    assert keep == sel_tag, (keep, sel_tag)
    seed_scores = {s: seed_scores[s] for s in seeds}
    runs = []
    for tag in sorted({_tag(lr, s0) for lr in lrs} | {_tag(best_lr, s) for s in seeds}):
        m = json.loads((_run_dir(cfg, exp, tag) / "manifest.json").read_text(encoding="utf-8"))
        runs.append({"run_tag": tag, "learning_rate": m["hyperparameters"]["learning_rate"], "seed": m["seed"],
                     "val_macro_f1": m["validation_metrics"]["macro_f1"],
                     "val_accuracy": m["validation_metrics"]["accuracy"],
                     "selected_epoch": m["selected_epoch"],
                     "epochs_run": len(m.get("training_log", [])),
                     "curve": [round(h["eval_macro_f1"], 4) for h in m.get("training_log", [])],
                     "runtime_seconds": m["training_runtime_seconds"]})
    vals = [seed_scores[s] for s in seeds]
    return {"experiment_id": exp, "checkpoint": CHECKPOINTS[exp], "learning_rate_search": {f"{lr:g}": results[_tag(lr, s0)] for lr in lrs},
            "chosen_learning_rate": best_lr, "confirm_seeds": seed_scores,
            "val_macro_f1_mean": statistics.fmean(vals), "val_macro_f1_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "selected_seed": sel_seed, "selected_run_tag": sel_tag, "runs": runs}


def export(cfg, summaries: list[dict], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    final_tag = cfg.get("transformer", "run_tag", default="v3-kaggle") + ("-smoke" if SMOKE else "")
    for s in summaries:
        exp = s["experiment_id"]
        src = _run_dir(cfg, exp, s["selected_run_tag"])
        dst = out / "models" / exp.lower() / final_tag
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src / "best", dst / "best")
        m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        m["source_run_tag"], m["run_tag"] = s["selected_run_tag"], final_tag
        m["version"] = f"{exp.lower()}-{final_tag}"
        m["artifact_dir"] = f"artifacts/models/{exp.lower()}/{final_tag}/best"
        m["v3_selection"] = {k: s[k] for k in ("learning_rate_search", "chosen_learning_rate", "confirm_seeds",
                                               "val_macro_f1_mean", "val_macro_f1_sd", "selected_seed")}
        m["status"]["selection"] = ("validation_macro_f1: lr search on seed 42, then best of seeds "
                                    f"{list(s['confirm_seeds'])} at the chosen lr")
        m.setdefault("notes", []).append("Trained by hinglish.kaggle_v3; the test split was not read.")
        (dst / "manifest.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    from .prepare_data import split_identity

    summary = {"protocol": os.environ.get("HINGLISH_CONFIG_OVERLAY"), "run_tag": final_tag, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "environment": _env(), "dataset": split_identity(cfg), "experiments": summaries}
    (out / "v3_search_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    zip_path = out.parent / f"hinglishsense_{final_tag}_models.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:  # weights do not compress
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out))
    return zip_path


def _env() -> dict:
    try:
        import torch

        return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__}
    except Exception:
        return {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/kaggle/working/export")
    p.add_argument("--experiments", nargs="+", default=["E4", "E3"], choices=["E3", "E4"])
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny dry run (64 rows, 1 epoch per run) to test the pipeline")
    args = p.parse_args(argv)
    global SMOKE, PREFIX
    SMOKE = args.smoke
    os.environ.setdefault("HINGLISH_CONFIG_OVERLAY", "configs/kaggle_v3.yaml")
    cfg = load_config()
    PREFIX = str(cfg.get("transformer", "run_tag")).split("-")[0]
    if cfg.get("v3_search") is None:
        raise SystemExit("v3_search missing: set HINGLISH_CONFIG_OVERLAY=configs/kaggle_v3.yaml")

    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No GPU visible. In Kaggle: Settings -> Accelerator -> GPU T4 x2 or P100.")
    print(f"[kaggle_v3] GPU: {_env().get('gpu')}  overlay: {os.environ['HINGLISH_CONFIG_OVERLAY']}", flush=True)

    # The split must be the frozen one; refuse otherwise.
    rc = subprocess.run([sys.executable, "-m", "hinglish.prepare_data", "--verify-files"]).returncode
    if rc != 0:
        raise SystemExit("Frozen split verification failed - upload data/processed/ unchanged.")

    log_dir = Path(args.out).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_experiment(cfg, exp, log_dir) for exp in args.experiments]
    zip_path = export(cfg, summaries, Path(args.out))
    for s in summaries:
        print(f"[kaggle_v3] {s['experiment_id']}: lr={s['chosen_learning_rate']:g}, val Macro-F1 seeds "
              f"{ {k: round(v, 4) for k, v in s['confirm_seeds'].items()} } mean {s['val_macro_f1_mean']:.4f} "
              f"sd {s['val_macro_f1_sd']:.4f}; served seed {s['selected_seed']}")
    print(f"[kaggle_v3] DONE. Download: {zip_path} ({zip_path.stat().st_size / 2**30:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
