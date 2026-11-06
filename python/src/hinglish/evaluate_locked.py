"""LOCKED test-set evaluation for E1-E4 (run once, after everything is frozen).

    python -m hinglish.evaluate_locked --experiment-id all
    python -m hinglish.evaluate_locked --experiment-id E4

Preconditions (enforced, otherwise the script refuses):
  * the frozen split verifies (file hashes unchanged since prepare_data);
  * the model manifest exists with training completed and validation-based
    selection/calibration recorded;
  * E1/E2: the Spark side already scored TEST with the saved PipelineModel
    (`bash scripts/run_spark.sh e1 --final-test-eval`), producing
    artifacts/predictions/e?_test_predictions.csv.

Outputs: artifacts/metrics/<exp>_test_metrics.json (common schema +
probabilistic metrics), artifacts/predictions/<exp>_test_predictions.csv, an
appended audit line in artifacts/metrics/locked_test_log.jsonl (so any repeat
evaluation is visible), and the manifest's status is updated.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import load_calibration, probabilistic_metrics, softmax
from .config import load_config
from .evaluate import compute_classification_metrics, save_confusion_matrix_plot
from .prepare_data import load_processed_split, load_split_manifest, sha256_file, split_identity

ALL = ["E1", "E2", "E3", "E4"]


def _verify_split(cfg) -> None:
    m = load_split_manifest(cfg)
    for name in ("train", "validation", "test"):
        f = cfg.root / m["splits"][name]["file"]
        if sha256_file(f) != m["splits"][name]["file_sha256"]:
            raise SystemExit(f"REFUSING: {f} changed after the split was frozen.")


def _run_dir(cfg, exp: str, run_tag: str | None) -> Path:
    family_key = "spark" if exp in ("E1", "E2") else "transformer"
    tag = run_tag or cfg.get(family_key, "run_tag", default="v2-clean-split")
    return cfg.path("paths", "models_dir") / exp.lower() / tag


def _transformer_scores(cfg, run_dir: Path, texts: list[str]) -> tuple[np.ndarray, float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    best = run_dir / "best"
    tok = AutoTokenizer.from_pretrained(str(best))
    model = AutoModelForSequenceClassification.from_pretrained(str(best))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    max_len = int(cfg.get("transformer", "max_seq_length", default=128))
    bs = int(cfg.get("transformer", "eval_batch_size", default=32))
    out = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out.append(model(**enc).logits.float().cpu().numpy())
    secs = time.perf_counter() - t0
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64), secs


def evaluate(cfg, exp: str, run_tag: str | None) -> dict:
    labels = cfg.labels
    run_dir = _run_dir(cfg, exp, run_tag)
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        return {"experiment_id": exp, "status": "not_run", "reason": f"no manifest at {mpath}"}
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("status", {}).get("training") != "completed":
        return {"experiment_id": exp, "status": "not_run", "reason": "training not completed"}

    test = load_processed_split(cfg, "test")
    y = test["label_id"].astype(int).to_numpy()
    duration = None

    if exp in ("E3", "E4"):
        logits, duration = _transformer_scores(cfg, run_dir, test["normalized_text"].tolist())
        pred = logits.argmax(1)
        raw_probs = softmax(logits)
        cal = load_calibration(run_dir / "best" / "calibration.json")
        cal_probs = cal.apply(logits)
        scores_out = cal_probs
        prob_block = {
            "uncalibrated_softmax": probabilistic_metrics(raw_probs, y),
            "calibrated": probabilistic_metrics(cal_probs, y),
            "calibration_method": cal.method,
        }
        preds_path = cfg.path("paths", "predictions_dir") / f"{exp.lower()}_test_predictions.csv"
        pd.DataFrame({"sample_id": test["sample_id"], "gold_label_id": y, "pred_label_id": pred,
                      **{f"score_{i}": cal_probs[:, i] for i in range(cal_probs.shape[1])}}
                     ).to_csv(preds_path, index=False)
    else:
        preds_path = cfg.path("paths", "predictions_dir") / f"{exp.lower()}_test_predictions.csv"
        if not preds_path.exists():
            return {"experiment_id": exp, "status": "not_run",
                    "reason": f"run `bash scripts/run_spark.sh {exp.lower()} --final-test-eval` first"}
        sp = pd.read_csv(preds_path).set_index("sample_id").loc[test["sample_id"]]
        if not np.array_equal(sp["gold_label_id"].to_numpy(), y):
            raise SystemExit(f"{exp}: Spark test predictions do not match the frozen test split")
        pred = sp["pred_label_id"].to_numpy()
        scores = sp[[c for c in sp.columns if c.startswith("score_")]].to_numpy(dtype=np.float64)
        if exp == "E1":
            scores_out = scores
            prob_block = {"model_probability": probabilistic_metrics(scores, y)}
        else:
            cal = load_calibration(run_dir / "calibration.json")
            scores_out = cal.apply(scores)
            prob_block = {"calibrated": probabilistic_metrics(scores_out, y),
                          "calibration_method": cal.method,
                          "note": "raw OvR margins are not probabilities; metrics shown are for the calibrated map"}

    metrics = compute_classification_metrics(y.tolist(), pred.tolist(), labels.id_to_label).to_dict()
    metrics_dir = cfg.path("paths", "metrics_dir")
    result = {
        "experiment_id": exp,
        "status": "completed",
        "split": "test",
        "split_identity": split_identity(cfg),
        "model_version": manifest.get("version"),
        "metrics": metrics,
        "probabilistic": prob_block,
        "n_samples": int(len(y)),
        "inference_seconds_total": duration,
        "inference_rows_per_second": (len(y) / duration) if duration else None,
        "evaluated_at_unix": time.time(),
    }
    (metrics_dir / f"{exp.lower()}_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_confusion_matrix_plot(metrics["confusion_matrix"], metrics["labels_order"],
                               metrics_dir / f"{exp.lower()}_test_confusion_matrix.png")
    with open(metrics_dir / "locked_test_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"experiment_id": exp, "model_version": manifest.get("version"),
                            "macro_f1": metrics["macro_f1"], "at_unix": time.time()}) + "\n")

    manifest.setdefault("status", {})["locked_test_evaluation"] = "completed"
    manifest["test_metrics_summary"] = {k: metrics[k] for k in
                                        ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")}
    mpath.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[evaluate_locked] {exp}: test macro-F1={metrics['macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment-id", default="all", choices=ALL + ["all"])
    p.add_argument("--run-tag", default=None)
    args = p.parse_args(argv)
    cfg = load_config()
    _verify_split(cfg)
    exps = ALL if args.experiment_id == "all" else [args.experiment_id]
    rc = 0
    for exp in exps:
        r = evaluate(cfg, exp, args.run_tag)
        if r["status"] != "completed":
            print(f"[evaluate_locked] {exp}: {r['status']} ({r['reason']})")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
