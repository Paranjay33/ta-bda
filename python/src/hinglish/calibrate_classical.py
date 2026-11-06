"""Validation-only score handling for the Spark experiments (run after E1/E2 training).

    python -m hinglish.calibrate_classical

* E2 (OneVsRest LinearSVC): exports raw per-class SVM margins (NOT probabilities).
  Temperature scaling softmax(margins / T) is fitted on the VALIDATION margins
  by minimising NLL and saved as calibration.json. It is monotone per row, so
  the calibrated argmax always equals the native OvR decision (bars never
  contradict the predicted class). A multinomial Platt map was also evaluated
  (slightly lower validation NLL) but changes the argmax on ~18% of rows, i.e.
  it would be a different, validation-trained classifier rather than E2; its
  numbers are recorded in the manifest for transparency only.
* E1 (multinomial LR): Spark's probability output is used as-is
  ("model_probability", not separately calibrated); its validation
  probabilistic metrics are recorded for transparency.

The test split is never read here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .calibration import fit_margin_platt, fit_temperature, probabilistic_metrics
from .config import load_config


def _run_dir(cfg, exp: str):
    tag = cfg.get("spark", "run_tag", default="v2-clean-split")
    return cfg.path("paths", "models_dir") / exp.lower() / tag


def _load_val(cfg, exp: str):
    path = cfg.path("paths", "predictions_dir") / f"{exp.lower()}_validation_predictions.csv"
    df = pd.read_csv(path)
    scores = df[[c for c in df.columns if c.startswith("score_")]].to_numpy(dtype=np.float64)
    return df, scores, df["gold_label_id"].to_numpy()


def main() -> int:
    cfg = load_config()
    out = {}

    # ---- E2: Platt calibration on OvR margins (validation only) ----
    run = _run_dir(cfg, "E2")
    if (run / "manifest.json").exists():
        df, margins, y = _load_val(cfg, "E2")
        cal = fit_temperature(margins, y)
        cal.method = "temperature_scaling_on_ovr_margins"
        cal.save(run / "calibration.json")
        probs = cal.apply(margins)
        disagree = float(np.mean(probs.argmax(1) != df["pred_label_id"].to_numpy()))
        platt = fit_margin_platt(margins, y)
        platt_probs = platt.apply(margins)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifest["calibration"] = {
            "method": cal.method, "temperature": cal.temperature, "fitted_on": "validation",
            "file": "calibration.json", "n_samples": cal.n_samples,
            "validation_log_loss_after": cal.val_nll_after,
            "validation_argmax_disagreement_with_native_ovr": disagree,
            "rejected_alternative": {
                "method": platt.method, "validation_log_loss": platt.val_nll_after,
                "validation_argmax_disagreement_with_native_ovr":
                    float(np.mean(platt_probs.argmax(1) != df["pred_label_id"].to_numpy())),
                "reason": "changes the predicted class -> would be a different (validation-trained) classifier",
            },
        }
        manifest["score_semantics"] = (
            "calibrated_probability (temperature-scaled softmax over OvR LinearSVC margins, fitted on validation; "
            "argmax identical to the native OvR decision)"
        )
        manifest.setdefault("validation_metrics", {})["probabilistic_calibrated"] = probabilistic_metrics(probs, y)
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out["E2"] = {"val_log_loss": cal.val_nll_after, "argmax_disagreement": disagree}
    else:
        out["E2"] = "not trained"

    # ---- E1: record probabilistic quality of native LR probabilities ----
    run = _run_dir(cfg, "E1")
    if (run / "manifest.json").exists():
        _df, probs, y = _load_val(cfg, "E1")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifest["calibration"] = {"method": "none", "note": "Spark multinomial LR probabilities used as-is."}
        manifest.setdefault("validation_metrics", {})["probabilistic_uncalibrated"] = probabilistic_metrics(probs, y)
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out["E1"] = manifest["validation_metrics"]["probabilistic_uncalibrated"]
    else:
        out["E1"] = "not trained"

    # Python-side validation metrics in the common schema (cross-check vs Spark's own).
    from .evaluate import compute_classification_metrics

    labels = cfg.labels
    for exp in ("E1", "E2"):
        p = cfg.path("paths", "predictions_dir") / f"{exp.lower()}_validation_predictions.csv"
        if p.exists():
            df = pd.read_csv(p)
            m = compute_classification_metrics(df["gold_label_id"].tolist(), df["pred_label_id"].tolist(),
                                               labels.id_to_label).to_dict()
            version = json.loads((_run_dir(cfg, exp) / "manifest.json").read_text(encoding="utf-8"))["version"]
            (cfg.path("paths", "metrics_dir") / f"{exp.lower()}_validation_metrics.json").write_text(
                json.dumps({"experiment_id": exp, "split": "validation", "model_version": version, "metrics": m},
                           indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
