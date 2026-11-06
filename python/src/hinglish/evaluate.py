"""Shared evaluation used by both branches at the aggregation stage.

The Spark branch computes its own metrics natively in Scala (see
spark/src/main/scala/hinglish/evaluation/EvaluationUtils.scala) and writes
them as JSON in the SAME schema produced here, so that
``build_comparison_table`` can consume results from all four experiments
uniformly regardless of which language produced them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

METRICS_SCHEMA_VERSION = 1


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels_order: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels_order": self.labels_order,
        }


def compute_classification_metrics(
    y_true: list[int], y_pred: list[int], id_to_label: dict[int, str]
) -> ClassificationMetrics:
    labels_sorted = sorted(id_to_label.keys())
    labels_order = [id_to_label[i] for i in labels_sorted]

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    per_class = {
        id_to_label[labels_sorted[i]]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels_sorted))
    }

    return ClassificationMetrics(
        accuracy=acc,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        labels_order=labels_order,
    )


def save_confusion_matrix_plot(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_comparison_table(metrics_dir: Path) -> pd.DataFrame:
    """Scan artifacts/metrics/*.json (written by E1-E4, any language) and
    build one machine-generated comparison table. Never hand-edit results
    into this table — it must be fully derived from saved artifacts."""
    rows = []
    # Only locked-test metrics (validation / Spark cross-check JSONs live in the
    # same folder and must never be mixed into the comparison).
    # The full report (with not-run rows, CIs, error analysis) is hinglish.report.
    for path in sorted(Path(metrics_dir).glob("e[1-4]_test_metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "experiment": data.get("experiment_id", path.stem),
                "accuracy": data.get("metrics", data).get("accuracy"),
                "macro_precision": data.get("metrics", data).get("macro_precision"),
                "macro_recall": data.get("metrics", data).get("macro_recall"),
                "macro_f1": data.get("metrics", data).get("macro_f1"),
                "weighted_f1": data.get("metrics", data).get("weighted_f1"),
                "runtime_seconds": data.get("runtime_seconds"),
                "source_file": str(path),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return df
