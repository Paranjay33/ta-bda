"""Score calibration + probabilistic metrics, shared by training, evaluation
and the serving layer.

* Temperature scaling (E3/E4): one scalar T > 0 fitted on VALIDATION logits by
  minimising NLL. argmax is unchanged; only the score scale is affected.
* Temperature scaling on OvR margins (E2): Spark's OneVsRest(LinearSVC) emits
  one raw margin per class, which are NOT probabilities. softmax(m / T) with T
  fitted on VALIDATION margins (argmax-preserving). A multinomial Platt map
  (fit_margin_platt) is kept for comparison only; it can change the argmax.
* Nothing here is ever fitted on the test split.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def probabilistic_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    return {
        "log_loss": nll(probs, y),
        "brier_multiclass": brier_multiclass(probs, y),
        "ece_15_bins": expected_calibration_error(probs, y),
    }


@dataclass
class TemperatureCalibration:
    method: str
    temperature: float
    fitted_on: str
    n_samples: int
    val_nll_before: float
    val_nll_after: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Always returns a 2-D (n, k) array."""
        return softmax(np.atleast_2d(np.asarray(logits, dtype=np.float64)) / self.temperature)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def fit_temperature(logits: np.ndarray, y: np.ndarray, fitted_on: str = "validation") -> TemperatureCalibration:
    """Golden-section search over log T in [-3, 3]; NLL in T is unimodal enough
    for this 1-D problem and needs no torch dependency."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    def f(log_t: float) -> float:
        return nll(softmax(logits / np.exp(log_t)), y)

    a, b = -3.0, 3.0
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    t = float(np.exp((a + b) / 2))
    return TemperatureCalibration(
        method="temperature_scaling",
        temperature=t,
        fitted_on=fitted_on,
        n_samples=int(len(y)),
        val_nll_before=f(0.0),
        val_nll_after=f(np.log(t)),
    )


@dataclass
class MarginPlattCalibration:
    """softmax(W @ margins + b) fitted on validation OvR margins."""

    method: str
    coef: list[list[float]]
    intercept: list[float]
    fitted_on: str
    n_samples: int
    val_nll_after: float

    def apply(self, margins: np.ndarray) -> np.ndarray:
        m = np.atleast_2d(np.asarray(margins, dtype=np.float64))
        return softmax(m @ np.asarray(self.coef).T + np.asarray(self.intercept))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def fit_margin_platt(margins: np.ndarray, y: np.ndarray, fitted_on: str = "validation") -> MarginPlattCalibration:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(np.asarray(margins, dtype=np.float64), np.asarray(y))
    assert list(clf.classes_) == list(range(len(clf.classes_))), "class order mismatch"
    cal = MarginPlattCalibration(
        method="multinomial_platt_on_ovr_margins",
        coef=clf.coef_.tolist(),
        intercept=clf.intercept_.tolist(),
        fitted_on=fitted_on,
        n_samples=int(len(y)),
        val_nll_after=0.0,
    )
    cal.val_nll_after = nll(cal.apply(margins), np.asarray(y))
    return cal


def load_calibration(path: Path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["method"] in ("temperature_scaling", "temperature_scaling_on_ovr_margins"):
        return TemperatureCalibration(**data)
    if data["method"] == "multinomial_platt_on_ovr_margins":
        return MarginPlattCalibration(**data)
    raise ValueError(f"unknown calibration method {data['method']}")
