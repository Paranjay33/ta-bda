"""Run-manifest construction.

Every experiment (E1-E4, scalability runs) must emit a manifest capturing
what actually happened: seed, dataset identity/hash, model name/revision,
library versions, hyperparameters, runtime and metrics. Everything here is
read from the live environment/config — nothing is hard-coded or invented,
per the build spec's academic-integrity requirement.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def capture_environment(root: Path) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(root),
    }
    for pkg in [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "evaluate",
        "scikit-learn",
        "pandas",
        "numpy",
        "pyspark",
    ]:
        v = _pkg_version(pkg)
        if v is not None:
            env[f"{pkg}_version"] = v

    try:
        import torch

        env["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch_cuda_available"] = None

    return env


@dataclass
class RunManifest:
    experiment_id: str
    seed: int
    dataset_identity: dict[str, Any]
    model_name: str | None
    model_revision: str | None
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    created_at_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "dataset_identity": self.dataset_identity,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "environment": self.environment,
            "notes": self.notes,
            "created_at_unix": self.created_at_unix,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)


class Timer:
    """Small context manager for wall-clock runtime capture used in manifests."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed: float | None = None
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start
