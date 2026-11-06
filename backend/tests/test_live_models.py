"""Opt-in tests against the REAL running stack (real artifacts, real Spark service).

    RUN_LIVE=1 bash scripts/test_all.sh        (server must be running)

Checks consistency between API responses, registered manifests and the database.
They do not assert particular sentiment outputs (that would be testing the model, not the
software), only contracts that must hold for every real prediction.
"""
import json
import os
import urllib.request
from pathlib import Path

import pytest

BASE = os.environ.get("HS_BASE_URL", "http://127.0.0.1:8000")
ROOT = Path(__file__).resolve().parents[2]


def _get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def _up():
    try:
        return _get("/health/live")["status"] == "alive"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason=f"no running server at {BASE}")


def _models():
    return {m["model_id"]: m for m in _get("/api/v1/models")["models"]}


@pytest.mark.parametrize("mid", ["e1", "e2", "e3", "e4"])
def test_real_prediction_matches_registered_artifact(mid):
    m = _models()[mid]
    if not m["available"]:
        pytest.skip(f"{mid}: {m['unavailable_reason']}")
    manifest = json.loads((ROOT / "artifacts" / "models" / mid / "v2-clean-split" / "manifest.json").read_text())
    d = _post("/api/v1/predict", {"text": "movie toh acchi thi but ending bakwas thi", "model": mid})
    assert d["model_used"] == mid and d["model_version"] == manifest["version"]
    assert list(d["scores"]) == ["negative", "neutral", "positive"]
    assert abs(sum(d["scores"].values()) - 1) < 1e-6
    assert d["prediction"] == max(d["scores"], key=d["scores"].get)  # bars never contradict the class
    assert d["model_input"] == "movi toh acchi thi end bakwa thi"
    hist = _get(f"/api/v1/analyses?model={mid}&page_size=1")["items"][0]
    assert hist["request_id"] == d["request_id"] and hist["model_version"] == d["model_version"]


def test_real_compare_same_input_independent_results():
    ms = _models()
    if not (ms["e3"]["available"] and ms["e4"]["available"]):
        pytest.skip("E3 and E4 both required")
    d = _post("/api/v1/predict/compare", {"text": "khana bilkul accha nahi tha"})
    e3, e4 = d["results"]["e3"], d["results"]["e4"]
    assert e3["model_version"].startswith("e3-") and e4["model_version"].startswith("e4-")
    assert e3["model_input"] == e4["model_input"] == d["model_input"]
    assert d["agreement"] == (e3["prediction"] == e4["prediction"])


def test_auto_resolves_to_preferred_when_ready():
    ms = _models()
    auto = _get("/api/v1/models")["auto"]
    if ms["e4"]["available"]:
        assert auto["resolves_to"] == "e4" and auto["warnings"] == []
    d = _post("/api/v1/predict", {"text": "bahut badhiya", "model": "auto"})
    assert d["model_requested"] == "auto" and d["model_used"] == auto["resolves_to"]
