"""API integration tests.

Model adapters are TEST DOUBLES (FakeAdapter) so the HTTP contract, Auto
policy, error handling, persistence and batch flow can be tested without GPU
or Spark. They exist only in this test module and are never used in serving.
Real-model behaviour is covered by tests/test_live_models.py (opt-in).
"""
import io
import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.registry import ModelEntry, ModelRegistry, RawPrediction
from app.settings import Settings


class FakeAdapter:
    def __init__(self, mid, label_id=2):
        self.mid, self.label_id, self.loaded, self.calls = mid, label_id, True, 0

    def load(self):
        pass

    def predict(self, texts, batch_size=32):
        self.calls += 1
        out = []
        for t in texts:
            lid = 0 if "bakwa" in t else self.label_id
            scores = [0.1, 0.1, 0.1]
            scores[lid] = 0.8
            out.append(RawPrediction(label_id=lid, scores=scores))
        return out

    def status(self):
        return {"runtime": "fake", "loaded": True}


class FakeRegistry(ModelRegistry):
    """Registry whose entries are fake manifests + FakeAdapters (selected ids only)."""

    def __init__(self, settings, available=("e1", "e2", "e3", "e4")):
        self._available = set(available)
        super().__init__(settings)

    def refresh(self):
        for mid in ("e1", "e2", "e3", "e4"):
            if mid in self._available:
                manifest = {"version": f"{mid}-test", "status": {"training": "completed", "selection": "validation_macro_f1"},
                            "score_semantics": "test"}
                e = ModelEntry(mid, self._run_dir(mid), manifest, None, {"metrics": {"macro_f1": 0.5}})
                e.adapter = FakeAdapter(mid)
            else:
                e = ModelEntry(mid, self._run_dir(mid), None, None, None)
            self.entries[mid] = e

    def availability(self, mid):
        return (True, None) if mid in self._available else (False, "not_trained")

    def warm_up(self):
        pass


def make_client(tmp_path, available=("e1", "e2", "e3", "e4"), **env):
    os.environ.update({"DATABASE_URL": f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                       "LOAD_MODELS_ON_STARTUP": "false", "LOG_LEVEL": "WARNING",
                       "BATCH_OUTPUT_DIR": str(tmp_path / "out"), "RATE_LIMIT_PER_MINUTE": "1000",
                       "ALLOW_HISTORY_STORAGE": "false", "AUTH_REQUIRED": "false", "SYNC_BATCH_MAX_ROWS": "100"})
    os.environ.pop("API_KEY", None)
    os.environ.update(env)
    settings = Settings.from_env()
    return TestClient(create_app(settings, FakeRegistry(settings, available)))


@pytest.fixture()
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def test_health_live_and_ready(client):
    assert client.get("/health/live").json()["status"] == "alive"
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["unavailable_models"] == []


def test_ready_reports_unavailable_models(tmp_path):
    with make_client(tmp_path, available=("e1",)) as c:
        body = c.get("/health/ready").json()
        assert set(body["unavailable_models"]) == {"e2", "e3", "e4"}


def test_models_endpoint_shape(client):
    body = client.get("/api/v1/models").json()
    ids = [m["model_id"] for m in body["models"]]
    assert ids == ["e1", "e2", "e3", "e4"]
    assert body["auto"]["resolves_to"] == "e4"
    e4 = body["models"][3]
    assert e4["score_type"] == "calibrated_probability" and e4["test_metrics"] is None


def test_predict_contract_and_canonical_scores(client):
    r = client.post("/api/v1/predict", json={"text": "Yeh movie bahut acchi thi yaar", "model": "e4"})
    assert r.status_code == 200
    d = r.json()
    assert d["model_requested"] == "e4" and d["model_used"] == "e4" and d["model_version"] == "e4-test"
    assert list(d["scores"]) == ["negative", "neutral", "positive"]
    assert d["prediction"] == "positive" and d["label_id"] == 2
    assert d["model_input"] == "yeh movi bahut acchi thi yaar"  # same dataset-style transform as serving
    assert d["confidence"] == pytest.approx(0.8)
    assert r.headers["X-Request-ID"] == d["request_id"]


def test_e1_scores_are_not_called_confidence(client):
    d = client.post("/api/v1/predict", json={"text": "acchi", "model": "e1"}).json()
    assert d["score_type"] == "model_probability" and d["confidence"] is None
    assert any("not calibrated" in w for w in d["warnings"])


def test_input_validation_errors(client):
    r = client.post("/api/v1/predict", json={"text": "   ", "model": "e4"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    r = client.post("/api/v1/predict", json={"text": "a" * 501, "model": "e4"})
    assert r.status_code == 400 and "500" in r.json()["error"]["message"]
    r = client.post("/api/v1/predict", json={"text": "ok", "model": "e9"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = client.post("/api/v1/predict", json={"model": "e4"})
    assert r.status_code == 422


def test_unavailable_model_is_503_never_substituted(tmp_path):
    with make_client(tmp_path, available=("e1", "e2")) as c:
        r = c.post("/api/v1/predict", json={"text": "acchi", "model": "e4"})
        assert r.status_code == 503 and "not_trained" in r.json()["error"]["message"]


def test_auto_fallback_is_explicit(tmp_path):
    with make_client(tmp_path, available=("e2", "e3")) as c:
        d = c.post("/api/v1/predict", json={"text": "acchi", "model": "auto"}).json()
        assert d["model_requested"] == "auto" and d["model_used"] == "e3"
        assert any("Auto fallback" in w for w in d["warnings"])


def test_auto_with_no_models_is_503(tmp_path):
    with make_client(tmp_path, available=()) as c:
        assert c.post("/api/v1/predict", json={"text": "acchi", "model": "auto"}).status_code == 503


def test_compare_full_and_partial(tmp_path):
    with make_client(tmp_path) as c:
        d = c.post("/api/v1/predict/compare", json={"text": "movie acchi thi"}).json()
        assert d["results"]["e3"]["status"] == "ok" and d["results"]["e4"]["status"] == "ok"
        assert d["results"]["e3"]["model_used"] == "e3" and d["results"]["e4"]["model_used"] == "e4"
        assert d["agreement"] is True
    with make_client(tmp_path, available=("e4",)) as c:
        d = c.post("/api/v1/predict/compare", json={"text": "movie acchi thi"}).json()
        assert d["results"]["e3"]["status"] == "unavailable" and d["results"]["e3"]["model_used"] is None
        assert d["results"]["e4"]["status"] == "ok" and d["agreement"] is None


def test_batch_csv_roundtrip_and_download(client):
    csv = "id,tweet\n1,movie acchi thi\n2,ending bakwas thi\n3,\n4," + "x" * 600 + "\n"
    r = client.post("/api/v1/predict/batch", files={"file": ("in.csv", io.BytesIO(csv.encode()), "text/csv")},
                    data={"model": "e2"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "completed" and job["text_column"] == "tweet"
    assert job["row_count"] == 4 and job["processed_count"] == 2 and job["failed_count"] == 2
    assert job["summary"]["distribution"] == {"negative": 1, "neutral": 0, "positive": 1}
    assert [row["status"] for row in job["rows"]] == ["ok", "ok", "skipped", "skipped"]  # order preserved
    dl = client.get(job["download_url"])
    assert dl.status_code == 200 and "text/csv" in dl.headers["content-type"]
    assert dl.text.splitlines()[0].startswith("row,text,status,error,prediction,label_id,score_negative")


def test_batch_blank_lines_are_reported_not_dropped(client):
    csv = 'text\nmovie acchi thi\n\n"kal office jana hai, normal din"\n'
    job = client.post("/api/v1/predict/batch", files={"file": ("b.csv", io.BytesIO(csv.encode()), "text/csv")},
                      data={"model": "e4"}).json()
    assert job["row_count"] == 3 and job["failed_count"] == 1
    assert [r["status"] for r in job["rows"]] == ["ok", "skipped", "ok"]
    assert job["rows"][2]["text"] == "kal office jana hai, normal din"  # quoted comma kept


def test_batch_errors(client):
    r = client.post("/api/v1/predict/batch", files={"file": ("in.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 400 and "No text column" in r.json()["error"]["message"]
    r = client.post("/api/v1/predict/batch", files={"file": ("in.csv", io.BytesIO(b"text\nx\n"), "text/csv")},
                    data={"text_column": "missing"})
    assert r.status_code == 400
    big = "text\n" + "\n".join(["acchi"] * 501) + "\n"
    r = client.post("/api/v1/predict/batch", files={"file": ("in.csv", io.BytesIO(big.encode()), "text/csv")})
    assert r.status_code == 400 and "limit" in r.json()["error"]["message"]
    r = client.post("/api/v1/predict/batch", files={"file": ("in.exe", io.BytesIO(b"x"), "application/octet-stream")})
    assert r.status_code == 400


def test_history_persistence_privacy_and_analytics(client):
    client.post("/api/v1/predict", json={"text": "movie acchi thi", "model": "e4"})
    client.post("/api/v1/predict", json={"text": "ending bakwas thi", "model": "e3"})
    h = client.get("/api/v1/analyses").json()
    assert h["total"] == 2 and h["text_storage_enabled"] is False
    assert all(item["normalized_text"] is None for item in h["items"])  # raw text not stored by default
    one = client.get(f"/api/v1/analyses/{h['items'][0]['id']}").json()
    assert one["model_used"] in ("e3", "e4")
    assert client.get("/api/v1/analyses?model=e3").json()["total"] == 1
    a = client.get("/api/v1/analytics/summary").json()
    assert a["total"] == 2 and a["by_prediction"] == {"negative": 1, "neutral": 0, "positive": 1}
    assert client.delete("/api/v1/analyses").json()["deleted"] == 2
    assert client.get("/api/v1/analyses").json()["total"] == 0


def test_history_text_storage_when_enabled(tmp_path):
    with make_client(tmp_path, ALLOW_HISTORY_STORAGE="true") as c:
        c.post("/api/v1/predict", json={"text": "Movie ACCHI thi", "model": "e4"})
        assert c.get("/api/v1/analyses").json()["items"][0]["normalized_text"] == "Movie ACCHI thi"


def test_auth_required(tmp_path):
    with make_client(tmp_path, AUTH_REQUIRED="true", API_KEY="s3cret") as c:
        assert c.post("/api/v1/predict", json={"text": "acchi", "model": "e4"}).status_code == 401
        r = c.post("/api/v1/predict", json={"text": "acchi", "model": "e4"}, headers={"X-API-Key": "s3cret"})
        assert r.status_code == 200
        assert c.get("/health/live").status_code == 200


def test_rate_limit(tmp_path):
    with make_client(tmp_path, RATE_LIMIT_PER_MINUTE="3") as c:
        codes = [c.post("/api/v1/predict", json={"text": "acchi", "model": "e4"}).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[-1] == 429


def test_oversized_body_is_413(tmp_path):
    with make_client(tmp_path, MAX_UPLOAD_MB="1") as c:
        r = c.post("/api/v1/predict/batch", files={"file": ("in.csv", io.BytesIO(b"text\n" + b"a" * (2 * 1024 * 1024)), "text/csv")})
        assert r.status_code == 413


def test_errors_never_leak_internals(client):
    r = client.get("/api/v1/batch/jobs/doesnotexist")
    body = r.json()
    assert r.status_code == 404 and set(body) == {"error"}
    assert "Traceback" not in r.text and "\\" not in body["error"]["message"]


def test_research_endpoints(client):
    body = client.get("/api/v1/metrics/comparison").json()
    assert [r["experiment"] for r in body["rows"]] == ["E1", "E2", "E3", "E4"]
    assert "NOT locked-test" in body["historical_midterm"]["status"]
    assert client.get("/api/v1/experiments").json()["experiments"][0]["experiment_id"] == "E1"
    assert client.get("/api/v1/experiments/e9").status_code == 404


def test_production_requires_auth(tmp_path):
    os.environ.update({"APP_ENV": "production", "AUTH_REQUIRED": "false"})
    try:
        with pytest.raises(RuntimeError):
            Settings.from_env().validate()
    finally:
        os.environ["APP_ENV"] = "development"


def test_frontend_is_served_same_origin(client):
    r = client.get("/")
    assert r.status_code == 200 and "HinglishSense" in r.text
    assert client.get("/js/app.js").status_code == 200
