"""End-to-end acceptance journey (spec §15.5) against a RUNNING stack.

    python scripts/e2e_acceptance.py [--base http://127.0.0.1:8000] [--phase before|after-restart]

Records real responses to artifacts/logs/e2e_acceptance_<phase>.json and takes UI screenshots.
Exit code 0 only if every check passes.
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENTENCE = "movie toh acchi thi but ending bakwas thi"


def call(base, path, body=None, raw=False):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
        return data if raw else json.loads(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--phase", default="before")
    a = ap.parse_args()
    out, checks = {"phase": a.phase, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, []

    def check(name, cond):
        checks.append({"check": name, "ok": bool(cond)})
        print(("PASS " if cond else "FAIL ") + name)

    models = {m["model_id"]: m for m in call(a.base, "/api/v1/models")["models"]}
    db_path = ROOT / "data" / "hinglishsense.db"

    if a.phase == "after-restart":
        prev = json.loads((ROOT / "artifacts/logs/e2e_acceptance_before.json").read_text(encoding="utf-8"))
        rid = prev["e4"]["request_id"]
        hist = call(a.base, "/api/v1/analyses?page_size=100")
        found = [h for h in hist["items"] if h["request_id"] == rid]
        out["persisted_record"] = found[0] if found else None
        check("E4 prediction from before restart is still in history", found)
        check("persisted model version/prediction unchanged",
              found and found[0]["model_version"] == prev["e4"]["model_version"]
              and found[0]["prediction"] == prev["e4"]["prediction"])
        job = call(a.base, f"/api/v1/batch/jobs/{prev['batch']['id']}")
        check("batch job persisted and still downloadable", job["status"] == "completed" and job["download_url"])
        again = call(a.base, "/api/v1/predict", {"text": SENTENCE, "model": "e4"})
        check("E4 gives the same prediction after restart (deterministic model)",
              again["prediction"] == prev["e4"]["prediction"]
              and abs(again["scores"]["negative"] - prev["e4"]["scores"]["negative"]) < 1e-4)
        out["e4_after_restart"] = again
    else:
        for mid in ("e4", "e3"):
            r = call(a.base, "/api/v1/predict", {"text": SENTENCE, "model": mid})
            out[mid] = r
            manifest = json.loads((ROOT / f"artifacts/models/{mid}/v2-clean-split/manifest.json").read_text())
            check(f"{mid.upper()} served by the registered fine-tuned artifact ({manifest['version']})",
                  r["model_used"] == mid and r["model_version"] == manifest["version"]
                  and models[mid]["status"]["runtime"]["loaded"])
            check(f"{mid.upper()} scores labelled calibrated (temperature fitted on validation)",
                  r["score_type"] == "calibrated_probability" and r["confidence"] is not None)
        cmp_ = call(a.base, "/api/v1/predict/compare", {"text": SENTENCE, "models": ["e3", "e4"]})
        out["compare"] = cmp_
        check("compare returns both models on the same model input",
              cmp_["results"]["e3"]["status"] == cmp_["results"]["e4"]["status"] == "ok"
              and cmp_["results"]["e3"]["model_input"] == cmp_["results"]["e4"]["model_input"])
        check("compare results equal the single-model results",
              cmp_["results"]["e4"]["prediction"] == out["e4"]["prediction"]
              and cmp_["results"]["e3"]["prediction"] == out["e3"]["prediction"])

        # Batch via multipart
        boundary = "hsBoundary7"
        csv = ("text\n" + SENTENCE + "\nyeh movie bahut acchi thi yaar\n\"kal office jaana hai, normal din hai\"\n"
               "\n\"khana bilkul accha nahi tha, kabhi mat jaana\"\n")
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"accept.csv\"\r\n"
                f"Content-Type: text/csv\r\n\r\n{csv}\r\n--{boundary}\r\nContent-Disposition: form-data; "
                f"name=\"model\"\r\n\r\ne4\r\n--{boundary}--\r\n").encode()
        req = urllib.request.Request(a.base + "/api/v1/predict/batch", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        job = json.loads(urllib.request.urlopen(req, timeout=180).read())
        out["batch"] = {k: job[k] for k in ("id", "status", "row_count", "processed_count", "failed_count",
                                            "model_used", "model_version", "summary")}
        check("batch CSV processed (4 ok, 1 empty row skipped)",
              job["status"] == "completed" and job["processed_count"] == 4 and job["failed_count"] == 1)
        first = job["rows"][0]
        check("batch row 1 equals the single E4 prediction", first["prediction"] == out["e4"]["prediction"])
        dl = call(a.base, job["download_url"], raw=True).decode("utf-8")
        check("batch CSV downloadable", dl.startswith("row,text,status"))

        con = sqlite3.connect(db_path)
        row = con.execute("select model_used, model_version, prediction, score_type, normalized_text from "
                          "analysis_history where request_id=? and source='single' and model_used='e4'",
                          (out["e4"]["request_id"],)).fetchone()
        con.close()
        out["db_row_e4"] = row
        check("DB row matches API response (model, version, prediction, score type)",
              row and row[:4] == ("e4", out["e4"]["model_version"], out["e4"]["prediction"], out["e4"]["score_type"]))
        check("raw text not stored (ALLOW_HISTORY_STORAGE=false)", row and row[4] is None)

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b = p.chromium.launch(channel="msedge", headless=True)
                pg = b.new_page(viewport={"width": 1280, "height": 900})
                pg.goto(a.base + "/", wait_until="networkidle")
                pg.wait_for_function("document.querySelector('[data-model-status=\"e4\"]').textContent !== 'Checking…'")
                pg.fill("#inputText", SENTENCE)
                pg.click("#modelTrigger"); pg.click("#opt-e4")
                pg.click("#analyzeBtn")
                pg.wait_for_selector("#stateDone:not([hidden])", timeout=120000)
                ui = {"badge": pg.inner_text("#sentimentBadge"), "model": pg.inner_text("#modelUsed"),
                      "version": pg.inner_text("#modelVersion"), "score_label": pg.inner_text("#confidenceText")}
                out["ui_e4"] = ui
                check("UI shows the same E4 prediction and version as the API",
                      ui["badge"].lower() == out["e4"]["prediction"] and ui["version"] == out["e4"]["model_version"])
                pg.locator("#result").screenshot(path=str(ROOT / "artifacts/logs/ui_e4_result.png"))
                pg.click("#compareBtn")
                pg.wait_for_selector(".compare-card", timeout=120000)
                pg.locator("#compare").screenshot(path=str(ROOT / "artifacts/logs/ui_compare.png"))
                pg.goto(a.base + "/#research", wait_until="networkidle")
                pg.wait_for_timeout(1500)
                pg.locator("#research").screenshot(path=str(ROOT / "artifacts/logs/ui_research.png"))
                pg.locator("#research-insight").screenshot(path=str(ROOT / "artifacts/logs/ui_insight.png"))
                b.close()
        except Exception as e:  # recorded, counts as failure
            check(f"browser journey ({type(e).__name__}: {e})", False)

    out["checks"] = checks
    (ROOT / f"artifacts/logs/e2e_acceptance_{a.phase}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                                                         encoding="utf-8")
    ok = all(c["ok"] for c in checks)
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
