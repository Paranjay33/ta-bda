"""Browser tests (Playwright, using the installed Microsoft Edge / Chromium).

Run against a running stack:
    bash scripts/start_local.sh            # in another terminal
    RUN_UI=1 bash scripts/test_all.sh      # or: cd backend && pytest tests/test_ui.py
Base URL: HS_BASE_URL (default http://127.0.0.1:8000). Skipped if unreachable.
"""
import os
import urllib.request

import pytest

BASE = os.environ.get("HS_BASE_URL", "http://127.0.0.1:8000")


def _up() -> bool:
    try:
        return urllib.request.urlopen(BASE + "/health/live", timeout=3).status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason=f"no HinglishSense server at {BASE}")


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        channel = os.environ.get("HS_BROWSER_CHANNEL", "msedge")
        try:
            b = p.chromium.launch(channel=channel, headless=True)
        except Exception:
            b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.goto(BASE + "/", wait_until="networkidle")
    pg.wait_for_function("document.querySelector('[data-model-status=\"e1\"]').textContent !== 'Checking…'")
    yield pg
    ctx.close()


def _available(page):
    return page.evaluate("Object.keys(window.HS_UI.registry.models).filter(k => window.HS_UI.registry.models[k].available)")


def test_page_loads_with_design_and_no_console_errors(page):
    assert page.title().startswith("HinglishSense")
    assert "Understand What Your Hinglish" in page.inner_text("h1")
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(250, 248, 243)"  # supplied cream background preserved
    assert page.errors == []


def test_counter_clear_and_chips(page):
    page.fill("#inputText", "abc")
    assert page.inner_text("#charCount") == "3/500"
    page.click("#clearBtn")
    assert page.input_value("#inputText") == "" and page.inner_text("#charCount") == "0/500"
    page.click(".chip >> nth=0")
    assert page.input_value("#inputText") == "yeh movie bahut acchi thi yaar"
    assert page.locator("#stateWaiting").is_visible()  # chips never auto-run analysis


def test_empty_input_shows_validation(page):
    page.click("#analyzeBtn")
    assert page.locator("#inputError").is_visible()


def test_model_status_comes_from_api(page):
    models = page.evaluate("window.HS_UI.registry.models")
    for mid, m in models.items():
        disabled = page.get_attribute(f"#opt-{mid}", "aria-disabled")
        assert disabled == ("false" if m["available"] else "true")


def test_listbox_keyboard(page):
    page.focus("#modelTrigger")
    page.keyboard.press("ArrowDown")
    assert page.get_attribute("#modelTrigger", "aria-expanded") == "true"
    page.keyboard.press("Escape")
    assert page.get_attribute("#modelTrigger", "aria-expanded") == "false"


@pytest.mark.parametrize("model", ["e1", "e2", "e3", "e4"])
def test_live_prediction_renders(page, model):
    if model not in _available(page):
        pytest.skip(f"{model} not available in this deployment")
    page.fill("#inputText", "movie toh acchi thi but ending bakwas thi")
    page.click("#modelTrigger")
    page.click(f"#opt-{model}")
    page.click("#analyzeBtn")
    page.wait_for_selector("#stateDone:not([hidden])", timeout=120000)
    assert page.inner_text("#sentimentBadge") in {"Positive", "Neutral", "Negative"}
    assert page.inner_text("#modelVersion").startswith(model)
    assert "ms" in page.inner_text("#processingTime")
    label = page.inner_text("#scoreTypeLabel")
    assert ("Calibrated probability" in label) if model != "e1" else ("not calibrated" in label)
    total = sum(float(page.inner_text(f"#val{k}").rstrip("%")) for k in ("Positive", "Neutral", "Negative"))
    assert abs(total - 100) < 0.5
    assert page.errors == []


def test_compare_mode(page):
    if not {"e3", "e4"} & set(_available(page)):
        pytest.skip("neither E3 nor E4 available")
    page.fill("#inputText", "service theek thi lekin delivery bahut late aayi")
    page.click("#compareBtn")
    page.wait_for_selector(".compare-card", timeout=120000)
    assert page.locator(".compare-card").count() == 2
    assert page.inner_text("#compareAgreement") != ""


def test_unavailable_model_error_state(page):
    """Force a 503 through the real API by requesting a model that the server reports unavailable."""
    unavailable = [m for m in ("e1", "e2", "e3", "e4") if m not in _available(page)]
    if not unavailable:
        pytest.skip("all models available")
    page.fill("#inputText", "acchi")
    page.evaluate(f"""async () => {{
        try {{ await window.HS_API.post('/predict', {{text: 'acchi', model: '{unavailable[0]}'}}); }}
        catch (e) {{ window.__err = e.status; }} }}""")
    assert page.evaluate("window.__err") == 503


def test_batch_upload_and_download(page, tmp_path):
    if not _available(page):
        pytest.skip("no model available")
    csv = tmp_path / "batch.csv"
    csv.write_text("text\nyeh movie bahut acchi thi\nbilkul bakwas film\n\n", encoding="utf-8")
    page.set_input_files("#batchFile", str(csv))
    page.click("#batchSubmit")
    page.wait_for_selector("#batchDownload:not([hidden])", timeout=120000)
    assert "completed" in page.inner_text("#batchStatus")
    with page.expect_download() as dl:
        page.click("#batchDownload")
    content = open(dl.value.path(), encoding="utf-8").read()
    assert content.startswith("row,text,status")


def test_mobile_layout_has_no_horizontal_overflow(browser):
    ctx = browser.new_context(viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    pg.goto(BASE + "/", wait_until="networkidle")
    assert pg.evaluate("document.documentElement.scrollWidth") <= 390
    pg.click("#navToggle")
    assert pg.locator("#navMobile").is_visible()
    ctx.close()
