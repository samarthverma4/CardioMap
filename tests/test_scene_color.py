"""End-to-end: the colour rendered on each artery mesh equals the API's scene_state colour, for every preset.

Needs the built frontend (frontend/dist) and a Chromium browser. Uses Playwright's bundled Chromium if
installed, otherwise a local Chrome/Brave/Chromium, or the path in $CARDIOMAP_BROWSER.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRESETS = json.loads((ROOT / "artifacts" / "presets.json").read_text())["presets"]
CANDIDATES = [
    os.environ.get("CARDIOMAP_BROWSER", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/chromium", "/usr/bin/google-chrome",
]

sync_api = pytest.importorskip("playwright.sync_api")
if not (ROOT / "frontend" / "dist" / "index.html").exists():
    pytest.skip("frontend not built (cd frontend && npm run build)", allow_module_level=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(url + "/api/meta", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("backend did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def page():
    with sync_api.sync_playwright() as p:
        browser = None
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except Exception:
            for exe in CANDIDATES:
                if exe and Path(exe).exists():
                    browser = p.chromium.launch(executable_path=exe, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
                    break
        if browser is None:
            pytest.skip("no Chromium browser available")
        pg = browser.new_page(viewport={"width": 1440, "height": 900})
        yield pg
        browser.close()


def _api_scene(url: str, features: dict) -> dict:
    req = urllib.request.Request(url + "/api/predict", data=json.dumps({"features": features}).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["scene_state"]


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_rendered_artery_colour_matches_api(server, page, preset):
    expected = {v: s["color"] for v, s in _api_scene(server, preset["features"]).items()}
    page.goto(server)
    page.wait_for_selector("#preset")
    page.select_option("#preset", preset["id"])
    page.wait_for_function(
        "exp => { const c = window.__cardiomap && window.__cardiomap.arteryColors();"
        " return c && Object.keys(exp).every(k => c[k] === exp[k]); }",
        arg=expected, timeout=15000,
    )
    rendered = page.evaluate("window.__cardiomap.arteryColors()")
    assert rendered == expected


def test_selected_artery_streams(server, page):
    """Selecting an artery starts the streaming-glow animation, and its clock keeps advancing."""
    page.goto(server)
    page.wait_for_selector("#preset")
    page.wait_for_function("() => window.__cardiomap && window.__cardiomap.arteryColors().LAD", timeout=15000)
    page.click(".vessel-row >> text=LAD")
    page.wait_for_function("() => (window.__cardiomapStream || 0) > 0.2", timeout=10000)
    t1 = page.evaluate("window.__cardiomapStream")
    page.wait_for_timeout(600)
    assert page.evaluate("window.__cardiomapStream") > t1
