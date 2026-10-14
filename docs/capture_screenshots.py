"""Capture app screenshots into docs/figures/ for the write-up and video.

Run with the backend serving the built frontend:  python docs/capture_screenshots.py [http://127.0.0.1:8000]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
LOCAL = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def launch(p):
    try:
        return p.chromium.launch(args=ARGS)
    except Exception:
        exe = next(e for e in LOCAL if Path(e).exists())
        return p.chromium.launch(executable_path=exe, args=ARGS)


def settle(page):
    page.wait_for_function("() => window.__cardiomap && window.__cardiomap.arteryColors().LAD")
    page.wait_for_timeout(800)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = launch(p)
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        pg.goto(URL)
        pg.wait_for_selector("#preset")
        settle(pg)
        pg.screenshot(path=OUT / "01_patient_defaults.png")

        pg.select_option("#preset", "multi_vessel")
        settle(pg)
        pg.click(".vessel-row >> text=LAD")
        pg.wait_for_timeout(600)
        pg.screenshot(path=OUT / "02_sample_c_lad_selected.png")
        pg.locator(".vessels").screenshot(path=OUT / "03_vessel_rows.png")
        pg.locator(".shap-overlay").screenshot(path=OUT / "04_shap_lad.png")

        pg.select_option("#preset", "lcx_rca")
        settle(pg)
        pg.click(".vessel-row >> text=RCA")
        pg.wait_for_timeout(600)
        pg.locator(".col-scene").screenshot(path=OUT / "05_heart_rca_territory.png")

        pg.click("text=Model card")
        pg.wait_for_selector(".model-card")
        pg.screenshot(path=OUT / "06_model_card_full.png", full_page=True)
        pg.locator(".chart").nth(0).screenshot(path=OUT / "07_vessel_auc_chart.png")
        pg.locator(".chart").nth(1).screenshot(path=OUT / "08_ablation_chart.png")

        m = b.new_page(viewport={"width": 375, "height": 812}, device_scale_factor=2, is_mobile=True, has_touch=True)
        m.goto(URL)
        m.wait_for_selector("#preset")
        m.select_option("#preset", "single_lad")
        settle(m)
        m.screenshot(path=OUT / "09_mobile.png", full_page=False)
        b.close()
    for f in sorted(OUT.glob("*.png")):
        print(f.relative_to(ROOT))


if __name__ == "__main__":
    main()
