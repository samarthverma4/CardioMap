"""Render video/film.html frame by frame (deterministic time) with headless Brave/Chrome on the GPU.

Usage:  python video/render.py stills 2.5,9,15 OUTDIR     -> PNG stills at given times
        python video/render.py frames OUTDIR [fps] [start] [end]  -> JPEG frames
Requires a static server at http://127.0.0.1:8123 rooted at the repo (python -m http.server 8123).
"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8123/video/film.html"
EXE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"


def open_page(p):
    b = p.chromium.launch(executable_path=EXE, args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.goto(URL)
    pg.wait_for_function("window.ready === true", timeout=60000)
    return b, pg, errs


def main():
    mode, out = sys.argv[1], Path(sys.argv[3] if sys.argv[1] == "stills" else sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b, pg, errs = open_page(p)
        if mode == "stills":
            for t in [float(x) for x in sys.argv[2].split(",")]:
                pg.evaluate(f"window.renderFrame({t})")
                pg.screenshot(path=str(out / f"still_{t:05.2f}.png"))
        else:
            fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
            start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            end = int(sys.argv[5]) if len(sys.argv) > 5 else 60 * fps
            t0 = time.time()
            for f in range(start, end):
                pg.evaluate(f"window.renderFrame({f / fps})")
                pg.screenshot(path=str(out / f"f{f:05d}.jpg"), type="jpeg", quality=93)
                if f % 150 == 0:
                    print(f"frame {f}/{end}  {time.time() - t0:.0f}s", flush=True)
        print("errors:", errs[:5])
        b.close()


if __name__ == "__main__":
    main()
