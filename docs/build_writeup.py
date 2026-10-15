"""Render docs/writeup.md to docs/writeup.pdf (A4) with headless Chromium.

Run:  python docs/build_writeup.py
"""
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent
LOCAL = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
CSS = """
@page { size: A4; margin: 16mm 16mm 18mm; }
body { font: 9.6pt/1.38 'Public Sans', 'Helvetica Neue', Arial, sans-serif; color: #15202a; }
h1 { font-size: 19pt; margin: 0 0 2pt; letter-spacing: -0.02em; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ccd5db; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; }
h2, h3 { break-after: avoid; }
p, li { margin: 0 0 5pt; }
blockquote { margin: 6pt 0; padding: 5pt 8pt; background: #fff4dc; border-left: 3px solid #e3b84f; color: #5b3d00; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.6pt; break-inside: avoid; }
th, td { border-bottom: 1px solid #ccd5db; padding: 3pt 5pt; text-align: left; }
th { color: #4a5864; }
code { font: 8.6pt 'IBM Plex Mono', Menlo, monospace; }
figure, img { max-width: 100%; break-inside: avoid; }
img { display: block; margin: 4pt auto 2pt; max-height: 60mm; }
.cap { font-size: 8.4pt; color: #4a5864; text-align: center; margin: 0 0 8pt; }
"""


def main():
    md = (DOCS / "writeup.md").read_text()
    body = markdown.markdown(md, extensions=["tables"])
    # turn each image's alt text into a visible caption
    import re
    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" /></p>',
                  r'<figure><img alt="\1" src="\2" /><p class="cap">\1</p></figure>', body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    tmp = DOCS / "_writeup.html"
    tmp.write_text(html)
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception:
            b = p.chromium.launch(executable_path=next(e for e in LOCAL if Path(e).exists()))
        pg = b.new_page()
        pg.goto(tmp.as_uri())
        pg.wait_for_load_state("networkidle")
        pg.pdf(path=str(DOCS / "writeup.pdf"), format="A4", print_background=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#6f7d88'>"
                               "CardioMap · <span class='pageNumber'></span>/<span class='totalPages'></span></div>",
               margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        b.close()
    tmp.unlink()
    print("wrote", DOCS / "writeup.pdf")


if __name__ == "__main__":
    main()
