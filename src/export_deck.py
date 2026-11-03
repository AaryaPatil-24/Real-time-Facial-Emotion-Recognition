"""Export the HTML slide deck to PNGs + a 16:9 PDF using headless Chromium.

Requires: pip install playwright && playwright install chromium
Usage:    python -m src.export_deck
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .config import PRESENTATION_DIR


def main() -> None:
    from playwright.sync_api import sync_playwright

    index = PRESENTATION_DIR / "index.html"
    if not index.exists():
        raise SystemExit("presentation/index.html not found — run `python -m src.build_deck` first.")

    slides_dir = PRESENTATION_DIR / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    url = index.resolve().as_uri() + "?export=1"
    images: list[Image.Image] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=2)
        page.goto(url, wait_until="load")
        page.wait_for_function("window.deckReady === true")
        page.evaluate("() => document.fonts.ready.then(() => true)")
        page.evaluate(
            """async () => {
                await Promise.all(Array.from(document.images).map(img =>
                    img.complete ? Promise.resolve() : new Promise(r => { img.onload = r; img.onerror = r; })
                ));
            }"""
        )
        page.wait_for_timeout(1000)
        count = int(page.evaluate("window.slideCount"))
        print(f"Exporting {count} slides ...")
        for i in range(count):
            page.evaluate(f"window.showSlide({i}, false)")
            page.wait_for_timeout(220)
            png = page.screenshot(type="png")
            img = Image.open(io.BytesIO(png)).convert("RGB")
            img.save(slides_dir / f"slide-{i + 1:02d}.png")
            images.append(img)
            print(f"  {i + 1:02d}/{count}")
        browser.close()

    pdf = PRESENTATION_DIR / "deck.pdf"
    images[0].save(
        pdf,
        "PDF",
        resolution=192.0,
        save_all=True,
        append_images=images[1:],
    )
    print(f"Wrote {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")

    # Contact sheet for a quick visual overview / README.
    cols, thumb_w, thumb_h, gap = 5, 320, 180, 12
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb_w + gap) + gap, rows * (thumb_h + gap) + gap), (7, 12, 23))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(img.resize((thumb_w, thumb_h), Image.LANCZOS), (gap + c * (thumb_w + gap), gap + r * (thumb_h + gap)))
    sheet.save(PRESENTATION_DIR / "contact_sheet.png")
    print(f"Wrote {PRESENTATION_DIR / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
