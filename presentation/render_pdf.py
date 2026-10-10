"""Render the deck to a clean 15-page PDF: one screenshot per slide.

Rendering each slide with headless Chrome and assembling the PNGs avoids every
reveal.js print-CSS pitfall (wrong page size, clipped slides, centered overflow):
every slide becomes exactly one 1280x800 page showing exactly what is on screen.
Run build_standalone.py first so hcc_dual_output_talk.html embeds all images.
"""
import pathlib
import shutil
import subprocess
import time

W, H = 1280, 800
N_SLIDES = 15

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    import os
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    for name in ("chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Chrome/Edge not found")


def main() -> None:
    base = pathlib.Path(__file__).resolve().parent
    html = base / "hcc_dual_output_talk.html"
    pdf = base / "hcc_dual_output_talk.pdf"
    shots = base / "_shots"
    profile = base / "._chrome_shot"
    shutil.rmtree(shots, ignore_errors=True)
    shutil.rmtree(profile, ignore_errors=True)
    shots.mkdir()

    browser = find_browser()
    url = html.resolve().as_uri() + f"?pdfshot&v={int(time.time() * 1000)}"
    pngs = []
    for i in range(N_SLIDES):
        png = shots / f"slide_{i:02d}.png"
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--force-device-scale-factor=1",
             f"--window-size={W},{H}", "--virtual-time-budget=8000",
             f"--user-data-dir={profile}", f"--screenshot={png}", f"{url}#/{i}"],
            check=False,
        )
        pngs.append(png)

    try:
        import pymupdf
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"pymupdf required to assemble the PDF: {e}")

    doc = pymupdf.open()
    missing = []
    for png in pngs:
        if not png.exists():
            missing.append(png.name)
            continue
        page = doc.new_page(width=W, height=H)
        page.insert_image(pymupdf.Rect(0, 0, W, H), filename=str(png))
    n_pages = doc.page_count
    doc.save(str(pdf))
    doc.close()

    shutil.rmtree(shots, ignore_errors=True)
    shutil.rmtree(profile, ignore_errors=True)
    print(f"Successfully generated {pdf} ({n_pages} pages)")
    if missing:
        print("WARNING missing screenshots:", missing)


if __name__ == "__main__":
    main()
