"""Render the deck to PDF using the browser PRINT engine + print stylesheet.

This does NOT screenshot: Chrome prints the HTML, so the output is a real
(text/vector) PDF, one 1280x800 slide per page, driven entirely by the deck's
`@media print` rules (the page flips into the reveal print layout by adding the
'print-pdf' class at print time -- see slides_src.html).

Run build_standalone.py first so hcc_dual_output_talk.html is self-contained.
"""
import os
import pathlib
import shutil
import subprocess
import tempfile
import time

CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str:
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    for name in ("chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Chrome/Edge not found")


def trim_trailing_blank(pdf: pathlib.Path) -> None:
    try:
        import pymupdf
    except Exception:
        return
    doc = pymupdf.open(str(pdf))
    first = doc.load_page(0)
    if not (first.get_text().strip() or first.get_images()):
        doc.close()
        return
    while doc.page_count > 1:
        page = doc.load_page(doc.page_count - 1)
        if page.get_text().strip() or page.get_images():
            break
        doc.delete_page(doc.page_count - 1)
    tmp = pdf.with_suffix(".tmp.pdf")
    doc.save(str(tmp))
    doc.close()
    for _ in range(15):
        try:
            os.replace(str(tmp), str(pdf))
            return
        except PermissionError:
            time.sleep(0.4)
    print("WARNING: trimmed PDF could not replace the locked file")


def main() -> None:
    base = pathlib.Path(__file__).resolve().parent
    html = base / "hcc_dual_output_talk.html"
    pdf = base / "hcc_dual_output_talk.pdf"
    profile = tempfile.mkdtemp(prefix="deck_print_")

    # 'printpdf' flips the deck into its reveal print layout deterministically
    url = html.resolve().as_uri() + f"?printpdf&v={int(time.time() * 1000)}"
    args = [
        find_browser(), "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000", f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf}", url,
    ]
    try:
        subprocess.run(args, check=False)
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    if pdf.exists():
        time.sleep(0.6)
        trim_trailing_blank(pdf)
        print(f"Successfully generated {pdf}")
    else:
        print("PDF generation failed")


if __name__ == "__main__":
    main()
