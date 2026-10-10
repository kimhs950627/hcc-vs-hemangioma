"""Render the self-contained reveal.js deck to PDF via headless Chrome/Edge.

Run build_standalone.py first so hcc_dual_output_talk.html embeds all images.
"""
import os
import pathlib
import shutil
import subprocess

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


def main() -> None:
    base = pathlib.Path(__file__).resolve().parent
    html = base / "hcc_dual_output_talk.html"
    pdf = base / "hcc_dual_output_talk.pdf"
    profile = base / "._chrome_profile"

    url = html.resolve().as_uri() + "?print-pdf"
    args = [
        find_browser(),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf}",
        url,
    ]
    subprocess.run(args, check=False)
    print(f"Successfully generated {pdf}" if pdf.exists() else "PDF generation failed")


if __name__ == "__main__":
    main()
