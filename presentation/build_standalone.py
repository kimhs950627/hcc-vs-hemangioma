"""Inline every local <img> into base64 data URIs to produce a self-contained deck.

Input : slides_src.html   (editable, images referenced by relative path)
Output: hcc_dual_output_talk.html   (single file, no figures/ folder needed)
"""
import base64
import pathlib
import re

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


def data_uri(path: pathlib.Path) -> str:
    mime = MIME.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> None:
    base = pathlib.Path(__file__).resolve().parent
    src = base / "slides_src.html"
    out = base / "hcc_dual_output_talk.html"
    html = src.read_text(encoding="utf-8")

    missing = []

    def repl(match: re.Match) -> str:
        prefix, path, suffix = match.group(1), match.group(2), match.group(3)
        if path.startswith(("http://", "https://", "data:", "//")):
            return match.group(0)
        fp = (base / path).resolve()
        if not fp.exists():
            missing.append(path)
            return match.group(0)
        return f"{prefix}{data_uri(fp)}{suffix}"

    html = re.sub(r'(<img[^>]*?\ssrc=")([^"]+)(")', repl, html)
    out.write_text(html, encoding="utf-8")

    print(f"[build] wrote {out.name}  ({len(html):,} chars)")
    if missing:
        print(f"[build] WARNING missing images: {missing}")


if __name__ == "__main__":
    main()
