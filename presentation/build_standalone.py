"""Build a fully self-contained, OFFLINE deck.

Inlines every local image (<img src>), stylesheet (<link href=...css>) and
script (<script src=...>) into the single output HTML, so that
hcc_dual_output_talk.html opens and runs with no network and no side files.

Input : slides_src.html                 (editable; references images + vendor/)
Output: hcc_dual_output_talk.html       (self-contained)
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

REMOTE = ("http://", "https://", "data:", "//")


_URI_CACHE: dict = {}


def data_uri(path: pathlib.Path) -> str:
    key = str(path)
    hit = _URI_CACHE.get(key)
    if hit is not None:
        return hit
    mime = MIME.get(path.suffix.lower(), "application/octet-stream")
    uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    _URI_CACHE[key] = uri
    return uri


def main() -> None:
    base = pathlib.Path(__file__).resolve().parent
    src = base / "slides_src.html"
    out = base / "hcc_dual_output_talk.html"
    html = src.read_text(encoding="utf-8")
    missing = []

    # 1) images -> base64 data URI
    def img_repl(m: re.Match) -> str:
        prefix, path, suffix = m.group(1), m.group(2), m.group(3)
        if path.startswith(REMOTE):
            return m.group(0)
        fp = (base / path).resolve()
        if not fp.exists():
            missing.append(path)
            return m.group(0)
        return f"{prefix}{data_uri(fp)}{suffix}"
    html = re.sub(r'(<img[^>]*?\ssrc=")([^"]+)(")', img_repl, html)

    # 1b) SVG <image href="..."> -> base64 data URI (inline SVG figures stay offline)
    html = re.sub(r'(<image[^>]*?\s(?:xlink:)?href=")([^"]+)(")', img_repl, html)

    # 2) local stylesheets -> inline <style>
    def css_repl(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(REMOTE):
            return m.group(0)
        fp = (base / href).resolve()
        if not fp.exists():
            missing.append(href)
            return m.group(0)
        return f"<style>\n{fp.read_text(encoding='utf-8')}\n</style>"
    html = re.sub(r'<link[^>]*?\shref="([^"]+\.css)"[^>]*>', css_repl, html)

    # 3) local scripts -> inline <script>
    def js_repl(m: re.Match) -> str:
        s = m.group(1)
        if s.startswith(REMOTE):
            return m.group(0)
        fp = (base / s).resolve()
        if not fp.exists():
            missing.append(s)
            return m.group(0)
        return f"<script>\n{fp.read_text(encoding='utf-8')}\n</script>"
    html = re.sub(r'<script[^>]*?\ssrc="([^"]+)"[^>]*>\s*</script>', js_repl, html)

    out.write_text(html, encoding="utf-8", newline="\n")  # keep LF: avoid whole-file CRLF churn
    print(f"[build] wrote {out.name}  ({len(html):,} chars)")
    if missing:
        print(f"[build] WARNING missing: {missing}")


if __name__ == "__main__":
    main()
