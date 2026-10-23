"""Emit the two-step inline SVG for the "cutoff is an interval" slide.

Step 1 (always visible): the validation score distribution over the full range,
with the empty gap marked by a dashed box.

Step 2 (Reveal fragment, appears on click): a magnified view of that box.
Inside the box the classification does not change at all, so any threshold in
it gives the same performance -- that span is the equivalence interval. The
Youden panel makes the same point: J is flat at 1.0 across the span, so the
criterion can only pick an edge, and the implementation picks the right edge.

Colours come from the deck's CSS custom properties, so the figure inherits the
slide theme instead of carrying its own palette.

Run: python3 presentation/slide_explainer_svg.py
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE.parent / "cosine_probe_result" / "raw_cosprobe_effnet_supcon.csv"
OUT = HERE / "slide8_figure.svg"

W, H = 1152, 452


def load() -> dict:
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    val = [r for r in rows if r["split"] == "Val"]
    return {
        "hcc": [float(r["hcc_cosine_mean"]) for r in val if int(float(r["true_label"])) == 1],
        "hem": [float(r["hcc_cosine_mean"]) for r in val if int(float(r["true_label"])) == 0],
    }


def bins(values, lo, hi, n):
    counts = [0] * n
    step = (hi - lo) / n
    for v in values:
        if lo <= v < hi:
            counts[min(n - 1, int((v - lo) / step))] += 1
    return counts, step


def j_curve(hcc, hem, lo, hi, n=400):
    out = []
    for i in range(n + 1):
        t = lo + (hi - lo) * i / n
        sens = sum(1 for s in hcc if s >= t) / len(hcc)
        spec = sum(1 for s in hem if s < t) / len(hem)
        out.append((t, sens + spec - 1))
    return out


def main() -> None:
    d = load()
    hcc, hem = d["hcc"], d["hem"]
    hem_max, hcc_min = max(hem), min(hcc)
    cutoff = hcc_min

    g = []
    add = g.append

    def txt(x, y, s, size=13, fill="var(--muted)", weight="400", anchor="start"):
        add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>')

    add(f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
        'font-family="inherit" role="img" '
        'aria-label="검증 세트 분포에서 빈 구간을 확대하면 어떤 임계값도 같은 판정을 준다">')

    # ── step 1: full-range histogram ─────────────────────────────────────
    x0, x1, top, base = 6, 1146, 62, 232
    lo, hi = -0.35, 1.05
    xf = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)

    txt(0, 18, "① 검증 세트 — 두 무리가 떨어져 있고, 사이에 빈 공간이 있다", 21, "var(--ink)", "700")
    txt(0, 40, "HCC cosine score · 막대 = 그 구간의 영상 수 · 점 하나가 영상 한 장", 13)

    gap0, gap1 = xf(hem_max), xf(hcc_min)
    add(f'<rect x="{gap0:.1f}" y="{top}" width="{max(gap1-gap0,3):.1f}" height="{base-top}" '
        'fill="var(--caution)" opacity="0.18"/>')

    for vals, color in ((hem, "var(--hem)"), (hcc, "var(--hcc)")):
        counts, step = bins(vals, lo, hi, 96)
        peak = max(counts)
        for i, c in enumerate(counts):
            if not c:
                continue
            x = xf(lo + i * step)
            w = xf(lo + (i + 1) * step) - x
            h = c / peak * (base - top - 8)
            add(f'<rect x="{x:.1f}" y="{base-h:.1f}" width="{max(w-1,1.2):.1f}" height="{h:.1f}" '
                f'fill="{color}" opacity="0.85"/>')

    add(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="var(--line)" stroke-width="1.6"/>')
    for t in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        x = xf(t)
        add(f'<line x1="{x:.1f}" y1="{base}" x2="{x:.1f}" y2="{base+5}" stroke="var(--line)" stroke-width="1.4"/>')
        txt(x, base + 20, f"{t:.2f}", 11.5, "var(--muted)", "400", "middle")

    # dashed zoom box around the gap
    add(f'<rect x="{gap0-10:.1f}" y="{top-10}" width="{max(gap1-gap0,3)+20:.1f}" '
        f'height="{base-top+16}" fill="none" stroke="var(--ink)" stroke-width="1.8" stroke-dasharray="6 4"/>')
    txt(gap1 + 22, top + 4, "이 사이", 13, "var(--ink)", "700")
    txt(gap1 + 22, top + 22, "클릭 →", 13, "var(--caution)", "700")

    add(f'<rect x="{x1-190}" y="{top-14}" width="9" height="9" fill="var(--hem)"/>')
    txt(x1 - 176, top - 6, "혈관종 253장", 12.5, "var(--muted)")
    add(f'<rect x="{x1-80}" y="{top-14}" width="9" height="9" fill="var(--hcc)"/>')
    txt(x1 - 66, top - 6, "HCC 277장", 12.5, "var(--muted)")

    # ── step 2: magnified interval + flat J (fragment) ───────────────────
    add('<g class="fragment">')

    # magnifier connector
    zx0, zx1, ztop = 40, 700, 300
    add(f'<path d="M{gap0:.1f},{base+8:.1f} L{zx0:.1f},{ztop-8:.1f} M{gap1:.1f},{base+8:.1f} '
        f'L{zx1:.1f},{ztop-8:.1f}" stroke="var(--line-strong, var(--line))" stroke-width="1.4" '
        'stroke-dasharray="4 4" fill="none"/>')

    zlo, zhi = -0.085, 0.045
    zx = lambda v: zx0 + (v - zlo) / (zhi - zlo) * (zx1 - zx0)
    zbase, ztopb = 396, 312

    txt(zx0, ztop - 14, "② 확대 — 간격 0.045", 15, "var(--ink)", "700")
    add(f'<rect x="{zx(hem_max):.1f}" y="{ztopb}" width="{zx(hcc_min)-zx(hem_max):.1f}" '
        f'height="{zbase-ztopb}" fill="var(--caution)" opacity="0.16"/>')

    for vals, color in ((hem, "var(--hem)"), (hcc, "var(--hcc)"),):
        counts, step = bins(vals, zlo, zhi, 44)
        peak = max(counts)
        for i, c in enumerate(counts):
            if not c:
                continue
            x = zx(zlo + i * step)
            w = zx(zlo + (i + 1) * step) - x
            h = c / peak * (zbase - ztopb)
            add(f'<rect x="{x:.1f}" y="{zbase-h:.1f}" width="{max(w-1,1.2):.1f}" height="{h:.1f}" '
                f'fill="{color}" opacity="0.9"/>')

    add(f'<line x1="{zx0}" y1="{zbase}" x2="{zx1}" y2="{zbase}" stroke="var(--line)" stroke-width="1.6"/>')
    for v, lab in ((hem_max, "−0.0498"), (hcc_min, "−0.0048")):
        x = zx(v)
        add(f'<line x1="{x:.1f}" y1="{zbase}" x2="{x:.1f}" y2="{zbase+6}" stroke="var(--ink)" stroke-width="1.6"/>')
        txt(x, zbase + 22, lab, 12.5, "var(--ink)", "700", "middle")

    txt((zx(hem_max) + zx(hcc_min)) / 2, zbase - 26,
        "이 안 어디에 그어도 판정은 같다", 13.5, "var(--ink)", "700", "middle")

    # J on the right
    jx0, jx1, jtop, jbase = 760, 1146, 326, 396
    xj = lambda v: jx0 + (v - zlo) / (zhi - zlo) * (jx1 - jx0)
    yj = lambda j: jbase - max(0.0, min(1.0, j)) * (jbase - jtop)

    txt(jx0, ztop - 14, "Youden J = 민감도 + 특이도 − 1", 15, "var(--ink)", "700")
    add(f'<line x1="{jx0}" y1="{jbase}" x2="{jx1}" y2="{jbase}" stroke="var(--line)" stroke-width="1.6"/>')
    add(f'<line x1="{jx0}" y1="{jbase}" x2="{jx0}" y2="{jtop-8}" stroke="var(--line)" stroke-width="1.6"/>')
    txt(jx0 - 8, jbase + 4, "0", 11.5, "var(--muted)", "400", "end")
    txt(jx0 - 8, jtop + 4, "1.0", 11.5, "var(--muted)", "400", "end")

    pts = j_curve(hcc, hem, zlo, zhi, 400)
    add('<polyline points="' + " ".join(f"{xj(t):.1f},{yj(j):.1f}" for t, j in pts) +
        '" fill="none" stroke="var(--accent)" stroke-width="2.4"/>')
    add(f'<line x1="{xj(hem_max):.1f}" y1="{yj(1.0):.1f}" x2="{xj(hcc_min):.1f}" y2="{yj(1.0):.1f}" '
        'stroke="var(--caution)" stroke-width="6" stroke-linecap="round"/>')
    add(f'<circle cx="{xj(cutoff):.1f}" cy="{yj(1.0):.1f}" r="6" fill="var(--caution)"/>')
    txt((xj(hem_max) + xj(hcc_min)) / 2, yj(1.0) + 22, "평평 — 최고점이 하나가 아니다", 12.5,
        "var(--caution)", "700", "middle")
    txt(xj(cutoff) + 10, yj(1.0) - 10, "구현이 고른 끝 = −0.0048", 12, "var(--ink)")

    add('</g>')

    # ── takeaway (always) ────────────────────────────────────────────────
    add('<text x="0" y="450" font-size="15.5" fill="var(--ink)">'
        '컷오프는 데이터가 정한 한 점이 아니다 — '
        '<tspan font-weight="700">같은 판정이 유지되는 구간</tspan>으로 제시한다.</text>')
    add("</svg>")

    OUT.write_text("\n".join(g), encoding="utf-8")
    print(f"gap ({hem_max:.4f}, {hcc_min:.4f}] · wrote {OUT}")


if __name__ == "__main__":
    main()
