"""Draw the "why an interval, not a point" explainer (1920x1080 PNG).

Data: cosine_probe_result/raw_cosprobe_effnet_supcon.csv (validation + test).
Reading order:

1. validation, full range: the two groups are far apart;
2. zoom on the gap: Youden's J is exactly 1.0 everywhere inside it, so the
   criterion cannot pick a point — the implementation takes the right edge,
   which is the lowest-scoring HCC. That entire gap is the equivalence interval;
3. test at the same zoom: the edge we took sits just under one hemangioma case,
   which becomes the single false positive.

Distributions are drawn as bin histograms (bars), not stacked dots, so a dense
cluster cannot bleed out of its panel.

Run: python3 presentation/explainer_youden.py
     chrome --headless --window-size=1920,1080 --screenshot=... explainer_youden.html
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RAW = REPO / "cosine_probe_result" / "raw_cosprobe_effnet_supcon.csv"
OUT_HTML = HERE / "explainer_youden.html"

INK = "#101828"
INK_2 = "#51555D"
LINE = "#DCD8CF"
PAPER = "#F7F6F3"
ACCENT = "#1D4ED8"      # HCC
HEM = "#0F766E"         # hemangioma
ALERT = "#B45309"
SANS = "Noto Sans CJK KR, Noto Sans KR, sans-serif"


def load() -> dict:
    rows = list(csv.DictReader(RAW.open(encoding="utf-8")))
    out = {}
    for split in ("Val", "Test"):
        part = [r for r in rows if r["split"] == split]
        out[split] = {
            "hcc": [float(r["hcc_cosine_mean"]) for r in part if int(float(r["true_label"])) == 1],
            "hem": [float(r["hcc_cosine_mean"]) for r in part if int(float(r["true_label"])) == 0],
        }
    return out


def bins(values, lo, hi, n):
    counts = [0] * n
    step = (hi - lo) / n
    for v in values:
        if lo <= v < hi:
            counts[min(n - 1, int((v - lo) / step))] += 1
    return counts, step


def bars(counts, step, lo, x_of, base_y, max_h, color, opacity=0.85):
    peak = max(counts) or 1
    out = []
    for i, c in enumerate(counts):
        if not c:
            continue
        x = x_of(lo + i * step)
        w = x_of(lo + (i + 1) * step) - x
        h = c / peak * max_h
        out.append(
            f'<rect x="{x:.1f}" y="{base_y-h:.1f}" width="{max(w-1.2, 1.2):.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="{opacity}"/>'
        )
    return out


def j_curve(hcc, hem, lo, hi, n=400):
    pts = []
    for i in range(n + 1):
        t = lo + (hi - lo) * i / n
        sens = sum(1 for s in hcc if s >= t) / len(hcc)
        spec = sum(1 for s in hem if s < t) / len(hem)
        pts.append((t, sens + spec - 1))
    return pts


def main() -> None:
    d = load()
    v_hcc, v_hem = d["Val"]["hcc"], d["Val"]["hem"]
    t_hcc, t_hem = d["Test"]["hcc"], d["Test"]["hem"]
    hem_max, hcc_min = max(v_hem), min(v_hcc)
    cutoff = hcc_min
    t_hem_max, t_hcc_min = max(t_hem), min(t_hcc)

    W, H = 1920, 1080
    s = []
    add = s.append
    add(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')

    def text(x, y, t, size=19, color=INK, weight="400", anchor="start", opacity=1.0):
        add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}" opacity="{opacity}">{t}</text>')

    # ── header ───────────────────────────────────────────────────────────
    text(96, 88, "왜 컷오프를 점이 아니라 구간으로 제시하는가", 23, ACCENT, "700")
    add(f'<text x="96" y="152" font-family="{SANS}" font-size="48" font-weight="700" fill="{INK}">'
        f'AUC가 1.000이면 선은 <tspan fill="{ACCENT}">어디에 그어도 같다</tspan></text>')
    text(96, 194, "Youden은 “이 안”까지만 알려준다 — 그래서 점 하나 대신 구간을 보고한다", 23, INK_2)
    add(f'<line x1="96" y1="216" x2="{W-96}" y2="216" stroke="{LINE}" stroke-width="2"/>')

    # ── panel 1: validation, full range ──────────────────────────────────
    x0, x1 = 140, 1800
    lo1, hi1 = -0.35, 1.05
    x_of1 = lambda v: x0 + (v - lo1) / (hi1 - lo1) * (x1 - x0)
    base1, top1 = 420, 300

    text(96, 262, "① 검증 세트 — 두 무리는 넓게 떨어져 있다", 25, INK, "700")
    text(96, 292, "HCC cosine score · 막대 = 그 구간의 영상 수", 18, INK_2)
    text(x1, 262, "■ 혈관종 253장", 18, HEM, "400", "end")
    text(x1, 292, "■ HCC 277장", 18, ACCENT, "400", "end")

    c_hem, st_hem = bins(v_hem, lo1, hi1, 110)
    c_hcc, st_hcc = bins(v_hcc, lo1, hi1, 110)
    s += bars(c_hem, st_hem, lo1, x_of1, base1, top1 and (base1 - 300), HEM)
    s += bars(c_hcc, st_hcc, lo1, x_of1, base1, base1 - 300, ACCENT)

    gx0, gx1 = x_of1(hem_max), x_of1(hcc_min)
    add(f'<rect x="{gx0:.1f}" y="300" width="{max(gx1-gx0,3):.1f}" height="{base1-300}" '
        f'fill="{ALERT}" opacity="0.10"/>')
    text((gx0 + gx1) / 2, base1 - 14, "빈 공간", 15, ALERT, "700", "middle")

    add(f'<line x1="{x0}" y1="{base1}" x2="{x1}" y2="{base1}" stroke="{LINE}" stroke-width="2"/>')
    for t in (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
        px = x_of1(t)
        add(f'<line x1="{px:.1f}" y1="{base1}" x2="{px:.1f}" y2="{base1+7}" stroke="{LINE}" stroke-width="2"/>')
        text(px, base1 + 30, f"{t:.2f}", 16, INK_2, "400", "middle")

    # callout to panel 2
    add(f'<rect x="{gx0-14:.1f}" y="302" width="{max(gx1-gx0,3)+28:.1f}" height="{base1-302+10}" '
        f'fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="6 5"/>')
    text(gx1 + 26, 330, "② 이 사이를 확대", 19, INK_2)

    # ── panel 2: zoom + J ────────────────────────────────────────────────
    zlo, zhi = -0.085, 0.045
    q0, q1 = 140, 860
    xz = lambda v: q0 + (v - zlo) / (zhi - zlo) * (q1 - q0)
    bz, tz = 660, 540

    text(96, 500, "② 확대 — 이 안에서는 판정이 한 장도 바뀌지 않는다", 25, INK, "700")
    text(96, 530, "빈 공간 전체가 등가 구간 (−0.0498, −0.0048] · 폭 0.045", 18, INK_2)

    add(f'<rect x="{xz(hem_max):.1f}" y="{tz}" width="{xz(hcc_min)-xz(hem_max):.1f}" '
        f'height="{bz-tz}" fill="{ALERT}" opacity="0.16"/>')
    c1, s1 = bins(v_hem, zlo, zhi, 52)
    c2, s2 = bins(v_hcc, zlo, zhi, 52)
    s += bars(c1, s1, zlo, xz, bz, bz - tz, HEM)
    s += bars(c2, s2, zlo, xz, bz, bz - tz, ACCENT)
    add(f'<line x1="{q0}" y1="{bz}" x2="{q1}" y2="{bz}" stroke="{LINE}" stroke-width="2"/>')

    for v, lab, col in ((hem_max, "−0.0498", HEM), (hcc_min, "−0.0048", ACCENT)):
        px = xz(v)
        add(f'<line x1="{px:.1f}" y1="{bz}" x2="{px:.1f}" y2="{bz+9}" stroke="{INK}" stroke-width="2"/>')
        text(px, bz + 32, lab, 18, INK, "700", "middle")
    text((xz(hem_max) + xz(hcc_min)) / 2, bz + 32, "간격 0.045", 17, INK_2, "400", "middle")

    # J panel
    jx0, jx1 = 1010, 1800
    jy0, jy1 = 640, 540
    text(1010, 500, "Youden J = 민감도 + 특이도 − 1", 21, INK, "700")
    add(f'<line x1="{jx0}" y1="{jy0}" x2="{jx1}" y2="{jy0}" stroke="{LINE}" stroke-width="2"/>')
    add(f'<line x1="{jx0}" y1="{jy0}" x2="{jx0}" y2="{jy1-10}" stroke="{LINE}" stroke-width="2"/>')
    text(jx0 - 12, jy0 + 6, "0", 16, INK_2, "400", "end")
    text(jx0 - 12, jy1 + 5, "1.0", 16, INK_2, "400", "end")

    xj = lambda v: jx0 + (v - zlo) / (zhi - zlo) * (jx1 - jx0)
    yj = lambda j: jy0 - max(0.0, min(1.0, j)) * (jy0 - jy1)
    pts = j_curve(v_hcc, v_hem, zlo, zhi, 400)
    add('<polyline points="' + " ".join(f"{xj(t):.1f},{yj(j):.1f}" for t, j in pts) +
        f'" fill="none" stroke="{ACCENT}" stroke-width="3.5"/>')
    add(f'<line x1="{xj(hem_max):.1f}" y1="{yj(1.0):.1f}" x2="{xj(hcc_min):.1f}" y2="{yj(1.0):.1f}" '
        f'stroke="{ALERT}" stroke-width="7" stroke-linecap="round"/>')
    add(f'<circle cx="{xj(cutoff):.1f}" cy="{yj(1.0):.1f}" r="8" fill="{ALERT}"/>')

    text((xj(hem_max) + xj(hcc_min)) / 2, yj(1.0) + 30, "최대가 평평 → 고를 수가 없다", 18, ALERT, "700", "middle")
    ax = xj(cutoff)
    add(f'<path d="M{ax+4:.1f},{yj(1.0)+10:.1f} L{ax+16:.1f},{jy0-14:.1f}" stroke="{INK}" stroke-width="2" fill="none"/>')
    text(ax + 24, jy0 - 4, "구현은 오른쪽 끝을 고른다", 18, INK, "400")

    # ── panel 3: test ────────────────────────────────────────────────────
    t0, t1 = 140, 1800
    tlo, thi = -0.35, 1.05
    xt = lambda v: t0 + (v - tlo) / (thi - tlo) * (t1 - t0)
    bt, tt = 900, 790

    text(96, 748, "③ 테스트 세트 — 그 끝에 그은 선은 한 장에 붙어 있다", 25, INK, "700")
    c3, s3 = bins(t_hem, tlo, thi, 110)
    c4, s4 = bins(t_hcc, tlo, thi, 110)
    s += bars(c3, s3, tlo, xt, bt, bt - tt, HEM)
    s += bars(c4, s4, tlo, xt, bt, bt - tt, ACCENT)
    add(f'<line x1="{t0}" y1="{bt}" x2="{t1}" y2="{bt}" stroke="{LINE}" stroke-width="2"/>')

    cx = xt(cutoff)
    add(f'<line x1="{cx:.1f}" y1="{tt-6}" x2="{cx:.1f}" y2="{bt}" stroke="{INK}" stroke-width="2.5" stroke-dasharray="8 6"/>')
    text(cx - 12, tt + 26, "컷오프 −0.0048", 17, INK, "700", "end")

    fx = xt(t_hem_max)
    add(f'<circle cx="{fx:.1f}" cy="{tt+34:.1f}" r="10" fill="none" stroke="{ALERT}" stroke-width="3"/>')
    text(fx + 22, tt + 38, "혈관종 1장이 위로 (+0.0325)", 19, ALERT, "700")
    text(fx + 22, tt + 64, "→ 위양성 1건", 19, ALERT, "700")

    text(xt(t_hcc_min) - 8, tt + 96, f"HCC 최저 {t_hcc_min:.4f}", 17, ACCENT, "400", "end")

    # ── takeaway ─────────────────────────────────────────────────────────
    add(f'<line x1="96" y1="972" x2="{W-96}" y2="972" stroke="{LINE}" stroke-width="2"/>')
    add(f'<text x="96" y="1016" font-family="{SANS}" font-size="22" fill="{INK}">'
        '컷오프 한 점은 데이터가 정한 값이 아니라 구현이 고른 위치다 — 그래서 '
        f'<tspan font-weight="700">같은 판정이 유지되는 구간</tspan>을 함께 보고한다.</text>')
    text(96, 1050, "검증 n=530 (HCC 277 · 혈관종 253) · 테스트 n=268 (HCC 140 · 혈관종 128)", 17, INK_2)

    html = (
        "<!doctype html><meta charset='utf-8'>"
        f"<style>html,body{{margin:0;background:{PAPER};}}svg{{display:block}}</style>"
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>"
        + "".join(s) + "</svg>"
    )
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"val gap ({hem_max:.4f}, {hcc_min:.4f}] · test max hem {t_hem_max:.4f} · wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
