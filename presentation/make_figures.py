"""Concept figures for the deck (matplotlib). Outputs into presentation/figures/.

  fig_embed_sphere.png  : image -> encoder -> number vector -> unit-hypersphere scatter
  fig_score_calc.png    : prototype (class mean) -> cosine geometry -> scores
  fig_three_outputs.png : the three model outputs (confidence / cosine / delta)
  fig_architecture.png  : Hybrid ViT (EfficientNetV2B0 + Cross-Attention), CV-paper style
All are schematic/illustrative, not real model outputs.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from PIL import Image

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

NAVY = "#0056b3"; INK = "#14202e"; MUTED = "#5b6b7d"; LINE = "#dbe3ee"
HCC = "#c9403a"; HEM = "#2f6fb3"; GOLD = "#a9781a"

BASE = pathlib.Path(__file__).resolve().parent
FIG = BASE / "figures"
rng = np.random.default_rng(7)


def thumb(name, size=120):
    return Image.open(FIG / name).convert("L").resize((size, size))


def cluster(direction, n, spread):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    pts = []
    while len(pts) < n:
        p = d + rng.normal(0, spread, 3)
        p /= np.linalg.norm(p)
        pts.append(p)
    return np.asarray(pts)


def arrow(ax, xy0, xy1, color=GOLD, lw=2.4, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, lw=lw, color=color,
                                 mutation_scale=16, shrinkA=0, shrinkB=0))


def box(ax, x0, x1, y0, y1, ec=LINE, fc="#ffffff", lw=1.6, r=0.12, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls))


# ─────────────────────────────── figure 1 ───────────────────────────────
def figure_embed_sphere():
    fig = plt.figure(figsize=(13, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.35], wspace=0.05)

    # (A) input images
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.5, 9.6, "① 입력 영상 (B-mode)", fontsize=13, color=INK, fontweight="bold")
    for i, name in enumerate(["ex_hcc_new.jpg", "ex_hcc_04.png"]):
        iax = ax.inset_axes([0.08, 0.60 - 0.40 * i, 0.5, 0.32])
        iax.imshow(thumb(name), cmap="gray"); iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_edgecolor(LINE)

    # (B) encoder + number vector
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.5, 9.6, "② 인코더 → 숫자의 나열", fontsize=13, color=INK, fontweight="bold")
    ax.text(5.0, 8.6, "인코더 f(·)  (CNN + ViT)", fontsize=12, color=GOLD,
            fontweight="bold", ha="center")
    for y, txt in [(7.6, "[0.82, 0.63, 0.54, ...]"), (3.6, "[0.70, 0.50, 1.20, ...]")]:
        arrow(ax, (0.7, y), (2.4, y), lw=2.4)
        ax.text(2.7, y, txt, fontsize=12.5, color=INK, ha="left", va="center",
                family="DejaVu Sans Mono")
    ax.text(5.0, 1.4, "각 영상 → 하나의 숫자 벡터('좌표')", fontsize=11.5, color=MUTED, ha="center")

    # (C) hypersphere scatter
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    u = np.linspace(0, 2 * np.pi, 48); v = np.linspace(0, np.pi, 24)
    ax3.plot_surface(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                     np.outer(np.ones_like(u), np.cos(v)), color="#eef4fb", alpha=0.30,
                     linewidth=0, shade=False)
    ax3.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v)), color=LINE, linewidth=0.4,
                       rstride=4, cstride=4)
    hcc = cluster([0.95, 0.15, 0.35], 46, 0.11)
    hem = cluster([-0.55, 0.60, 0.55], 46, 0.14)
    ax3.scatter(*hcc.T, s=16, color=HCC, alpha=0.75, label="HCC")
    ax3.scatter(*hem.T, s=16, color=HEM, alpha=0.75, label="혈관종")
    ph = hcc.mean(0); ph /= np.linalg.norm(ph)
    pm = hem.mean(0); pm /= np.linalg.norm(pm)
    ax3.scatter(*ph, s=200, marker="*", color=HCC, edgecolor="white", linewidth=1.2, zorder=5)
    ax3.scatter(*pm, s=200, marker="*", color=HEM, edgecolor="white", linewidth=1.2, zorder=5)
    ax3.set_box_aspect([1, 1, 1]); ax3.set_axis_off()
    ax3.view_init(elev=18, azim=32)
    ax3.set_xlim(-1.15, 1.15); ax3.set_ylim(-1.15, 1.15); ax3.set_zlim(-1.15, 1.15)
    ax3.text2D(0.02, 0.98, "③ 초구면 위에 점으로 표시", transform=ax3.transAxes,
               fontsize=13, color=INK, fontweight="bold")
    ax3.text2D(0.02, 0.88, "★ HCC prototype", transform=ax3.transAxes,
               fontsize=11.5, color=HCC, fontweight="bold")
    ax3.text2D(0.02, 0.79, "★ 혈관종 prototype", transform=ax3.transAxes,
               fontsize=11.5, color=HEM, fontweight="bold")
    ax3.text2D(0.02, 0.05, "닮음 = 두 점 사이의 각도(cosine)", transform=ax3.transAxes,
               fontsize=11, color=MUTED)
    ax3.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.savefig(FIG / "fig_embed_sphere.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────── figure 2 ───────────────────────────────
def figure_score_calc():
    fig = plt.figure(figsize=(11.4, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0]); ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(th), np.sin(th), color=LINE, lw=1.6)
    ax.plot(0, 0, ".", color=INK, ms=4)
    ang_hcc, ang_hem, ang_z = np.deg2rad(25), np.deg2rad(155), np.deg2rad(72)

    def vec(a): return np.array([np.cos(a), np.sin(a)])
    vz, vh, vm = vec(ang_z), vec(ang_hcc), vec(ang_hem)

    def arr(v, color, label, off):
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color=color))
        ax.text(*(v * 1.12 + off), label, color=color, fontsize=12, fontweight="bold", ha="center")
    arr(vh, HCC, "HCC prototype", np.array([0.05, 0.06]))
    arr(vm, HEM, "혈관종 prototype", np.array([-0.06, 0.06]))
    ax.annotate("", xy=vz * 0.92, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color=GOLD, linestyle=(0, (5, 3))))
    ax.text(*(vz * 0.90 + np.array([0.14, -0.02])), "이 병변 z", color=GOLD, fontsize=12,
            fontweight="bold", ha="left")
    ax.plot(*vz, "o", color=GOLD, ms=6)
    for a0, a1, col, txt, la in [
            (ang_hcc, ang_z, HCC, r"$\theta_{HCC}$", np.deg2rad(36)),
            (ang_hem, ang_z, HEM, r"$\theta_{Hem}$", np.deg2rad(144))]:
        tt = np.linspace(min(a0, a1), max(a0, a1), 40)
        ax.plot(0.30 * np.cos(tt), 0.30 * np.sin(tt), color=col, lw=1.8)
        ax.text(0.62 * np.cos(la), 0.62 * np.sin(la), txt, color=col, fontsize=15,
                fontweight="bold", ha="center", va="center")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.3, 1.4)
    ax.text(0, 1.28, "cosine similarity = 두 벡터가 이루는 각의 코사인",
            ha="center", fontsize=12.5, color=INK, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    labels = ["HCC cosine score", "Hemangioma cosine", "Δscore"]
    values = [0.92, 0.68, 0.24]
    colors = [HCC, HEM, GOLD]
    ax2.text(0.0, 0.95, "산출되는 세 가지 점수", fontsize=12.5, color=INK,
             fontweight="bold", transform=ax2.transAxes)
    for lab, val, col, y in zip(labels, values, colors, [0.78, 0.50, 0.22]):
        ax2.add_patch(Rectangle((0.0, y), 0.98 * val, 0.11, transform=ax2.transAxes,
                                color=col, alpha=0.85, clip_on=False))
        ax2.text(0.0, y + 0.135, lab, fontsize=11.5, color=INK, transform=ax2.transAxes)
        ax2.text(0.98 * val + 0.02, y + 0.02, f"{val:+.2f}" if lab == "Δscore" else f"{val:.2f}",
                 fontsize=11.5, color=col, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.10, 0.03, "Δscore > 0 → HCC 쪽에 더 가까움", fontsize=11, color=MUTED,
             transform=ax2.transAxes)
    ax2.text(0.0, -0.08, "prototype = 훈련 세트 각 클래스 임베딩의 평균 벡터 (mean prototype)",
             fontsize=10.5, color=MUTED, transform=ax2.transAxes)

    fig.savefig(FIG / "fig_score_calc.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────── figure 3 ───────────────────────────────
def figure_three_outputs():
    fig = plt.figure(figsize=(13, 4.0), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[0.85, 1.0, 1.0, 1.0], wspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
    iax = ax0.inset_axes([0.05, 0.30, 0.9, 0.55])
    iax.imshow(thumb("ex_hcc_new.jpg"), cmap="gray"); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(LINE)
    ax0.text(0.5, 0.14, "병변 영상", fontsize=12.5, color=INK, fontweight="bold",
             ha="center", transform=ax0.transAxes)

    # ① confidence
    ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.97, 0.1, 0.92, ec=LINE, fc="#ffffff")
    ax.text(0.5, 0.86, "① Confidence", fontsize=12.5, color=HCC, fontweight="bold", ha="center")
    ax.add_patch(Rectangle((0.12, 0.60), 0.76, 0.09, color=HCC, alpha=0.85))
    ax.add_patch(Rectangle((0.12, 0.60), 0.76 * 0.95, 0.09, color=HCC))
    ax.plot([0.12 + 0.76 * 0.95] * 2, [0.56, 0.73], color=INK, lw=2)
    ax.text(0.5, 0.47, "0.95", fontsize=16, color=HCC, fontweight="bold", ha="center")
    ax.text(0.5, 0.30, "softmax = 결정 강도", fontsize=10.5, color=MUTED, ha="center")
    ax.text(0.5, 0.19, "(HCC일 확률로\n직접 해석은 안 됨)", fontsize=10, color=MUTED, ha="center")

    # ② cosine
    ax = fig.add_subplot(gs[0, 2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.97, 0.1, 0.92, ec=LINE, fc="#ffffff")
    ax.text(0.5, 0.86, "② HCC/Hem cosine", fontsize=12.5, color=NAVY, fontweight="bold", ha="center")
    for lab, val, col, y in [("HCC 닮음", 0.92, HCC, 0.60), ("혈관종 닮음", 0.68, HEM, 0.40)]:
        ax.add_patch(Rectangle((0.12, y), 0.76 * val, 0.09, color=col, alpha=0.85))
        ax.text(0.12, y + 0.10, f"{lab}  {val:.2f}", fontsize=10.5, color=col)
    ax.text(0.5, 0.24, "prototype과의 유사도", fontsize=10.5, color=MUTED, ha="center")
    ax.text(0.5, 0.14, "= 전형 소견과의 닮음", fontsize=10, color=MUTED, ha="center")

    # ③ delta
    ax = fig.add_subplot(gs[0, 3]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.03, 0.97, 0.1, 0.92, ec=LINE, fc="#ffffff")
    ax.text(0.5, 0.86, "③ Δscore", fontsize=12.5, color=GOLD, fontweight="bold", ha="center")
    ax.plot([0.12, 0.88], [0.55, 0.55], color=LINE, lw=2)
    ax.plot([0.5], [0.55], "|", color=MUTED, ms=12)
    ax.plot([0.5 + 0.76 * 0.24 / 2], [0.55], "o", color=GOLD, ms=10)
    ax.text(0.5, 0.38, "+0.24", fontsize=16, color=GOLD, fontweight="bold", ha="center")
    ax.text(0.5, 0.24, "HCC 닮음 - 혈관종 닮음", fontsize=10.5, color=MUTED, ha="center")
    ax.text(0.5, 0.14, "> 0 → HCC 쪽", fontsize=10, color=MUTED, ha="center")

    fig.savefig(FIG / "fig_three_outputs.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────── figure 4 ───────────────────────────────
def figure_architecture():
    fig, ax = plt.subplots(figsize=(13, 4.0), facecolor="white")
    ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(10, 5.75, "Hybrid Vision Transformer — EfficientNetV2B0 + Cross-Attention",
            fontsize=13.5, color=INK, fontweight="bold", ha="center")

    # input image
    iax = ax.inset_axes([0.022, 0.42, 0.075, 0.42])
    iax.imshow(thumb("ex_hcc_new.jpg", 100), cmap="gray"); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor(LINE)
    ax.text(1.15, 1.55, "입력 영상\n384×384 (B-mode)", fontsize=10, color=MUTED, ha="center")

    arrow(ax, (1.95, 3.0), (2.45, 3.0))

    # CNN encoder trapezoid (wider left -> narrower right)
    ax.add_patch(Polygon([(2.5, 1.5), (4.9, 2.25), (4.9, 3.75), (2.5, 4.5)],
                         closed=True, facecolor="#fff6e6", edgecolor=GOLD, lw=2))
    ax.text(3.7, 3.0, "EfficientNetV2B0\n(CNN encoder)\n7.1M · ImageNet", fontsize=10.5,
            color="#6b4d12", ha="center", va="center", fontweight="bold")
    ax.text(3.7, 1.15, "특징 맵  12×12×1280", fontsize=10, color=MUTED, ha="center")

    arrow(ax, (4.95, 3.0), (5.45, 3.0))

    # patch tokenization
    box(ax, 5.5, 7.7, 2.05, 3.95, ec=LINE, fc="#f4f7fb")
    ax.text(6.6, 3.35, "패치 토큰화", fontsize=11, color=INK, ha="center", fontweight="bold")
    ax.text(6.6, 2.95, "Conv 1×1 + LN", fontsize=9.5, color=MUTED, ha="center")
    ax.text(6.6, 2.6, "144 patches", fontsize=9.5, color=MUTED, ha="center")
    ax.text(6.6, 2.25, "[CLS] + PE → 145", fontsize=9.5, color=MUTED, ha="center")

    arrow(ax, (7.75, 3.0), (8.25, 3.0))

    # transformer encoder (cross-attention)
    box(ax, 8.3, 12.6, 1.35, 4.65, ec=GOLD, fc="#fffdf7", lw=2)
    ax.text(10.45, 4.35, "Transformer Encoder × 4", fontsize=11.5, color="#6b4d12",
            ha="center", fontweight="bold")
    box(ax, 8.6, 12.3, 2.45, 4.0, ec=NAVY, fc="#eef4fb", lw=1.4)
    ax.text(10.45, 3.35, "Cross-Attention", fontsize=11, color=NAVY, ha="center", fontweight="bold")
    ax.text(10.45, 2.95, "Q = CLS   ·   K, V = patches", fontsize=10, color=INK, ha="center")
    ax.text(10.45, 2.6, "8 heads  +  FFN", fontsize=9.5, color=MUTED, ha="center")
    ax.text(10.45, 1.9, "d = 128", fontsize=9.5, color=MUTED, ha="center")

    arrow(ax, (12.65, 3.0), (13.15, 3.0))

    # CLS embedding
    box(ax, 13.2, 14.9, 2.2, 3.8, ec=HEM, fc="#eef4fb", lw=2)
    ax.text(14.05, 3.2, "CLS 임베딩", fontsize=11, color=HEM, ha="center", fontweight="bold")
    ax.text(14.05, 2.75, "128차원", fontsize=10, color=MUTED, ha="center")

    # heads
    arrow(ax, (14.95, 3.15), (15.5, 4.2))
    arrow(ax, (14.95, 2.85), (15.5, 1.8))
    box(ax, 15.6, 19.7, 3.6, 4.75, ec=HCC, fc="#ffffff", lw=2)
    ax.text(17.65, 4.45, "분류 헤드 → Confidence", fontsize=11, color=HCC, ha="center", fontweight="bold")
    ax.text(17.65, 3.95, "Dense → softmax (HCC)", fontsize=9.5, color=MUTED, ha="center")
    box(ax, 15.6, 19.7, 1.25, 2.4, ec=GOLD, fc="#ffffff", lw=2)
    ax.text(17.65, 2.1, "프로젝션 헤드 → 임베딩", fontsize=11, color=GOLD, ha="center", fontweight="bold")
    ax.text(17.65, 1.6, "HCC/Hem cosine · Δscore", fontsize=9.5, color=MUTED, ha="center")

    fig.savefig(FIG / "fig_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_supcon_loss():
    fig = plt.figure(figsize=(11.5, 1.9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.68, r"$\mathcal{L}\;=\;\mathcal{L}_{\mathrm{CE}}\;+\;\lambda\,\mathcal{L}_{\mathrm{SupCon}}$",
            ha="center", va="center", fontsize=25, color=INK)
    ax.text(0.5, 0.22,
            r"$\mathcal{L}_{\mathrm{SupCon}}=\sum_{i}\frac{-1}{|P(i)|}\sum_{p\in P(i)}"
            r"\log\frac{\exp(\mathbf{z}_i\cdot\mathbf{z}_p/\tau)}"
            r"{\sum_{a\in A(i)}\exp(\mathbf{z}_i\cdot\mathbf{z}_a/\tau)}$",
            ha="center", va="center", fontsize=17, color=INK)
    fig.savefig(FIG / "fig_supcon_loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figure_embed_sphere()
    figure_score_calc()
    figure_three_outputs()
    figure_architecture()
    figure_supcon_loss()
    print("figures written to", FIG)
