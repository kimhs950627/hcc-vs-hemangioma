# -*- coding: utf-8 -*-
"""Export the figure set for KJFP (JPEG, 600 dpi tag).

Figure 2 is re-rendered from the raw score CSV at 600 dpi so it has true
600-dpi pixels. The other four are raster sources that cannot gain real
resolution: they are re-tagged at 600 dpi and their effective print width at
600 dpi is reported, so the limit is visible instead of hidden.

Run: python3 make_figures_kjfp.py
"""
import pathlib, csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures_kjfp"
OUT.mkdir(exist_ok=True)
CSV = HERE.parents[1] / "cosine_probe_result" / "raw_cosprobe_effnet_supcon.csv"
DPI = 600

PANELS = [
    ("(A) Confidence score", "confidence_score", 0.50, "$\\geq$ 0.50 (prespecified)"),
    ("(B) HCC cosine score", "hcc_cosine_mean", -0.004843, "Youden point (exploratory)"),
    ("(C) $\\Delta$score", "delta_mean", -0.958471, "Youden point (exploratory)"),
]

rows = [r for r in csv.DictReader(CSV.open()) if r["split"] == "Test"]
y = np.array([int(float(r["true_label"])) for r in rows])
n = len(y)

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1), dpi=DPI)
for ax, (label, col, cut, note) in zip(axes, PANELS):
    score = np.array([float(r[col]) for r in rows])
    pred = (score >= cut).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    prop = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(prop, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{prop[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                    fontsize=8, color="black" if prop[i, j] < 0.6 else "white")
    ax.set_xticks([0, 1], ["Hemangioma", "HCC"])
    ax.set_yticks([0, 1], ["Hemangioma", "HCC"])
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.set_title(f"{label}\n{note}\nn = {n}", fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(OUT / "Figure2A-C.jpg", format="jpg", dpi=DPI, pil_kwargs={"quality": 95})
plt.close(fig)
print("Figure2A-C.jpg  rendered at true", DPI, "dpi")


# ---- Figure 3: triple ROC re-rendered from the per-case scores at 600 dpi ----
def roc(yv, s):
    order = np.argsort(-s)
    yv, s = yv[order], s[order]
    tps = np.cumsum(yv); fps = np.cumsum(1 - yv)
    P, N = yv.sum(), (1 - yv).sum()
    return np.r_[0, fps / N], np.r_[0, tps / P]

fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=DPI)
for col, label, style in [
        ("confidence_score", "ROC-A confidence score", "-"),
        ("hcc_cosine_mean", "ROC-B HCC cosine score", "--"),
        ("delta_mean", "ROC-C $\\Delta$score", "-.")]:
    sc = np.array([float(r[col]) for r in rows])
    fpr, tpr = roc(y, sc)
    ax.plot(fpr, tpr, style, linewidth=1.2, label=f"{label} (AUROC 1.000)")
ax.plot([0, 1], [0, 1], ":", color="grey", linewidth=0.8)
ax.set_xlabel("1 - Specificity", fontsize=8)
ax.set_ylabel("Sensitivity", fontsize=8)
ax.set_title("Test set (n = 268)", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6, loc="lower right", frameon=False)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
fig.tight_layout()
fig.savefig(OUT / "Figure3.jpg", format="jpg", dpi=DPI, pil_kwargs={"quality": 95})
plt.close(fig)
print("Figure3.jpg  re-rendered at true", DPI, "dpi")

SRC_MAP = {
    "Figure1.jpg": "fig 1 TSNE visualization.jpg",
    "Figure4.jpg": "fig_4_hcc_cosine_distribution_comparison.png",
    "Figure5.jpg": "fig_5_hemangioma_cosine_distribution_comparison.png",
}
for dst, src in SRC_MAP.items():
    im = Image.open(HERE / "figures" / src).convert("RGB")
    w_cm = im.width / DPI * 2.54
    im.save(OUT / dst, format="JPEG", quality=95, dpi=(DPI, DPI))
    print(f"{dst:12s} source {im.width}x{im.height} px -> {w_cm:.1f} cm wide at {DPI} dpi")
