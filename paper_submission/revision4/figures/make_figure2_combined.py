"""Figure 2 (combined): three confusion matrices of the primary model as one
three-panel figure, replacing the separate Fig 2-A / 2-B / 2-C files.

Panel A  confidence score, prespecified 0.50
Panel B  HCC cosine score (mean prototype), validation Youden point
Panel C  delta score (mean prototype), validation Youden point

All 268 test images, same style/colour map as revision 3 so the numbers are
directly comparable with the printed tables.

Run: python3 make_figure2_combined.py
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE.parents[2] / "cosine_probe_result" / "raw_cosprobe_effnet_supcon.csv"

PANELS = [
    ("(A) Confidence score", "confidence_score", 0.50, "$\\geq$ 0.50 (prespecified)"),
    ("(B) HCC cosine score", "hcc_cosine_mean", -0.004843, "Youden point (exploratory)"),
    ("(C) $\\Delta$score", "delta_mean", -0.958471, "Youden point (exploratory)"),
]

rows = [r for r in csv.DictReader(CSV.open()) if r["split"] == "Test"]
y = np.array([int(float(r["true_label"])) for r in rows])  # 0 hemangioma, 1 HCC
n = len(y)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=200)
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
                    fontsize=11, color="black" if prop[i, j] < 0.6 else "white")
    ax.set_xticks([0, 1], ["Hemangioma", "HCC"])
    ax.set_yticks([0, 1], ["Hemangioma", "HCC"])
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"{label}\n{note}\nn = {n}", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    tn, fp = cm[0]
    fn, tp = cm[1]
    print(f"{label}: TN={tn} FP={fp} FN={fn} TP={tp}  cutoff={cut}")

fig.tight_layout()
out = HERE / "fig 2 combined confusion matrices.png"
fig.savefig(out, format="png", dpi=200)
print("wrote", out)
