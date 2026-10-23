"""Figure 2-A: ROC-A confusion matrix at the prespecified 0.50 operating point.

Reads the per-case score CSV and writes a figure in the same style as the
revision-2 matrices (cream-to-red map, proportion + count in each cell).
Run: python3 make_confusion_matrix_a.py
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE.parents[2] / "cosine_probe_result" / "raw_cosprobe_effnet_supcon.csv"
CUTOFF = 0.50

rows = [r for r in csv.DictReader(CSV.open()) if r["split"] == "Test"]
y = np.array([int(float(r["true_label"])) for r in rows])          # 0 hem, 1 HCC
conf = np.array([float(r["confidence_score"]) for r in rows])
pred = (conf >= CUTOFF).astype(int)

cm = np.zeros((2, 2), dtype=int)
for t, p in zip(y, pred):
    cm[t, p] += 1
prop = cm / cm.sum(axis=1, keepdims=True)
tn, fp = cm[0]
fn, tp = cm[1]

fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
im = ax.imshow(prop, cmap="YlOrRd", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{prop[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center",
                fontsize=13, color="black" if prop[i, j] < 0.6 else "white")
ax.set_xticks([0, 1], ["Hemangioma", "HCC"])
ax.set_yticks([0, 1], ["Hemangioma", "HCC"])
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix — ROC-A (Confidence)\n(Test, cutoff={CUTOFF:.2f}, n={len(y)})", fontsize=12)
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.tight_layout()
out = HERE / "fig 2-A Confusion matrix-A.jpg"
fig.savefig(out, format="jpg", pil_kwargs={"quality": 95})
print(f"wrote {out}  TN/FP/FN/TP = {tn}/{fp}/{fn}/{tp}")
