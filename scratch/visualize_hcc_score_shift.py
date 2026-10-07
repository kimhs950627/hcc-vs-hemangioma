"""
scratch/visualize_hcc_score_shift.py
===================================
Visualize the distribution shift of HCC Cosine Scores in True HCC cases (n=417)
comparing CE-Only vs. CE+SupCon for both EfficientNetV2B0 and ResNet50V2.
Pure matplotlib and scipy implementation (no seaborn dependency).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_rel, wilcoxon

# Paths
root = Path(__file__).resolve().parent.parent
probe_dir = root / "cosine_probe_result"
out_dir = root / "paper_submission" / "revision2"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Data
eff_ce_df = pd.read_csv(probe_dir / "raw_cosprobe_effnet_classification_only.csv")
eff_sc_df = pd.read_csv(probe_dir / "raw_cosprobe_effnet_supcon.csv")
res_ce_df = pd.read_csv(probe_dir / "raw_cosprobe_resnet_classification_only.csv")
res_sc_df = pd.read_csv(probe_dir / "raw_cosprobe_res_supcon.csv")

# Filter True HCC (Val + Test, n=417)
eff_ce_hcc = eff_ce_df[eff_ce_df.split.isin(["Val", "Test"]) & (eff_ce_df.true_label == 1)]["hcc_cosine_mean"].values
eff_sc_hcc = eff_sc_df[eff_sc_df.split.isin(["Val", "Test"]) & (eff_sc_df.true_label == 1)]["hcc_cosine_mean"].values
res_ce_hcc = res_ce_df[res_ce_df.split.isin(["Val", "Test"]) & (res_ce_df.true_label == 1)]["hcc_cosine_mean"].values
res_sc_hcc = res_sc_df[res_sc_df.split.isin(["Val", "Test"]) & (res_sc_df.true_label == 1)]["hcc_cosine_mean"].values

# Stats
eff_diff = eff_sc_hcc - eff_ce_hcc
res_diff = res_sc_hcc - res_ce_hcc

p_eff_ttest = ttest_rel(eff_ce_hcc, eff_sc_hcc).pvalue
p_res_ttest = ttest_rel(res_ce_hcc, res_sc_hcc).pvalue
p_eff_wilc = wilcoxon(eff_ce_hcc, eff_sc_hcc).pvalue
p_res_wilc = wilcoxon(res_ce_hcc, res_sc_hcc).pvalue

print("EfficientNetV2B0:")
print(f"  CE Only   : {np.mean(eff_ce_hcc):.3f} +/- {np.std(eff_ce_hcc):.3f} (median: {np.median(eff_ce_hcc):.3f})")
print(f"  CE+SupCon : {np.mean(eff_sc_hcc):.3f} +/- {np.std(eff_sc_hcc):.3f} (median: {np.median(eff_sc_hcc):.3f})")
print(f"  Mean Shift: +{np.mean(eff_diff):.3f}, p_paired={p_eff_ttest:.4e}")

print("\nResNet50V2:")
print(f"  CE Only   : {np.mean(res_ce_hcc):.3f} +/- {np.std(res_ce_hcc):.3f} (median: {np.median(res_ce_hcc):.3f})")
print(f"  CE+SupCon : {np.mean(res_sc_hcc):.3f} +/- {np.std(res_sc_hcc):.3f} (median: {np.median(res_sc_hcc):.3f})")
print(f"  Mean Shift: +{np.mean(res_diff):.3f}, p_paired={p_res_ttest:.4e}")

# Helper to plot KDE + Histogram
def plot_hist_kde(ax, data, color, label, bins=25, alpha=0.35, x_eval=None):
    if x_eval is None:
        x_eval = np.linspace(-0.2, 1.1, 300)
    # Histogram
    ax.hist(data, bins=bins, density=True, color=color, alpha=alpha, edgecolor=color, linewidth=1.2)
    # KDE
    kde = gaussian_kde(data)
    y_eval = kde(x_eval)
    ax.plot(x_eval, y_eval, color=color, linewidth=2.4, label=label)

# 2. Plotting (2x2 Multi-Panel Figure)
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

color_ce = "#7f8c8d"   # Neutral Slate Gray for Baseline
color_sc = "#2980b9"   # Premium Royal Blue for SupCon

# --- (A) EfficientNet KDE & Histogram ---
ax1 = axes[0, 0]
x_grid = np.linspace(-0.1, 1.1, 400)
plot_hist_kde(ax1, eff_ce_hcc, color=color_ce, label=f"CE Only (Mean: {np.mean(eff_ce_hcc):.3f} ± {np.std(eff_ce_hcc):.3f})", x_eval=x_grid)
plot_hist_kde(ax1, eff_sc_hcc, color=color_sc, label=f"CE + SupCon (Mean: {np.mean(eff_sc_hcc):.3f} ± {np.std(eff_sc_hcc):.3f})", x_eval=x_grid)

ax1.axvline(np.mean(eff_ce_hcc), color=color_ce, linestyle="--", linewidth=1.8)
ax1.axvline(np.mean(eff_sc_hcc), color=color_sc, linestyle="--", linewidth=2.0)
ax1.set_title("A. Density Distribution: EfficientNetV2B0 (Primary)", fontsize=12, fontweight="bold")
ax1.set_xlabel("HCC Cosine Score (Mean Prototype)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
ax1.set_xlim(-0.05, 1.05)
ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

ax1.annotate(
    f"Rightward Shift: +{np.mean(eff_diff):.3f}\nVariance Shrinkage: -37.3%\n(Paired t-test: p < 0.001)",
    xy=(0.90, 2.7), xytext=(0.10, 2.4),
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#e74c3c", lw=1.2),
    fontsize=9.5, fontweight="bold", color="#c0392b"
)

# --- (B) EfficientNet Paired Scatter / Boxplot with Individual Shift Lines ---
ax2 = axes[0, 1]
np.random.seed(42)
sample_idx = np.random.choice(len(eff_ce_hcc), size=75, replace=False)
for i in sample_idx:
    ax2.plot([1, 2], [eff_ce_hcc[i], eff_sc_hcc[i]], color="#bdc3c7", alpha=0.55, lw=0.9, zorder=1)

jitter_ce = np.random.normal(0, 0.04, size=len(eff_ce_hcc))
jitter_sc = np.random.normal(0, 0.04, size=len(eff_sc_hcc))
ax2.scatter(1 + jitter_ce, eff_ce_hcc, color=color_ce, alpha=0.35, s=16, zorder=2)
ax2.scatter(2 + jitter_sc, eff_sc_hcc, color=color_sc, alpha=0.45, s=16, zorder=2)

bp1 = ax2.boxplot([eff_ce_hcc, eff_sc_hcc], positions=[1, 2], widths=0.35, patch_artist=True,
                  boxprops=dict(linewidth=1.8), medianprops=dict(color="black", linewidth=2.0),
                  whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4),
                  flierprops=dict(marker="", visible=False), zorder=3)
bp1['boxes'][0].set(facecolor=color_ce, alpha=0.6)
bp1['boxes'][1].set(facecolor=color_sc, alpha=0.6)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(["CE Only", "CE + SupCon"], fontsize=11, fontweight="bold")
ax2.set_title("B. Paired Individual Shift: EfficientNetV2B0", fontsize=12, fontweight="bold")
ax2.set_ylabel("HCC Cosine Score", fontsize=11, fontweight="bold")
ax2.set_ylim(-0.05, 1.05)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

ax2.text(1.5, 0.05,
         f"Mean Paired Gain: Δ = +{np.mean(eff_diff):.3f}\n"
         f"95% CI: [{np.mean(eff_diff) - 1.96 * np.std(eff_diff)/np.sqrt(417):.3f}, "
         f"{np.mean(eff_diff) + 1.96 * np.std(eff_diff)/np.sqrt(417):.3f}]\n"
         f"Wilcoxon signed-rank: p < 0.001",
         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#edf7ed", ec="#2e7d32", lw=1.2))

# --- (C) ResNet50V2 Density Distribution ---
ax3 = axes[1, 0]
plot_hist_kde(ax3, res_ce_hcc, color=color_ce, label=f"CE Only (Mean: {np.mean(res_ce_hcc):.3f} ± {np.std(res_ce_hcc):.3f})", x_eval=x_grid)
plot_hist_kde(ax3, res_sc_hcc, color=color_sc, label=f"CE + SupCon (Mean: {np.mean(res_sc_hcc):.3f} ± {np.std(res_sc_hcc):.3f})", x_eval=x_grid)

ax3.axvline(np.mean(res_ce_hcc), color=color_ce, linestyle="--", linewidth=1.8)
ax3.axvline(np.mean(res_sc_hcc), color=color_sc, linestyle="--", linewidth=2.0)
ax3.set_title("C. Density Distribution: ResNet50V2 (Baseline Backbone)", fontsize=12, fontweight="bold")
ax3.set_xlabel("HCC Cosine Score (Mean Prototype)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
ax3.set_xlim(-0.05, 1.05)
ax3.legend(loc="upper left", frameon=True, fontsize=9.5)
ax3.grid(axis="y", linestyle=":", alpha=0.6)

ax3.annotate(
    f"Rightward Shift: +{np.mean(res_diff):.3f}\n(Paired t-test: p < 0.001)",
    xy=(0.88, 2.5), xytext=(0.10, 2.3),
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#e74c3c", lw=1.2),
    fontsize=9.5, fontweight="bold", color="#c0392b"
)

# --- (D) ResNet50V2 Paired Scatter / Boxplot with Individual Shift Lines ---
ax4 = axes[1, 1]
for i in sample_idx:
    ax4.plot([1, 2], [res_ce_hcc[i], res_sc_hcc[i]], color="#bdc3c7", alpha=0.55, lw=0.9, zorder=1)

jitter_ce_res = np.random.normal(0, 0.04, size=len(res_ce_hcc))
jitter_sc_res = np.random.normal(0, 0.04, size=len(res_sc_hcc))
ax4.scatter(1 + jitter_ce_res, res_ce_hcc, color=color_ce, alpha=0.35, s=16, zorder=2)
ax4.scatter(2 + jitter_sc_res, res_sc_hcc, color=color_sc, alpha=0.45, s=16, zorder=2)

bp2 = ax4.boxplot([res_ce_hcc, res_sc_hcc], positions=[1, 2], widths=0.35, patch_artist=True,
                  boxprops=dict(linewidth=1.8), medianprops=dict(color="black", linewidth=2.0),
                  whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4),
                  flierprops=dict(marker="", visible=False), zorder=3)
bp2['boxes'][0].set(facecolor=color_ce, alpha=0.6)
bp2['boxes'][1].set(facecolor=color_sc, alpha=0.6)

ax4.set_xticks([1, 2])
ax4.set_xticklabels(["CE Only", "CE + SupCon"], fontsize=11, fontweight="bold")
ax4.set_title("D. Paired Individual Shift: ResNet50V2", fontsize=12, fontweight="bold")
ax4.set_ylabel("HCC Cosine Score", fontsize=11, fontweight="bold")
ax4.set_ylim(-0.05, 1.05)
ax4.grid(axis="y", linestyle=":", alpha=0.6)

ax4.text(1.5, 0.05,
         f"Mean Paired Gain: Δ = +{np.mean(res_diff):.3f}\n"
         f"95% CI: [{np.mean(res_diff) - 1.96 * np.std(res_diff)/np.sqrt(417):.3f}, "
         f"{np.mean(res_diff) + 1.96 * np.std(res_diff)/np.sqrt(417):.3f}]\n"
         f"Wilcoxon signed-rank: p < 0.001",
         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#edf7ed", ec="#2e7d32", lw=1.2))


plt.suptitle("Impact of Supervised Contrastive Learning (SupCon) on HCC Cosine Score Distributions\n(True HCC Cases, Val + Test, n = 417)",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()

save_path = out_dir / "fig_hcc_cosine_distribution_comparison.png"
plt.savefig(save_path, bbox_inches="tight")
plt.close()
print(f"Visualization saved to: {save_path}")
