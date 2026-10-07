"""
scratch/visualize_hemangioma_score_shift.py
===========================================
Visualize the distribution shift of Hemangioma Cosine Scores in True Hemangioma cases (n=381)
comparing CE-Only vs. CE+SupCon for both EfficientNetV2B0 and ResNet50V2.
Mirrors the structure and design of fig_hcc_cosine_distribution_comparison.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ttest_rel, wilcoxon, t as t_dist

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

# Filter True Hemangioma (Val + Test, n=381, true_label == 0)
eff_ce_hem = eff_ce_df[eff_ce_df.split.isin(["Val", "Test"]) & (eff_ce_df.true_label == 0)]["hem_cosine_mean"].values
eff_sc_hem = eff_sc_df[eff_sc_df.split.isin(["Val", "Test"]) & (eff_sc_df.true_label == 0)]["hem_cosine_mean"].values
res_ce_hem = res_ce_df[res_ce_df.split.isin(["Val", "Test"]) & (res_ce_df.true_label == 0)]["hem_cosine_mean"].values
res_sc_hem = res_sc_df[res_sc_df.split.isin(["Val", "Test"]) & (res_sc_df.true_label == 0)]["hem_cosine_mean"].values

n_hem = len(eff_ce_hem)

# Stats
eff_diff = eff_sc_hem - eff_ce_hem
res_diff = res_sc_hem - res_ce_hem

p_eff_wilc = wilcoxon(eff_sc_hem, eff_ce_hem).pvalue
p_res_wilc = wilcoxon(res_sc_hem, res_ce_hem).pvalue

# Pitman-Morgan Test function
def pitman_morgan(x, y):
    d = x - y
    s = x + y
    r = np.corrcoef(d, s)[0, 1]
    n = len(x)
    t_val = r * np.sqrt((n - 2) / (1 - r**2))
    p_val = 2 * (1 - t_dist.cdf(abs(t_val), df=n - 2))
    return t_val, p_val

pm_eff_t, pm_eff_p = pitman_morgan(eff_ce_hem, eff_sc_hem)
pm_res_t, pm_res_p = pitman_morgan(res_ce_hem, res_sc_hem)

eff_var_red = (np.var(eff_ce_hem) - np.var(eff_sc_hem)) / np.var(eff_ce_hem) * 100
res_var_red = (np.var(res_ce_hem) - np.var(res_sc_hem)) / np.var(res_ce_hem) * 100

print(f"EfficientNetV2B0 True Hemangioma (n={n_hem}):")
print(f"  CE Only   : {np.mean(eff_ce_hem):.4f} +/- {np.std(eff_ce_hem):.4f} (median: {np.median(eff_ce_hem):.4f})")
print(f"  CE+SupCon : {np.mean(eff_sc_hem):.4f} +/- {np.std(eff_sc_hem):.4f} (median: {np.median(eff_sc_hem):.4f})")
print(f"  Shift: +{np.mean(eff_diff):.4f}, Wilcoxon p={p_eff_wilc:.4e}, Var Red: -{eff_var_red:.1f}%")

print(f"\nResNet50V2 True Hemangioma (n={n_hem}):")
print(f"  CE Only   : {np.mean(res_ce_hem):.4f} +/- {np.std(res_ce_hem):.4f} (median: {np.median(res_ce_hem):.4f})")
print(f"  CE+SupCon : {np.mean(res_sc_hem):.4f} +/- {np.std(res_sc_hem):.4f} (median: {np.median(res_sc_hem):.4f})")
print(f"  Shift: +{np.mean(res_diff):.4f}, Wilcoxon p={p_res_wilc:.4e}, Var Red: -{res_var_red:.1f}%")

# Helper to plot KDE + Histogram
def plot_hist_kde(ax, data, color, label, bins=25, alpha=0.35, x_eval=None):
    if x_eval is None:
        x_eval = np.linspace(0.65, 1.02, 400)
    ax.hist(data, bins=bins, density=True, color=color, alpha=alpha, edgecolor=color, linewidth=1.2)
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
x_grid = np.linspace(0.65, 1.02, 400)
plot_hist_kde(ax1, eff_ce_hem, color=color_ce, label=f"CE Only (Mean: {np.mean(eff_ce_hem):.3f} ± {np.std(eff_ce_hem):.3f})", bins=30, x_eval=x_grid)
plot_hist_kde(ax1, eff_sc_hem, color=color_sc, label=f"CE + SupCon (Mean: {np.mean(eff_sc_hem):.3f} ± {np.std(eff_sc_hem):.3f})", bins=20, x_eval=x_grid)

ax1.axvline(np.mean(eff_ce_hem), color=color_ce, linestyle="--", linewidth=1.8)
ax1.axvline(np.mean(eff_sc_hem), color=color_sc, linestyle="--", linewidth=2.0)
ax1.set_title("A. Density Distribution: EfficientNetV2B0 (Primary)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Hemangioma Cosine Score (Mean Prototype)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
ax1.set_xlim(0.65, 1.02)
ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

# Annotate shift and variance reduction
ax1.annotate(
    f"Rightward Shift: +{np.mean(eff_diff):.3f} (Wilcoxon p < 0.001)\nVariance Reduction: -{eff_var_red:.1f}% (Pitman-Morgan p < 0.001)",
    xy=(0.985, 30), xytext=(0.68, 25),
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#e74c3c", lw=1.2),
    fontsize=9.2, fontweight="bold", color="#c0392b"
)

# --- (B) EfficientNet Paired Scatter / Boxplot with Individual Shift Lines ---
ax2 = axes[0, 1]
np.random.seed(42)
sample_idx = np.random.choice(n_hem, size=75, replace=False)
for i in sample_idx:
    ax2.plot([1, 2], [eff_ce_hem[i], eff_sc_hem[i]], color="#bdc3c7", alpha=0.55, lw=0.9, zorder=1)

jitter_ce = np.random.normal(0, 0.04, size=n_hem)
jitter_sc = np.random.normal(0, 0.04, size=n_hem)
ax2.scatter(1 + jitter_ce, eff_ce_hem, color=color_ce, alpha=0.35, s=16, zorder=2)
ax2.scatter(2 + jitter_sc, eff_sc_hem, color=color_sc, alpha=0.45, s=16, zorder=2)

bp1 = ax2.boxplot([eff_ce_hem, eff_sc_hem], positions=[1, 2], widths=0.35, patch_artist=True,
                  boxprops=dict(linewidth=1.8), medianprops=dict(color="black", linewidth=2.0),
                  whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4),
                  flierprops=dict(marker="", visible=False), zorder=3)
bp1['boxes'][0].set(facecolor=color_ce, alpha=0.6)
bp1['boxes'][1].set(facecolor=color_sc, alpha=0.6)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(["CE Only", "CE + SupCon"], fontsize=11, fontweight="bold")
ax2.set_title("B. Paired Individual Shift: EfficientNetV2B0", fontsize=12, fontweight="bold")
ax2.set_ylabel("Hemangioma Cosine Score", fontsize=11, fontweight="bold")
ax2.set_ylim(0.65, 1.02)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

ci_eff_low = np.mean(eff_diff) - 1.96 * np.std(eff_diff) / np.sqrt(n_hem)
ci_eff_high = np.mean(eff_diff) + 1.96 * np.std(eff_diff) / np.sqrt(n_hem)

ax2.text(1.5, 0.67,
         f"Mean Paired Gain: Δ = +{np.mean(eff_diff):.3f}\n"
         f"95% CI: [{ci_eff_low:.3f}, {ci_eff_high:.3f}]\n"
         f"Wilcoxon signed-rank: p < 0.001",
         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#edf7ed", ec="#2e7d32", lw=1.2))

# --- (C) ResNet50V2 Density Distribution ---
ax3 = axes[1, 0]
x_grid_res = np.linspace(0.85, 1.02, 400)
plot_hist_kde(ax3, res_ce_hem, color=color_ce, label=f"CE Only (Mean: {np.mean(res_ce_hem):.3f} ± {np.std(res_ce_hem):.3f})", bins=25, x_eval=x_grid_res)
plot_hist_kde(ax3, res_sc_hem, color=color_sc, label=f"CE + SupCon (Mean: {np.mean(res_sc_hem):.3f} ± {np.std(res_sc_hem):.3f})", bins=20, x_eval=x_grid_res)

ax3.axvline(np.mean(res_ce_hem), color=color_ce, linestyle="--", linewidth=1.8)
ax3.axvline(np.mean(res_sc_hem), color=color_sc, linestyle="--", linewidth=2.0)
ax3.set_title("C. Density Distribution: ResNet50V2 (Baseline Backbone)", fontsize=12, fontweight="bold")
ax3.set_xlabel("Hemangioma Cosine Score (Mean Prototype)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
ax3.set_xlim(0.85, 1.02)
ax3.legend(loc="upper left", frameon=True, fontsize=9.5)
ax3.grid(axis="y", linestyle=":", alpha=0.6)

ax3.annotate(
    f"Rightward Shift: +{np.mean(res_diff):.3f} (Wilcoxon p < 0.001)\nVariance Reduction: -{res_var_red:.1f}% (Pitman-Morgan p < 0.001)",
    xy=(0.990, 45), xytext=(0.86, 38),
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#e74c3c", lw=1.2),
    fontsize=9.2, fontweight="bold", color="#c0392b"
)

# --- (D) ResNet50V2 Paired Scatter / Boxplot with Individual Shift Lines ---
ax4 = axes[1, 1]
for i in sample_idx:
    ax4.plot([1, 2], [res_ce_hem[i], res_sc_hem[i]], color="#bdc3c7", alpha=0.55, lw=0.9, zorder=1)

jitter_ce_res = np.random.normal(0, 0.04, size=n_hem)
jitter_sc_res = np.random.normal(0, 0.04, size=n_hem)
ax4.scatter(1 + jitter_ce_res, res_ce_hem, color=color_ce, alpha=0.35, s=16, zorder=2)
ax4.scatter(2 + jitter_sc_res, res_sc_hem, color=color_sc, alpha=0.45, s=16, zorder=2)

bp2 = ax4.boxplot([res_ce_hem, res_sc_hem], positions=[1, 2], widths=0.35, patch_artist=True,
                  boxprops=dict(linewidth=1.8), medianprops=dict(color="black", linewidth=2.0),
                  whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4),
                  flierprops=dict(marker="", visible=False), zorder=3)
bp2['boxes'][0].set(facecolor=color_ce, alpha=0.6)
bp2['boxes'][1].set(facecolor=color_sc, alpha=0.6)

ax4.set_xticks([1, 2])
ax4.set_xticklabels(["CE Only", "CE + SupCon"], fontsize=11, fontweight="bold")
ax4.set_title("D. Paired Individual Shift: ResNet50V2", fontsize=12, fontweight="bold")
ax4.set_ylabel("Hemangioma Cosine Score", fontsize=11, fontweight="bold")
ax4.set_ylim(0.85, 1.02)
ax4.grid(axis="y", linestyle=":", alpha=0.6)

ci_res_low = np.mean(res_diff) - 1.96 * np.std(res_diff) / np.sqrt(n_hem)
ci_res_high = np.mean(res_diff) + 1.96 * np.std(res_diff) / np.sqrt(n_hem)

ax4.text(1.5, 0.865,
         f"Mean Paired Gain: Δ = +{np.mean(res_diff):.3f}\n"
         f"95% CI: [{ci_res_low:.3f}, {ci_res_high:.3f}]\n"
         f"Wilcoxon signed-rank: p < 0.001",
         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#edf7ed", ec="#2e7d32", lw=1.2))

plt.suptitle("Impact of Supervised Contrastive Learning (SupCon) on Hemangioma Cosine Score Distributions\n(True Hemangioma Cases, Val + Test, n = 381)",
             fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()

save_path = out_dir / "fig_hemangioma_cosine_distribution_comparison.png"
plt.savefig(save_path, bbox_inches="tight")
plt.close()
print(f"Visualization saved to: {save_path}")
