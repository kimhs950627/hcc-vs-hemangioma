"""
Recompute all metrics, tables, and generate updated Figures 4 and 5 for manuscript revision.
Target: Korean Journal of Family Practice (KJFP) revision.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import beta, ttest_ind, ttest_rel, wilcoxon

BASE_DIR = r"d:\fm_paper_works\hcc-vs-hemangioma"
PROBE_DIR = os.path.join(BASE_DIR, "cosine_probe_result")
OUT_DIR = os.path.join(BASE_DIR, "paper_submission", "revision2")
os.makedirs(OUT_DIR, exist_ok=True)

def clopper_pearson_ci(k, n, alpha=0.05):
    """Exact Clopper-Pearson binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo * 100.0, hi * 100.0

def compute_binary_metrics(y_true, y_score, threshold):
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    
    sens = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0
    ppv = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    npv = (tn / (tn + fn)) * 100.0 if (tn + fn) > 0 else 0.0
    acc = ((tp + tn) / n) * 100.0 if n > 0 else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    
    sens_ci = clopper_pearson_ci(tp, tp + fn)
    spec_ci = clopper_pearson_ci(tn, tn + fp)
    acc_ci = clopper_pearson_ci(tp + tn, n)
    
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
        "sensitivity": sens, "sens_ci": sens_ci,
        "specificity": spec, "spec_ci": spec_ci,
        "ppv": ppv, "npv": npv,
        "accuracy": acc, "acc_ci": acc_ci,
        "f1": f1
    }

# 1. Load data
files = {
    "effnet_ce": os.path.join(PROBE_DIR, "raw_cosprobe_effnet_classification_only.csv"),
    "effnet_supcon": os.path.join(PROBE_DIR, "raw_cosprobe_effnet_supcon.csv"),
    "effnet_nnclr": os.path.join(PROBE_DIR, "raw_cosprobe_effnet_nnclr.csv"),
    "resnet_ce": os.path.join(PROBE_DIR, "raw_cosprobe_resnet_classification_only.csv"),
    "resnet_supcon": os.path.join(PROBE_DIR, "raw_cosprobe_res_supcon.csv"),
    "resnet_nnclr": os.path.join(PROBE_DIR, "raw_cosprobe_resnet_nnclr.csv"),
}

dfs = {k: pd.read_csv(v) for k, v in files.items()}

# --- Table 2 Recomputation (Validation set, confidence_score) ---
table2_rows = []
model_names = [
    (1, "ResNet50V2", "CE Only", "resnet_ce"),
    (2, "EfficientNetV2B0", "CE Only", "effnet_ce"),
    (3, "ResNet50V2", "CE + SupCon", "resnet_supcon"),
    (4, "EfficientNetV2B0", "CE + SupCon", "effnet_supcon"),
    (5, "ResNet50V2", "NNCLR (SSL)", "resnet_nnclr"),
    (6, "EfficientNetV2B0", "NNCLR (SSL)", "effnet_nnclr"),
]

for idx, backbone, mode, key in model_names:
    df = dfs[key]
    val = df[df.split == "Val"]
    fpr, tpr, thr = roc_curve(val.true_label, val.confidence_score)
    j_idx = np.argmax(tpr - fpr)
    cut = thr[j_idx]
    auc = roc_auc_score(val.true_label, val.confidence_score)
    m = compute_binary_metrics(val.true_label.values, val.confidence_score.values, cut)
    table2_rows.append({
        "#": idx,
        "Backbone": backbone,
        "Training Mode": mode,
        "Cutoff": cut,
        "AUROC": f"{auc:.4f}",
        "Sensitivity (%)": f"{m['sensitivity']:.2f}",
        "Specificity (%)": f"{m['specificity']:.2f}",
        "F1 Score": f"{m['f1']:.4f}",
        "TP/FP/FN/TN": f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"
    })

df_table2 = pd.DataFrame(table2_rows)

# --- Table 3 Recomputation (Primary Model: EfficientNetV2B0 + CE + SupCon) ---
primary_df = dfs["effnet_supcon"]
val_prim = primary_df[primary_df.split == "Val"]
test_prim = primary_df[primary_df.split == "Test"]

fpr, tpr, thr = roc_curve(val_prim.true_label, val_prim.confidence_score)
prim_cut = thr[np.argmax(tpr - fpr)] # 0.0010883796...

val_m = compute_binary_metrics(val_prim.true_label.values, val_prim.confidence_score.values, prim_cut)
test_m = compute_binary_metrics(test_prim.true_label.values, test_prim.confidence_score.values, prim_cut)

val_auc = roc_auc_score(val_prim.true_label, val_prim.confidence_score)
test_auc = roc_auc_score(test_prim.true_label, test_prim.confidence_score)

table3_rows = [
    {"Metric": "AUROC", "Validation Set": f"{val_auc:.3f}", "Test Set": f"{test_auc:.3f}"},
    {"Metric": "Accuracy (%) (95% CI)", 
     "Validation Set": f"{val_m['accuracy']:.2f} ({val_m['acc_ci'][0]:.2f}–{val_m['acc_ci'][1]:.2f})", 
     "Test Set": f"{test_m['accuracy']:.2f} ({test_m['acc_ci'][0]:.2f}–{test_m['acc_ci'][1]:.2f})"},
    {"Metric": "Sensitivity (%) (95% CI)", 
     "Validation Set": f"{val_m['sensitivity']:.2f} ({val_m['sens_ci'][0]:.2f}–{val_m['sens_ci'][1]:.2f})", 
     "Test Set": f"{test_m['sensitivity']:.2f} ({test_m['sens_ci'][0]:.2f}–{test_m['sens_ci'][1]:.2f})"},
    {"Metric": "Specificity (%) (95% CI)", 
     "Validation Set": f"{val_m['specificity']:.2f} ({val_m['spec_ci'][0]:.2f}–{val_m['spec_ci'][1]:.2f})", 
     "Test Set": f"{test_m['specificity']:.2f} ({test_m['spec_ci'][0]:.2f}–{test_m['spec_ci'][1]:.2f})"},
    {"Metric": "PPV (%)", "Validation Set": f"{val_m['ppv']:.2f}", "Test Set": f"{test_m['ppv']:.2f}"},
    {"Metric": "NPV (%)", "Validation Set": f"{val_m['npv']:.2f}", "Test Set": f"{test_m['npv']:.2f}"},
    {"Metric": "F1 Score", "Validation Set": f"{val_m['f1']:.4f}", "Test Set": f"{test_m['f1']:.4f}"},
    {"Metric": "TP / FP / FN / TN", 
     "Validation Set": f"{val_m['tp']} / {val_m['fp']} / {val_m['fn']} / {val_m['tn']}", 
     "Test Set": f"{test_m['tp']} / {test_m['fp']} / {test_m['fn']} / {test_m['tn']}"},
]
df_table3 = pd.DataFrame(table3_rows)

# --- Table 4 Recomputation (Outputs comparison for Primary Model) ---
score_cols = [
    ("ROC-A", "Confidence Score", "—", "confidence_score"),
    ("ROC-B", "HCC Cosine Score", "mean", "hcc_cosine_mean"),
    ("ROC-C", "Δscore", "mean", "delta_mean"),
    ("ROC-B", "HCC Cosine Score", "k-means (k=4)", "hcc_cosine_kmeans"),
    ("ROC-C", "Δscore", "k-means (k=4)", "delta_kmeans"),
]

table4_rows = []
for code, stype, proto, col in score_cols:
    fpr, tpr, thr = roc_curve(val_prim.true_label, val_prim[col])
    cut = thr[np.argmax(tpr - fpr)]
    v_auc = roc_auc_score(val_prim.true_label, val_prim[col])
    t_auc = roc_auc_score(test_prim.true_label, test_prim[col])
    vm = compute_binary_metrics(val_prim.true_label.values, val_prim[col].values, cut)
    tm = compute_binary_metrics(test_prim.true_label.values, test_prim[col].values, cut)
    table4_rows.append({
        "Output": code,
        "Score Type": stype,
        "Prototype": proto,
        "Cutoff (Val)": f"{cut:.4f}",
        "AUROC (Val)": f"{v_auc:.3f}",
        "AUROC (Test)": f"{t_auc:.3f}",
        "Test Sens (%)": f"{tm['sensitivity']:.2f}",
        "Test Spec (%)": f"{tm['specificity']:.2f}",
        "Test Errors (FP/FN)": f"{tm['fp']}/{tm['fn']}"
    })
df_table4 = pd.DataFrame(table4_rows)

# --- Save Summary to Markdown ---
summary_md_path = os.path.join(OUT_DIR, "recalculated_tables.md")
with open(summary_md_path, "w", encoding="utf-8") as f:
    f.write("# Recomputed Metrics and Tables for Revision\n\n")
    f.write("## Table 2. Backbone × Training Mode (Validation Set, Confidence Score)\n")
    f.write(df_table2.to_markdown(index=False))
    f.write("\n\n## Table 3. Primary Model Confidence Score Performance (EfficientNetV2B0 + CE + SupCon)\n")
    f.write(f"Operating Cutoff (Youden J on Val): {prim_cut:.6f}\n\n")
    f.write(df_table3.to_markdown(index=False))
    f.write("\n\n## Table 4. Dual-Output Performance Comparison (Primary Model)\n")
    f.write(df_table4.to_markdown(index=False))
    f.write("\n\n*Note: In the test set (n=268), all 5 outputs misclassify the exact same single image (index 576, Hemangioma predicted as HCC, FP=1, FN=0).*")

print("Saved recalculated tables to markdown.")

# --- Figure 4 & Figure 5 Regeneration: True HCC Cases (Val + Test, n=417) ---
# Filter True HCC cases
eff_ce_hcc = dfs["effnet_ce"][dfs["effnet_ce"].split.isin(["Val", "Test"]) & (dfs["effnet_ce"].true_label == 1)]
eff_sc_hcc = dfs["effnet_supcon"][dfs["effnet_supcon"].split.isin(["Val", "Test"]) & (dfs["effnet_supcon"].true_label == 1)]
res_ce_hcc = dfs["resnet_ce"][dfs["resnet_ce"].split.isin(["Val", "Test"]) & (dfs["resnet_ce"].true_label == 1)]
res_sc_hcc = dfs["resnet_supcon"][dfs["resnet_supcon"].split.isin(["Val", "Test"]) & (dfs["resnet_supcon"].true_label == 1)]

# Paired t-tests
p_eff_cos = ttest_rel(eff_ce_hcc.hcc_cosine_mean.values, eff_sc_hcc.hcc_cosine_mean.values).pvalue
p_res_cos = ttest_rel(res_ce_hcc.hcc_cosine_mean.values, res_sc_hcc.hcc_cosine_mean.values).pvalue

p_eff_delta = ttest_rel(eff_ce_hcc.delta_mean.values, eff_sc_hcc.delta_mean.values).pvalue
p_res_delta = ttest_rel(res_ce_hcc.delta_mean.values, res_sc_hcc.delta_mean.values).pvalue

print(f"Paired t-test HCC Cosine Score: EffNet p={p_eff_cos:.4e}, ResNet p={p_res_cos:.4e}")
print(f"Paired t-test Delta Score: EffNet p={p_eff_delta:.4e}, ResNet p={p_res_delta:.4e}")

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def plot_boxplot_with_jitter(data_dict, title, ylabel, p_vals, out_filename, ylim=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    # 4 groups: EffNet CE, EffNet SupCon, ResNet CE, ResNet SupCon
    positions = [1, 2, 3.8, 4.8]
    keys = ["eff_ce", "eff_sc", "res_ce", "res_sc"]
    colors = ['#c2c2c2', '#4a90e2', '#c2c2c2', '#4a90e2']
    
    data = [data_dict[k] for k in keys]
    
    bp = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='', visible=False))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor('#222222' if color == '#c2c2c2' else '#1a4e8a')
        
    # Jitter points
    np.random.seed(42)
    for pos, vals in zip(positions, data):
        jitter = np.random.normal(0, 0.08, size=len(vals))
        ax.scatter(pos + jitter, vals, color='#333333', alpha=0.18, s=12, zorder=3)
        
    ax.set_xticks([1.5, 4.3])
    ax.set_xticklabels(['EfficientNetV2B0', 'ResNet50V2'], fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=18)
    
    # Annotate p-values
    # EffNet
    y_max_eff = max(np.max(data[0]), np.max(data[1]))
    ax.text(1.5, y_max_eff + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            f"CE vs SupCon (paired)\n{p_vals[0]}", 
            ha='center', va='bottom', fontsize=9.5, color='#a00000', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff2f2", ec="#e0a0a0", lw=0.8))
    
    # ResNet
    y_max_res = max(np.max(data[2]), np.max(data[3]))
    ax.text(4.3, y_max_res + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            f"CE vs SupCon (paired)\n{p_vals[1]}", 
            ha='center', va='bottom', fontsize=9.5, color='#a00000', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff2f2", ec="#e0a0a0", lw=0.8))
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#c2c2c2', edgecolor='#222222', alpha=0.65, label='CE only'),
        Patch(facecolor='#4a90e2', edgecolor='#1a4e8a', alpha=0.65, label='SupCon + CE')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(ylim)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, out_filename))
    plt.close()
    print(f"Generated {out_filename}")

# Generate Figure 4 (HCC Cosine Score in True HCC)
p_str_eff_cos = "p < 0.001 ***" if p_eff_cos < 0.001 else f"p = {p_eff_cos:.3f}"
p_str_res_cos = "p < 0.001 ***" if p_res_cos < 0.001 else f"p = {p_res_cos:.3f}"
plot_boxplot_with_jitter(
    {
        "eff_ce": eff_ce_hcc.hcc_cosine_mean.values,
        "eff_sc": eff_sc_hcc.hcc_cosine_mean.values,
        "res_ce": res_ce_hcc.hcc_cosine_mean.values,
        "res_sc": res_sc_hcc.hcc_cosine_mean.values,
    },
    title="Comparison of HCC Cosine Scores in True HCC Cases (n = 417)",
    ylabel="HCC Cosine Score (Mean Prototype)",
    p_vals=[p_str_eff_cos, p_str_res_cos],
    out_filename="fig4_true_hcc_cosine.png",
    ylim=(0.0, 1.25)
)

# Generate Figure 5 (Delta Score in True HCC)
p_str_eff_delta = "p < 0.001 ***" if p_eff_delta < 0.001 else f"p = {p_eff_delta:.3f}"
p_str_res_delta = "p < 0.001 ***" if p_res_delta < 0.001 else f"p = {p_res_delta:.3f}"
plot_boxplot_with_jitter(
    {
        "eff_ce": eff_ce_hcc.delta_mean.values,
        "eff_sc": eff_sc_hcc.delta_mean.values,
        "res_ce": res_ce_hcc.delta_mean.values,
        "res_sc": res_sc_hcc.delta_mean.values,
    },
    title="Comparison of ΔScores in True HCC Cases (n = 417)",
    ylabel="Δscore (HCC Cosine − Hemangioma Cosine)",
    p_vals=[p_str_eff_delta, p_str_res_delta],
    out_filename="fig5_true_hcc_delta.png",
    ylim=(0.0, 2.25)
)

print("All figures and table recomputations completed successfully.")
