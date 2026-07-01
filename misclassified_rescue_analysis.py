import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"acc": np.nan, "f1": np.nan, "ppv": np.nan, "npv": np.nan, "sens": np.nan, "spec": np.nan, "cm": "N/A", "rescue_rate": np.nan}
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ppv = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        cm_str = f"TP:{tp} FP:{fp} FN:{fn} TN:{tn}"
    except ValueError:
        spec = np.nan
        npv = np.nan
        cm_str = "Error"
        
    return {"acc": acc, "f1": f1, "ppv": ppv, "npv": npv, "sens": sens, "spec": spec, "cm": cm_str, "rescue_rate": acc}

def get_cutoff(df, score_col):
    val_df = df[df['split'] == 'Val']
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(val_df['true_label'], val_df[score_col])
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def analyze_rescue(name, supcon_path):
    print(f"=== {name} ===")
    
    df_sup = pd.read_csv(supcon_path)
    
    # Calculate cutoffs on Val
    cutoff_conf = get_cutoff(df_sup, 'confidence_score')
    cutoff_hcc = get_cutoff(df_sup, 'hcc_cosine_mean')
    cutoff_delta = get_cutoff(df_sup, 'delta_mean')
    
    print(f"Cutoffs -> Conf: {cutoff_conf:.4f}, HCC: {cutoff_hcc:.4f}, Delta (Youden): {cutoff_delta:.4f}")
    
    # Evaluate on Val + Test
    df_eval = df_sup[df_sup['split'].isin(['Val', 'Test'])].copy()
    
    # Predict using Confidence
    df_eval['pred_conf'] = (df_eval['confidence_score'] >= cutoff_conf).astype(int)
    
    # Extract misclassified by Confidence (FP / FN)
    misclassified = df_eval[df_eval['pred_conf'] != df_eval['true_label']].copy()
    total_errors = len(misclassified)
    print(f"Total misclassified by Confidence (FP/FN): {total_errors}")
    
    if total_errors > 0:
        y_true = misclassified['true_label']
        
        # 1. Predict using HCC score cutoff
        y_pred_hcc = (misclassified['hcc_cosine_mean'] >= cutoff_hcc).astype(int)
        metrics_hcc = calculate_metrics(y_true, y_pred_hcc)
        print(f"  [Rescue by HCC Score Cutoff ({cutoff_hcc:.4f})]")
        print(f"  Rescue Rate (Acc): {metrics_hcc['rescue_rate']*100:.1f}%, F1: {metrics_hcc['f1']:.4f}, PPV: {metrics_hcc['ppv']:.4f}, NPV: {metrics_hcc['npv']:.4f}, Sens: {metrics_hcc['sens']:.4f}, Spec: {metrics_hcc['spec']:.4f} | CM: {metrics_hcc['cm']}")
        
        # 2. Predict using Delta score cutoff
        y_pred_delta = (misclassified['delta_mean'] >= cutoff_delta).astype(int)
        metrics_delta = calculate_metrics(y_true, y_pred_delta)
        print(f"  [Rescue by Delta Score Cutoff ({cutoff_delta:.4f})]")
        print(f"  Rescue Rate (Acc): {metrics_delta['rescue_rate']*100:.1f}%, F1: {metrics_delta['f1']:.4f}, PPV: {metrics_delta['ppv']:.4f}, NPV: {metrics_delta['npv']:.4f}, Sens: {metrics_delta['sens']:.4f}, Spec: {metrics_delta['spec']:.4f} | CM: {metrics_delta['cm']}")
        
        # 3. Predict using Delta score sign (0.0)
        y_pred_delta_sign = (misclassified['delta_mean'] >= 0.0).astype(int)
        metrics_delta_sign = calculate_metrics(y_true, y_pred_delta_sign)
        print(f"  [Rescue by Delta Score Sign (0.0)]")
        print(f"  Rescue Rate (Acc): {metrics_delta_sign['rescue_rate']*100:.1f}%, F1: {metrics_delta_sign['f1']:.4f}, PPV: {metrics_delta_sign['ppv']:.4f}, NPV: {metrics_delta_sign['npv']:.4f}, Sens: {metrics_delta_sign['sens']:.4f}, Spec: {metrics_delta_sign['spec']:.4f} | CM: {metrics_delta_sign['cm']}")
    else:
        print("  No misclassified cases to rescue.")
    print("\n")

if __name__ == "__main__":
    eff_sup = "d:/fm_paper_works/hcc-vs-hemangioma/cosine_probe_result/effnet_supcon/all_scores.csv"
    res_sup = "d:/fm_paper_works/hcc-vs-hemangioma/cosine_probe_result/resnet_supcon/all_scores.csv"
    
    analyze_rescue("EfficientNetV2B0 (SupCon)", eff_sup)
    analyze_rescue("ResNet50V2 (SupCon)", res_sup)
