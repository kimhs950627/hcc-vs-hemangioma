import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"acc": np.nan, "f1": np.nan, "ppv": np.nan, "npv": np.nan, "sens": np.nan, "spec": np.nan, "cm": "N/A"}
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
        
    return {"acc": acc, "f1": f1, "ppv": ppv, "npv": npv, "sens": sens, "spec": spec, "cm": cm_str}

def get_cutoff(df, score_col):
    val_df = df[df['split'] == 'Val']
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(val_df['true_label'], val_df[score_col])
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def analyze_model(name, supcon_path, class_only_path):
    print(f"=== {name} ===")
    
    df_sup = pd.read_csv(supcon_path)
    df_class = pd.read_csv(class_only_path)
    
    # Check column names in class_only
    # It might be different, let's assume 'confidence_score' and 'true_label'
    if 'confidence_score' not in df_class.columns:
        # maybe it's named differently, let's print columns if it fails
        pass

    # Calculate cutoffs on Val
    cutoff_hcc = get_cutoff(df_sup, 'hcc_cosine_mean')
    cutoff_delta = get_cutoff(df_sup, 'delta_mean')
    print(f"Cutoffs from Val -> HCC Cosine: {cutoff_hcc:.4f}, Delta: {cutoff_delta:.4f}")
    
    # 1. Supcon: ambiguous cases (0.4 <= confidence <= 0.6)
    sup_ambig = df_sup[(df_sup['confidence_score'] >= 0.4) & (df_sup['confidence_score'] <= 0.6)]
    print(f"Supcon ambiguous cases: {len(sup_ambig)}")
    
    if len(sup_ambig) > 0:
        y_true_sup = sup_ambig['true_label']
        
        # Predict using HCC score
        y_pred_hcc = (sup_ambig['hcc_cosine_mean'] >= cutoff_hcc).astype(int)
        metrics_hcc = calculate_metrics(y_true_sup, y_pred_hcc)
        print(f"  [Supcon - HCC Score] Acc: {metrics_hcc['acc']:.4f}, F1: {metrics_hcc['f1']:.4f}, PPV: {metrics_hcc['ppv']:.4f}, NPV: {metrics_hcc['npv']:.4f}, Sens: {metrics_hcc['sens']:.4f}, Spec: {metrics_hcc['spec']:.4f} | CM: {metrics_hcc['cm']}")
        
        # Predict using Delta score
        y_pred_delta = (sup_ambig['delta_mean'] >= cutoff_delta).astype(int)
        metrics_delta = calculate_metrics(y_true_sup, y_pred_delta)
        print(f"  [Supcon - Delta Score] Acc: {metrics_delta['acc']:.4f}, F1: {metrics_delta['f1']:.4f}, PPV: {metrics_delta['ppv']:.4f}, NPV: {metrics_delta['npv']:.4f}, Sens: {metrics_delta['sens']:.4f}, Spec: {metrics_delta['spec']:.4f} | CM: {metrics_delta['cm']}")
    else:
        print("  No ambiguous cases for Supcon.")

    # 2. Classification Only: ambiguous cases (0.4 <= confidence <= 0.6)
    # Some raw files might have confidence in 'pred_prob' or something. Let's handle it.
    conf_col = 'confidence_score' if 'confidence_score' in df_class.columns else 'prob_1' # guess
    if conf_col not in df_class.columns:
        if 'probability' in df_class.columns: conf_col = 'probability'
        elif 'score' in df_class.columns: conf_col = 'score'
    
    class_ambig = df_class[(df_class[conf_col] >= 0.4) & (df_class[conf_col] <= 0.6)]
    print(f"Classification-only ambiguous cases: {len(class_ambig)}")
    
    if len(class_ambig) > 0:
        y_true_class = class_ambig['true_label'] if 'true_label' in class_ambig.columns else class_ambig['label']
        # Predict using confidence 0.5
        y_pred_class = (class_ambig[conf_col] >= 0.5).astype(int)
        metrics_class = calculate_metrics(y_true_class, y_pred_class)
        print(f"  [Class Only - Conf 0.5] Acc: {metrics_class['acc']:.4f}, F1: {metrics_class['f1']:.4f}, PPV: {metrics_class['ppv']:.4f}, NPV: {metrics_class['npv']:.4f}, Sens: {metrics_class['sens']:.4f}, Spec: {metrics_class['spec']:.4f} | CM: {metrics_class['cm']}")
    else:
        print("  No ambiguous cases for Classification Only.")
        
    print("\n")

if __name__ == "__main__":
    eff_sup = "d:/fm_paper_works/hcc-vs-hemangioma/cosine_probe_result/effnet_supcon/all_scores.csv"
    eff_class = "d:/fm_paper_works/hcc-vs-hemangioma/cosine_probe_result/raw_cosprobe_effnet_classification_only.csv"
    analyze_model("EfficientNetV2B0", eff_sup, eff_class)
    
    res_sup = "d:/fm_paper_works/hcc-vs-hemangioma/cosine_probe_result/resnet_supcon/all_scores.csv"
    res_class = "d:/fm_paper_works/hcc-vs-hemangioma/cosine_probe_result/raw_cosprobe_resnet_classification_only.csv"
    analyze_model("ResNet50V2", res_sup, res_class)
