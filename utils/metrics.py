"""
utils/metrics.py
Binary classification metrics for HCC vs. Hemangioma benchmark.

Computes:
  - Accuracy
  - ROC-AUC
  - Sensitivity  (Recall / TPR)
  - Specificity  (TNR)
  - PPV          (Positive Predictive Value / Precision)
  - NPV          (Negative Predictive Value)
  - Optimal Youden-index threshold (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core metric function
# ─────────────────────────────────────────────────────────────────────────────

def binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    compute_youden: bool = True,
) -> dict:
    """Compute a full set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 or 1), shape (N,).
        y_prob: Predicted probabilities for class 1, shape (N,).
        threshold: Decision threshold for hard predictions (default 0.5).
        compute_youden: If True, also return the Youden-optimal threshold and
                        metrics at that threshold.

    Returns:
        dict with keys:
            accuracy, roc_auc, sensitivity, specificity, ppv, npv,
            tp, tn, fp, fn, threshold
            (optionally: youden_threshold, youden_sensitivity, youden_specificity,
             youden_ppv, youden_npv, youden_accuracy)
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float32).ravel()

    # ── Confusion matrix at the given threshold ───────────────────────────────
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv         = tp / max(tp + fp, 1)
    npv         = tn / max(tn + fn, 1)

    # ── ROC-AUC via trapezoidal rule ──────────────────────────────────────────
    roc_auc = _roc_auc(y_true, y_prob)

    result = dict(
        accuracy=float(accuracy),
        roc_auc=float(roc_auc),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        ppv=float(ppv),
        npv=float(npv),
        tp=tp, tn=tn, fp=fp, fn=fn,
        threshold=float(threshold),
    )

    # ── Youden-optimal threshold ───────────────────────────────────────────────
    if compute_youden:
        best_thresh, fpr_arr, tpr_arr = _youden_threshold(y_true, y_prob)
        y_pred_y = (y_prob >= best_thresh).astype(np.int32)
        tp_y = int(np.sum((y_pred_y == 1) & (y_true == 1)))
        tn_y = int(np.sum((y_pred_y == 0) & (y_true == 0)))
        fp_y = int(np.sum((y_pred_y == 1) & (y_true == 0)))
        fn_y = int(np.sum((y_pred_y == 0) & (y_true == 1)))
        result.update(
            youden_threshold=float(best_thresh),
            youden_sensitivity=tp_y / max(tp_y + fn_y, 1),
            youden_specificity=tn_y / max(tn_y + fp_y, 1),
            youden_ppv=tp_y / max(tp_y + fp_y, 1),
            youden_npv=tn_y / max(tn_y + fn_y, 1),
            youden_accuracy=(tp_y + tn_y) / max(tp_y + tn_y + fp_y + fn_y, 1),
            roc_fpr=fpr_arr.tolist(),
            roc_tpr=tpr_arr.tolist(),
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC-AUC via the trapezoidal rule (no sklearn dependency)."""
    thresholds = np.concatenate([[1.0 + 1e-6], np.sort(np.unique(y_prob))[::-1], [0.0]])
    fprs, tprs = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tprs.append(tp / max(n_pos, 1))
        fprs.append(fp / max(n_neg, 1))
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    # Sort by FPR for trapz
    idx = np.argsort(fprs_arr)
    return float(np.trapz(tprs_arr[idx], fprs_arr[idx]))


def _youden_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the Youden-optimal threshold and (fpr, tpr) arrays."""
    thresholds = np.concatenate([[1.0 + 1e-6], np.sort(np.unique(y_prob))[::-1], [0.0]])
    fprs, tprs = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tprs.append(tp / max(n_pos, 1))
        fprs.append(fp / max(n_neg, 1))
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    youden = tprs_arr - fprs_arr
    best_idx = int(np.argmax(youden))
    best_thresh = float(thresholds[best_idx])
    return best_thresh, fprs_arr, tprs_arr


# ─────────────────────────────────────────────────────────────────────────────
# Save / load helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_json(metrics: dict, path: str | Path) -> None:
    """Save a metrics dict to JSON, stripping large list fields."""
    slim = {k: v for k, v in metrics.items() if k not in ("roc_fpr", "roc_tpr")}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)


def format_metrics_string(metrics: dict) -> str:
    """Return a compact one-line summary string for logging."""
    keys = ["accuracy", "roc_auc", "sensitivity", "specificity", "ppv", "npv"]
    parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
    return "  ".join(parts)
