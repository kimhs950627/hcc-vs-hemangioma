# validation/score_probe.py
"""Post-hoc classifier score analysis for radiologic marker development.

Pipeline
--------
1. Extract P(HCC) scores from train / val / test tf.data.Dataset
2. Determine optimal cutoff from **val set only** (no test-data leakage)
   - Strategy: Youden's J  or  sensitivity-first
3. Evaluate all three splits at the val-determined cutoff
4. Produce ROC curves (3-split overlay) and P(HCC) KDE distribution plots
5. Log everything to WandB (scalars + wandb.Image + wandb.Table)

Usage (inside Kaggle notebook, after training)
----------------------------------------------
from validation.score_probe import ScoreProbeConfig, run_score_probe

probe_cfg = ScoreProbeConfig(
    model_id         = model_id,           # e.g. "0x8xv1u0"
    work_dir         = cfg.work_dir,
    cutoff_strategy  = "youden",           # or "sens_first"
    sens_target      = 0.90,               # used only with "sens_first"
    wandb_prefix     = "score_probe",
    positive_class   = 1,                  # 1 = HCC
)

results = run_score_probe(
    stage2_model = classifier,             # PureClassifier or ClassifierTrainer
    train_ds     = train_ds,
    val_ds       = val_ds,
    test_ds      = test_ds,
    cfg          = probe_cfg,
)
# results: dict[cutoff, val_metrics, test_metrics, train_metrics]
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

try:
    import wandb as _wandb
except Exception:
    _wandb = None


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CutoffStrategy = Literal["youden", "sens_first"]


@dataclass
class ScoreProbeConfig:
    """Configuration for score_probe analysis.

    Parameters
    ----------
    model_id        : Experiment identifier (used in file names and WandB keys).
    work_dir        : Root output directory.
    cutoff_strategy :
        "youden"     -> maximise Sensitivity + Specificity - 1 (default)
        "sens_first" -> lowest threshold s.t. sensitivity >= sens_target
    sens_target     : Target sensitivity; used only when cutoff_strategy=="sens_first".
    wandb_prefix    : WandB log key namespace prefix.
    positive_class  : Index of the positive class in the softmax output (1 = HCC).
    class_names     : Display names for [negative, positive].
    """
    model_id        : str
    work_dir        : str
    cutoff_strategy : CutoffStrategy     = "youden"
    sens_target     : float              = 0.90
    wandb_prefix    : str                = "score_probe"
    positive_class  : int                = 1
    class_names     : tuple[str, str]    = ("Hemangioma", "HCC")

    @property
    def out_dir(self) -> Path:
        return Path(self.work_dir) / "output" / "score_probe" / self.model_id


# ─────────────────────────────────────────────────────────────
# Score extraction
# ─────────────────────────────────────────────────────────────

def _extract_scores(
    stage2_model,
    dataset: tf.data.Dataset,
    positive_class: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate dataset and collect P(positive_class) scores and labels.

    Supports
    --------
    - PureClassifier  : output dict with key "probabilities" (B, num_classes)
    - ClassifierTrainer : wraps PureClassifier; unwraps via .model attribute
    - Any callable returning dict with "probabilities" key

    Returns
    -------
    scores : (N,) float32 -- P(positive class)
    labels : (N,) int32
    """
    model = getattr(stage2_model, "model", stage2_model)

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        out = model(x_batch, training=False)
        if isinstance(out, dict):
            probs = out["probabilities"].numpy()           # (B, C)
        else:
            probs = tf.nn.softmax(out, axis=-1).numpy()    # fallback: raw logits
        all_scores.append(probs[:, positive_class].astype(np.float32))
        all_labels.append(y_batch.numpy().astype(np.int32))

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────
# Cutoff determination (Val set only)
# ─────────────────────────────────────────────────────────────

def _find_cutoff(
    val_scores : np.ndarray,
    val_labels : np.ndarray,
    strategy   : CutoffStrategy = "youden",
    sens_target: float          = 0.90,
) -> tuple[float, dict]:
    """Find optimal threshold from val set only.

    Strategies
    ----------
    youden     : argmax(TPR - FPR)              <- balanced; standard for publication
    sens_first : highest threshold s.t. TPR >= sens_target  <- HCC screening priority

    Returns
    -------
    cutoff  : float
    metrics : dict at this cutoff
    """
    fpr, tpr, thresholds = roc_curve(val_labels, val_scores, pos_label=1)

    if strategy == "youden":
        j = tpr - fpr
        best_idx = int(np.argmax(j))
    elif strategy == "sens_first":
        cands = np.where(tpr >= sens_target)[0]
        best_idx = (
            int(cands[np.argmax(thresholds[cands])]) if len(cands) > 0
            else int(np.argmax(tpr))
        )
    else:
        raise ValueError(f"Unknown cutoff_strategy: {strategy!r}")

    cutoff = float(thresholds[best_idx])
    return cutoff, _metrics_at_cutoff(val_scores, val_labels, cutoff)


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _metrics_at_cutoff(
    scores : np.ndarray,
    labels : np.ndarray,
    cutoff : float,
) -> dict:
    preds = (scores >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens  = tp / (tp + fn + 1e-8)
    spec  = tn / (tn + fp + 1e-8)
    ppv   = tp / (tp + fp + 1e-8)
    npv   = tn / (tn + fn + 1e-8)
    f1    = 2.0 * ppv * sens / (ppv + sens + 1e-8)
    auroc = float(roc_auc_score(labels, scores))
    return dict(
        cutoff=cutoff, auroc=auroc,
        sensitivity=float(sens), specificity=float(spec),
        ppv=float(ppv), npv=float(npv), f1=float(f1),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        n=int(len(labels)),
    )


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

_SPLIT_STYLE: dict[str, dict] = {
    "Train": dict(color="#4f98a3", lw=1.5, ls="--", alpha=0.80),
    "Val"  : dict(color="#e8af34", lw=2.0, ls="-",  alpha=0.90),
    "Test" : dict(color="#dd6974", lw=2.5, ls="-",  alpha=1.00),
}
_CLS_COLOR: dict[str, str] = {
    "Hemangioma": "#4f98a3",
    "HCC"        : "#dd6974",
}
_BG     = "#1c1b19"
_GRID   = "#262523"
_TEXT   = "#cdccca"
_BORDER = "#393836"


def _apply_dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(color=_GRID, lw=0.5)


def _fig_to_wandb_image(fig: plt.Figure, caption: str = "") -> "_wandb.Image":
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return _wandb.Image(buf, caption=caption)


def plot_roc_curves(
    score_dict : dict[str, tuple[np.ndarray, np.ndarray]],
    cutoff     : float,
    save_path  : Path,
) -> plt.Figure:
    """Three-split ROC overlay + val operating point.

    Parameters
    ----------
    score_dict : {"Train": (scores, labels), "Val": ..., "Test": ...}
    cutoff     : threshold determined from val
    save_path  : PNG output path
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_BG)
    _apply_dark_ax(ax)
    ax.plot([0, 1], [0, 1], color=_BORDER, lw=1, ls=":")

    for split, (sc, lb) in score_dict.items():
        if len(sc) == 0:
            continue
        fpr, tpr, thrs = roc_curve(lb, sc, pos_label=1)
        auroc_val = auc(fpr, tpr)
        st = _SPLIT_STYLE[split]
        ax.plot(fpr, tpr,
                label=f"{split}  AUROC={auroc_val:.3f}",
                color=st["color"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"])

        if split == "Val":
            idx = int(np.argmin(np.abs(thrs - cutoff)))
            ax.scatter(fpr[idx], tpr[idx],
                       s=110, zorder=6, color=st["color"],
                       edgecolors="white", linewidths=1.5,
                       label=f"Val cutoff = {cutoff:.3f}")

    ax.set_xlabel("1 - Specificity  (FPR)", fontsize=11)
    ax.set_ylabel("Sensitivity  (TPR)",     fontsize=11)
    ax.set_title("ROC Curve -- HCC vs Hemangioma", fontsize=12, pad=10)
    ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT, fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[score_probe] ROC saved -> {save_path}")
    return fig


def plot_score_distribution(
    score_dict  : dict[str, tuple[np.ndarray, np.ndarray]],
    cutoff      : float,
    class_names : tuple[str, str],
    save_path   : Path,
) -> plt.Figure:
    """KDE + rug score distribution (3 rows: Train / Val / Test).

    Parameters
    ----------
    class_names : (negative_name, positive_name)
    """
    splits = list(score_dict.keys())
    fig, axes = plt.subplots(len(splits), 1, figsize=(8, 9),
                              facecolor=_BG, sharex=True)
    fig.suptitle(f"P({class_names[1]}) Score Distribution",
                 color=_TEXT, fontsize=13, y=1.01)

    xx = np.linspace(0.0, 1.0, 500)

    for ax, split in zip(axes, splits):
        _apply_dark_ax(ax)
        sc, lb = score_dict[split]

        for cls_idx, cls_name in enumerate(class_names):
            mask  = lb == cls_idx
            s     = sc[mask]
            color = _CLS_COLOR.get(cls_name, "#a06fdf")

            if len(s) >= 5:
                bw  = max(float(s.std()) * (len(s) ** -0.2), 0.02)
                kde = gaussian_kde(s, bw_method=bw)
                ax.fill_between(xx, kde(xx), alpha=0.30, color=color)
                ax.plot(xx, kde(xx), color=color, lw=1.8,
                        label=f"{cls_name}  n={int(mask.sum())}")
            elif len(s) > 0:
                ax.hist(s, bins=10, density=True, alpha=0.55, color=color,
                        label=f"{cls_name}  n={int(mask.sum())}")

            if len(s) > 0:
                ax.plot(s, np.full_like(s, -0.18 - cls_idx * 0.18),
                        "|", color=color, alpha=0.35, markersize=4)

        ax.axvline(cutoff, color=_SPLIT_STYLE["Val"]["color"],
                   lw=1.5, ls="--", label=f"cutoff = {cutoff:.3f}")
        ax.set_title(split, color=_TEXT, fontsize=11, loc="left", pad=3)
        ax.set_ylabel("Density", color=_TEXT, fontsize=9)
        ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT,
                  fontsize=8, loc="upper right")

    axes[-1].set_xlabel(f"P({class_names[1]}) Score", color=_TEXT, fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[score_probe] Distribution saved -> {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# WandB logging
# ─────────────────────────────────────────────────────────────

def _log_wandb(
    cfg        : ScoreProbeConfig,
    cutoff     : float,
    all_metrics: dict[str, dict],
    roc_fig    : plt.Figure,
    dist_fig   : plt.Figure,
    score_dict : dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if _wandb is None or _wandb.run is None:
        print("[score_probe] WandB run not active -- skipping logging.")
        return

    pfx = f"{cfg.wandb_prefix}/{cfg.model_id}"

    # scalar metrics per split
    scalar_keys = ["auroc", "sensitivity", "specificity", "ppv", "npv", "f1",
                   "tp", "tn", "fp", "fn"]
    log_dict: dict = {f"{pfx}/cutoff": cutoff}
    for split, m in all_metrics.items():
        sl = split.lower()
        for k in scalar_keys:
            if k in m:
                log_dict[f"{pfx}/{sl}_{k}"] = m[k]

    # figures
    log_dict[f"{pfx}/roc_curves"]         = _fig_to_wandb_image(roc_fig,  "ROC (Train/Val/Test)")
    log_dict[f"{pfx}/score_distribution"] = _fig_to_wandb_image(dist_fig, "P(HCC) KDE distribution")

    # per-sample score table (val + test, publication splits)
    rows = []
    for split in ("Val", "Test"):
        if split not in score_dict:
            continue
        sc, lb = score_dict[split]
        preds  = (sc >= cutoff).astype(int)
        for score_val, label, pred in zip(sc, lb, preds):
            rows.append({
                "split"     : split,
                "true_label": int(label),
                "hcc_score" : float(score_val),
                "predicted" : int(pred),
                "correct"   : int(label) == int(pred),
            })
    if rows:
        log_dict[f"{pfx}/score_table"] = _wandb.Table(
            dataframe=pd.DataFrame(rows)
        )

    # summary metrics table (all splits)
    summary_rows = []
    for split, m in all_metrics.items():
        row = {"split": split}
        row.update(m)
        summary_rows.append(row)
    log_dict[f"{pfx}/metrics_table"] = _wandb.Table(
        dataframe=pd.DataFrame(summary_rows)
    )

    _wandb.log(log_dict)
    print(f"[score_probe] WandB logged under prefix: {pfx}/")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_score_probe(
    stage2_model,
    train_ds    : tf.data.Dataset,
    val_ds      : tf.data.Dataset,
    test_ds     : tf.data.Dataset,
    cfg         : ScoreProbeConfig,
) -> dict:
    """Full pipeline: score extraction -> cutoff (val) -> eval -> plots -> WandB.

    Parameters
    ----------
    stage2_model : PureClassifier or ClassifierTrainer (Keras model).
                   Must return dict with "probabilities" key from its call().
    train_ds     : tf.data.Dataset  -- yields (x_batch, y_batch)
    val_ds       : tf.data.Dataset
    test_ds      : tf.data.Dataset
    cfg          : ScoreProbeConfig

    Returns
    -------
    dict with keys:
        cutoff        : float  -- val-determined threshold
        val_metrics   : dict
        test_metrics  : dict   -- evaluated at val cutoff (no leakage)
        train_metrics : dict
    """
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = cfg.positive_class

    print("=" * 58)
    print("  Score Probe -- HCC vs Hemangioma")
    print(f"  model_id : {cfg.model_id}")
    print(
        f"  strategy : {cfg.cutoff_strategy}"
        + (f"  (sens_target={cfg.sens_target})"
           if cfg.cutoff_strategy == "sens_first" else "")
    )
    print("=" * 58)

    # 1. Extract scores
    print("\n[1/5] Extracting scores ...")
    tr_sc, tr_lb   = _extract_scores(stage2_model, train_ds, pos)
    val_sc, val_lb = _extract_scores(stage2_model, val_ds,   pos)
    te_sc,  te_lb  = _extract_scores(stage2_model, test_ds,  pos)
    print(f"  train n={len(tr_sc)}   class_dist={np.bincount(tr_lb).tolist()}")
    print(f"  val   n={len(val_sc)}   class_dist={np.bincount(val_lb).tolist()}")
    print(f"  test  n={len(te_sc)}   class_dist={np.bincount(te_lb).tolist()}")

    # 2. Save raw scores
    for tag, sc, lb in [("train", tr_sc, tr_lb),
                         ("val",   val_sc, val_lb),
                         ("test",  te_sc,  te_lb)]:
        pd.DataFrame({"hcc_score": sc, "label": lb}).to_csv(
            out_dir / f"{tag}_scores.csv", index=False
        )

    # 3. Cutoff from val
    print(f"\n[2/5] Finding cutoff from Val ({cfg.cutoff_strategy}) ...")
    cutoff, val_metrics = _find_cutoff(
        val_sc, val_lb,
        strategy    = cfg.cutoff_strategy,
        sens_target = cfg.sens_target,
    )
    print(f"  -> cutoff = {cutoff:.4f}")
    print(f"  Val  AUROC={val_metrics['auroc']:.3f}  "
          f"Sens={val_metrics['sensitivity']:.3f}  "
          f"Spec={val_metrics['specificity']:.3f}  "
          f"F1={val_metrics['f1']:.3f}")

    # 4. Evaluate all splits at val cutoff
    print("\n[3/5] Evaluating all splits at val cutoff ...")
    train_metrics = _metrics_at_cutoff(tr_sc, tr_lb, cutoff)
    test_metrics  = _metrics_at_cutoff(te_sc, te_lb, cutoff)
    print(f"  Train  AUROC={train_metrics['auroc']:.3f}  "
          f"Sens={train_metrics['sensitivity']:.3f}  "
          f"Spec={train_metrics['specificity']:.3f}")
    print(f"  Test   AUROC={test_metrics['auroc']:.3f}  "
          f"Sens={test_metrics['sensitivity']:.3f}  "
          f"Spec={test_metrics['specificity']:.3f}")

    all_metrics: dict[str, dict] = {
        "Train": train_metrics,
        "Val"  : val_metrics,
        "Test" : test_metrics,
    }
    pd.DataFrame(
        list(all_metrics.values()), index=list(all_metrics.keys())
    ).to_csv(out_dir / "metrics_at_cutoff.csv")

    # 5. Plots
    print("\n[4/5] Generating plots ...")
    score_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Train": (tr_sc, tr_lb),
        "Val"  : (val_sc, val_lb),
        "Test" : (te_sc,  te_lb),
    }
    roc_fig  = plot_roc_curves(
        score_dict, cutoff, out_dir / "roc_curves.png"
    )
    dist_fig = plot_score_distribution(
        score_dict, cutoff, cfg.class_names, out_dir / "score_distribution.png"
    )

    # 6. WandB
    print("\n[5/5] Logging to WandB ...")
    _log_wandb(cfg, cutoff, all_metrics, roc_fig, dist_fig, score_dict)

    plt.close(roc_fig)
    plt.close(dist_fig)

    print("\n[score_probe] Done.")
    print(f"  Outputs -> {out_dir}")

    return dict(
        cutoff        = cutoff,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        train_metrics = train_metrics,
    )
