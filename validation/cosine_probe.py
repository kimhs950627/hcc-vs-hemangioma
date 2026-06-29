"""validation/cosine_probe.py

Cosine-similarity–based dual-score probe
=========================================

Previous work expressed model output as ``P(HCC) = softmax[:, 1]``, treating
it as a calibrated probability.  This module formalises the distinction:

    confidence_score  = softmax output (model's certainty, NOT a probability)
    hcc_cosine_score  = mean cosine sim(z_query, prototype_bank_HCC)
    hem_cosine_score  = mean cosine sim(z_query, prototype_bank_Hem)
    delta_score       = hcc_cosine_score − hem_cosine_score  (discriminative)

Three ROC curves are produced:
    ROC-A  Confidence score          (existing baseline)
    ROC-B  HCC Cosine Score          (radiologist-aligned similarity)
    ROC-C  Delta Score (Δ)           (discriminative margin)

DeLong test is applied to compare AUROC-A vs AUROC-B and AUROC-A vs AUROC-C.

Embedding source  (auto-detected, configurable)
------------------------------------------------
VICReg / ConvHybrid models
    Embeddings were already saved by extval.py:
        {work_dir}/output/extval/{model_id}/groupA/hcc_rep.npy
        {work_dir}/output/extval/{model_id}/groupA/hemangioma_rep.npy
    Prototype bank is fitted from those saved matrices (K-Means or mean).

SupCon / Benchmark models
    No pre-saved embeddings; embeddings are extracted on-the-fly from a
    tf.data.Dataset using the provided model's encoder.

Usage
-----
from validation.cosine_probe import CosineProbeConfig, run_cosine_probe

# ── VICReg / ConvHybrid (pre-saved embeddings) ──
cfg = CosineProbeConfig(
    model_id        = model_id,
    work_dir        = cfg.work_dir,
    embedding_source= "saved",          # load from extval output dir
)
results = run_cosine_probe(
    stage2_model  = classifier,
    val_ds        = val_ds,
    test_ds       = test_ds,
    cfg           = cfg,
    train_ds      = train_ds,           # optional; needed for softmax extraction
)

# ── SupCon / Benchmark (on-the-fly extraction) ──
cfg = CosineProbeConfig(
    model_id        = model_id,
    work_dir        = cfg.work_dir,
    embedding_source= "live",           # extract from train_ds
)
results = run_cosine_probe(
    stage2_model  = classifier,
    train_ds      = train_ds,           # used to build prototype bank
    val_ds        = val_ds,
    test_ds       = test_ds,
    cfg           = cfg,
)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as _PIL_Image
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gaussian_kde
from sklearn.metrics import auc, roc_auc_score, roc_curve

try:
    import wandb as _wandb
except Exception:
    _wandb = None


# ─────────────────────────────────────────────────────────────
# Types & constants
# ─────────────────────────────────────────────────────────────

EmbeddingSource = Literal["saved", "live"]
PrototypeMode   = Literal["mean", "kmeans"]

_BG     = "#1c1b19"
_GRID   = "#262523"
_TEXT   = "#cdccca"
_BORDER = "#393836"

# ROC-A / B / C style  (matches score_probe.py split-style spirit)
_ROC_STYLE: dict[str, dict] = {
    "ROC-A (Confidence)": dict(color="#4f98a3", lw=2.0, ls="--", alpha=0.90),
    "ROC-B (HCC Cosine)": dict(color="#e8af34", lw=2.5, ls="-",  alpha=0.95),
    "ROC-C (ΔScore)"    : dict(color="#dd6974", lw=2.5, ls="-",  alpha=1.00),
}

_CLS_COLOR = {"HCC": "#dd6974", "Hemangioma": "#4f98a3"}


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class CosineProbeConfig:
    """Configuration for cosine_probe analysis.

    Parameters
    ----------
    model_id          : Experiment identifier.
    work_dir          : Root output directory.
    embedding_source  :
        "saved"  -> load pre-computed embeddings from
                    {work_dir}/output/extval/{model_id}/groupA/
                    (VICReg / ConvHybrid)
        "live"   -> extract embeddings on-the-fly from train_ds
                    (SupCon / Benchmark / any model not run through extval)
    prototype_mode    :
        "mean"   -> class prototype = mean of all train embeddings
        "kmeans" -> K-Means cluster centres (n_proto clusters per class)
    n_proto           : Number of K-Means clusters (used only with "kmeans").
    random_state      : Reproducibility seed.
    positive_class    : Softmax index for HCC (default 1).
    cutoff_strategy   : "youden" | "sens_first" — same as score_probe.
    sens_target       : Target sensitivity for "sens_first".
    wandb_prefix      : WandB log-key namespace prefix.
    class_names       : (negative_name, positive_name) display strings.
    """
    model_id         : str
    work_dir         : str
    embedding_source : EmbeddingSource   = "saved"
    prototype_mode   : PrototypeMode     = "mean"
    n_proto          : int               = 8
    random_state     : int               = 42
    positive_class   : int               = 1
    cutoff_strategy  : Literal["youden", "sens_first"] = "youden"
    sens_target      : float             = 0.90
    wandb_prefix     : str               = "cosine_probe"
    class_names      : tuple[str, str]   = ("Hemangioma", "HCC")

    @property
    def out_dir(self) -> Path:
        return Path(self.work_dir) / "output" / "cosine_probe" / self.model_id

    @property
    def extval_groupA_dir(self) -> Path:
        """Pre-saved embedding directory (VICReg/ConvHybrid path convention)."""
        return Path(self.work_dir) / "output" / "extval" / self.model_id / "groupA"


# ─────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────

def _l2_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(denom, eps)


def _cosine_sim_to_bank(
    queries   : np.ndarray,   # (N, D)
    bank      : np.ndarray,   # (K, D) prototype bank
) -> np.ndarray:              # (N,) mean cosine similarity
    """Mean cosine similarity of each query to all prototypes in bank."""
    q = _l2_norm(queries)          # (N, D)
    b = _l2_norm(bank)             # (K, D)
    sim = q @ b.T                  # (N, K)
    return sim.mean(axis=-1)       # (N,)


# ─────────────────────────────────────────────────────────────
# Embedding loading / extraction
# ─────────────────────────────────────────────────────────────

def _load_saved_embeddings(
    cfg: CosineProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-saved embeddings from extval groupA output.

    Expected files (written by extval._eval_block):
        {extval_groupA_dir}/hcc_rep.npy
        {extval_groupA_dir}/hemangioma_rep.npy

    Returns
    -------
    hcc_emb : (N_hcc, D)
    hem_emb : (N_hem, D)
    """
    d = cfg.extval_groupA_dir
    hcc_path = d / "hcc_rep.npy"
    hem_path = d / "hemangioma_rep.npy"

    if not hcc_path.exists() or not hem_path.exists():
        raise FileNotFoundError(
            f"[cosine_probe] Pre-saved embeddings not found under:\n"
            f"  {d}\n"
            f"  Expected: hcc_rep.npy and hemangioma_rep.npy\n"
            f"  → Run run_extval() first, or set embedding_source='live'."
        )

    hcc_emb = np.load(hcc_path).astype(np.float32)
    hem_emb = np.load(hem_path).astype(np.float32)
    print(f"[cosine_probe] Loaded saved embeddings from {d}")
    print(f"  hcc_emb : {hcc_emb.shape}  hem_emb : {hem_emb.shape}")
    return hcc_emb, hem_emb


def _extract_encoder_repr(model, x_batch: tf.Tensor) -> np.ndarray:
    """Extract CLS / GAP embedding from model or its encoder sub-model."""
    encoder = (
        model.model.encoder if (hasattr(model, "model") and hasattr(model.model, "encoder"))
        else model.encoder if hasattr(model, "encoder")
        else model
    )
    out = encoder(x_batch, training=False)
    if isinstance(out, dict):
        emb = out.get("embedding", out.get("cls_token"))
        if emb is None:
            raise KeyError('encoder dict must contain "embedding" or "cls_token"')
    else:
        emb = out
    if len(emb.shape) > 2:
        emb = emb[:, 0]
    return emb.numpy().astype(np.float32)


def _extract_live_embeddings_by_class(
    model,
    dataset: tf.data.Dataset,
    positive_class: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and split embeddings by label from a tf.data.Dataset.

    Returns
    -------
    hcc_emb : embeddings where label == positive_class
    hem_emb : embeddings where label != positive_class
    """
    all_emb: list[np.ndarray] = []
    all_lbl: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        emb = _extract_encoder_repr(model, x_batch)
        all_emb.append(emb)
        all_lbl.append(y_batch.numpy().astype(np.int32))

    embs   = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_lbl, axis=0)

    hcc_emb = embs[labels == positive_class]
    hem_emb = embs[labels != positive_class]
    print(f"[cosine_probe] Live extracted embeddings  hcc={hcc_emb.shape}  hem={hem_emb.shape}")
    return hcc_emb, hem_emb


# ─────────────────────────────────────────────────────────────
# Prototype bank construction
# ─────────────────────────────────────────────────────────────

def _build_prototype_bank(
    embeddings  : np.ndarray,
    mode        : PrototypeMode,
    n_proto     : int,
    random_state: int,
) -> np.ndarray:
    """Build prototype bank from embeddings.

    mode="mean"   : single prototype = mean vector  → (1, D)
    mode="kmeans" : K cluster centres               → (min(n_proto, N), D)
    """
    if mode == "mean":
        return embeddings.mean(axis=0, keepdims=True).astype(np.float32)  # (1, D)

    # kmeans
    from sklearn.cluster import KMeans
    k   = min(n_proto, len(embeddings))
    km  = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(embeddings)
    return km.cluster_centers_.astype(np.float32)                          # (k, D)


# ─────────────────────────────────────────────────────────────
# Softmax confidence extraction
# ─────────────────────────────────────────────────────────────

def _extract_confidence_scores(
    model       : object,
    dataset     : tf.data.Dataset,
    positive_cls: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract softmax confidence P(positive_cls) and labels."""
    m = getattr(model, "model", model)
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        out = m(x_batch, training=False)
        if isinstance(out, dict):
            probs = out["probabilities"].numpy()
        else:
            probs = tf.nn.softmax(out, axis=-1).numpy()
        all_scores.append(probs[:, positive_cls].astype(np.float32))
        all_labels.append(y_batch.numpy().astype(np.int32))

    return np.concatenate(all_scores), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────
# Cosine score extraction for a dataset split
# ─────────────────────────────────────────────────────────────

def _extract_cosine_scores_from_ds(
    model      : object,
    dataset    : tf.data.Dataset,
    hcc_bank   : np.ndarray,
    hem_bank   : np.ndarray,
    positive_cls: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract hcc_cosine, hem_cosine, delta, and labels for a dataset split.

    Returns
    -------
    hcc_cosine : (N,)
    hem_cosine : (N,)
    delta      : (N,)  = hcc_cosine − hem_cosine
    labels     : (N,)
    """
    all_emb: list[np.ndarray] = []
    all_lbl: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        emb = _extract_encoder_repr(model, x_batch)
        all_emb.append(emb)
        all_lbl.append(y_batch.numpy().astype(np.int32))

    embs   = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_lbl, axis=0)

    hcc_cos = _cosine_sim_to_bank(embs, hcc_bank)
    hem_cos = _cosine_sim_to_bank(embs, hem_bank)
    delta   = hcc_cos - hem_cos

    return hcc_cos.astype(np.float32), hem_cos.astype(np.float32), delta.astype(np.float32), labels


# ─────────────────────────────────────────────────────────────
# DeLong test  (non-parametric AUROC comparison)
# ─────────────────────────────────────────────────────────────

def _delong_auc_variance(
    labels : np.ndarray,
    scores : np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute AUROC and its variance via DeLong (1988) structural components.

    Returns
    -------
    auroc    : float
    var      : (1,) variance of AUROC
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), np.array([float("nan")])

    # structural components (pairwise comparisons)
    mat_pos = np.zeros(n_pos)   # V10  — positives evaluated against negatives
    mat_neg = np.zeros(n_neg)   # V01  — negatives evaluated against positives
    for i, p in enumerate(pos):
        mat_pos[i] = np.mean((p > neg) + 0.5 * (p == neg))
    for j, n in enumerate(neg):
        mat_neg[j] = np.mean((pos > n) + 0.5 * (pos == n))

    auroc = float(np.mean(mat_pos))
    var   = (np.var(mat_pos, ddof=1) / n_pos + np.var(mat_neg, ddof=1) / n_neg)
    return auroc, np.array([var])


def delong_test(
    labels   : np.ndarray,
    scores_a : np.ndarray,
    scores_b : np.ndarray,
) -> dict:
    """Two-sided DeLong test: H0 = AUROC(a) == AUROC(b).

    Returns
    -------
    dict with auroc_a, auroc_b, z, p_value
    """
    from scipy.stats import norm

    auc_a, var_a = _delong_auc_variance(labels, scores_a)
    auc_b, var_b = _delong_auc_variance(labels, scores_b)
    se = np.sqrt(var_a + var_b)
    z  = (auc_a - auc_b) / (se + 1e-12)
    p  = float(2.0 * norm.sf(np.abs(z)))
    return {
        "auroc_a"  : float(auc_a),
        "auroc_b"  : float(auc_b),
        "z_stat"   : float(z),
        "p_value"  : p,
        "se"       : float(se),
    }


# ─────────────────────────────────────────────────────────────
# Cutoff helpers  (re-use score_probe logic)
# ─────────────────────────────────────────────────────────────

def _find_cutoff(
    scores      : np.ndarray,
    labels      : np.ndarray,
    strategy    : str = "youden",
    sens_target : float = 0.90,
) -> float:
    fpr, tpr, thrs = roc_curve(labels, scores, pos_label=1)
    if strategy == "youden":
        idx = int(np.argmax(tpr - fpr))
    else:  # sens_first
        cands = np.where(tpr >= sens_target)[0]
        idx   = int(cands[np.argmax(thrs[cands])]) if len(cands) else int(np.argmax(tpr))
    return float(thrs[idx])


def _metrics_at_cutoff(scores: np.ndarray, labels: np.ndarray, cutoff: float) -> dict:
    from sklearn.metrics import confusion_matrix
    preds = (scores >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens  = tp / (tp + fn + 1e-8)
    spec  = tn / (tn + fp + 1e-8)
    ppv   = tp / (tp + fp + 1e-8)
    npv   = tn / (tn + fn + 1e-8)
    f1    = 2.0 * ppv * sens / (ppv + sens + 1e-8)
    return dict(
        cutoff=cutoff,
        auroc=float(roc_auc_score(labels, scores)),
        sensitivity=float(sens), specificity=float(spec),
        ppv=float(ppv), npv=float(npv), f1=float(f1),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        n=int(len(labels)),
    )


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────

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
    pil_img = _PIL_Image.open(buf).copy()
    return _wandb.Image(pil_img, caption=caption)


def plot_triple_roc(
    scores_dict : dict[str, tuple[np.ndarray, np.ndarray]],
    cutoffs     : dict[str, float],
    split       : str,
    save_path   : Path,
) -> plt.Figure:
    """Overlay ROC-A / ROC-B / ROC-C on a single panel for one split.

    Parameters
    ----------
    scores_dict : {"ROC-A (Confidence)": (scores, labels), ...}
    cutoffs     : {"ROC-A (Confidence)": cutoff_float, ...}  (val-determined)
    split       : "Val" | "Test"  — used in title
    save_path   : PNG output path
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_BG)
    _apply_dark_ax(ax)
    ax.plot([0, 1], [0, 1], color=_BORDER, lw=1, ls=":")

    for roc_name, (sc, lb) in scores_dict.items():
        if len(sc) == 0 or lb.sum() == 0:
            continue
        fpr, tpr, thrs = roc_curve(lb, sc, pos_label=1)
        auroc_val = auc(fpr, tpr)
        st = _ROC_STYLE[roc_name]
        ax.plot(fpr, tpr,
                label=f"{roc_name}  AUROC={auroc_val:.3f}",
                color=st["color"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"])

        # operating point on val-determined cutoff
        if roc_name in cutoffs and split == "Val":
            c   = cutoffs[roc_name]
            idx = int(np.argmin(np.abs(thrs - c)))
            ax.scatter(fpr[idx], tpr[idx], s=110, zorder=6,
                       color=st["color"], edgecolors="white", linewidths=1.5,
                       label=f"  cutoff={c:.3f}")

    ax.set_xlabel("1 − Specificity  (FPR)", fontsize=11)
    ax.set_ylabel("Sensitivity  (TPR)",      fontsize=11)
    ax.set_title(f"Triple ROC ({split})  —  Confidence vs Cosine",
                 fontsize=11, pad=8, color=_TEXT)
    ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT, fontsize=8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[cosine_probe] Triple ROC ({split}) saved → {save_path}")
    return fig


def plot_cosine_distribution(
    hcc_cos  : np.ndarray,
    hem_cos  : np.ndarray,
    delta    : np.ndarray,
    labels   : np.ndarray,
    cutoffs  : dict[str, float],
    split    : str,
    save_path: Path,
) -> plt.Figure:
    """KDE distribution plot: HCC Cosine Score | ΔScore by true label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    fig.suptitle(f"Cosine Score Distribution ({split})",
                 color=_TEXT, fontsize=12, y=1.01)

    panel_data = [
        ("HCC Cosine Score",  hcc_cos, "ROC-B (HCC Cosine)"),
        ("ΔScore (HCC−Hem)",  delta,   "ROC-C (ΔScore)"),
    ]

    xx = np.linspace(-1.0, 1.0, 600)

    for ax, (title, scores, roc_key) in zip(axes, panel_data):
        _apply_dark_ax(ax)
        for cls_idx, cls_name in enumerate(("Hemangioma", "HCC")):
            mask  = labels == cls_idx
            s     = scores[mask]
            color = _CLS_COLOR.get(cls_name, "#a06fdf")
            if len(s) >= 5:
                bw  = max(float(s.std()) * (len(s) ** -0.2), 0.01)
                kde = gaussian_kde(s, bw_method=bw)
                ax.fill_between(xx, kde(xx), alpha=0.28, color=color)
                ax.plot(xx, kde(xx), color=color, lw=1.8,
                        label=f"{cls_name}  n={int(mask.sum())}")
            elif len(s) > 0:
                ax.hist(s, bins=10, density=True, alpha=0.55, color=color,
                        label=f"{cls_name}  n={int(mask.sum())}")
            if len(s) > 0:
                ax.plot(s, np.full_like(s, -0.10 - cls_idx * 0.12),
                        "|", color=color, alpha=0.30, markersize=4)

        # cutoff line
        if roc_key in cutoffs:
            c = cutoffs[roc_key]
            ax.axvline(c, color=_ROC_STYLE[roc_key]["color"],
                       lw=1.5, ls="--", label=f"cutoff={c:.3f}")

        ax.set_title(title, color=_TEXT, fontsize=10, loc="left", pad=3)
        ax.set_xlabel(title, color=_TEXT, fontsize=9)
        ax.set_ylabel("Density", color=_TEXT, fontsize=9)
        ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT,
                  fontsize=8, loc="upper right")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[cosine_probe] Distribution ({split}) saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# WandB logging
# ─────────────────────────────────────────────────────────────

def _log_wandb(
    cfg           : CosineProbeConfig,
    val_results   : dict,
    test_results  : dict,
    delong_ab     : dict,
    delong_ac     : dict,
    roc_figs      : dict[str, plt.Figure],
    dist_figs     : dict[str, plt.Figure],
    score_records : list[dict],
) -> None:
    if _wandb is None or _wandb.run is None:
        print("[cosine_probe] WandB run not active — skipping logging.")
        return

    pfx = f"{cfg.wandb_prefix}/{cfg.model_id}"
    log_dict: dict = {}

    # scalar metrics
    for split_tag, res in [("val", val_results), ("test", test_results)]:
        for roc_key, m in res["metrics"].items():
            tag = roc_key.lower().replace(" ", "_").replace("(", "").replace(")", "")
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{pfx}/{split_tag}/{tag}_{k}"] = v

    # DeLong
    for k, v in delong_ab.items():
        log_dict[f"{pfx}/delong_A_vs_B_{k}"] = v
    for k, v in delong_ac.items():
        log_dict[f"{pfx}/delong_A_vs_C_{k}"] = v

    # figures
    for split, fig in roc_figs.items():
        log_dict[f"{pfx}/triple_roc_{split.lower()}"] = _fig_to_wandb_image(
            fig, f"Triple ROC ({split})")
    for split, fig in dist_figs.items():
        log_dict[f"{pfx}/cosine_dist_{split.lower()}"] = _fig_to_wandb_image(
            fig, f"Cosine distribution ({split})")

    # per-sample table
    if score_records:
        log_dict[f"{pfx}/cosine_score_table"] = _wandb.Table(
            dataframe=pd.DataFrame(score_records)
        )

    # DeLong summary table
    delong_rows = [
        {"comparison": "A vs B (Conf vs HCC Cos)", **delong_ab},
        {"comparison": "A vs C (Conf vs ΔScore)",   **delong_ac},
    ]
    log_dict[f"{pfx}/delong_table"] = _wandb.Table(
        dataframe=pd.DataFrame(delong_rows)
    )

    _wandb.log(log_dict)
    print(f"[cosine_probe] WandB logged under prefix: {pfx}/")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_cosine_probe(
    stage2_model : object,
    val_ds       : tf.data.Dataset,
    test_ds      : tf.data.Dataset,
    cfg          : CosineProbeConfig,
    train_ds     : Optional[tf.data.Dataset] = None,
) -> dict:
    """Full cosine probe pipeline.

    Parameters
    ----------
    stage2_model :
        Keras model (PureClassifier / ClassifierTrainer / bare encoder).
        Must return dict with "probabilities" from its __call__
        (used for ROC-A confidence score).
    val_ds       : tf.data.Dataset  (x_batch, y_batch)  — cutoff source
    test_ds      : tf.data.Dataset
    cfg          : CosineProbeConfig
    train_ds     :
        Required when cfg.embedding_source == "live" (prototype bank source).
        Optional when "saved" (used only if you also want train-split scores).

    Returns
    -------
    dict:
        hcc_bank       : np.ndarray  (K, D)
        hem_bank       : np.ndarray  (K, D)
        cutoffs        : dict[roc_name → float]   (val-determined)
        val_metrics    : dict[roc_name → metrics_dict]
        test_metrics   : dict[roc_name → metrics_dict]
        delong_A_vs_B  : dict
        delong_A_vs_C  : dict
    """
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = cfg.positive_class

    print("=" * 60)
    print("  Cosine Probe  —  Confidence vs Cosine Similarity ROC")
    print(f"  model_id         : {cfg.model_id}")
    print(f"  embedding_source : {cfg.embedding_source}")
    print(f"  prototype_mode   : {cfg.prototype_mode}")
    print("=" * 60)

    # ── 1. Build prototype bank ──────────────────────────────
    print("\n[1/6] Building prototype bank …")
    if cfg.embedding_source == "saved":
        hcc_train_emb, hem_train_emb = _load_saved_embeddings(cfg)
    else:
        if train_ds is None:
            raise ValueError(
                "train_ds is required when embedding_source='live'.\n"
                "Pass train_ds= to run_cosine_probe()."
            )
        hcc_train_emb, hem_train_emb = _extract_live_embeddings_by_class(
            stage2_model, train_ds, pos
        )
        np.save(out_dir / "hcc_train_emb.npy", hcc_train_emb)
        np.save(out_dir / "hem_train_emb.npy", hem_train_emb)

    hcc_bank = _build_prototype_bank(
        hcc_train_emb, cfg.prototype_mode, cfg.n_proto, cfg.random_state
    )
    hem_bank = _build_prototype_bank(
        hem_train_emb, cfg.prototype_mode, cfg.n_proto, cfg.random_state
    )
    np.save(out_dir / "hcc_bank.npy", hcc_bank)
    np.save(out_dir / "hem_bank.npy", hem_bank)
    print(f"  hcc_bank : {hcc_bank.shape}  hem_bank : {hem_bank.shape}")

    # ── 2. Extract scores (confidence + cosine) ──────────────
    print("\n[2/6] Extracting scores for val / test …")
    split_data: dict[str, dict[str, np.ndarray]] = {}
    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        conf, lbl         = _extract_confidence_scores(stage2_model, ds, pos)
        hcc_cos, hem_cos, delta, lbl2 = _extract_cosine_scores_from_ds(
            stage2_model, ds, hcc_bank, hem_bank, pos
        )
        assert np.array_equal(lbl, lbl2), "label mismatch between conf and cosine passes"
        split_data[split_tag] = dict(
            conf    = conf,
            hcc_cos = hcc_cos,
            hem_cos = hem_cos,
            delta   = delta,
            labels  = lbl,
        )
        pd.DataFrame({
            "confidence_score" : conf,
            "hcc_cosine_score" : hcc_cos,
            "hem_cosine_score" : hem_cos,
            "delta_score"      : delta,
            "label"            : lbl,
        }).to_csv(out_dir / f"{split_tag.lower()}_scores.csv", index=False)
        print(f"  {split_tag}  n={len(lbl)}  class_dist={np.bincount(lbl).tolist()}")

    # ── 3. Cutoffs from val ──────────────────────────────────
    print(f"\n[3/6] Determining cutoffs from Val ({cfg.cutoff_strategy}) …")
    val_d = split_data["Val"]
    cutoffs: dict[str, float] = {}
    score_map: dict[str, np.ndarray] = {
        "ROC-A (Confidence)": val_d["conf"],
        "ROC-B (HCC Cosine)": val_d["hcc_cos"],
        "ROC-C (ΔScore)"    : val_d["delta"],
    }
    for roc_name, sc in score_map.items():
        c = _find_cutoff(sc, val_d["labels"], cfg.cutoff_strategy, cfg.sens_target)
        cutoffs[roc_name] = c
        print(f"  {roc_name:30s}  cutoff = {c:.4f}")
    pd.DataFrame([{"roc": k, "cutoff": v} for k, v in cutoffs.items()]).to_csv(
        out_dir / "cutoffs.csv", index=False
    )

    # ── 4. Metrics at cutoff ─────────────────────────────────
    print("\n[4/6] Computing metrics at val cutoff …")
    def _eval_split(sd: dict) -> dict[str, dict]:
        sm: dict[str, np.ndarray] = {
            "ROC-A (Confidence)": sd["conf"],
            "ROC-B (HCC Cosine)": sd["hcc_cos"],
            "ROC-C (ΔScore)"    : sd["delta"],
        }
        return {rn: _metrics_at_cutoff(sc, sd["labels"], cutoffs[rn])
                for rn, sc in sm.items()}

    val_metrics  = _eval_split(split_data["Val"])
    test_metrics = _eval_split(split_data["Test"])

    for split_tag, metrics in [("Val", val_metrics), ("Test", test_metrics)]:
        rows = [{"roc": rn, **m} for rn, m in metrics.items()]
        pd.DataFrame(rows).to_csv(
            out_dir / f"{split_tag.lower()}_metrics.csv", index=False
        )
        print(f"  ── {split_tag} ──")
        for rn, m in metrics.items():
            print(f"  {rn:30s}  AUROC={m['auroc']:.3f}  "
                  f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                  f"F1={m['f1']:.3f}")

    # ── 5. DeLong test (val labels, val scores) ──────────────
    print("\n[5/6] DeLong test (A vs B, A vs C)  [val set] …")
    vl  = split_data["Val"]["labels"]
    delong_ab = delong_test(vl, split_data["Val"]["conf"], split_data["Val"]["hcc_cos"])
    delong_ac = delong_test(vl, split_data["Val"]["conf"], split_data["Val"]["delta"])
    print(f"  A vs B  z={delong_ab['z_stat']:.3f}  p={delong_ab['p_value']:.4f}  "
          f"ΔAUROC={delong_ab['auroc_a']-delong_ab['auroc_b']:+.4f}")
    print(f"  A vs C  z={delong_ac['z_stat']:.3f}  p={delong_ac['p_value']:.4f}  "
          f"ΔAUROC={delong_ac['auroc_a']-delong_ac['auroc_b']:+.4f}")
    pd.DataFrame([
        {"comparison": "A_vs_B", **delong_ab},
        {"comparison": "A_vs_C", **delong_ac},
    ]).to_csv(out_dir / "delong.csv", index=False)

    # ── 6. Plots & WandB ─────────────────────────────────────
    print("\n[6/6] Generating plots …")
    roc_figs : dict[str, plt.Figure] = {}
    dist_figs: dict[str, plt.Figure] = {}
    score_records: list[dict] = []

    for split_tag, sd in split_data.items():
        sd_scores = {
            "ROC-A (Confidence)": (sd["conf"],    sd["labels"]),
            "ROC-B (HCC Cosine)": (sd["hcc_cos"], sd["labels"]),
            "ROC-C (ΔScore)"    : (sd["delta"],   sd["labels"]),
        }
        roc_figs[split_tag] = plot_triple_roc(
            sd_scores, cutoffs, split=split_tag,
            save_path=out_dir / f"triple_roc_{split_tag.lower()}.png",
        )
        dist_figs[split_tag] = plot_cosine_distribution(
            hcc_cos  = sd["hcc_cos"],
            hem_cos  = sd["hem_cos"],
            delta    = sd["delta"],
            labels   = sd["labels"],
            cutoffs  = cutoffs,
            split    = split_tag,
            save_path=out_dir / f"cosine_dist_{split_tag.lower()}.png",
        )
        # per-sample records
        for i in range(len(sd["labels"])):
            score_records.append({
                "split"            : split_tag,
                "true_label"       : int(sd["labels"][i]),
                "confidence_score" : float(sd["conf"][i]),
                "hcc_cosine_score" : float(sd["hcc_cos"][i]),
                "hem_cosine_score" : float(sd["hem_cos"][i]),
                "delta_score"      : float(sd["delta"][i]),
            })

    _log_wandb(
        cfg           = cfg,
        val_results   = {"metrics": val_metrics},
        test_results  = {"metrics": test_metrics},
        delong_ab     = delong_ab,
        delong_ac     = delong_ac,
        roc_figs      = roc_figs,
        dist_figs     = dist_figs,
        score_records = score_records,
    )

    for fig in list(roc_figs.values()) + list(dist_figs.values()):
        plt.close(fig)

    print("\n[cosine_probe] Done.")
    print(f"  Outputs → {out_dir}")

    return dict(
        hcc_bank      = hcc_bank,
        hem_bank      = hem_bank,
        cutoffs       = cutoffs,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        delong_A_vs_B = delong_ab,
        delong_A_vs_C = delong_ac,
    )
