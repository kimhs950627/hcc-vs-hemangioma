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
    Prototype bank is fitted from those saved matrices (both mean AND kmeans).

    FALLBACK: if saved embeddings are not found (e.g. VICReg failed or
    extval was not run), embeddings are re-extracted on-the-fly from
    train_ds automatically.

SupCon / Benchmark models
    No pre-saved embeddings; embeddings are extracted on-the-fly from a
    tf.data.Dataset using the provided model's encoder.

    FALLBACK: same mechanism — any extraction failure triggers a retry
    using train_ds directly.

prototype_mode
--------------
Both "mean" AND "kmeans" prototypes are always computed and all downstream
metrics / plots are produced for each mode independently.

Usage
-----
from validation.cosine_probe import CosineProbeConfig, run_cosine_probe

cfg = CosineProbeConfig(
    model_id        = model_id,
    work_dir        = cfg.work_dir,
    embedding_source= "saved",          # tries saved first, falls back to live
)
results = run_cosine_probe(
    stage2_model  = classifier,
    val_ds        = val_ds,
    test_ds       = test_ds,
    cfg           = cfg,
    train_ds      = train_ds,           # always pass; used for fallback / live
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
# prototype_mode is no longer a single choice — both are always computed.
# The type alias is kept for config documentation.
PrototypeMode   = Literal["mean", "kmeans"]

_BG     = "#1c1b19"
_GRID   = "#262523"
_TEXT   = "#cdccca"
_BORDER = "#393836"

# ROC style per mode × score type
_ROC_STYLE: dict[str, dict] = {
    "ROC-A (Confidence)"         : dict(color="#4f98a3", lw=2.0, ls="--", alpha=0.90),
    "ROC-B (HCC Cosine / mean)"  : dict(color="#e8af34", lw=2.5, ls="-",  alpha=0.95),
    "ROC-C (ΔScore / mean)"      : dict(color="#dd6974", lw=2.5, ls="-",  alpha=1.00),
    "ROC-B (HCC Cosine / kmeans)": dict(color="#fdc551", lw=2.0, ls="-.", alpha=0.90),
    "ROC-C (ΔScore / kmeans)"    : dict(color="#c24a59", lw=2.0, ls="-.", alpha=0.85),
}

_CLS_COLOR = {"HCC": "#dd6974", "Hemangioma": "#4f98a3"}

# cluster scatter markers for t-SNE
_CLUSTER_MARKER_STYLE = dict(s=220, zorder=8, linewidths=2.5)


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
                    (VICReg / ConvHybrid).
                    Falls back to live extraction if files are missing.
        "live"   -> always extract on-the-fly from train_ds.
    n_proto           : Number of K-Means clusters per class.
    random_state      : Reproducibility seed.
    positive_class    : Softmax index for HCC (default 1).
    cutoff_strategy   : "youden" | "sens_first".
    sens_target       : Target sensitivity for "sens_first".
    tsne_perplexity   : t-SNE perplexity (default 30).
    wandb_prefix      : WandB log-key namespace prefix.
    class_names       : (negative_name, positive_name) display strings.
    """
    model_id         : str
    work_dir         : str
    embedding_source : EmbeddingSource   = "saved"
    n_proto          : int               = 8
    random_state     : int               = 42
    positive_class   : int               = 1
    cutoff_strategy  : Literal["youden", "sens_first"] = "youden"
    sens_target      : float             = 0.90
    tsne_perplexity  : float             = 30.0
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
# Embedding loading / extraction  (with fallback)
# ─────────────────────────────────────────────────────────────

def _load_saved_embeddings(
    cfg: CosineProbeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-saved embeddings from extval groupA output.

    Raises FileNotFoundError if files are absent (caller handles fallback).
    """
    d = cfg.extval_groupA_dir
    hcc_path = d / "hcc_rep.npy"
    hem_path = d / "hemangioma_rep.npy"

    if not hcc_path.exists() or not hem_path.exists():
        raise FileNotFoundError(
            f"[cosine_probe] Pre-saved embeddings not found under:\n"
            f"  {d}\n"
            f"  Expected: hcc_rep.npy and hemangioma_rep.npy"
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
    """Extract and split embeddings by label from a tf.data.Dataset."""
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


def _get_train_embeddings(
    cfg          : CosineProbeConfig,
    stage2_model : object,
    train_ds     : Optional[tf.data.Dataset],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve train embeddings with automatic fallback.

    Priority:
    1. cfg.embedding_source == "saved"  →  try _load_saved_embeddings()
       on failure (FileNotFoundError OR any exception)  →  fallback to live
    2. cfg.embedding_source == "live"   →  always extract from train_ds

    Fallback always requires train_ds to be non-None.
    """
    if cfg.embedding_source == "saved":
        try:
            return _load_saved_embeddings(cfg)
        except Exception as e:
            print(f"[cosine_probe] WARNING: saved-embedding load failed: {e}")
            print("[cosine_probe] → Falling back to live extraction from train_ds.")

    # reach here if source=="live" OR fallback triggered
    if train_ds is None:
        raise ValueError(
            "train_ds is required for live embedding extraction "
            "(embedding_source='live' or saved-embedding fallback).\n"
            "Pass train_ds= to run_cosine_probe()."
        )
    hcc_emb, hem_emb = _extract_live_embeddings_by_class(
        stage2_model, train_ds, cfg.positive_class
    )
    # cache for reproducibility
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "hcc_train_emb.npy", hcc_emb)
    np.save(out_dir / "hem_train_emb.npy", hem_emb)
    return hcc_emb, hem_emb


# ─────────────────────────────────────────────────────────────
# Prototype bank construction  (mean AND kmeans, always both)
# ─────────────────────────────────────────────────────────────

def _build_prototype_bank(
    embeddings  : np.ndarray,
    mode        : PrototypeMode,
    n_proto     : int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single prototype bank.

    Returns
    -------
    bank        : (K, D)  prototype vectors
    assignments : (N,)    cluster assignment per training sample (0-based)
                  For mode="mean", all samples are assigned to cluster 0.
    """
    if mode == "mean":
        bank = embeddings.mean(axis=0, keepdims=True).astype(np.float32)  # (1, D)
        assignments = np.zeros(len(embeddings), dtype=np.int32)
        return bank, assignments

    # kmeans
    from sklearn.cluster import KMeans
    k  = min(n_proto, len(embeddings))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(embeddings)
    bank        = km.cluster_centers_.astype(np.float32)            # (k, D)
    assignments = km.labels_.astype(np.int32)                       # (N,)
    return bank, assignments


def _build_all_prototype_banks(
    hcc_emb     : np.ndarray,
    hem_emb     : np.ndarray,
    n_proto     : int,
    random_state: int,
) -> dict[str, dict]:
    """Build both mean and kmeans banks for HCC and Hemangioma.

    Returns
    -------
    {
        "mean": {
            "hcc_bank": (1, D),  "hem_bank": (1, D),
            "hcc_assign": (N_hcc,),  "hem_assign": (N_hem,),
        },
        "kmeans": {
            "hcc_bank": (k, D),  "hem_bank": (k, D),
            "hcc_assign": (N_hcc,),  "hem_assign": (N_hem,),
        },
    }
    """
    banks: dict[str, dict] = {}
    for mode in ("mean", "kmeans"):
        hcc_bank, hcc_assign = _build_prototype_bank(
            hcc_emb, mode, n_proto, random_state)
        hem_bank, hem_assign = _build_prototype_bank(
            hem_emb, mode, n_proto, random_state)
        banks[mode] = dict(
            hcc_bank   = hcc_bank,
            hem_bank   = hem_bank,
            hcc_assign = hcc_assign,
            hem_assign = hem_assign,
        )
        print(f"  [bank/{mode}]  hcc={hcc_bank.shape}  hem={hem_bank.shape}")
    return banks


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
    model       : object,
    dataset     : tf.data.Dataset,
    hcc_bank    : np.ndarray,
    hem_bank    : np.ndarray,
    positive_cls: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract hcc_cosine, hem_cosine, delta, raw embeddings, and labels.

    Returns
    -------
    hcc_cosine : (N,)
    hem_cosine : (N,)
    delta      : (N,)  = hcc_cosine − hem_cosine
    embs       : (N, D) raw embeddings (used for t-SNE)
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

    return (hcc_cos.astype(np.float32), hem_cos.astype(np.float32),
            delta.astype(np.float32), embs.astype(np.float32), labels)


# ─────────────────────────────────────────────────────────────
# DeLong test  (non-parametric AUROC comparison)
# ─────────────────────────────────────────────────────────────

def _delong_auc_variance(
    labels : np.ndarray,
    scores : np.ndarray,
) -> tuple[float, np.ndarray]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), np.array([float("nan")])

    mat_pos = np.zeros(n_pos)
    mat_neg = np.zeros(n_neg)
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
# Cutoff helpers
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
    else:
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


# ─────────────────────────────────────────────────────────────
# ROC plot  (all 5 curves on one figure per split)
# ─────────────────────────────────────────────────────────────

def plot_triple_roc(
    scores_dict : dict[str, tuple[np.ndarray, np.ndarray]],
    cutoffs     : dict[str, float],
    split       : str,
    save_path   : Path,
) -> plt.Figure:
    """Overlay all ROC curves (A / B-mean / B-kmeans / C-mean / C-kmeans)."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_BG)
    _apply_dark_ax(ax)
    ax.plot([0, 1], [0, 1], color=_BORDER, lw=1, ls=":")

    for roc_name, (sc, lb) in scores_dict.items():
        if len(sc) == 0 or lb.sum() == 0:
            continue
        fpr, tpr, thrs = roc_curve(lb, sc, pos_label=1)
        auroc_val = auc(fpr, tpr)
        st = _ROC_STYLE.get(roc_name, dict(color="#a86fdf", lw=2, ls="-", alpha=0.8))
        ax.plot(fpr, tpr,
                label=f"{roc_name}  AUROC={auroc_val:.3f}",
                color=st["color"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"])

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
    ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT, fontsize=7.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[cosine_probe] Triple ROC ({split}) saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# Cosine distribution plot
# ─────────────────────────────────────────────────────────────

def plot_cosine_distribution(
    hcc_cos  : np.ndarray,
    hem_cos  : np.ndarray,
    delta    : np.ndarray,
    labels   : np.ndarray,
    cutoffs  : dict[str, float],
    split    : str,
    save_path: Path,
) -> plt.Figure:
    """KDE distribution: HCC Cosine Score | ΔScore by true label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
    fig.suptitle(f"Cosine Score Distribution ({split})",
                 color=_TEXT, fontsize=12, y=1.01)

    panel_data = [
        ("HCC Cosine Score",  hcc_cos, "ROC-B (HCC Cosine / mean)"),
        ("ΔScore (HCC−Hem)",  delta,   "ROC-C (ΔScore / mean)"),
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
# t-SNE visualisation  (enhanced: subgroup + cluster centres)
# ─────────────────────────────────────────────────────────────

def _run_tsne(
    embeddings  : np.ndarray,
    perplexity  : float = 30.0,
    random_state: int   = 42,
) -> np.ndarray:
    """Run t-SNE and return 2-D coordinates."""
    from sklearn.manifold import TSNE
    n = len(embeddings)
    perp = min(perplexity, max(5.0, n / 4))
    tsne = TSNE(n_components=2, perplexity=perp,
                random_state=random_state, n_iter=1000,
                learning_rate="auto", init="pca")
    return tsne.fit_transform(embeddings.astype(np.float64))


def plot_tsne(
    train_hcc_emb  : np.ndarray,           # (N_hcc, D)
    train_hem_emb  : np.ndarray,           # (N_hem, D)
    hcc_assign_mean: np.ndarray,           # (N_hcc,) all zeros  [mean mode]
    hem_assign_mean: np.ndarray,           # (N_hem,) all zeros
    hcc_assign_km  : np.ndarray,           # (N_hcc,) k-means labels
    hem_assign_km  : np.ndarray,           # (N_hem,) k-means labels
    hcc_bank_mean  : np.ndarray,           # (1, D)
    hem_bank_mean  : np.ndarray,           # (1, D)
    hcc_bank_km    : np.ndarray,           # (k_hcc, D)
    hem_bank_km    : np.ndarray,           # (k_hem, D)
    cfg            : CosineProbeConfig,
    save_path      : Path,
    val_embs       : Optional[np.ndarray] = None,  # (N_val, D)
    val_labels     : Optional[np.ndarray] = None,  # (N_val,)
) -> plt.Figure:
    """t-SNE scatter with:
    - per-sample dots coloured by subgroup (HCC / Hemangioma)
    - cluster membership shown as enclosing circles (one per k-means cluster)
    - subgroup centre (mean prototype) marked with a large star ★
    - k-means cluster centres marked with a diamond ◆
    - optional val/test samples shown as triangles ▲

    Two panels: left = mean-mode view, right = k-means view.
    """
    print("[cosine_probe] Running t-SNE …")

    # ── pool all embeddings (train + prototypes) for joint t-SNE ──
    all_embs_list  = [train_hcc_emb, train_hem_emb]
    all_labels_list = (
        [np.zeros(len(train_hcc_emb), dtype=np.int32),   # HCC → 1
         np.ones (len(train_hem_emb), dtype=np.int32)]    # Hem → 0
    )
    n_hcc_train = len(train_hcc_emb)
    n_hem_train = len(train_hem_emb)
    n_train     = n_hcc_train + n_hem_train

    proto_embs = np.concatenate(
        [hcc_bank_mean, hem_bank_mean, hcc_bank_km, hem_bank_km], axis=0
    )
    # tags: 2 = HCC mean centre, 3 = Hem mean centre, 4 = HCC km centre, 5 = Hem km centre
    proto_tags = np.array(
        [2] * len(hcc_bank_mean) + [3] * len(hem_bank_mean) +
        [4] * len(hcc_bank_km)  + [5] * len(hem_bank_km),
        dtype=np.int32,
    )
    n_proto_each = [len(hcc_bank_mean), len(hem_bank_mean),
                    len(hcc_bank_km),   len(hem_bank_km)]

    pool_embs = np.concatenate(all_embs_list + [proto_embs], axis=0)
    if val_embs is not None and len(val_embs) > 0:
        pool_embs = np.concatenate([pool_embs, val_embs], axis=0)

    tsne_all = _run_tsne(pool_embs, cfg.tsne_perplexity, cfg.random_state)

    # ── unpack ──
    hcc_2d  = tsne_all[:n_hcc_train]
    hem_2d  = tsne_all[n_hcc_train:n_train]
    proto_start = n_train
    hcc_mean_2d = tsne_all[proto_start : proto_start + n_proto_each[0]]
    hem_mean_2d = tsne_all[proto_start + n_proto_each[0] :
                            proto_start + n_proto_each[0] + n_proto_each[1]]
    hcc_km_2d   = tsne_all[proto_start + n_proto_each[0] + n_proto_each[1] :
                            proto_start + sum(n_proto_each[:3])]
    hem_km_2d   = tsne_all[proto_start + sum(n_proto_each[:3]) :
                            proto_start + sum(n_proto_each)]
    if val_embs is not None and len(val_embs) > 0:
        val_2d = tsne_all[proto_start + sum(n_proto_each):]
    else:
        val_2d = None

    # ── figure: 2 panels ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=_BG)
    fig.suptitle("t-SNE Embedding Space  —  Mean vs K-Means Prototypes",
                 color=_TEXT, fontsize=13, y=1.01)

    _alpha_scatter = 0.45
    _s_scatter     = 18

    for ax_idx, (ax, mode_label) in enumerate(zip(axes, ("Mean prototype", "K-Means clusters"))):
        _apply_dark_ax(ax)

        # ── train scatter ──
        ax.scatter(hcc_2d[:, 0], hcc_2d[:, 1],
                   c=_CLS_COLOR["HCC"], s=_s_scatter, alpha=_alpha_scatter,
                   label="Train HCC", zorder=2)
        ax.scatter(hem_2d[:, 0], hem_2d[:, 1],
                   c=_CLS_COLOR["Hemangioma"], s=_s_scatter, alpha=_alpha_scatter,
                   label="Train Hemangioma", zorder=2)

        # ── val/test overlay ──
        if val_2d is not None and val_labels is not None:
            for cls_idx, cls_nm in enumerate(("Hemangioma", "HCC")):
                m = val_labels == cls_idx
                if m.sum() > 0:
                    ax.scatter(val_2d[m, 0], val_2d[m, 1],
                               c=_CLS_COLOR[cls_nm], s=55, alpha=0.85,
                               marker="^", edgecolors="white", linewidths=0.8,
                               label=f"Val {cls_nm}", zorder=4)

        if ax_idx == 0:
            # ── mean mode: single centre per subgroup ──
            ax.scatter(*hcc_mean_2d[0], marker="*", s=480, zorder=9,
                       c=_CLS_COLOR["HCC"], edgecolors="white", linewidths=1.5,
                       label="HCC mean centre ★")
            ax.scatter(*hem_mean_2d[0], marker="*", s=480, zorder=9,
                       c=_CLS_COLOR["Hemangioma"], edgecolors="white", linewidths=1.5,
                       label="Hem mean centre ★")

            # enclosing circle = std-radius ellipse around all train samples
            for pts, color, label in [
                (hcc_2d, _CLS_COLOR["HCC"],         "HCC cluster"),
                (hem_2d, _CLS_COLOR["Hemangioma"],   "Hem cluster"),
            ]:
                cx, cy = pts.mean(0)
                rx, ry = pts.std(0)
                theta  = np.linspace(0, 2 * np.pi, 200)
                ax.plot(cx + rx * np.cos(theta), cy + ry * np.sin(theta),
                        color=color, lw=1.5, ls="--", alpha=0.55, label=label + " (1σ)")

        else:
            # ── kmeans mode: cluster circles + cluster centres ──
            # colour gradient within each subgroup
            n_km_hcc = len(hcc_bank_km)
            n_km_hem = len(hem_bank_km)

            for ci in range(n_km_hcc):
                mask_pts = hcc_assign_km == ci
                pts = hcc_2d[mask_pts]
                if len(pts) < 2:
                    continue
                cx_pt, cy_pt = pts.mean(0)
                rx, ry = pts.std(0) + 1e-3
                theta  = np.linspace(0, 2 * np.pi, 200)
                ax.plot(cx_pt + rx * np.cos(theta), cy_pt + ry * np.sin(theta),
                        color=_CLS_COLOR["HCC"], lw=1.2, ls=":", alpha=0.45)

            for ci in range(n_km_hem):
                mask_pts = hem_assign_km == ci
                pts = hem_2d[mask_pts]
                if len(pts) < 2:
                    continue
                cx_pt, cy_pt = pts.mean(0)
                rx, ry = pts.std(0) + 1e-3
                theta  = np.linspace(0, 2 * np.pi, 200)
                ax.plot(cx_pt + rx * np.cos(theta), cy_pt + ry * np.sin(theta),
                        color=_CLS_COLOR["Hemangioma"], lw=1.2, ls=":", alpha=0.45)

            # subgroup mean centres (stars)
            ax.scatter(*hcc_2d.mean(0), marker="*", s=480, zorder=9,
                       c=_CLS_COLOR["HCC"], edgecolors="white", linewidths=1.5,
                       label="HCC subgroup centre ★")
            ax.scatter(*hem_2d.mean(0), marker="*", s=480, zorder=9,
                       c=_CLS_COLOR["Hemangioma"], edgecolors="white", linewidths=1.5,
                       label="Hem subgroup centre ★")

            # per-cluster centres (diamonds)
            ax.scatter(hcc_km_2d[:, 0], hcc_km_2d[:, 1],
                       marker="D", s=160, zorder=8,
                       c=_CLS_COLOR["HCC"], edgecolors="white", linewidths=1.5,
                       label="HCC cluster centres ◆")
            ax.scatter(hem_km_2d[:, 0], hem_km_2d[:, 1],
                       marker="D", s=160, zorder=8,
                       c=_CLS_COLOR["Hemangioma"], edgecolors="white", linewidths=1.5,
                       label="Hem cluster centres ◆")

            # annotate cluster index
            for ci, pt in enumerate(hcc_km_2d):
                ax.text(pt[0], pt[1] + 1.5, f"H{ci}", color=_CLS_COLOR["HCC"],
                        fontsize=7, ha="center", va="bottom", zorder=10)
            for ci, pt in enumerate(hem_km_2d):
                ax.text(pt[0], pt[1] + 1.5, f"h{ci}", color=_CLS_COLOR["Hemangioma"],
                        fontsize=7, ha="center", va="bottom", zorder=10)

        ax.set_title(mode_label, color=_TEXT, fontsize=10, pad=4)
        ax.set_xlabel("t-SNE dim 1", color=_TEXT, fontsize=9)
        ax.set_ylabel("t-SNE dim 2", color=_TEXT, fontsize=9)
        ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT,
                  fontsize=7.5, loc="best", ncol=2)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[cosine_probe] t-SNE saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# WandB logging
# ─────────────────────────────────────────────────────────────

def _log_wandb(
    cfg           : CosineProbeConfig,
    val_results   : dict,
    test_results  : dict,
    delong_records: list[dict],
    roc_figs      : dict[str, plt.Figure],
    dist_figs     : dict[str, plt.Figure],
    tsne_fig      : Optional[plt.Figure],
    score_records : list[dict],
) -> None:
    if _wandb is None or _wandb.run is None:
        print("[cosine_probe] WandB run not active — skipping logging.")
        return

    pfx = f"{cfg.wandb_prefix}/{cfg.model_id}"
    log_dict: dict = {}

    for split_tag, res in [("val", val_results), ("test", test_results)]:
        for roc_key, m in res["metrics"].items():
            tag = (roc_key.lower()
                   .replace(" ", "_").replace("(", "").replace(")", "")
                   .replace("/", "_").replace("−", "minus"))
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{pfx}/{split_tag}/{tag}_{k}"] = v

    for row in delong_records:
        comp = row["comparison"].replace(" ", "_")
        for k, v in row.items():
            if k != "comparison" and isinstance(v, (int, float)):
                log_dict[f"{pfx}/delong_{comp}_{k}"] = v

    for split, fig in roc_figs.items():
        log_dict[f"{pfx}/triple_roc_{split.lower()}"] = _fig_to_wandb_image(
            fig, f"Triple ROC ({split})")
    for split, fig in dist_figs.items():
        log_dict[f"{pfx}/cosine_dist_{split.lower()}"] = _fig_to_wandb_image(
            fig, f"Cosine distribution ({split})")
    if tsne_fig is not None:
        log_dict[f"{pfx}/tsne"] = _fig_to_wandb_image(tsne_fig, "t-SNE embedding space")

    if score_records:
        log_dict[f"{pfx}/cosine_score_table"] = _wandb.Table(
            dataframe=pd.DataFrame(score_records))
    if delong_records:
        log_dict[f"{pfx}/delong_table"] = _wandb.Table(
            dataframe=pd.DataFrame(delong_records))

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

    Both 'mean' and 'kmeans' prototype modes are always computed.
    Embedding source falls back automatically to live extraction if
    saved files are missing or any loading error occurs.

    Parameters
    ----------
    stage2_model : Keras classifier (or encoder-wrapper).
    val_ds       : (x, y) dataset — cutoff source.
    test_ds      : (x, y) dataset — held-out evaluation.
    cfg          : CosineProbeConfig.
    train_ds     : Required for live extraction or as fallback.

    Returns
    -------
    dict with banks, cutoffs, val/test metrics, DeLong results.
    """
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = cfg.positive_class

    print("=" * 64)
    print("  Cosine Probe  —  Confidence vs Cosine Similarity ROC")
    print(f"  model_id         : {cfg.model_id}")
    print(f"  embedding_source : {cfg.embedding_source} (auto-fallback enabled)")
    print(f"  prototype_modes  : mean  AND  kmeans (n_proto={cfg.n_proto})")
    print("=" * 64)

    # ── 1. Build prototype banks ─────────────────────────────
    print("\n[1/7] Resolving train embeddings …")
    hcc_train_emb, hem_train_emb = _get_train_embeddings(cfg, stage2_model, train_ds)

    print("\n[1b/7] Building prototype banks (mean + kmeans) …")
    banks = _build_all_prototype_banks(
        hcc_train_emb, hem_train_emb, cfg.n_proto, cfg.random_state
    )
    for mode, bd in banks.items():
        np.save(out_dir / f"hcc_bank_{mode}.npy",  bd["hcc_bank"])
        np.save(out_dir / f"hem_bank_{mode}.npy",  bd["hem_bank"])
        np.save(out_dir / f"hcc_assign_{mode}.npy", bd["hcc_assign"])
        np.save(out_dir / f"hem_assign_{mode}.npy", bd["hem_assign"])

    # ── 2. Extract scores for val / test ─────────────────────
    print("\n[2/7] Extracting scores (confidence + cosine, both modes) …")

    # confidence pass (labels only needed once)
    split_data: dict[str, dict[str, np.ndarray]] = {}
    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        conf, lbl = _extract_confidence_scores(stage2_model, ds, pos)
        entry = dict(conf=conf, labels=lbl)

        for mode, bd in banks.items():
            hcc_cos, hem_cos, delta, embs, lbl2 = _extract_cosine_scores_from_ds(
                stage2_model, ds, bd["hcc_bank"], bd["hem_bank"], pos
            )
            assert np.array_equal(lbl, lbl2), f"label mismatch at {split_tag}/{mode}"
            entry[f"hcc_cos_{mode}"] = hcc_cos
            entry[f"hem_cos_{mode}"] = hem_cos
            entry[f"delta_{mode}"]   = delta
            if mode == "mean":          # store once (same embeddings regardless of bank)
                entry["embs"] = embs

        split_data[split_tag] = entry
        # save CSV
        pd.DataFrame({
            "confidence_score"      : conf,
            "hcc_cosine_mean"       : entry["hcc_cos_mean"],
            "hem_cosine_mean"       : entry["hem_cos_mean"],
            "delta_mean"            : entry["delta_mean"],
            "hcc_cosine_kmeans"     : entry["hcc_cos_kmeans"],
            "hem_cosine_kmeans"     : entry["hem_cos_kmeans"],
            "delta_kmeans"          : entry["delta_kmeans"],
            "label"                 : lbl,
        }).to_csv(out_dir / f"{split_tag.lower()}_scores.csv", index=False)
        print(f"  {split_tag}  n={len(lbl)}  class_dist={np.bincount(lbl).tolist()}")

    # ── 3. Cutoffs from val ──────────────────────────────────
    print(f"\n[3/7] Determining cutoffs from Val ({cfg.cutoff_strategy}) …")
    val_d = split_data["Val"]
    score_map_val: dict[str, np.ndarray] = {
        "ROC-A (Confidence)"          : val_d["conf"],
        "ROC-B (HCC Cosine / mean)"   : val_d["hcc_cos_mean"],
        "ROC-C (ΔScore / mean)"       : val_d["delta_mean"],
        "ROC-B (HCC Cosine / kmeans)" : val_d["hcc_cos_kmeans"],
        "ROC-C (ΔScore / kmeans)"     : val_d["delta_kmeans"],
    }
    cutoffs: dict[str, float] = {}
    for roc_name, sc in score_map_val.items():
        c = _find_cutoff(sc, val_d["labels"], cfg.cutoff_strategy, cfg.sens_target)
        cutoffs[roc_name] = c
        print(f"  {roc_name:38s}  cutoff = {c:.4f}")
    pd.DataFrame([{"roc": k, "cutoff": v} for k, v in cutoffs.items()]).to_csv(
        out_dir / "cutoffs.csv", index=False
    )

    # ── 4. Metrics at cutoff ─────────────────────────────────
    print("\n[4/7] Computing metrics …")
    def _eval_split(sd: dict) -> dict[str, dict]:
        sm: dict[str, np.ndarray] = {
            "ROC-A (Confidence)"          : sd["conf"],
            "ROC-B (HCC Cosine / mean)"   : sd["hcc_cos_mean"],
            "ROC-C (ΔScore / mean)"       : sd["delta_mean"],
            "ROC-B (HCC Cosine / kmeans)" : sd["hcc_cos_kmeans"],
            "ROC-C (ΔScore / kmeans)"     : sd["delta_kmeans"],
        }
        return {rn: _metrics_at_cutoff(sc, sd["labels"], cutoffs[rn])
                for rn, sc in sm.items()}

    val_metrics  = _eval_split(split_data["Val"])
    test_metrics = _eval_split(split_data["Test"])

    for split_tag, metrics in [("Val", val_metrics), ("Test", test_metrics)]:
        rows = [{"roc": rn, **m} for rn, m in metrics.items()]
        pd.DataFrame(rows).to_csv(
            out_dir / f"{split_tag.lower()}_metrics.csv", index=False)
        print(f"  ── {split_tag} ──")
        for rn, m in metrics.items():
            print(f"  {rn:38s}  AUROC={m['auroc']:.3f}  "
                  f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                  f"F1={m['f1']:.3f}")

    # ── 5. DeLong tests ──────────────────────────────────────
    print("\n[5/7] DeLong test (A vs B/C, mean and kmeans) …")
    vl = split_data["Val"]["labels"]
    delong_pairs = [
        ("A vs B-mean",   val_d["conf"], val_d["hcc_cos_mean"]),
        ("A vs C-mean",   val_d["conf"], val_d["delta_mean"]),
        ("A vs B-kmeans", val_d["conf"], val_d["hcc_cos_kmeans"]),
        ("A vs C-kmeans", val_d["conf"], val_d["delta_kmeans"]),
        ("B-mean vs B-kmeans", val_d["hcc_cos_mean"], val_d["hcc_cos_kmeans"]),
        ("C-mean vs C-kmeans", val_d["delta_mean"],   val_d["delta_kmeans"]),
    ]
    delong_records: list[dict] = []
    for name, sa, sb in delong_pairs:
        res = delong_test(vl, sa, sb)
        delong_records.append({"comparison": name, **res})
        print(f"  {name:30s}  z={res['z_stat']:+.3f}  p={res['p_value']:.4f}  "
              f"ΔAUROC={res['auroc_a'] - res['auroc_b']:+.4f}")
    pd.DataFrame(delong_records).to_csv(out_dir / "delong.csv", index=False)

    # ── 6. Plots ─────────────────────────────────────────────
    print("\n[6/7] Generating plots …")
    roc_figs : dict[str, plt.Figure] = {}
    dist_figs: dict[str, plt.Figure] = {}
    score_records: list[dict] = []

    for split_tag, sd in split_data.items():
        sd_scores = {
            "ROC-A (Confidence)"          : (sd["conf"],           sd["labels"]),
            "ROC-B (HCC Cosine / mean)"   : (sd["hcc_cos_mean"],   sd["labels"]),
            "ROC-C (ΔScore / mean)"       : (sd["delta_mean"],     sd["labels"]),
            "ROC-B (HCC Cosine / kmeans)" : (sd["hcc_cos_kmeans"], sd["labels"]),
            "ROC-C (ΔScore / kmeans)"     : (sd["delta_kmeans"],   sd["labels"]),
        }
        roc_figs[split_tag] = plot_triple_roc(
            sd_scores, cutoffs, split=split_tag,
            save_path=out_dir / f"triple_roc_{split_tag.lower()}.png",
        )
        dist_figs[split_tag] = plot_cosine_distribution(
            hcc_cos  = sd["hcc_cos_mean"],
            hem_cos  = sd["hem_cos_mean"],
            delta    = sd["delta_mean"],
            labels   = sd["labels"],
            cutoffs  = cutoffs,
            split    = split_tag,
            save_path=out_dir / f"cosine_dist_{split_tag.lower()}.png",
        )
        for i in range(len(sd["labels"])):
            score_records.append({
                "split"                 : split_tag,
                "true_label"            : int(sd["labels"][i]),
                "confidence_score"      : float(sd["conf"][i]),
                "hcc_cosine_mean"       : float(sd["hcc_cos_mean"][i]),
                "hem_cosine_mean"       : float(sd["hem_cos_mean"][i]),
                "delta_mean"            : float(sd["delta_mean"][i]),
                "hcc_cosine_kmeans"     : float(sd["hcc_cos_kmeans"][i]),
                "hem_cosine_kmeans"     : float(sd["hem_cos_kmeans"][i]),
                "delta_kmeans"          : float(sd["delta_kmeans"][i]),
            })

    # t-SNE
    print("\n[7/7] t-SNE visualisation …")
    tsne_fig = None
    try:
        tsne_fig = plot_tsne(
            train_hcc_emb   = hcc_train_emb,
            train_hem_emb   = hem_train_emb,
            hcc_assign_mean = banks["mean"]["hcc_assign"],
            hem_assign_mean = banks["mean"]["hem_assign"],
            hcc_assign_km   = banks["kmeans"]["hcc_assign"],
            hem_assign_km   = banks["kmeans"]["hem_assign"],
            hcc_bank_mean   = banks["mean"]["hcc_bank"],
            hem_bank_mean   = banks["mean"]["hem_bank"],
            hcc_bank_km     = banks["kmeans"]["hcc_bank"],
            hem_bank_km     = banks["kmeans"]["hem_bank"],
            cfg             = cfg,
            save_path       = out_dir / "tsne.png",
            val_embs        = split_data["Val"].get("embs"),
            val_labels      = split_data["Val"].get("labels"),
        )
    except Exception as e:
        print(f"[cosine_probe] WARNING: t-SNE failed: {e}  (skipping)")

    _log_wandb(
        cfg           = cfg,
        val_results   = {"metrics": val_metrics},
        test_results  = {"metrics": test_metrics},
        delong_records= delong_records,
        roc_figs      = roc_figs,
        dist_figs     = dist_figs,
        tsne_fig      = tsne_fig,
        score_records = score_records,
    )

    for fig in list(roc_figs.values()) + list(dist_figs.values()):
        plt.close(fig)
    if tsne_fig is not None:
        plt.close(tsne_fig)

    print("\n[cosine_probe] Done.")
    print(f"  Outputs → {out_dir}")

    return dict(
        banks         = banks,
        cutoffs       = cutoffs,
        val_metrics   = val_metrics,
        test_metrics  = test_metrics,
        delong_records= delong_records,
    )
