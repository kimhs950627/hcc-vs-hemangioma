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

Design note (v2 refactor)
--------------------------
``run_cosine_probe`` now returns ``split_data`` as part of its result dict so
that ``run_cosine_probe_v2`` can reuse the already-extracted embeddings and
scores without a second forward pass through the model.  The [ext-1] block
in v2 is therefore removed entirely.

Usage
-----
from validation.cosine_probe import CosineProbeConfig, run_cosine_probe_v2

cfg = CosineProbeConfig(
    model_id        = model_id,
    work_dir        = cfg.work_dir,
    embedding_source= "saved",          # tries saved first, falls back to live
)
results = run_cosine_probe_v2(
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
# Score extraction helper  (single pass, all modes)
# ─────────────────────────────────────────────────────────────

def _extract_all_scores_for_split(
    model  : object,
    dataset: tf.data.Dataset,
    banks  : dict[str, dict],
    pos    : int,
) -> dict[str, np.ndarray]:
    """Extract confidence + cosine scores for one dataset split in a single pass.

    Embeddings are extracted once (mean-mode bank pass) then reused for
    kmeans-mode cosine computation.  Confidence requires a separate forward
    pass through the full classifier head (not the encoder), so it is
    collected independently but still in a single loop.

    Returns a flat dict:
        conf, labels, embs,
        hcc_cos_mean, hem_cos_mean, delta_mean,
        hcc_cos_kmeans, hem_cos_kmeans, delta_kmeans
    """
    # ── single embedding pass (mean bank — embs are bank-agnostic) ──
    hcc_cos_m, hem_cos_m, delta_m, embs, labels = _extract_cosine_scores_from_ds(
        model, dataset, banks["mean"]["hcc_bank"], banks["mean"]["hem_bank"], pos
    )
    # ── kmeans cosine computed from already-extracted embs (no second pass) ──
    hcc_cos_km = _cosine_sim_to_bank(embs, banks["kmeans"]["hcc_bank"])
    hem_cos_km = _cosine_sim_to_bank(embs, banks["kmeans"]["hem_bank"])
    delta_km   = (hcc_cos_km - hem_cos_km).astype(np.float32)

    # ── confidence (separate classifier head forward pass) ──
    conf, lbl2 = _extract_confidence_scores(model, dataset, pos)
    assert np.array_equal(labels, lbl2), "label mismatch between embedding and confidence pass"

    return dict(
        conf           = conf,
        labels         = labels,
        embs           = embs,
        hcc_cos_mean   = hcc_cos_m,
        hem_cos_mean   = hem_cos_m,
        delta_mean     = delta_m,
        hcc_cos_kmeans = hcc_cos_km.astype(np.float32),
        hem_cos_kmeans = hem_cos_km.astype(np.float32),
        delta_kmeans   = delta_km,
    )


# ─────────────────────────────────────────────────────────────
# DeLong test  (non-parametric AUROC comparison)
# ─────────────────────────────────────────────────────────────

def _delong_auc_variance(
    labels : np.ndarray,
    scores : np.ndarray,
) -> tuple[float, float]:
    """DeLong AUC estimator with variance (scalar outputs).

    Returns
    -------
    auroc : float
    var   : float  (scalar, NOT array — avoids downstream shape issues)
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    mat_pos = np.zeros(n_pos)
    mat_neg = np.zeros(n_neg)
    for i, p in enumerate(pos):
        mat_pos[i] = np.mean((p > neg) + 0.5 * (p == neg))
    for j, n in enumerate(neg):
        mat_neg[j] = np.mean((pos > n) + 0.5 * (pos == n))

    auroc = float(np.mean(mat_pos))
    # explicit float() cast — prevents shape-(1,) array propagation
    var   = float(np.var(mat_pos, ddof=1) / n_pos + np.var(mat_neg, ddof=1) / n_neg)
    return auroc, var


def delong_test(
    labels   : np.ndarray,
    scores_a : np.ndarray,
    scores_b : np.ndarray,
) -> dict:
    """Non-parametric DeLong AUROC comparison.

    All intermediate values are explicitly cast to Python float to avoid
    numpy scalar / 0-dim array type errors in downstream float() calls.
    """
    from scipy.stats import norm
    auc_a, var_a = _delong_auc_variance(labels, scores_a)
    auc_b, var_b = _delong_auc_variance(labels, scores_b)
    # var_a / var_b are now guaranteed scalar floats
    se = float(np.sqrt(float(var_a) + float(var_b)))
    z  = float((float(auc_a) - float(auc_b)) / (se + 1e-12))
    p  = float(2.0 * norm.sf(abs(z)))   # builtin abs() — safe for any numeric type
    return {
        "auroc_a"  : float(auc_a),
        "auroc_b"  : float(auc_b),
        "z_stat"   : z,
        "p_value"  : p,
        "se"       : se,
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

def _interpret_delong_row(row: dict) -> str:
    delta = float(row["auroc_a"]) - float(row["auroc_b"])
    p     = float(row["p_value"])
    better = "first metric (A)" if delta > 0 else "second metric (B/C)"
    signif = "statistically significant (p<0.05)" if p < 0.05 else "not statistically significant (p≥0.05)"
    return (
        f"{row['comparison']}: ΔAUROC={delta:+.4f}, p={p:.4f}. "
        f"The {better} had higher AUROC. The difference was {signif}."
    )


def _interpret_calibration(split: str, calib: dict) -> str:
    ece = float(calib.get("ece", float("nan")))
    mce = float(calib.get("mce", float("nan")))
    ace = float(calib.get("ace", float("nan")))
    if ece < 0.05:
        level = "well calibrated — softmax output approximates true fraction positive"
    elif ece < 0.10:
        level = "moderately calibrated — some overconfidence present"
    else:
        level = ("poorly calibrated / significantly overconfident — "
                 "softmax output should NOT be interpreted as a true probability without temperature scaling")
    return (
        f"{split} calibration: ECE={ece:.4f}, MCE={mce:.4f}, ACE={ace:.4f}. "
        f"Model is {level}."
    )


def _build_interpretation_df(
    val_delong   : list[dict],
    test_delong  : list[dict],
    calib_results: dict[str, dict],
) -> pd.DataFrame:
    """Build human-readable interpretation table for cosine_probe/abstract/summary."""
    rows: list[dict] = []

    rows.append({
        "section": "Concept",
        "item": "confidence_vs_probability",
        "interpretation": (
            "Softmax output P(HCC) is NOT a calibrated clinical probability. "
            "It reflects the model's relative certainty between classes (logit ratio), "
            "not the true frequency of HCC among cases with that score. "
            "Calibration metrics (ECE/MCE/ACE) quantify this gap. "
            "The cosine-based HCC Score provides a complementary, representation-space view "
            "aligned with how radiologists reason: 'how much does this lesion resemble typical HCC?'"
        ),
    })

    for row in val_delong:
        rows.append({
            "section": "DeLong/Val",
            "item": row["comparison"],
            "interpretation": _interpret_delong_row(row),
        })
    for row in test_delong:
        rows.append({
            "section": "DeLong/Test",
            "item": row["comparison"],
            "interpretation": _interpret_delong_row(row),
        })

    for split_tag in ("Val", "Test"):
        if split_tag in calib_results:
            rows.append({
                "section": "Calibration",
                "item": split_tag,
                "interpretation": _interpret_calibration(split_tag, calib_results[split_tag]),
            })

    rows.append({
        "section": "Metrics",
        "item": "ROC-B vs ROC-A",
        "interpretation": (
            "ROC-B (HCC Cosine Score, mean prototype) measures embedding-space similarity "
            "to the HCC prototype cluster. If AUROC-B ≈ AUROC-A, cosine similarity is a "
            "competitive and interpretable alternative. If AUROC-B < AUROC-A, both metrics "
            "provide complementary discriminative information."
        ),
    })
    rows.append({
        "section": "Metrics",
        "item": "ROC-C vs ROC-A",
        "interpretation": (
            "ROC-C (ΔScore = HCC cosine − Hemangioma cosine) captures the discriminative "
            "margin in embedding space. High ΔScore means the lesion is simultaneously more "
            "similar to HCC prototypes AND less similar to Hemangioma prototypes, directly "
            "mirroring the radiologist's differential diagnosis reasoning."
        ),
    })

    return pd.DataFrame(rows)


def _log_wandb(
    cfg             : CosineProbeConfig,
    val_results     : dict,
    test_results    : dict,
    delong_records  : list[dict],
    roc_figs        : dict[str, plt.Figure],
    dist_figs       : dict[str, plt.Figure],
    tsne_fig        : Optional[plt.Figure],
    score_records   : list[dict],
    # v2 extras (defaulted to None for backward compat with run_cosine_probe)
    scatter_figs    : Optional[dict[str, plt.Figure]] = None,
    cm_figs         : Optional[dict[str, plt.Figure]] = None,
    calib_figs      : Optional[dict[str, plt.Figure]] = None,
    calib_results   : Optional[dict[str, dict]]       = None,
    test_delong     : Optional[list[dict]]             = None,
    out_dir         : Optional[Path]                   = None,
) -> None:
    """Structured W&B logging with sub-prefix hierarchy.

    Sub-prefix layout
    -----------------
    cosine_probe/ROC/               Triple ROC images (Val / Test)
    cosine_probe/scatter/           Conf-vs-cosine scatter plots
    cosine_probe/confusion_matrix/  Confusion matrices per ROC key
    cosine_probe/images/            Reliability diagrams, cosine distributions, t-SNE
    cosine_probe/table/             All metric / DeLong / calibration / score tables
    cosine_probe/abstract/          Human-readable interpretation table
    cosine_probe/original_csv       W&B Artifact — all CSV exports for post-hoc reuse
    """
    if _wandb is None or _wandb.run is None:
        print("[cosine_probe] WandB run not active — skipping logging.")
        return

    pfx = cfg.wandb_prefix.rstrip("/")   # e.g. "cosine_probe"
    log_dict: dict = {}

    # ── ROC images ──────────────────────────────────────────────
    for split_tag, fig in roc_figs.items():
        log_dict[f"{pfx}/ROC/{split_tag.lower()}_triple_roc"] = _fig_to_wandb_image(
            fig, f"Triple ROC ({split_tag})"
        )

    # ── Scatter ─────────────────────────────────────────────────
    if scatter_figs:
        for split_tag, fig in scatter_figs.items():
            log_dict[f"{pfx}/scatter/{split_tag.lower()}_conf_vs_cosine"] = _fig_to_wandb_image(
                fig, f"Confidence vs Cosine Score ({split_tag})"
            )

    # ── Confusion matrices ───────────────────────────────────────
    if cm_figs:
        for key, fig in cm_figs.items():
            # key format: "{split_tag}/{roc_name_slug}"
            safe_key = key.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            log_dict[f"{pfx}/confusion_matrix/{safe_key}"] = _fig_to_wandb_image(
                fig, f"Confusion Matrix ({key})"
            )

    # ── Other images (reliability, cosine distribution, t-SNE) ──
    if calib_figs:
        for split_tag, fig in calib_figs.items():
            log_dict[f"{pfx}/images/{split_tag.lower()}_reliability"] = _fig_to_wandb_image(
                fig, f"Reliability Diagram ({split_tag})"
            )
    for split_tag, fig in dist_figs.items():
        log_dict[f"{pfx}/images/{split_tag.lower()}_cosine_distribution"] = _fig_to_wandb_image(
            fig, f"Cosine Distribution ({split_tag})"
        )
    if tsne_fig is not None:
        log_dict[f"{pfx}/images/tsne"] = _fig_to_wandb_image(tsne_fig, "t-SNE embedding space")

    # ── Tables ───────────────────────────────────────────────────
    if val_results.get("metrics"):
        log_dict[f"{pfx}/table/val_metrics"] = _wandb.Table(
            dataframe=pd.DataFrame(
                [{"roc": rn, **m} for rn, m in val_results["metrics"].items()]
            )
        )
    if test_results.get("metrics"):
        log_dict[f"{pfx}/table/test_metrics"] = _wandb.Table(
            dataframe=pd.DataFrame(
                [{"roc": rn, **m} for rn, m in test_results["metrics"].items()]
            )
        )
    if delong_records:
        log_dict[f"{pfx}/table/delong_val"] = _wandb.Table(
            dataframe=pd.DataFrame(delong_records)
        )
    if test_delong:
        log_dict[f"{pfx}/table/delong_test"] = _wandb.Table(
            dataframe=pd.DataFrame(test_delong)
        )
    if calib_results:
        log_dict[f"{pfx}/table/calibration"] = _wandb.Table(
            dataframe=pd.DataFrame(
                [{"split": sp, **m} for sp, m in calib_results.items()]
            )
        )
    if score_records:
        log_dict[f"{pfx}/table/all_scores"] = _wandb.Table(
            dataframe=pd.DataFrame(score_records)
        )

    # ── Abstract (human-readable interpretation) ─────────────────
    _test_dl = test_delong or []
    _calib   = calib_results or {}
    interpretation_df = _build_interpretation_df(delong_records, _test_dl, _calib)
    log_dict[f"{pfx}/abstract/summary"] = _wandb.Table(dataframe=interpretation_df)

    _wandb.log(log_dict)
    print(f"[cosine_probe] WandB logged under prefix: {pfx}/")

    # ── CSV Artifact ─────────────────────────────────────────────
    if out_dir is not None and out_dir.exists():
        csv_files = sorted(out_dir.glob("*.csv"))
        if csv_files:
            art = _wandb.Artifact(
                name=f"{cfg.model_id}-cosine-probe-csv",
                type="analysis",
                description="Original CSV exports from cosine_probe for post-hoc reuse",
                metadata={"model_id": cfg.model_id},
            )
            for csv_file in csv_files:
                art.add_file(str(csv_file),
                             name=f"cosine_probe/original_csv/{csv_file.name}")
            _wandb.run.log_artifact(art)
            print(f"[cosine_probe] CSV artifact logged: {len(csv_files)} files")



# ─────────────────────────────────────────────────────────────
# Public API  —  run_cosine_probe
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
    dict with keys:
        banks, cutoffs, val_metrics, test_metrics, delong_records,
        split_data   ← NEW: raw per-sample scores reused by v2 (no re-extraction)
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

    # ── 2. Extract scores for val / test (single pass per split) ─
    print("\n[2/7] Extracting scores (single pass per split) …")
    split_data: dict[str, dict[str, np.ndarray]] = {}
    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        entry = _extract_all_scores_for_split(stage2_model, ds, banks, pos)
        split_data[split_tag] = entry
        # save CSV
        pd.DataFrame({
            "confidence_score"      : entry["conf"],
            "hcc_cosine_mean"       : entry["hcc_cos_mean"],
            "hem_cosine_mean"       : entry["hem_cos_mean"],
            "delta_mean"            : entry["delta_mean"],
            "hcc_cosine_kmeans"     : entry["hcc_cos_kmeans"],
            "hem_cosine_kmeans"     : entry["hem_cos_kmeans"],
            "delta_kmeans"          : entry["delta_kmeans"],
            "label"                 : entry["labels"],
        }).to_csv(out_dir / f"{split_tag.lower()}_scores.csv", index=False)
        print(f"  {split_tag}  n={len(entry['labels'])}  "
              f"class_dist={np.bincount(entry['labels']).tolist()}")

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
        res = delong_test(val_d["labels"], sa, sb)
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
        banks          = banks,
        cutoffs        = cutoffs,
        val_metrics    = val_metrics,
        test_metrics   = test_metrics,
        delong_records = delong_records,
        split_data     = split_data,   # ← v2 reuses this; no second forward pass
    )

# ─────────────────────────────────────────────────────────────
# Calibration reliability diagram
# ─────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    conf_scores : np.ndarray,   # (N,) softmax P(HCC)
    labels      : np.ndarray,   # (N,) binary ground-truth
    n_bins      : int = 10,
    split       : str = "Val",
    save_path   : Optional[Path] = None,
) -> plt.Figure:
    """Reliability diagram to visualise softmax calibration error.

    Shows observed positive fraction vs mean predicted confidence per bin.
    A perfectly calibrated model should sit on the y=x diagonal.

    Also reports:
        ECE  — Expected Calibration Error
        MCE  — Maximum Calibration Error
        ACE  — Adaptive (equal-sample) Calibration Error
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres: list[float] = []
    bin_accuracy: list[float] = []
    bin_confidence: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf_scores >= lo) & (conf_scores < hi)
        if not mask.any():
            continue
        bin_centres.append(float((lo + hi) / 2))
        bin_accuracy.append(float(labels[mask].mean()))
        bin_confidence.append(float(conf_scores[mask].mean()))
        bin_counts.append(int(mask.sum()))

    bin_counts_arr    = np.array(bin_counts, dtype=np.float64)
    bin_accuracy_arr  = np.array(bin_accuracy)
    bin_confidence_arr = np.array(bin_confidence)
    total             = bin_counts_arr.sum()

    ece = float(np.sum(bin_counts_arr * np.abs(bin_accuracy_arr - bin_confidence_arr)) / (total + 1e-8))
    mce = float(np.max(np.abs(bin_accuracy_arr - bin_confidence_arr))) if len(bin_accuracy_arr) else 0.0

    # Adaptive CE: equal-population bins
    sorted_idx = np.argsort(conf_scores)
    ace_bins   = np.array_split(sorted_idx, n_bins)
    ace = 0.0
    for b in ace_bins:
        if len(b) == 0:
            continue
        ace += len(b) / len(conf_scores) * abs(labels[b].mean() - conf_scores[b].mean())

    # ── plot ──
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=_BG)
    _apply_dark_ax(ax)

    ax.bar(bin_centres, bin_accuracy, width=1.0 / n_bins * 0.85,
           color="#4f98a3", alpha=0.75, label="Fraction positive (observed)")
    ax.bar(bin_centres, bin_confidence, width=1.0 / n_bins * 0.85,
           color="#e8af34", alpha=0.40, label="Mean confidence (model)")

    ax.plot([0, 1], [0, 1], ls="--", lw=1.5, color=_BORDER, label="Perfect calibration")

    ax.set_xlim([0, 1]);  ax.set_ylim([0, 1])
    ax.set_xlabel("Confidence (softmax P(HCC))", fontsize=11)
    ax.set_ylabel("Fraction positive",           fontsize=11)
    ax.set_title(
        f"Reliability Diagram ({split})\n"
        f"ECE={ece:.4f}  MCE={mce:.4f}  ACE={ace:.4f}",
        fontsize=10, pad=8, color=_TEXT,
    )
    ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"[cosine_probe] Reliability diagram ({split}) → {save_path}")

    return fig, {"ece": ece, "mce": mce, "ace": ace}


# ─────────────────────────────────────────────────────────────
# Confidence vs Cosine scatter plot  (disagreement visualisation)
# ─────────────────────────────────────────────────────────────

def plot_conf_vs_cosine_scatter(
    conf_scores  : np.ndarray,   # (N,)
    cosine_scores: np.ndarray,   # (N,)  HCC cosine (mean-mode)
    delta_scores : np.ndarray,   # (N,)
    labels       : np.ndarray,   # (N,)
    cutoff_conf  : float,
    cutoff_cos   : float,
    split        : str,
    save_path    : Optional[Path] = None,
) -> plt.Figure:
    """Scatter: confidence (x) vs HCC cosine score (y), coloured by true label.

    Quadrant lines at (cutoff_conf, cutoff_cos) highlight four cases:
        Q1 (↑conf, ↑cos)  : agreement — positive
        Q2 (↓conf, ↑cos)  : disagreement — cos says HCC, conf says Hem
        Q3 (↓conf, ↓cos)  : agreement — negative
        Q4 (↑conf, ↓cos)  : disagreement — conf says HCC, cos says Hem
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=_BG)
    fig.suptitle(
        f"Confidence vs Cosine Score Scatter ({split})",
        color=_TEXT, fontsize=12, y=1.01,
    )

    panel_data = [
        ("HCC Cosine Score",  cosine_scores, cutoff_cos,
         ["Q1: agree (+)", "Q2: cos+/conf−", "Q3: agree (−)", "Q4: conf+/cos−"]),
        ("ΔScore (HCC−Hem)",  delta_scores,  None,
         None),
    ]

    for ax, (ylabel, ysc, ycut, quad_labels) in zip(axes, panel_data):
        _apply_dark_ax(ax)
        for cls_idx, (cls_name, marker) in enumerate([("Hemangioma", "o"), ("HCC", "^")]):
            m = labels == cls_idx
            ax.scatter(conf_scores[m], ysc[m],
                       c=_CLS_COLOR[cls_name], s=28, alpha=0.65, marker=marker,
                       edgecolors="none", label=f"{cls_name}  n={m.sum()}", zorder=3)

        ax.axvline(cutoff_conf, color="#a86fdf", lw=1.5, ls="--",
                   label=f"conf cutoff={cutoff_conf:.3f}", alpha=0.85)
        if ycut is not None:
            ax.axhline(ycut, color="#fdc551", lw=1.5, ls="--",
                       label=f"cos cutoff={ycut:.3f}", alpha=0.85)

        ax.set_xlabel("Confidence (softmax)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(framealpha=0.15, facecolor=_BG, labelcolor=_TEXT, fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"[cosine_probe] Conf-vs-cosine scatter ({split}) → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    scores    : np.ndarray,
    labels    : np.ndarray,
    cutoff    : float,
    roc_name  : str,
    split     : str,
    class_names: tuple[str, str] = ("Hemangioma", "HCC"),
    save_path : Optional[Path] = None,
) -> plt.Figure:
    """Normalised confusion matrix as a 2×2 heatmap."""
    from sklearn.metrics import confusion_matrix as _cm
    preds = (scores >= cutoff).astype(int)
    cm = _cm(labels, preds, labels=[0, 1])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=_BG)
    _apply_dark_ax(ax)

    im = ax.imshow(cm_norm, vmin=0, vmax=1, aspect="auto",
                   cmap="YlOrRd", alpha=0.85)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(color=_TEXT)

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})",
                    ha="center", va="center",
                    color=_TEXT, fontsize=10)

    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, color=_TEXT)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names, color=_TEXT)
    ax.set_xlabel("Predicted", fontsize=10, color=_TEXT)
    ax.set_ylabel("True",      fontsize=10, color=_TEXT)
    ax.set_title(f"Confusion Matrix — {roc_name}\n({split}, cutoff={cutoff:.3f})",
                 fontsize=9, color=_TEXT, pad=6)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        print(f"[cosine_probe] Confusion matrix [{roc_name}] ({split}) → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# Per-sample WandB image score table
# ─────────────────────────────────────────────────────────────

def _build_wandb_score_image_table(
    score_records  : list[dict],
    model          : object,
    dataset        : tf.data.Dataset,
    cfg            : CosineProbeConfig,
    split          : str,
    max_samples    : int = 80,
) -> Optional["_wandb.Table"]:
    """Build a wandb.Table with per-sample image + all cosine scores.

    Columns: image | true_label | confidence | hcc_cos_mean | delta_mean |
             hcc_cos_kmeans | delta_kmeans | pred_conf | pred_cos
    """
    if _wandb is None or _wandb.run is None:
        return None

    rows = [r for r in score_records if r["split"] == split][:max_samples]
    if not rows:
        return None

    # Re-collect images in order (best effort; assumes dataset is not shuffled)
    all_imgs: list[np.ndarray] = []
    for x_batch, _ in dataset:
        imgs_np = x_batch.numpy()
        for img in imgs_np:
            if len(all_imgs) >= len(rows):
                break
            all_imgs.append(img)
        if len(all_imgs) >= len(rows):
            break

    table = _wandb.Table(columns=[
        "image", "split", "true_label",
        "confidence_score",
        "hcc_cosine_mean", "hem_cosine_mean", "delta_mean",
        "hcc_cosine_kmeans", "hem_cosine_kmeans", "delta_kmeans",
    ])

    label_names = {0: cfg.class_names[0], 1: cfg.class_names[1]}

    for idx, row in enumerate(rows):
        if idx < len(all_imgs):
            img = all_imgs[idx]
            if img.dtype != np.uint8:
                lo, hi = img.min(), img.max()
                img_disp = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
            else:
                img_disp = img
            wimg = _wandb.Image(
                _PIL_Image.fromarray(img_disp if img_disp.shape[-1] == 3
                                     else img_disp[:, :, 0]),
                caption=f"{label_names.get(row['true_label'], row['true_label'])} "
                        f"| conf={row['confidence_score']:.3f} "
                        f"| Δcos={row['delta_mean']:.3f}",
            )
        else:
            wimg = None

        table.add_data(
            wimg,
            split,
            label_names.get(row["true_label"], row["true_label"]),
            row["confidence_score"],
            row["hcc_cosine_mean"],
            row["hem_cosine_mean"],
            row["delta_mean"],
            row["hcc_cosine_kmeans"],
            row["hem_cosine_kmeans"],
            row["delta_kmeans"],
        )

    return table


# ─────────────────────────────────────────────────────────────
# Test-set DeLong  (separate from val DeLong)
# ─────────────────────────────────────────────────────────────

def run_test_delong(
    test_data   : dict,           # split_data["Test"]
) -> list[dict]:
    """DeLong test on held-out test set."""
    tl = test_data["labels"]
    pairs = [
        ("A vs B-mean",       test_data["conf"],          test_data["hcc_cos_mean"]),
        ("A vs C-mean",       test_data["conf"],          test_data["delta_mean"]),
        ("A vs B-kmeans",     test_data["conf"],          test_data["hcc_cos_kmeans"]),
        ("A vs C-kmeans",     test_data["conf"],          test_data["delta_kmeans"]),
        ("B-mean vs B-kmeans",test_data["hcc_cos_mean"],  test_data["hcc_cos_kmeans"]),
        ("C-mean vs C-kmeans",test_data["delta_mean"],    test_data["delta_kmeans"]),
    ]
    records: list[dict] = []
    for name, sa, sb in pairs:
        r = delong_test(tl, sa, sb)
        records.append({"comparison": name, **r})
        print(f"  [test] {name:30s}  z={r['z_stat']:+.3f}  p={r['p_value']:.4f}  "
              f"ΔAUROC={r['auroc_a'] - r['auroc_b']:+.4f}")
    return records



# ─────────────────────────────────────────────────────────────
# Extended run_cosine_probe_v2  (모든 기능 통합)
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Extended run_cosine_probe_v2  (모든 기능 통합)
# ─────────────────────────────────────────────────────────────

def run_cosine_probe_v2(
    stage2_model : object,
    val_ds       : tf.data.Dataset,
    test_ds      : tf.data.Dataset,
    cfg          : CosineProbeConfig,
    train_ds     : Optional[tf.data.Dataset] = None,
    base_results : Optional[dict]            = None,
) -> dict:
    """Extended cosine probe: adds scatter, confusion matrix, calibration,
    test DeLong, and fully structured WandB logging.

    Parameters
    ----------
    base_results : dict returned by run_cosine_probe().
        If provided, embeddings / scores are reused (no second forward pass).
        If None, run_cosine_probe() is called internally.

    WandB sub-prefix layout
    -----------------------
    cosine_probe/ROC/               Triple ROC images
    cosine_probe/scatter/           Conf-vs-cosine scatter plots
    cosine_probe/confusion_matrix/  Per-ROC confusion matrices
    cosine_probe/images/            Reliability diagrams, distributions, t-SNE
    cosine_probe/table/             Metric / DeLong / calibration / score tables
    cosine_probe/abstract/          Human-readable interpretation
    cosine_probe/original_csv       W&B Artifact — all CSV files
    """
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── [ext-0] Base probe (reuse if already run) ─────────────
    if base_results is None:
        print("[cosine_probe_v2] Running base probe …")
        base_results = run_cosine_probe(
            stage2_model=stage2_model,
            val_ds=val_ds, test_ds=test_ds,
            cfg=cfg, train_ds=train_ds,
        )

    split_data    = base_results["split_data"]
    banks         = base_results["banks"]
    cutoffs       = base_results["cutoffs"]
    val_metrics   = base_results["val_metrics"]
    test_metrics  = base_results["test_metrics"]
    delong_records= base_results["delong_records"]

    # ── [ext-1] Test-set DeLong ──────────────────────────────
    print("\n[v2-1/5] Test-set DeLong …")
    test_delong = run_test_delong(split_data["Test"])
    pd.DataFrame(test_delong).to_csv(out_dir / "delong_test.csv", index=False)

    # ── [ext-2] Calibration (reliability diagram) ────────────
    print("\n[v2-2/5] Calibration analysis …")
    calib_figs: dict[str, plt.Figure]    = {}
    calib_results: dict[str, dict]       = {}

    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        fig, stats = plot_reliability_diagram(
            conf_scores=split_data[split_tag]["conf"],
            labels     =split_data[split_tag]["labels"],
            split      =split_tag,
            save_path  =out_dir / f"reliability_{split_tag.lower()}.png",
        )
        calib_figs[split_tag]    = fig
        calib_results[split_tag] = stats
    pd.DataFrame(
        [{"split": sp, **m} for sp, m in calib_results.items()]
    ).to_csv(out_dir / "calibration.csv", index=False)

    # ── [ext-3] Scatter: conf vs cosine ──────────────────────
    print("\n[v2-3/5] Scatter plots …")
    scatter_figs: dict[str, plt.Figure] = {}

    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        sd = split_data[split_tag]
        fig = plot_conf_vs_cosine_scatter(
            conf_scores  =sd["conf"],
            cosine_scores=sd["hcc_cos_mean"],
            delta_scores =sd["delta_mean"],
            labels       =sd["labels"],
            cutoff_conf  =cutoffs["ROC-A (Confidence)"],
            cutoff_cos   =cutoffs["ROC-B (HCC Cosine / mean)"],
            split        =split_tag,
            save_path    =out_dir / f"scatter_{split_tag.lower()}.png",
        )
        scatter_figs[split_tag] = fig

    # ── [ext-4] Confusion matrices ───────────────────────────
    print("\n[v2-4/5] Confusion matrices …")
    cm_figs: dict[str, plt.Figure] = {}
    roc_keys_to_plot = [
        "ROC-A (Confidence)",
        "ROC-B (HCC Cosine / mean)",
        "ROC-C (ΔScore / mean)",
    ]

    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        sd = split_data[split_tag]
        score_lookup = {
            "ROC-A (Confidence)"         : sd["conf"],
            "ROC-B (HCC Cosine / mean)"  : sd["hcc_cos_mean"],
            "ROC-C (ΔScore / mean)"      : sd["delta_mean"],
            "ROC-B (HCC Cosine / kmeans)": sd["hcc_cos_kmeans"],
            "ROC-C (ΔScore / kmeans)"    : sd["delta_kmeans"],
        }
        for roc_name in roc_keys_to_plot:
            slug = (roc_name.lower()
                    .replace(" ", "_").replace("(", "").replace(")", "")
                    .replace("/", "_").replace("−", "minus").replace("δ", "delta"))
            cm_key = f"{split_tag}/{slug}"
            fig = plot_confusion_matrix(
                scores     =score_lookup[roc_name],
                labels     =sd["labels"],
                cutoff     =cutoffs[roc_name],
                roc_name   =roc_name,
                split      =split_tag,
                class_names=cfg.class_names,
                save_path  =out_dir / f"cm_{split_tag.lower()}_{slug}.png",
            )
            cm_figs[cm_key] = fig

    # ── [ext-5] WandB structured logging ────────────────────
    print("\n[v2-5/5] WandB structured logging …")

    # Rebuild roc_figs / dist_figs for v2 (re-generate from saved PNGs if closed)
    roc_figs : dict[str, plt.Figure] = {}
    dist_figs: dict[str, plt.Figure] = {}

    for split_tag, ds in [("Val", val_ds), ("Test", test_ds)]:
        sd = split_data[split_tag]
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
            hcc_cos  =sd["hcc_cos_mean"],
            hem_cos  =sd["hem_cos_mean"],
            delta    =sd["delta_mean"],
            labels   =sd["labels"],
            cutoffs  =cutoffs,
            split    =split_tag,
            save_path=out_dir / f"cosine_dist_{split_tag.lower()}.png",
        )

    score_records: list[dict] = []
    for split_tag, sd in split_data.items():
        for i in range(len(sd["labels"])):
            score_records.append({
                "split"             : split_tag,
                "true_label"        : int(sd["labels"][i]),
                "confidence_score"  : float(sd["conf"][i]),
                "hcc_cosine_mean"   : float(sd["hcc_cos_mean"][i]),
                "hem_cosine_mean"   : float(sd["hem_cos_mean"][i]),
                "delta_mean"        : float(sd["delta_mean"][i]),
                "hcc_cosine_kmeans" : float(sd["hcc_cos_kmeans"][i]),
                "hem_cosine_kmeans" : float(sd["hem_cos_kmeans"][i]),
                "delta_kmeans"      : float(sd["delta_kmeans"][i]),
            })
    pd.DataFrame(score_records).to_csv(out_dir / "all_scores.csv", index=False)

    # t-SNE
    tsne_fig: Optional[plt.Figure] = None
    try:
        hcc_train_emb = (
            np.load(out_dir / "hcc_train_emb.npy")
            if (out_dir / "hcc_train_emb.npy").exists()
            else split_data["Val"]["embs"][split_data["Val"]["labels"] == 1]
        )
        hem_train_emb = (
            np.load(out_dir / "hem_train_emb.npy")
            if (out_dir / "hem_train_emb.npy").exists()
            else split_data["Val"]["embs"][split_data["Val"]["labels"] == 0]
        )
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
        print(f"[cosine_probe_v2] WARNING: t-SNE failed: {e}  (skipping)")

    _log_wandb(
        cfg           = cfg,
        val_results   = {"metrics": val_metrics},
        test_results  = {"metrics": test_metrics},
        delong_records= delong_records,
        roc_figs      = roc_figs,
        dist_figs     = dist_figs,
        tsne_fig      = tsne_fig,
        score_records = score_records,
        scatter_figs  = scatter_figs,
        cm_figs       = cm_figs,
        calib_figs    = calib_figs,
        calib_results = calib_results,
        test_delong   = test_delong,
        out_dir       = out_dir,
    )

    for fig in (list(roc_figs.values()) + list(dist_figs.values()) +
                list(scatter_figs.values()) + list(cm_figs.values()) +
                list(calib_figs.values())):
        plt.close(fig)
    if tsne_fig is not None:
        plt.close(tsne_fig)

    print("\n[cosine_probe_v2] Done.")
    print(f"  Outputs → {out_dir}")

    return dict(
        banks          = banks,
        cutoffs        = cutoffs,
        val_metrics    = val_metrics,
        test_metrics   = test_metrics,
        delong_records = delong_records,
        test_delong    = test_delong,
        calib_results  = calib_results,
        split_data     = split_data,
    )
