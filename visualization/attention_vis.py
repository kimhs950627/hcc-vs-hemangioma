"""attention_vis.py

Attention Rollout + EigenAttention + per-head overlay for ConvHybridViTBackbone.

References
----------
* Attention Rollout : Abnar & Zuidema, 2020
    "Quantifying Attention Flow in Transformers"
    https://arxiv.org/abs/2005.00928

* EigenAttention : inspired by DINO (Caron et al., 2021) self-attention
    visualisation and Ghiasi et al. (2022).
    Key idea: extract top-k eigenvectors of the symmetrised, head-averaged
    attention matrix via SVD, then map patch-token components back to a
    spatial grid.  Unlike CLS-rollout, this is NOT anchored to a single
    token and therefore captures scene-level structure.

Usage
-----
    from visualization.attention_vis import (
        attention_rollout,
        eigen_attention,
        build_attention_wandb_table,
    )

    out = encoder(images, training=False, return_attention=True)
    # Rollout  [B, N]
    rollout = attention_rollout(out['attention_weights'])
    # EigenAttention  [B, N, k_components]
    eigenmaps = eigen_attention(out['attention_weights'], k_components=3)
"""
from __future__ import annotations

import numpy as np
import cv2
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _to_np(t) -> np.ndarray:
    if hasattr(t, "numpy"):
        return t.numpy()
    return np.asarray(t, dtype=np.float32)


def _head_fuse(a: np.ndarray, mode: str = "mean") -> np.ndarray:
    """[B, n_heads, seq, seq] → [B, seq, seq] via head fusion."""
    if mode == "mean":
        return a.mean(axis=1)
    elif mode == "max":
        return a.max(axis=1)
    else:
        raise ValueError(f"Unknown head_fusion: {mode}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention Rollout  (unchanged — kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────
def attention_rollout(
    attn_weights_list: list,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Compute Attention Rollout for CLS→patch attention.

    Parameters
    ----------
    attn_weights_list : list of tensors
        Each element [B, n_heads, seq, seq].  seq = 1 + N (CLS first).
    discard_ratio : float
        Fraction of lowest-attention tokens zeroed before rollout.
    head_fusion : 'mean' | 'max'

    Returns
    -------
    rollout : np.ndarray  [B, N]
        CLS-to-patch attention after rollout.  NOT yet reshaped to spatial.
    """
    assert head_fusion in ("mean", "max", "min")

    attn_list = [_to_np(a).astype(np.float32) for a in attn_weights_list]
    B, n_heads, seq, _ = attn_list[0].shape

    fused = []
    for a in attn_list:
        if head_fusion == "mean":
            f = a.mean(axis=1)
        elif head_fusion == "max":
            f = a.max(axis=1)
        else:
            f = a.min(axis=1)

        if discard_ratio > 0.0:
            flat = f.reshape(B, seq * seq)
            threshold = np.quantile(flat, discard_ratio, axis=1, keepdims=True)
            flat = np.where(flat >= threshold, flat, 0.0)
            f = flat.reshape(B, seq, seq)

        fused.append(f)

    result = np.eye(seq)[None].repeat(B, axis=0)   # [B, seq, seq]
    for f in fused:
        a_hat = 0.5 * f + 0.5 * np.eye(seq)[None]
        a_hat = a_hat / (a_hat.sum(axis=-1, keepdims=True) + 1e-8)
        result = np.matmul(a_hat, result)

    cls_attn = result[:, 0, 1:]   # [B, N]
    return cls_attn


# ─────────────────────────────────────────────────────────────────────────────
# 2. EigenAttention  (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def eigen_attention(
    attn_weights_list: list,
    k_components: int = 3,
    head_fusion: str = "mean",
    layer_fusion: str = "last",
    take_abs: bool = True,
    flip_sign: bool = True,
) -> np.ndarray:
    """EigenAttention: top-k eigenvectors of the symmetrised attention matrix.

    Algorithm
    ---------
    1. Head-fuse per layer: A_l = head_fuse(attn_l)       [B, seq, seq]
    2. Layer fusion (choose one of: last / mean-all / rollout-style):
       - 'last'     : use the final layer only
       - 'mean'     : average all layers
       - 'rollout'  : chain-multiply (same as rollout but without I residual)
    3. Symmetrise: A_sym = 0.5 * (A + A^T)
    4. Extract patch rows/cols (rows 1.., cols 1..):
       A_patch = A_sym[:, 1:, 1:]                          [B, N, N]
    5. Eigendecomposition via SVD on A_patch:
       U, s, Vt = np.linalg.svd(A_patch)
       top-k left singular vectors U[:, :, :k]            [B, N, k]
    6. Optional: flip sign so max-abs component is positive.

    Parameters
    ----------
    attn_weights_list : list of tensors, each [B, n_heads, seq, seq]
    k_components      : how many eigenvectors to return
    head_fusion       : 'mean' | 'max'
    layer_fusion      : 'last' | 'mean' | 'rollout'
    take_abs          : if True, return |eigenvectors| (for unsigned heatmap)
    flip_sign         : flip each eigenvector so dominant value is positive

    Returns
    -------
    eigenmaps : np.ndarray  [B, N, k_components]  float32
        Each column is a spatial eigenvector over N patch tokens.
        Visualise by reshaping column [:, c] to (gh, gw).
    eigenvalues : np.ndarray  [B, k_components]  float32
        Corresponding singular values (descending order).
    """
    attn_list = [_to_np(a).astype(np.float32) for a in attn_weights_list]
    B, n_heads, seq, _ = attn_list[0].shape

    # ── Step 1: head fusion per layer ──────────────────────────────────────
    fused = [_head_fuse(a, head_fusion) for a in attn_list]   # list of [B,seq,seq]

    # ── Step 2: layer fusion ───────────────────────────────────────────────
    if layer_fusion == "last":
        A = fused[-1]                                          # [B, seq, seq]
    elif layer_fusion == "mean":
        A = np.mean(np.stack(fused, axis=0), axis=0)          # [B, seq, seq]
    elif layer_fusion == "rollout":
        A = fused[0]
        for f in fused[1:]:
            A = np.matmul(f, A)
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)
    else:
        raise ValueError(f"Unknown layer_fusion: {layer_fusion}")

    # ── Step 3: symmetrise ─────────────────────────────────────────────────
    A_sym = 0.5 * (A + A.transpose(0, 2, 1))                  # [B, seq, seq]

    # ── Step 4: extract patch sub-matrix ───────────────────────────────────
    # Skip CLS (index 0); keep patch tokens 1..
    A_patch = A_sym[:, 1:, 1:]                                 # [B, N, N]

    # ── Step 5: SVD → top-k left singular vectors ──────────────────────────
    # np.linalg.svd returns U [B,N,N], s [B,N], Vt [B,N,N]
    # singular values are in *descending* order already.
    U, s, Vt = np.linalg.svd(A_patch, full_matrices=False)    # [B,N,N], [B,N]
    eigenmaps   = U[:, :, :k_components]                       # [B, N, k]
    eigenvalues = s[:, :k_components]                          # [B, k]

    # ── Step 6: sign convention ────────────────────────────────────────────
    if flip_sign:
        for b in range(B):
            for c in range(k_components):
                vec = eigenmaps[b, :, c]
                if vec[np.abs(vec).argmax()] < 0:
                    eigenmaps[b, :, c] = -vec

    if take_abs:
        eigenmaps = np.abs(eigenmaps)

    return eigenmaps.astype(np.float32), eigenvalues.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-head CLS attention  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def per_head_cls_attention(
    attn_weights: np.ndarray,
) -> np.ndarray:
    """[B, n_heads, seq, seq] → [B, n_heads, N] CLS-to-patch per head."""
    if hasattr(attn_weights, "numpy"):
        attn_weights = attn_weights.numpy()
    attn_weights = np.array(attn_weights, dtype=np.float32)
    return attn_weights[:, :, 0, 1:]   # [B, n_heads, N]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _attn_to_heatmap(
    attn_1d: np.ndarray,
    patch_grid: tuple[int, int],
    input_hw: tuple[int, int],
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """[N] float → [H, W, 3] uint8 heatmap, upsampled to input_hw."""
    gh, gw = patch_grid
    H, W   = input_hw
    amap = attn_1d.reshape(gh, gw).astype(np.float32)
    mn, mx = amap.min(), amap.max()
    amap = (amap - mn) / (mx - mn + 1e-8)
    amap_u8    = (amap * 255).astype(np.uint8)
    heat_small = cv2.applyColorMap(amap_u8, colormap)
    heat_large = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return heat_large   # BGR uint8


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend heatmap on image.  Returns [H, W, 3] uint8 RGB."""
    if image_rgb.dtype != np.uint8:
        img_u8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image_rgb.copy()
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 5. WandB Table builder  (extended with EigenAttention columns)
# ─────────────────────────────────────────────────────────────────────────────
def build_attention_wandb_table(
    paths: list[str],
    images_np: np.ndarray,
    attn_weights_list: list,
    patch_grid: tuple[int, int],
    input_hw: tuple[int, int] = (384, 384),
    discard_ratio: float = 0.0,
    alpha: float = 0.5,
    k_eigen: int = 3,
    eigen_layer_fusion: str = "last",
    last_layer_only_heads: bool = True,
) -> wandb.Table:
    """Build WandB Table with Rollout + EigenAttention + per-head overlays.

    Columns
    -------
    path | raw_image
    | overlay_rollout
    | eigen_map_1 | eigen_map_2 | eigen_map_3          (k_eigen columns)
    | overlay_eigen_1 | overlay_eigen_2 | overlay_eigen_3
    | overlay_head_1 .. overlay_head_N

    Parameters
    ----------
    paths              : list[str]  image file paths
    images_np          : [B, H, W, 3]  float32  0-1
    attn_weights_list  : list of tensors (all layers), each [B, n_heads, seq, seq]
    patch_grid         : (gh, gw)  e.g. (12, 12)
    input_hw           : (H, W)    e.g. (384, 384)
    discard_ratio      : rollout discard fraction
    alpha              : overlay blend weight
    k_eigen            : number of eigenvectors to visualise
    eigen_layer_fusion : 'last' | 'mean' | 'rollout'
    last_layer_only_heads : use last layer for per-head maps
    """
    B       = images_np.shape[0]
    n_heads = int(_to_np(attn_weights_list[0]).shape[1])

    # ── Rollout [B, N] ───────────────────────────────────────────────────
    rollout_all = attention_rollout(
        attn_weights_list, discard_ratio=discard_ratio, head_fusion="mean"
    )

    # ── EigenAttention [B, N, k_eigen] ──────────────────────────────────
    eigenmaps, eigenvalues = eigen_attention(
        attn_weights_list,
        k_components    = k_eigen,
        head_fusion     = "mean",
        layer_fusion    = eigen_layer_fusion,
        take_abs        = True,
        flip_sign       = True,
    )
    # eigenvalues logged as metadata (not per-image column)
    # eigenmaps: [B, N, k_eigen]

    # ── Per-head from last layer [B, n_heads, N] ─────────────────────────
    last_attn    = attn_weights_list[-1]
    per_head_all = per_head_cls_attention(last_attn)   # [B, n_heads, N]

    # ── Column names ─────────────────────────────────────────────────────
    eigen_raw_cols     = [f"eigen_map_{k+1}"     for k in range(k_eigen)]
    eigen_overlay_cols = [f"overlay_eigen_{k+1}" for k in range(k_eigen)]
    head_cols          = [f"overlay_head_{h+1}"  for h in range(n_heads)]
    columns = (
        ["path", "raw_image", "overlay_rollout"]
        + eigen_raw_cols
        + eigen_overlay_cols
        + head_cols
    )
    table = wandb.Table(columns=columns)

    for i in range(B):
        img = images_np[i]   # [H, W, 3] float32 0-1

        # rollout overlay
        rollout_heat    = _attn_to_heatmap(rollout_all[i], patch_grid, input_hw)
        rollout_overlay = overlay_heatmap(img, rollout_heat, alpha=alpha)

        # eigen raw heatmaps + overlays
        eigen_raw_imgs     = []
        eigen_overlay_imgs = []
        for k in range(k_eigen):
            e_heat = _attn_to_heatmap(eigenmaps[i, :, k], patch_grid, input_hw)
            e_raw  = cv2.cvtColor(e_heat, cv2.COLOR_BGR2RGB)
            e_ov   = overlay_heatmap(img, e_heat, alpha=alpha)
            eigen_raw_imgs.append(wandb.Image(e_raw))
            eigen_overlay_imgs.append(wandb.Image(e_ov))

        # per-head overlays
        head_overlays = []
        for h in range(n_heads):
            h_heat = _attn_to_heatmap(per_head_all[i, h], patch_grid, input_hw)
            h_ov   = overlay_heatmap(img, h_heat, alpha=alpha)
            head_overlays.append(wandb.Image(h_ov))

        raw_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        row = (
            [paths[i], wandb.Image(raw_u8), wandb.Image(rollout_overlay)]
            + eigen_raw_imgs
            + eigen_overlay_imgs
            + head_overlays
        )
        table.add_data(*row)

    return table


# ─────────────────────────────────────────────────────────────────────────────
# 6. Standalone EigenAttention WandB Table  (lightweight — no head maps)
# ─────────────────────────────────────────────────────────────────────────────
def build_eigen_wandb_table(
    paths: list[str],
    images_np: np.ndarray,
    attn_weights_list: list,
    patch_grid: tuple[int, int],
    input_hw: tuple[int, int] = (384, 384),
    k_components: int = 5,
    layer_fusion: str = "last",
    alpha: float = 0.5,
) -> tuple[wandb.Table, np.ndarray]:
    """Lightweight table: raw eigenmap + overlay for top-k eigenvectors.

    Useful for quick inspection without per-head columns.

    Returns
    -------
    table       : wandb.Table
    eigenvalues : np.ndarray [B, k_components]  (for logging as bar chart)
    """
    B = images_np.shape[0]

    eigenmaps, eigenvalues = eigen_attention(
        attn_weights_list,
        k_components = k_components,
        head_fusion  = "mean",
        layer_fusion = layer_fusion,
        take_abs     = True,
        flip_sign    = True,
    )

    raw_cols     = [f"eigen_raw_{k+1}"     for k in range(k_components)]
    overlay_cols = [f"eigen_overlay_{k+1}" for k in range(k_components)]
    ev_cols      = [f"sigma_{k+1}"         for k in range(k_components)]
    columns      = ["path", "raw_image"] + raw_cols + overlay_cols + ev_cols
    table        = wandb.Table(columns=columns)

    for i in range(B):
        img = images_np[i]
        raw_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        raw_imgs, ov_imgs = [], []
        for k in range(k_components):
            heat = _attn_to_heatmap(eigenmaps[i, :, k], patch_grid, input_hw)
            raw_imgs.append(wandb.Image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))
            ov_imgs.append(wandb.Image(overlay_heatmap(img, heat, alpha=alpha)))

        ev_vals = [float(eigenvalues[i, k]) for k in range(k_components)]
        row = [paths[i], wandb.Image(raw_u8)] + raw_imgs + ov_imgs + ev_vals
        table.add_data(*row)

    return table, eigenvalues
