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

Shape contract
--------------
SA mode  (token_attention_mode='sa'):
    attn per layer: [B, n_heads, seq, seq]   seq = 1 + N_patches  (square)
CA mode  (token_attention_mode='ca'):
    attn per layer: [B, n_heads, 1, N_patches]  (Q dim = 1, non-square)

All public functions auto-detect mode from tensor shape (Q dim == 1 → CA).

Usage
-----
    from visualization.attention_vis import (
        attention_rollout,
        eigen_attention,
        build_attention_wandb_table,
    )

    out = encoder(images, training=False, return_attention=True)
    # SA Rollout  [B, N]
    rollout = attention_rollout(out['attention_weights'])
    # CA Rollout  [B, N]  — same call; shape auto-detected
    rollout = attention_rollout(out['attention_weights'])
    # EigenAttention  [B, N, k_components]
    eigenmaps, eigenvalues = eigen_attention(out['attention_weights'], k_components=3)
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


def _is_ca_shape(a: np.ndarray) -> bool:
    """True if attention tensor is CA shape [B, H, 1, N] (Q dim == 1)."""
    return a.ndim == 4 and a.shape[2] == 1


def _head_fuse(a: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Head-axis fusion.

    SA mode: [B, H, T, T] → [B, T, T]
    CA mode: [B, H, 1, N] → [B, N]   (squeeze Q dim after head fusion)
    """
    if _is_ca_shape(a):
        # CA: fuse heads then squeeze Q dim
        if mode == "mean":
            return a.mean(axis=1)[:, 0, :]   # [B, N]
        else:
            return a.max(axis=1)[:, 0, :]    # [B, N]
    # SA: standard
    if mode == "mean":
        return a.mean(axis=1)
    elif mode == "max":
        return a.max(axis=1)
    else:
        raise ValueError(f"Unknown head_fusion: {mode}")


def _infer_patch_grid_from_n(n_patches: int) -> tuple[int, int]:
    side = int(np.sqrt(n_patches))
    if side * side == n_patches:
        return side, side
    for h in range(side, 0, -1):
        if n_patches % h == 0:
            return h, n_patches // h
    return 1, n_patches


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention Rollout  (SA + CA unified)
# ─────────────────────────────────────────────────────────────────────────────
def attention_rollout(
    attn_weights_list: list,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Compute attention rollout / layer-accumulation for CLS→patch attention.

    SA mode: standard Abnar & Zuidema rollout over square [B,H,T,T] matrices.
    CA mode: [B,H,1,N] — square-matrix rollout is undefined; instead accumulate
             layer-wise head-mean CLS→patch maps with weighted averaging
             (same fallback as wandb_viz.compute_attention_rollout).

    Parameters
    ----------
    attn_weights_list : list of tensors
        SA: each [B, n_heads, seq, seq].  seq = 1 + N (CLS first).
        CA: each [B, n_heads, 1, N].
    discard_ratio : float  (0.0~1.0)
        Fraction of lowest-attention tokens zeroed before accumulation.
    head_fusion : 'mean' | 'max'

    Returns
    -------
    rollout : np.ndarray  [B, N]
        CLS-to-patch attention.  NOT yet reshaped to spatial grid.
    """
    assert head_fusion in ("mean", "max", "min")
    attn_list = [_to_np(a).astype(np.float32) for a in attn_weights_list]

    first = attn_list[0]

    # ── CA mode fallback ───────────────────────────────────────────────────
    if _is_ca_shape(first):
        per_layer: list[np.ndarray] = []
        for a in attn_list:
            # [B, H, 1, N] → head-fuse → [B, N]
            layer_map = _head_fuse(a, "mean")
            if discard_ratio > 0.0:
                thresh = np.quantile(layer_map, discard_ratio, axis=-1, keepdims=True)
                layer_map = np.where(layer_map >= thresh, layer_map, 0.0)
            row_sum = layer_map.sum(axis=-1, keepdims=True) + 1e-8
            per_layer.append(layer_map / row_sum)
        accumulated = per_layer[0]
        for i, lm in enumerate(per_layer[1:], start=1):
            alpha = i / (i + 1)
            accumulated = alpha * accumulated + (1 - alpha) * lm
        return accumulated  # [B, N]

    # ── SA mode: standard rollout ─────────────────────────────────────────
    B, n_heads, seq, _ = first.shape

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
# 2. EigenAttention  (SA + CA unified)
# ─────────────────────────────────────────────────────────────────────────────
def eigen_attention(
    attn_weights_list: list,
    k_components: int = 3,
    head_fusion: str = "mean",
    layer_fusion: str = "last",
    take_abs: bool = True,
    flip_sign: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """EigenAttention: top-k eigenvectors of the symmetrised attention matrix.

    SA mode: patch-to-patch sub-matrix [B,N,N] is available → full SVD.
    CA mode: [B,H,1,N] — patch-to-patch square matrix is undefined.
             Fallback: cosine-modulated rollout maps are used as synthetic
             eigenmaps (same strategy as wandb_viz.compute_eigen_attention).

    Parameters
    ----------
    attn_weights_list : list of tensors
        SA: each [B, n_heads, seq, seq]
        CA: each [B, n_heads, 1, N]
    k_components  : number of eigenvectors / synthetic components to return
    head_fusion   : 'mean' | 'max'
    layer_fusion  : 'last' | 'mean' | 'rollout'  (SA only; CA uses rollout always)
    take_abs      : return |eigenvectors| for unsigned heatmap
    flip_sign     : flip each vector so dominant value is positive

    Returns
    -------
    eigenmaps   : np.ndarray  [B, N, k_components]  float32
    eigenvalues : np.ndarray  [B, k_components]     float32
        SA: true SVD singular values.  CA: synthetic (normalised rollout norm).
    """
    attn_list = [_to_np(a).astype(np.float32) for a in attn_weights_list]
    first = attn_list[0]

    # ── CA mode: cosine-modulated rollout fallback ─────────────────────────
    if _is_ca_shape(first):
        accumulated = attention_rollout(attn_list, discard_ratio=0.0, head_fusion=head_fusion)
        B, N = accumulated.shape
        eigenmaps   = np.zeros((B, N, k_components), dtype=np.float32)
        eigenvalues = np.zeros((B, k_components), dtype=np.float32)
        freq_x = np.arange(N, dtype=np.float32)
        for ki in range(k_components):
            if ki == 0:
                ev = accumulated.copy()
            else:
                modulator = np.cos(ki * np.pi * freq_x / max(N - 1, 1))
                ev = accumulated * modulator[None, :]
            if flip_sign:
                for b in range(B):
                    if ev[b, np.argmax(np.abs(ev[b]))] < 0:
                        ev[b] = -ev[b]
            if take_abs:
                ev = np.abs(ev)
            eigenmaps[:, :, ki]   = ev.astype(np.float32)
            eigenvalues[:, ki]    = np.linalg.norm(ev, axis=-1).astype(np.float32)
        return eigenmaps, eigenvalues

    # ── SA mode: standard SVD EigenAttention ──────────────────────────────
    B, n_heads, seq, _ = first.shape

    fused = [_head_fuse(a, head_fusion) for a in attn_list]   # list of [B,seq,seq]

    if layer_fusion == "last":
        A = fused[-1]
    elif layer_fusion == "mean":
        A = np.mean(np.stack(fused, axis=0), axis=0)
    elif layer_fusion == "rollout":
        A = fused[0]
        for f in fused[1:]:
            A = np.matmul(f, A)
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)
    else:
        raise ValueError(f"Unknown layer_fusion: {layer_fusion}")

    A_sym   = 0.5 * (A + A.transpose(0, 2, 1))
    A_patch = A_sym[:, 1:, 1:]                    # [B, N, N]

    U, s, Vt = np.linalg.svd(A_patch, full_matrices=False)
    eigenmaps   = U[:, :, :k_components]           # [B, N, k]
    eigenvalues = s[:, :k_components]              # [B, k]

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
# 3. Per-head CLS attention  (SA + CA unified)
# ─────────────────────────────────────────────────────────────────────────────
def per_head_cls_attention(
    attn_weights: np.ndarray,
) -> np.ndarray:
    """CLS→patch attention per head.

    SA mode: [B, n_heads, seq, seq] → [B, n_heads, N]  (skip CLS self-attn col)
    CA mode: [B, n_heads, 1, N]     → [B, n_heads, N]  (squeeze Q dim)
    """
    if hasattr(attn_weights, "numpy"):
        attn_weights = attn_weights.numpy()
    attn_weights = np.array(attn_weights, dtype=np.float32)

    if _is_ca_shape(attn_weights):
        # CA: Q dim is already 1; squeeze it
        return attn_weights[:, :, 0, :]   # [B, n_heads, N]
    else:
        # SA: CLS row, skip CLS self-attn column
        return attn_weights[:, :, 0, 1:]  # [B, n_heads, N]


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
# 5. WandB Table builder  (SA + CA unified, mode_suffix column labels)
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
    token_attention_mode: str = "sa",
) -> wandb.Table:
    """Build WandB Table with Rollout + EigenAttention + per-head overlays.

    Supports both SA mode (square attn [B,H,T,T]) and CA mode ([B,H,1,N]).
    Auto-detects mode from tensor shape; column names suffixed with _SA or _CA.

    Columns
    -------
    path | raw_image
    | overlay_rollout_{SA|CA}
    | eigen_map_1_{SA|CA} .. eigen_map_k_{SA|CA}
    | overlay_eigen_1_{SA|CA} .. overlay_eigen_k_{SA|CA}
    | overlay_head_1_{SA|CA} .. overlay_head_N_{SA|CA}

    Parameters
    ----------
    paths               : list[str]  image file paths
    images_np           : [B, H, W, 3]  float32  0-1
    attn_weights_list   : list of tensors (all layers)
                          SA: each [B, n_heads, seq, seq]
                          CA: each [B, n_heads, 1, N]
    patch_grid          : (gh, gw)  e.g. (16, 16)
    input_hw            : (H, W)    e.g. (512, 512)
    discard_ratio       : rollout discard fraction
    alpha               : overlay blend weight
    k_eigen             : number of eigenvectors to visualise
    eigen_layer_fusion  : 'last' | 'mean' | 'rollout'  (SA only)
    last_layer_only_heads : use last layer for per-head maps
    token_attention_mode  : 'sa' | 'ca'  — used for column suffix only;
                            actual shape is auto-detected from tensor.
    """
    B       = images_np.shape[0]
    n_heads = int(_to_np(attn_weights_list[0]).shape[1])
    sfx     = f"_{token_attention_mode.upper()}"   # '_SA' or '_CA'

    # ── Rollout [B, N] ────────────────────────────────────────────────────
    rollout_all = attention_rollout(
        attn_weights_list, discard_ratio=discard_ratio, head_fusion="mean"
    )

    # ── EigenAttention [B, N, k_eigen] ───────────────────────────────────
    eigenmaps, eigenvalues = eigen_attention(
        attn_weights_list,
        k_components   = k_eigen,
        head_fusion    = "mean",
        layer_fusion   = eigen_layer_fusion,
        take_abs       = True,
        flip_sign      = True,
    )

    # ── Per-head from last layer [B, n_heads, N] ──────────────────────────
    last_attn    = attn_weights_list[-1]
    per_head_all = per_head_cls_attention(last_attn)   # [B, n_heads, N]

    # ── Column names ──────────────────────────────────────────────────────
    eigen_raw_cols     = [f"eigen_map_{k+1}{sfx}"     for k in range(k_eigen)]
    eigen_overlay_cols = [f"overlay_eigen_{k+1}{sfx}" for k in range(k_eigen)]
    head_cols          = [f"overlay_head_{h+1}{sfx}"  for h in range(n_heads)]
    columns = (
        ["path", "raw_image", f"overlay_rollout{sfx}"]
        + eigen_raw_cols
        + eigen_overlay_cols
        + head_cols
    )
    table = wandb.Table(columns=columns)

    for i in range(B):
        img = images_np[i]   # [H, W, 3] float32 0-1

        rollout_heat    = _attn_to_heatmap(rollout_all[i], patch_grid, input_hw)
        rollout_overlay = overlay_heatmap(img, rollout_heat, alpha=alpha)

        eigen_raw_imgs     = []
        eigen_overlay_imgs = []
        for k in range(k_eigen):
            e_heat = _attn_to_heatmap(eigenmaps[i, :, k], patch_grid, input_hw)
            e_raw  = cv2.cvtColor(e_heat, cv2.COLOR_BGR2RGB)
            e_ov   = overlay_heatmap(img, e_heat, alpha=alpha)
            eigen_raw_imgs.append(wandb.Image(e_raw))
            eigen_overlay_imgs.append(wandb.Image(e_ov))

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
    token_attention_mode: str = "sa",
) -> tuple[wandb.Table, np.ndarray]:
    """Lightweight table: raw eigenmap + overlay for top-k eigenvectors.

    SA + CA mode unified.  Column names suffixed with _SA or _CA.

    Returns
    -------
    table       : wandb.Table
    eigenvalues : np.ndarray [B, k_components]
    """
    B   = images_np.shape[0]
    sfx = f"_{token_attention_mode.upper()}"

    eigenmaps, eigenvalues = eigen_attention(
        attn_weights_list,
        k_components = k_components,
        head_fusion  = "mean",
        layer_fusion = layer_fusion,
        take_abs     = True,
        flip_sign    = True,
    )

    raw_cols     = [f"eigen_raw_{k+1}{sfx}"     for k in range(k_components)]
    overlay_cols = [f"eigen_overlay_{k+1}{sfx}" for k in range(k_components)]
    ev_cols      = [f"sigma_{k+1}{sfx}"         for k in range(k_components)]
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
