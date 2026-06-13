"""attention_vis.py

Attention Rollout + per-head overlay for ConvHybridViTBackbone.

Reference: Abnar & Zuidema, 2020 ("Quantifying Attention Flow in Transformers")
           https://arxiv.org/abs/2005.00928

Usage
-----
    from visualization.attention_vis import attention_rollout, build_attention_wandb_table

    out = encoder(images, training=False, return_attention=True)
    rollout = attention_rollout(out['attention_weights'])  # [B, gh, gw]
    table   = build_attention_wandb_table(
        paths, images_np, out['attention_weights'],
        patch_grid=(gh, gw), input_hw=(384, 384)
    )
    wandb.log({'Stage1_attention': table})
"""
from __future__ import annotations

import numpy as np
import cv2
import wandb


# ──────────────────────────────────────────────────────────────────────────────
# Core math
# ──────────────────────────────────────────────────────────────────────────────
def attention_rollout(
    attn_weights_list: list,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> np.ndarray:
    """Compute Attention Rollout for CLS→patch attention.

    Parameters
    ----------
    attn_weights_list : list of tensors / np.ndarray
        Each element shape [B, n_heads, seq, seq].  seq = 1 + N (CLS first).
    discard_ratio : float
        Fraction of lowest-attention tokens to set to 0 before rollout.
    head_fusion : str
        How to aggregate heads. 'mean' | 'max' | 'min'

    Returns
    -------
    rollout : np.ndarray  [B, N]
        CLS-to-patch attention after rollout.  NOT yet reshaped to spatial.
    """
    assert head_fusion in ("mean", "max", "min")

    def _to_np(t):
        if hasattr(t, "numpy"):
            return t.numpy()
        return np.array(t)

    attn_list = [_to_np(a).astype(np.float32) for a in attn_weights_list]
    # attn_list[l]: [B, n_heads, seq, seq]

    B, n_heads, seq, _ = attn_list[0].shape

    # Head fusion per layer
    fused = []
    for a in attn_list:
        if head_fusion == "mean":
            f = a.mean(axis=1)           # [B, seq, seq]
        elif head_fusion == "max":
            f = a.max(axis=1)
        else:
            f = a.min(axis=1)

        # Optional: discard bottom-k tokens
        if discard_ratio > 0.0:
            flat = f.reshape(B, seq * seq)
            threshold = np.quantile(flat, discard_ratio, axis=1, keepdims=True)
            flat = np.where(flat >= threshold, flat, 0.0)
            f = flat.reshape(B, seq, seq)

        fused.append(f)

    # Rollout: A_hat = 0.5 * A + 0.5 * I  (residual connection)
    result = np.eye(seq)[None].repeat(B, axis=0)  # [B, seq, seq]
    for f in fused:
        a_hat = 0.5 * f + 0.5 * np.eye(seq)[None]
        # Row-normalize
        a_hat = a_hat / (a_hat.sum(axis=-1, keepdims=True) + 1e-8)
        result = np.matmul(a_hat, result)

    # Extract CLS row → patch attentions: [B, N]
    cls_attn = result[:, 0, 1:]   # skip CLS-to-CLS
    return cls_attn


def per_head_cls_attention(
    attn_weights: np.ndarray,
) -> np.ndarray:
    """Extract CLS-to-patch attention per head for a single layer.

    Parameters
    ----------
    attn_weights : [B, n_heads, seq, seq]

    Returns
    -------
    [B, n_heads, N]  (N = seq - 1, patch tokens)
    """
    if hasattr(attn_weights, "numpy"):
        attn_weights = attn_weights.numpy()
    attn_weights = np.array(attn_weights, dtype=np.float32)
    return attn_weights[:, :, 0, 1:]   # [B, n_heads, N]


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap helpers
# ──────────────────────────────────────────────────────────────────────────────
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
    # Min-max normalise
    mn, mx = amap.min(), amap.max()
    if mx - mn > 1e-8:
        amap = (amap - mn) / (mx - mn)
    else:
        amap = np.zeros_like(amap)
    amap_u8  = (amap * 255).astype(np.uint8)
    heat_small = cv2.applyColorMap(amap_u8, colormap)                # [gh, gw, 3]
    heat_large = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_LINEAR)
    return heat_large   # BGR uint8


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap_bgr: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend heatmap on image.

    Parameters
    ----------
    image_rgb  : [H, W, 3] float32 0-1  or  uint8 0-255
    heatmap_bgr: [H, W, 3] uint8 (BGR from cv2)
    alpha      : blending weight for heatmap

    Returns
    -------
    [H, W, 3] uint8 RGB
    """
    if image_rgb.dtype != np.uint8:
        img_u8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    else:
        img_u8 = image_rgb.copy()

    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# WandB Table builder
# ──────────────────────────────────────────────────────────────────────────────
def build_attention_wandb_table(
    paths: list[str],
    images_np: np.ndarray,
    attn_weights_list: list,
    patch_grid: tuple[int, int],
    input_hw: tuple[int, int] = (384, 384),
    discard_ratio: float = 0.0,
    alpha: float = 0.5,
    last_layer_only_heads: bool = True,
) -> wandb.Table:
    """Build WandB Table with rollout + per-head overlays.

    Columns
    -------
    path | raw_image | attention_map_rollout | overlay_rollout
    | overlay_head_1 | ... | overlay_head_N

    Parameters
    ----------
    paths            : list of image file paths (strings)
    images_np        : [B, H, W, 3] float32 0-1
    attn_weights_list: list of tensors (all layers), each [B, n_heads, seq, seq]
    patch_grid       : (gh, gw)  e.g. (12, 12)
    input_hw         : (H, W)  e.g. (384, 384)
    last_layer_only_heads : if True, extract per-head maps from last layer only
    """
    B = images_np.shape[0]
    n_heads = int(attn_weights_list[0].shape[1] if hasattr(attn_weights_list[0], 'shape')
                  else np.array(attn_weights_list[0]).shape[1])

    # Rollout [B, N]
    rollout_all = attention_rollout(
        attn_weights_list, discard_ratio=discard_ratio, head_fusion="mean"
    )

    # Per-head from last layer: [B, n_heads, N]
    last_attn = attn_weights_list[-1]
    per_head_all = per_head_cls_attention(last_attn)  # [B, n_heads, N]

    # Build columns
    head_cols = [f"overlay_head_{h+1}" for h in range(n_heads)]
    columns   = ["path", "raw_image", "attention_map_rollout", "overlay_rollout"] + head_cols
    table     = wandb.Table(columns=columns)

    for i in range(B):
        img = images_np[i]  # [H, W, 3] float32 0-1

        # --- rollout heatmap ---
        rollout_heat = _attn_to_heatmap(rollout_all[i], patch_grid, input_hw)
        attn_map_img = cv2.cvtColor(rollout_heat, cv2.COLOR_BGR2RGB)  # pure heatmap
        rollout_overlay = overlay_heatmap(img, rollout_heat, alpha=alpha)

        # --- per-head heatmaps ---
        head_overlays = []
        for h in range(n_heads):
            h_heat = _attn_to_heatmap(per_head_all[i, h], patch_grid, input_hw)
            h_ov   = overlay_heatmap(img, h_heat, alpha=alpha)
            head_overlays.append(wandb.Image(h_ov))

        # --- raw image (uint8) ---
        raw_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        row = [
            paths[i],
            wandb.Image(raw_u8),
            wandb.Image(attn_map_img),
            wandb.Image(rollout_overlay),
        ] + head_overlays

        table.add_data(*row)

    return table
