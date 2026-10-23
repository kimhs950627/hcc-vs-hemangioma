"""scratch/vis_mimic_cam_attn.py
================================================================
Publication-quality Grad-CAM + attention-rollout for hyperechoic
(hemangioma-like) HCC cases.

Why this script
---------------
The previous explainability overlay used a naive global min-max
normalisation followed by a flat alpha blend over the WHOLE image::

    overlay = 0.55*img + 0.45*jet(cam)

Because ``cam/max`` always stretches the map to span the full colormap
range, even flat / uniform "no attention here" regions end up colored
(the background turns blue and the lesion turns red regardless of the
actual saliency). This is the classic artefact described in DINO-style
visualisations.

Fixes implemented here
----------------------
1. DINO-style mass threshold (Caron et al., ICCV 2021,
   "Emerging Properties in Self-Supervised Vision Transformers"):
   keep only patches whose cumulativeattention mass exceeds
   (1 - keep_mass); everything else is zeroed. A flat map therefore
   renders dark instead of spanning the full colormap.
2. Per-pixel alpha blending: ``out = img*(1-a) + cmap*a`` with
   ``a = heat * max_alpha``. Low-saliency background stays grayscale.
3. Gaussian smoothing after upsampling so the 12x12 patch grid does not
   show hard blocks, plus a gamma knob to sharpen the hot spot.

Maps produced
-------------
* Grad-CAM  : gradient of P(HCC) wrt the projected feature map
              (EfficientNetV2B0 + CE + SupCon backbone).
* Attention : CA cross-attention rollout (CLS -> patch), layer-weighted
              accumulation, reshaped to the 12x12 patch grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.explainable_predictor import ExplainableHCCPredictor  # noqa: E402


# --------------------------------------------------------------------------
# Core map computation
# --------------------------------------------------------------------------
def gradcam_raw(predictor: ExplainableHCCPredictor, img_tensor_np: np.ndarray,
                target_class: int = 1) -> tuple[np.ndarray, tuple[int, int]]:
    """Raw (un-normalised, ReLU'd) Grad-CAM at patch-grid resolution."""
    x_var = tf.Variable(img_tensor_np[None], dtype=tf.float32, trainable=True)
    patch_embed = predictor.encoder.patch_embed
    norm_layer = predictor.encoder.norm
    pool_mode = getattr(predictor.encoder, "pool_mode", "cls")
    head = predictor.model.classifier_head

    with tf.GradientTape() as tape:
        tokens, fmap, gh, gw = patch_embed(x_var, training=False)
        B = tf.shape(x_var)[0]
        d = patch_embed.d_model
        N = gh * gw
        patches = tf.reshape(fmap, [B, N, d])

        attn_mode = getattr(patch_embed, "token_attention_mode", "sa")
        if attn_mode == "ca":
            cls_tok = tf.reduce_mean(patches, axis=1, keepdims=True)
        else:
            cls_tok = tf.cast(tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype)

        seq = tf.concat([cls_tok, patches], axis=1)
        if patch_embed.pos_embed is not None:
            seq = seq + tf.cast(patch_embed.pos_embed, seq.dtype)

        for block in predictor.encoder.blocks:
            res = block(seq, training=False, return_attention=False)
            seq = res if not isinstance(res, tuple) else res[0]

        seq = norm_layer(seq, training=False)
        emb = seq[:, 0, :] if pool_mode == "cls" else tf.reduce_mean(seq[:, 1:], axis=1)
        logits = head(emb, training=False)
        score = logits[:, target_class]

    grads = tape.gradient(score, fmap)
    fh, fw = int(fmap.shape[1]), int(fmap.shape[2])
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * fmap, axis=-1)
    cam = tf.nn.relu(cam)
    cam2d = tf.reshape(cam, [fh, fw]).numpy().astype(np.float32)
    print(f"    [gradcam] fmap={tuple(fmap.shape)} gh,gw={int(gh)},{int(gw)} cam2d={cam2d.shape}")
    return cam2d, (fh, fw)


def attention_rollout_patches(predictor: ExplainableHCCPredictor,
                              img_tensor_np: np.ndarray,
                              discard_ratio: float = 0.0) -> np.ndarray:
    """CA cross-attention rollout: [B,H,1,N] list -> [N] patch saliency."""
    x = tf.constant(img_tensor_np[None], dtype=tf.float32)
    out = predictor.encoder(x, training=False, return_attention=True)
    attn_list = out["attention_weights"]
    per_layer = []
    for a in attn_list:
        an = a.numpy() if hasattr(a, "numpy") else np.asarray(a, np.float32)
        layer_map = an.mean(axis=1)[:, 0, :]  # head mean -> [B, N]
        if discard_ratio > 0.0:
            thr = np.quantile(layer_map, discard_ratio, axis=-1, keepdims=True)
            layer_map = np.where(layer_map >= thr, layer_map, 0.0)
        row_sum = layer_map.sum(axis=-1, keepdims=True) + 1e-8
        per_layer.append(layer_map / row_sum)
    acc = per_layer[0]
    for i, lm in enumerate(per_layer[1:], start=1):
        alpha = i / (i + 1)
        acc = alpha * acc + (1 - alpha) * lm
    return acc[0].astype(np.float32)


# --------------------------------------------------------------------------
# DINO-style normalisation + publication rendering
# --------------------------------------------------------------------------
def dino_normalize(f: np.ndarray, keep_mass: float = 0.55,
                   gamma: float = 1.0) -> np.ndarray:
    """Mass-threshold then renormalise. Flat maps -> dark, not full-range."""
    orig_shape = np.asarray(f).shape
    flat = np.maximum(np.asarray(f, np.float32).ravel(), 0.0)
    if flat.sum() < 1e-8:
        return np.zeros(orig_shape, dtype=np.float32)
    prob = flat / flat.sum()
    order = np.argsort(prob)                       # ascending index order
    cum = np.cumsum(prob[order])
    keep = np.zeros_like(flat)
    keep[order[cum > (1.0 - keep_mass)]] = 1.0
    masked = flat * keep
    mx = masked.max()
    if mx < 1e-8:
        return np.zeros(orig_shape, dtype=np.float32)
    return (np.clip(masked / mx, 0.0, 1.0) ** gamma).reshape(orig_shape)


def upsample_heat(heat: np.ndarray, out_hw: tuple[int, int],
                  sigma_frac: float = 0.012) -> np.ndarray:
    h, w = out_hw
    up = cv2.resize(heat.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=max(h, w) * sigma_frac)
    mx = up.max()
    return np.clip(up / mx, 0.0, 1.0) if mx > 1e-8 else up


def colorize(heat: np.ndarray, cmap_name: str) -> np.ndarray:
    return plt.get_cmap(cmap_name)(heat)[..., :3].astype(np.float32)


def overlay_per_pixel_alpha(img_display: np.ndarray, heat_full: np.ndarray,
                            cmap_name: str = "jet", max_alpha: float = 0.62) -> np.ndarray:
    """Blend with per-pixel alpha so flat background stays grayscale."""
    rgb = colorize(heat_full, cmap_name)
    a = (heat_full ** 0.9 * max_alpha)[..., None]
    return np.clip(img_display * (1.0 - a) + rgb * a, 0.0, 1.0)


# --------------------------------------------------------------------------
# Per-image pipeline
# --------------------------------------------------------------------------
def analyze(predictor: ExplainableHCCPredictor, image_path: Path,
            keep_mass: float = 0.55):
    img_tensor_np, img_display = predictor.preprocess_image(image_path)
    H, W = img_display.shape[:2]

    cam, (gh, gw) = gradcam_raw(predictor, img_tensor_np, target_class=1)
    cam_norm = dino_normalize(cam.reshape(gh, gw), keep_mass=keep_mass)
    cam_full = upsample_heat(cam_norm, (H, W))

    attn = attention_rollout_patches(predictor, img_tensor_np)
    side = int(round(np.sqrt(attn.shape[0])))
    attn_norm = dino_normalize(attn.reshape(side, side), keep_mass=keep_mass)
    attn_full = upsample_heat(attn_norm, (H, W))

    # scores
    x = tf.constant(img_tensor_np[None], dtype=tf.float32)
    out = predictor.model(x, training=False)
    probs = out["probabilities"].numpy()[0]
    emb = out["embedding"].numpy()[0]
    emb_n = emb / (np.linalg.norm(emb) + 1e-8)
    hcc_cos = float(np.dot(emb_n, predictor.hcc_proto_norm))
    hem_cos = float(np.dot(emb_n, predictor.hem_proto_norm))

    return {
        "path": image_path,
        "img_display": img_display,
        "cam_full": cam_full,
        "attn_full": attn_full,
        "cam_overlay": overlay_per_pixel_alpha(img_display, cam_full, "jet", 0.62),
        "attn_overlay": overlay_per_pixel_alpha(img_display, attn_full, "magma", 0.62),
        "p_hcc": float(probs[1]),
        "p_hem": float(probs[0]),
        "hcc_cos": hcc_cos,
        "hem_cos": hem_cos,
        "delta": hcc_cos - hem_cos,
    }


def render_row(axs, res, show_titles: bool = True):
    axs[0].imshow(res["img_display"])
    axs[1].imshow(res["cam_full"], cmap="jet", vmin=0, vmax=1)
    axs[2].imshow(res["cam_overlay"])
    axs[3].imshow(res["attn_full"], cmap="magma", vmin=0, vmax=1)
    axs[4].imshow(res["attn_overlay"])
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#bbbbbb"); s.set_linewidth(0.6)
    if show_titles:
        axs[0].set_title("Input (B-mode US)", fontsize=11, fontweight="bold")
        axs[1].set_title("Grad-CAM heat", fontsize=11, fontweight="bold")
        axs[2].set_title(f"Grad-CAM overlay\nP(HCC)={res['p_hcc']:.3f}",
                         fontsize=11, fontweight="bold")
        axs[3].set_title("Attention rollout", fontsize=11, fontweight="bold")
        axs[4].set_title(f"Attention overlay\nHCC cos={res['hcc_cos']:.3f}",
                         fontsize=11, fontweight="bold")


def main():
    out_dir = PROJECT_ROOT / "scratch" / "mimic_cam_attn"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = ExplainableHCCPredictor()

    mimic_dir = PROJECT_ROOT / "example_images_hcc_case" / "hcc_mimicking_hemangioma_web" / "images"
    images = sorted(mimic_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"No images found in {mimic_dir}")

    results = []
    for p in images:
        print(f"[analyze] {p.name}")
        res = analyze(predictor, p)
        results.append(res)
        # individual figure
        fig, axs = plt.subplots(1, 5, figsize=(22, 4.8), dpi=300)
        render_row(axs, res, show_titles=True)
        fig.suptitle(p.name, fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"{p.stem}_cam_attn.png", bbox_inches="tight")
        plt.close(fig)

    # combined contact sheet
    n = len(results)
    fig, axs = plt.subplots(n, 5, figsize=(22, 4.6 * n), dpi=200)
    if n == 1:
        axs = axs[None, :]
    for i, res in enumerate(results):
        render_row(axs[i], res, show_titles=(i == 0))
        axs[i, 0].set_ylabel(res["path"].stem[:26], fontsize=8, rotation=90, va="center")
    fig.tight_layout()
    combined = out_dir / "mimic_cam_attn_combined.png"
    fig.savefig(combined, bbox_inches="tight")
    plt.close(fig)

    print("\n=== SUMMARY ===")
    for res in results:
        print(f"{res['path'].name:55s} P(HCC)={res['p_hcc']:.4f} "
              f"HCCcos={res['hcc_cos']:.3f} Hemcos={res['hem_cos']:.3f} "
              f"dScore={res['delta']:.3f}")
    print(f"\ncombined figure: {combined}")


if __name__ == "__main__":
    main()
