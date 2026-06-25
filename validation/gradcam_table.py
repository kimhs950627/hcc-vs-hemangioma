# validation/gradcam_table.py
"""GradCAM WandB table for test_clean directory images.

Pipeline
--------
1. test_clean/HCC/ 와 test_clean/Hemangioma/ 에서 파일명 순서대로 각 8개씩 선택
2. 각 이미지에 대해 class_index=0 (Hemangioma) 과 class_index=1 (HCC) 두 가지 GradCAM overlay 생성
3. wandb.Table 4-column 생성:
   Col 0: Original Image
   Col 1: True Label (str)
   Col 2: GradCAM overlay for class=0 (Hemangioma)
   Col 3: GradCAM overlay for class=1 (HCC)
4. WandB에 logging

Dependency
----------
visualization.interpret.gradcam_overlay  (기존 구현 재사용)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as _PIL_Image

try:
    import wandb as _wandb
except Exception:
    _wandb = None

try:
    import tensorflow as tf
except Exception:
    tf = None


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class GradCAMTableConfig:
    """Configuration for GradCAM WandB table logging.

    Parameters
    ----------
    test_clean_dir  : Path to test_clean/ root containing HCC/ and Hemangioma/ subdirs.
    n_per_class     : Number of images to sample per class (sorted by filename). Default 8.
    image_size      : Resize target (H, W) for model input AND overlay output. Default (224, 224).
    overlay_alpha   : Weight of heatmap in overlay blend. Default 0.45.
    wandb_key       : WandB log key for the table. Default "gradcam_table".
    class_names     : (negative_name, positive_name). Default ("Hemangioma", "HCC").
    wandb_prefix    : Optional prefix prepended to wandb_key. Default "".
    """
    test_clean_dir : str
    n_per_class    : int             = 8
    image_size     : tuple           = (224, 224)
    overlay_alpha  : float           = 0.45
    wandb_key      : str             = "gradcam_table"
    class_names    : tuple           = ("Hemangioma", "HCC")
    wandb_prefix   : str             = ""

    @property
    def table_key(self) -> str:
        if self.wandb_prefix:
            return f"{self.wandb_prefix}/{self.wandb_key}"
        return self.wandb_key


# ─────────────────────────────────────────────────────────────
# Image I/O helpers
# ─────────────────────────────────────────────────────────────

def _load_image_np(path: Path, size: tuple) -> np.ndarray:
    """Load image as float32 (H, W, 3) in [0, 1] range, resized to `size`."""
    img = _PIL_Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), _PIL_Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _np_to_wandb_image(img_np: np.ndarray, caption: str = "") -> "_wandb.Image":
    """Convert (H, W, 3) float32 [0,1] ndarray -> wandb.Image."""
    arr_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    pil = _PIL_Image.fromarray(arr_uint8)
    return _wandb.Image(pil, caption=caption)


# ─────────────────────────────────────────────────────────────
# File collection
# ─────────────────────────────────────────────────────────────

def _collect_images(
    test_clean_dir: Path,
    class_names: tuple,
    n_per_class: int,
) -> list:
    """Return sorted list of (image_path, class_index, class_name) tuples.

    class_index: 0 = negative (Hemangioma), 1 = positive (HCC)
    Images are sorted by filename (ascending) and the first n_per_class are taken.
    """
    records = []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = test_clean_dir / cls_name
        if not cls_dir.exists():
            # Case-insensitive fallback
            for d in test_clean_dir.iterdir():
                if d.is_dir() and d.name.lower() == cls_name.lower():
                    cls_dir = d
                    break
        if not cls_dir.exists():
            print(f"[gradcam_table] WARNING: directory not found: {cls_dir}")
            continue
        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        files = sorted(
            [f for f in cls_dir.iterdir() if f.suffix.lower() in img_exts],
            key=lambda p: p.name,
        )[:n_per_class]
        for f in files:
            records.append((f, cls_idx, cls_name))
    return records


# ─────────────────────────────────────────────────────────────
# GradCAM core
# ─────────────────────────────────────────────────────────────

def _gradcam_overlay_np(
    model,
    img_np: np.ndarray,
    class_index: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Compute GradCAM overlay for a single image.

    Parameters
    ----------
    model       : Keras model returning dict with keys 'logits' and 'feature_map'.
                  feature_map: last CNN feature map before global pooling (1, h, w, C).
    img_np      : (H, W, 3) float32 in [0, 1].
    class_index : Target class for gradient computation. 0=Hemangioma, 1=HCC.
    alpha       : Heatmap blend weight (overlay_alpha in config).

    Returns
    -------
    overlay : (H, W, 3) float32 in [0, 1]

    Notes
    -----
    Heatmap colormap:
      class_index=1 (HCC)        -> warm  (Red high, Blue low)
      class_index=0 (Hemangioma) -> cool  (Blue high, Red low)
    This helps visually distinguish which class the activation targets.
    """
    x = tf.constant(img_np[None], dtype=tf.float32)   # (1, H, W, 3)

    with tf.GradientTape() as tape:
        out   = model(x, training=False)
        fmap  = out.get("feature_map", None)
        if fmap is None:
            raise ValueError(
                "[gradcam_table] GradCAM requires 'feature_map' key in model output. "
                "Verify the model call() returns feature_map (last CNN feature map)."
            )
        tape.watch(fmap)
        score = out["logits"][:, class_index]           # scalar per sample

    grads   = tape.gradient(score, fmap)                # (1, h, w, C)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)   # (1, 1, 1, C)
    cam     = tf.reduce_sum(weights * fmap, axis=-1)    # (1, h, w)
    cam     = tf.nn.relu(cam)
    cam     = cam / (tf.reduce_max(cam) + 1e-8)
    # Upsample CAM to input resolution
    cam_up  = tf.image.resize(cam[..., None], x.shape[1:3]).numpy()[0, ..., 0]  # (H, W)

    # Colormap: warm=red for HCC (class=1), cool=blue for Hemangioma (class=0)
    if class_index == 1:
        heat = np.stack([cam_up, np.zeros_like(cam_up), 1.0 - cam_up], axis=-1)
    else:
        heat = np.stack([1.0 - cam_up, np.zeros_like(cam_up), cam_up], axis=-1)

    overlay = np.clip((1.0 - alpha) * img_np + alpha * heat, 0.0, 1.0)
    return overlay.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_gradcam_table(
    stage2_model,
    cfg: GradCAMTableConfig,
) -> "_wandb.Table | None":
    """Build and log a 16x4 WandB GradCAM table to the active WandB run.

    Table columns:
        [0] "Original"                       -- wandb.Image of original US frame
        [1] "True Label"                     -- string label ("HCC" or "Hemangioma")
        [2] f"GradCAM ({class_names[0]})"    -- overlay targeting negative class (cool colormap)
        [3] f"GradCAM ({class_names[1]})"    -- overlay targeting positive class (warm colormap)

    Parameters
    ----------
    stage2_model : Trained Keras model.  PureClassifier or ClassifierTrainer wrapper.
                   call() must return dict with "logits" and "feature_map" keys.
    cfg          : GradCAMTableConfig

    Returns
    -------
    wandb.Table or None if WandB run is not active.

    Usage (Cell 13)
    ---------------
    from validation.gradcam_table import GradCAMTableConfig, run_gradcam_table

    gradcam_cfg = GradCAMTableConfig(
        test_clean_dir = "/kaggle/input/.../test_clean",
        n_per_class    = 8,
        wandb_prefix   = f"Benchmark/{model_id}",
    )
    run_gradcam_table(stage2_model=wrapper, cfg=gradcam_cfg)
    """
    if _wandb is None or _wandb.run is None:
        print("[gradcam_table] WandB run not active -- skipping GradCAM table.")
        return None

    if tf is None:
        print("[gradcam_table] TensorFlow not available -- skipping GradCAM table.")
        return None

    # Unwrap trainer wrapper (same pattern as score_probe._extract_scores)
    model = getattr(stage2_model, "model", stage2_model)

    test_clean_dir = Path(cfg.test_clean_dir)
    if not test_clean_dir.exists():
        print(f"[gradcam_table] ERROR: test_clean_dir not found: {test_clean_dir}")
        return None

    records = _collect_images(test_clean_dir, cfg.class_names, cfg.n_per_class)
    if not records:
        print("[gradcam_table] No images found. Check test_clean_dir and class_names.")
        return None

    print(f"[gradcam_table] Found {len(records)} images. Generating GradCAM overlays ...")

    columns = [
        "Original",
        "True Label",
        f"GradCAM ({cfg.class_names[0]})",
        f"GradCAM ({cfg.class_names[1]})",
    ]
    table = _wandb.Table(columns=columns)

    for i, (img_path, cls_idx, cls_name) in enumerate(records):
        print(f"  [{i+1:>2}/{len(records)}] {cls_name:<12} | {img_path.name}", end=" ... ")
        try:
            img_np = _load_image_np(img_path, cfg.image_size)

            overlay_neg = _gradcam_overlay_np(
                model, img_np, class_index=0, alpha=cfg.overlay_alpha
            )
            overlay_pos = _gradcam_overlay_np(
                model, img_np, class_index=1, alpha=cfg.overlay_alpha
            )

            w_orig    = _np_to_wandb_image(img_np,      caption=f"{cls_name} | {img_path.name}")
            w_neg_cam = _np_to_wandb_image(overlay_neg, caption=f"GradCAM\u2192{cfg.class_names[0]}")
            w_pos_cam = _np_to_wandb_image(overlay_pos, caption=f"GradCAM\u2192{cfg.class_names[1]}")

            table.add_data(w_orig, cls_name, w_neg_cam, w_pos_cam)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            # Still add a row so the table stays N rows (no silent gap)
            table.add_data(None, cls_name, None, None)

    _wandb.log({cfg.table_key: table})
    print(
        f"[gradcam_table] WandB table logged -> "
        f"key: '{cfg.table_key}'  rows={len(records)}"
    )
    return table
