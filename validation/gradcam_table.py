# validation/gradcam_table.py
"""GradCAM WandB table for test_clean directory images.

Pipeline
--------
1. test_clean/HCC/ 와 test_clean/Hemangioma/ 에서 파일명 순서대로 n_per_class 개씩 선택
2. 각 이미지에 대해 **1회 forward pass** (persistent GradientTape) 로:
   - pred_label  : argmax(logits)
   - GradCAM overlay for class=0 (Hemangioma)  ← tape.gradient(logits[:,0], fmap)
   - GradCAM overlay for class=1 (HCC)          ← tape.gradient(logits[:,1], fmap)
3. wandb.Table 5-column 생성:
   Col 0: Original Image
   Col 1: True Label  (str)
   Col 2: Pred Label  (str, ✓/✗ mark)
   Col 3: GradCAM overlay for class=0 (Hemangioma)
   Col 4: GradCAM overlay for class=1 (HCC)
4. WandB에 logging

Colormap
--------
matplotlib 'jet' colormap 사용:
  - cam=0.0 (low activation)  → 파랑(blue)
  - cam=0.5 (mid activation)  → 초록(green)
  - cam=1.0 (high activation) → 빨강(red)
  - 두 클래스 모두 동일한 jet colormap 적용 (class별 구분은 caption으로 표시)
  - RGBA → RGB slicing으로 alpha 채널 제거

이미지당 Forward Pass 최적화
-----------------------------
이전 버전:
  _predict_label()          → 1회 (tape 없음)
  _gradcam_overlay_np(cls=0) → 1회 (tape)
  _gradcam_overlay_np(cls=1) → 1회 (tape)
  합계: **3회/image**

현재 버전:
  _gradcam_both_overlays_and_pred() → persistent tape **1회**
    ├─ logits 전체 계산 → pred_label = argmax
    ├─ tape.gradient(logits[:,0], fmap) → cam0
    └─ tape.gradient(logits[:,1], fmap) → cam1
  합계: **1회/image**  (3× 절감)

persistent tape 주의사항
------------------------
  - tf.GradientTape(persistent=True) 사용 시 tape는 backward pass 후에도
    그래프를 메모리에 유지한다.
  - .gradient() 호출 후 반드시 `del tape` 로 해제해야 GPU 메모리 누수 방지.

GradCAM target tensor
---------------------
encoder.patch_embed(x) returns:
    tokens : [B, 1+N, d_model]
    fmap   : [B, gh, gw, d_model]  ← Conv1x1 proj + LN 후 spatial feature map
    gh, gw : scalar tensors

`fmap`을 GradCAM target으로 사용하는 이유:
  - Spatial 구조 유지 [B, gh, gw, C]
  - CNN projection 후, Transformer 입력 직전 → 최종 분류에 직접 기여
  - encoded_patches [B, seq, d]는 flatten되어 spatial 정보 소실

지원 wrapper 구조
-----------------
ClassifierTrainer.model  → SupervisedClassifier (또는 SupConClassifier)
  .encoder               → ConvHybridViTBackbone
    .patch_embed         → ConvPatchEmbedding   (CNN + proj + CLS + PE)
    .blocks              → list of TransformerEncoderBlock / CrossAttentionEncoderBlock
    .norm                → LayerNormalization
  .classifier_head       → Dense head

score_probe.py에서의 import 방법
---------------------------------
from validation.gradcam_table import GradCAMTableConfig, run_gradcam_table

gradcam_cfg = GradCAMTableConfig(
    test_clean_dir = cfg.work_dir + "/data/test_clean",
    n_per_class    = 8,
    image_size     = (384, 384),
    wandb_prefix   = f"Benchmark/{model_id}",
)
run_gradcam_table(stage2_model=wrapper, cfg=gradcam_cfg)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image as _PIL_Image

try:
    import matplotlib.cm as _mpl_cm
    _JET = _mpl_cm.get_cmap("jet")
except Exception:
    _JET = None

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
    n_per_class     : Number of images per class (sorted by filename). Default 8.
    image_size      : Resize target (H, W) for model input AND overlay. Default (384, 384).
    overlay_alpha   : Heatmap blend weight (0=original only, 1=heatmap only). Default 0.45.
    wandb_key       : WandB log key for the table. Default "gradcam_table".
    class_names     : (negative_name, positive_name). Default ("Hemangioma", "HCC").
    wandb_prefix    : Optional prefix prepended to wandb_key. Default "".
    input_scale     : "0_1"   -> model receives [0,1] float32.
                      "0_255" -> model receives [0,255] float32 (default).
    colormap        : matplotlib colormap name for GradCAM heatmap. Default "jet".
                      Examples: "jet", "inferno", "hot", "RdYlGn"
    """
    test_clean_dir : str
    n_per_class    : int   = 8
    image_size     : tuple = (384, 384)
    overlay_alpha  : float = 0.45
    wandb_key      : str   = "gradcam_table"
    class_names    : tuple = ("Hemangioma", "HCC")
    wandb_prefix   : str   = ""
    input_scale    : str   = "0_255"   # "0_1" or "0_255"
    colormap       : str   = "jet"     # matplotlib colormap name

    @property
    def table_key(self) -> str:
        if self.wandb_prefix:
            return f"{self.wandb_prefix}/{self.wandb_key}"
        return self.wandb_key


# ─────────────────────────────────────────────────────────────
# Image I/O helpers
# ─────────────────────────────────────────────────────────────

def _load_image_np(path: Path, size: tuple, input_scale: str = "0_255") -> np.ndarray:
    """Load image as float32 (H, W, 3), resized to `size`.

    Returns
    -------
    img : (H, W, 3) float32
      If input_scale == "0_255"  ->  range [0, 255]
      If input_scale == "0_1"    ->  range [0, 1]
    """
    img = _PIL_Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), _PIL_Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if input_scale == "0_1":
        arr = arr / 255.0
    return arr


def _to_display_np(img_model: np.ndarray, input_scale: str) -> np.ndarray:
    """Normalize model-input array to [0,1] for display / overlay."""
    if input_scale == "0_255":
        return np.clip(img_model / 255.0, 0.0, 1.0)
    return np.clip(img_model, 0.0, 1.0)


def _np_to_wandb_image(img_np: np.ndarray, caption: str = "") -> "_wandb.Image":
    """(H, W, 3) float32 [0,1] -> wandb.Image via PIL."""
    arr_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    pil = _PIL_Image.fromarray(arr_uint8)
    return _wandb.Image(pil, caption=caption)


# ─────────────────────────────────────────────────────────────
# Colormap helper
# ─────────────────────────────────────────────────────────────

def _cam_to_rgb(cam_up: np.ndarray, colormap_name: str = "jet") -> np.ndarray:
    """Apply matplotlib colormap to a [0,1] CAM array.

    Parameters
    ----------
    cam_up       : (H, W) float32, range [0, 1]
    colormap_name: matplotlib colormap name. Default "jet".
                   jet:     blue(low) → green(mid) → red(high)
                   inferno: black → purple → orange → yellow(high)
                   hot:     black → red → yellow → white(high)

    Returns
    -------
    heat : (H, W, 3) float32, range [0, 1]  — RGBA alpha channel stripped
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap_name)
    except Exception:
        # fallback: grayscale if matplotlib unavailable
        return np.stack([cam_up, cam_up, cam_up], axis=-1).astype(np.float32)

    rgba = cmap(cam_up)                          # (H, W, 4) float64, range [0, 1]
    return rgba[..., :3].astype(np.float32)      # drop alpha → (H, W, 3)


# ─────────────────────────────────────────────────────────────
# File collection
# ─────────────────────────────────────────────────────────────

def _collect_images(
    test_clean_dir: Path,
    class_names: tuple,
    n_per_class: int,
) -> list:
    """Return sorted list of (image_path, class_index, class_name) tuples."""
    records = []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = test_clean_dir / cls_name
        if not cls_dir.exists():
            for d in test_clean_dir.iterdir():
                if d.is_dir() and d.name.lower() == cls_name.lower():
                    cls_dir = d
                    break
        if not cls_dir.exists():
            warnings.warn(
                f"[gradcam_table] directory not found: {cls_dir}", stacklevel=2
            )
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
# Model unwrapping
# ─────────────────────────────────────────────────────────────

def _unwrap_encoder_and_head(stage2_model):
    """Unwrap stage2_model to get (encoder, classifier_head, pool_mode).

    Supported wrapper hierarchies
    ------------------------------
    ClassifierTrainer (or PureClassifier)
      └─ .model  →  SupervisedClassifier / SupConClassifier
            ├─ .encoder         → ConvHybridViTBackbone
            │     ├─ .patch_embed  → ConvPatchEmbedding
            │     ├─ .blocks       → list of Transformer blocks
            │     └─ .norm         → LayerNormalization
            └─ .classifier_head → Dense head

    Returns
    -------
    encoder         : ConvHybridViTBackbone
    classifier_head : Dense Keras layer / Sequential
    pool_mode       : "cls" or "gap"
    """
    model = getattr(stage2_model, "model", stage2_model)

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise AttributeError(
            "[gradcam_table] Cannot find .encoder on stage2_model. "
            "Expected stage2_model.model.encoder → ConvHybridViTBackbone."
        )

    classifier_head = getattr(model, "classifier_head", None)
    if classifier_head is None:
        raise AttributeError(
            "[gradcam_table] Cannot find .classifier_head on model. "
            "Expected stage2_model.model.classifier_head → Dense head."
        )

    pool_mode = getattr(encoder, "pool_mode", "cls")
    return encoder, classifier_head, pool_mode


# ─────────────────────────────────────────────────────────────
# Core: 1 forward pass → pred_label + GradCAM overlay ×2
# ─────────────────────────────────────────────────────────────

def _gradcam_both_overlays_and_pred(
    encoder,
    classifier_head,
    pool_mode: str,
    img_np: np.ndarray,
    class_names: tuple,
    alpha: float = 0.45,
    input_scale: str = "0_255",
    colormap: str = "jet",
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Single persistent-tape forward pass → pred_label + two GradCAM overlays.

    Colormap
    --------
    matplotlib 'jet' (default) 사용:
      cam=0.0 → blue  (low activation)
      cam=0.5 → green (mid activation)
      cam=1.0 → red   (high activation)

    두 클래스(HCC, Hemangioma) 모두 동일한 jet colormap 적용.
    class별 구분은 WandB table의 column caption으로 표시.

    Strategy
    --------
    persistent=True tape를 사용해 forward pass를 1회만 수행하고,
    두 클래스(0, 1)에 대한 gradient를 순차적으로 추출한다.

    Tape lifecycle
    --------------
    1. GradientTape(persistent=True) context 진입
    2. patch_embed(x) → fmap 생성 (INSIDE tape)
    3. tape.watch(fmap)
    4. fmap → Transformer → norm → pool → classifier_head → logits  (1회)
    5. pred_label = argmax(logits)
    6. grads_0 = tape.gradient(logits[:,0], fmap)   ← class=0 gradient
    7. grads_1 = tape.gradient(logits[:,1], fmap)   ← class=1 gradient
    8. del tape  ← 반드시 명시적 해제 (persistent tape 메모리 누수 방지)

    Parameters
    ----------
    encoder         : ConvHybridViTBackbone
    classifier_head : Dense head layer
    pool_mode       : "cls" or "gap"
    img_np          : (H, W, 3) float32 in model input range
    class_names     : (negative_name, positive_name)
    alpha           : Heatmap blend weight
    input_scale     : "0_255" or "0_1"
    colormap        : matplotlib colormap name (default "jet")

    Returns
    -------
    pred_name   : str           — class_names[argmax(logits)]
    overlay_neg : (H,W,3) f32  — GradCAM for class=0  (jet colormap)
    overlay_pos : (H,W,3) f32  — GradCAM for class=1  (jet colormap)
    """
    patch_embed = encoder.patch_embed
    blocks      = encoder.blocks
    norm_layer  = encoder.norm

    x = tf.constant(img_np[None], dtype=tf.float32)  # [1, H, W, 3]

    # ── persistent tape: 1 forward pass ──────────────────────────────────────
    with tf.GradientTape(persistent=True) as tape:
        # Step 1: CNN + Conv1x1 proj → fmap (INSIDE tape)
        tokens, fmap, gh, gw = patch_embed(x, training=False)

        # Step 2: watch fmap
        tape.watch(fmap)

        # Step 3: rebuild sequence from watched fmap
        B = tf.shape(x)[0]
        d = patch_embed.d_model
        N = gh * gw
        patches = tf.reshape(fmap, [B, N, d])  # [B, N, d]

        attn_mode = getattr(patch_embed, "token_attention_mode", "sa")
        if attn_mode == "ca":
            cls = tf.reduce_mean(patches, axis=1, keepdims=True)   # [B,1,d]
        else:
            cls = tf.cast(
                tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype
            )  # [B,1,d]

        seq = tf.concat([cls, patches], axis=1)  # [B, 1+N, d]

        if patch_embed.pos_embed is not None:
            seq = seq + tf.cast(patch_embed.pos_embed, seq.dtype)

        # Step 4: Transformer blocks
        for block in blocks:
            result = block(seq, training=False, return_attention=False)
            seq = result if not isinstance(result, tuple) else result[0]

        # Step 5: norm → pool → head → logits  (1회, persistent tape 내부)
        seq        = norm_layer(seq, training=False)
        embedding  = seq[:, 0, :] if pool_mode == "cls" \
                     else tf.reduce_mean(seq[:, 1:, :], axis=1)
        logits     = classifier_head(embedding, training=False)  # [B, num_cls]

    # ── pred label (argmax, outside tape) ────────────────────────────────────
    pred_idx  = int(tf.argmax(logits, axis=-1).numpy()[0])
    pred_name = class_names[pred_idx]

    # ── gradients for both classes (persistent tape 재사용) ──────────────────
    grads_0 = tape.gradient(logits[:, 0], fmap)  # [1, gh, gw, d]
    grads_1 = tape.gradient(logits[:, 1], fmap)  # [1, gh, gw, d]
    del tape  # 반드시 해제

    # ── CAM builder helper ────────────────────────────────────────────────────
    H, W        = img_np.shape[:2]
    img_display = _to_display_np(img_np, input_scale)

    def _build_overlay(grads, class_index: int) -> np.ndarray:
        if grads is None:
            warnings.warn(
                f"[gradcam_table] GradCAM gradients are None for "
                f"class_index={class_index}. Falling back to original image.",
                RuntimeWarning, stacklevel=3,
            )
            return img_display.copy()
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # [1,1,1,d]
        cam     = tf.reduce_sum(weights * fmap, axis=-1)               # [1,gh,gw]
        cam     = tf.nn.relu(cam)
        cam     = cam / (tf.reduce_max(cam) + 1e-8)
        cam_up  = tf.image.resize(cam[..., None], [H, W]).numpy()[0, :, :, 0]

        # ── matplotlib colormap (default: jet) ───────────────────────────────
        # jet:  blue(0.0) → green(0.5) → red(1.0)
        # 두 클래스 모두 동일한 colormap 적용; class 구분은 caption으로 표시
        heat = _cam_to_rgb(cam_up, colormap_name=colormap)   # (H, W, 3) [0,1]

        return np.clip(
            (1.0 - alpha) * img_display + alpha * heat, 0.0, 1.0
        ).astype(np.float32)

    overlay_neg = _build_overlay(grads_0, class_index=0)
    overlay_pos = _build_overlay(grads_1, class_index=1)

    return pred_name, overlay_neg, overlay_pos


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_gradcam_table(
    stage2_model,
    cfg: GradCAMTableConfig,
) -> "_wandb.Table | None":
    """Build and log a GradCAM WandB table to the active WandB run.

    Table columns:
        [0] "Original"                       -- wandb.Image of original US frame
        [1] "True Label"                     -- string label ("HCC" or "Hemangioma")
        [2] "Pred Label"                     -- argmax prediction + ✓/✗ mark
        [3] f"GradCAM ({class_names[0]})"    -- jet heatmap targeting negative class
        [4] f"GradCAM ({class_names[1]})"    -- jet heatmap targeting positive class

    Colormap: matplotlib 'jet' (configurable via GradCAMTableConfig.colormap)
        blue (low activation) → green (mid) → red (high activation)

    Forward pass cost: **1회/image** (persistent GradientTape)

    Parameters
    ----------
    stage2_model : ClassifierTrainer or PureClassifier wrapper.
    cfg          : GradCAMTableConfig

    Returns
    -------
    wandb.Table or None if WandB run is not active.
    """
    if _wandb is None or _wandb.run is None:
        print("[gradcam_table] WandB run not active -- skipping GradCAM table.")
        return None

    if tf is None:
        print("[gradcam_table] TensorFlow not available -- skipping GradCAM table.")
        return None

    try:
        encoder, classifier_head, pool_mode = _unwrap_encoder_and_head(stage2_model)
    except AttributeError as e:
        print(f"[gradcam_table] Model unwrap failed: {e}")
        return None

    test_clean_dir = Path(cfg.test_clean_dir)
    if not test_clean_dir.exists():
        print(f"[gradcam_table] ERROR: test_clean_dir not found: {test_clean_dir}")
        return None

    records = _collect_images(test_clean_dir, cfg.class_names, cfg.n_per_class)
    if not records:
        print("[gradcam_table] No images found. Check test_clean_dir and class_names.")
        return None

    attn_mode = getattr(encoder.patch_embed, "token_attention_mode", "sa")
    print(
        f"[gradcam_table] {len(records)} images | "
        f"pool={pool_mode!r}  attn={attn_mode!r}  scale={cfg.input_scale!r}  "
        f"colormap={cfg.colormap!r}  forward_pass=1/image (persistent tape)"
    )
    print("[gradcam_table] Generating GradCAM overlays ...")

    columns = [
        "Original",
        "True Label",
        "Pred Label",
        f"GradCAM ({cfg.class_names[0]})",
        f"GradCAM ({cfg.class_names[1]})",
    ]
    table = _wandb.Table(columns=columns)

    for i, (img_path, cls_idx, cls_name) in enumerate(records):
        print(f"  [{i+1:>2}/{len(records)}] {cls_name:<12} | {img_path.name}", end=" ... ")
        try:
            img_np = _load_image_np(img_path, cfg.image_size, cfg.input_scale)

            # ── 1 forward pass → pred + overlay×2 ────────────────────────────
            pred_name, overlay_neg, overlay_pos = _gradcam_both_overlays_and_pred(
                encoder         = encoder,
                classifier_head = classifier_head,
                pool_mode       = pool_mode,
                img_np          = img_np,
                class_names     = cfg.class_names,
                alpha           = cfg.overlay_alpha,
                input_scale     = cfg.input_scale,
                colormap        = cfg.colormap,
            )

            correct_mark  = "✓" if pred_name == cls_name else "✗"
            img_display   = _to_display_np(img_np, cfg.input_scale)

            w_orig    = _np_to_wandb_image(img_display,  caption=f"{cls_name} | {img_path.name}")
            w_neg_cam = _np_to_wandb_image(overlay_neg,  caption=f"GradCAM→{cfg.class_names[0]} ({cfg.colormap})")
            w_pos_cam = _np_to_wandb_image(overlay_pos,  caption=f"GradCAM→{cfg.class_names[1]} ({cfg.colormap})")

            table.add_data(w_orig, cls_name, f"{pred_name} {correct_mark}", w_neg_cam, w_pos_cam)
            print(f"OK  (pred={pred_name} {correct_mark})")

        except Exception as e:
            print(f"FAILED: {e}")
            table.add_data(None, cls_name, "ERROR", None, None)

    _wandb.log({cfg.table_key: table})
    print(
        f"[gradcam_table] WandB table logged -> "
        f"key: '{cfg.table_key}'  rows={len(records)}"
    )
    return table
