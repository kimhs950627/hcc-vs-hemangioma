# validation/gradcam_table.py
"""GradCAM WandB table for test_clean directory images.

Pipeline
--------
1. test_clean/HCC/ 와 test_clean/Hemangioma/ 에서 파일명 순서대로 n_per_class 개씩 선택
2. 각 이미지에 대해 class_index=0 (Hemangioma) 과 class_index=1 (HCC) 두 가지 GradCAM overlay 생성
3. wandb.Table 4-column 생성:
   Col 0: Original Image
   Col 1: True Label (str)
   Col 2: GradCAM overlay for class=0 (Hemangioma)
   Col 3: GradCAM overlay for class=1 (HCC)
4. WandB에 logging

GradCAM 구현 핵심 (FIXED — 기존 버그 수정)
-----------------------------------------
기존 코드의 버그:
  with tape:
      out  = model(x)        # forward pass 완료 후
      fmap = out["feature_map"]
      tape.watch(fmap)       # ← 이미 계산된 tensor → gradient = None

올바른 구현:
  Step 1: encoder.patch_embed 만 실행 → fmap 생성 (tape 내부)
  Step 2: tape.watch(fmap)
  Step 3: fmap → Transformer blocks → norm → CLS → classifier_head → logit
  → gradient graph가 fmap으로 완전히 연결됨

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
    n_per_class     : Number of images per class (sorted by filename). Default 8.
    image_size      : Resize target (H, W) for model input AND overlay. Default (384, 384).
    overlay_alpha   : Heatmap blend weight (0=original only, 1=heatmap only). Default 0.45.
    wandb_key       : WandB log key for the table. Default "gradcam_table".
    class_names     : (negative_name, positive_name). Default ("Hemangioma", "HCC").
    wandb_prefix    : Optional prefix prepended to wandb_key. Default "".
    input_scale     : "0_1"   -> model receives [0,1] float32.
                      "0_255" -> model receives [0,255] float32 (default).
    """
    test_clean_dir : str
    n_per_class    : int   = 8
    image_size     : tuple = (384, 384)
    overlay_alpha  : float = 0.45
    wandb_key      : str   = "gradcam_table"
    class_names    : tuple = ("Hemangioma", "HCC")
    wandb_prefix   : str   = ""
    input_scale    : str   = "0_255"   # "0_1" or "0_255"

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
# GradCAM core — FIXED
# ─────────────────────────────────────────────────────────────

def _gradcam_overlay_np(
    encoder,
    classifier_head,
    pool_mode: str,
    img_np: np.ndarray,
    class_index: int,
    alpha: float = 0.45,
    input_scale: str = "0_255",
) -> np.ndarray:
    """Compute GradCAM overlay for a single image.

    GradCAM target tensor
    ---------------------
    encoder.patch_embed(x) returns:
        tokens : [B, 1+N, d_model]
        fmap   : [B, gh, gw, d_model]   <- Conv1x1 proj + LN 후 spatial feature map
        gh, gw : scalar tensors

    `fmap`을 GradCAM target으로 사용하는 이유:
      - Spatial 구조 유지 [B, gh, gw, C]
      - CNN projection 후, Transformer 입력 직전 → 최종 분류에 직접 기여
      - encoded_patches [B, seq, d]는 flatten되어 spatial 정보 소실

    Gradient tape chain (올바른 순서)
    ---------------------------------
    1. GradientTape open
    2. patch_embed(x) → fmap이 tape INSIDE에서 생성  ← 핵심 수정
    3. tape.watch(fmap)
    4. fmap → reshape → concat CLS → PE → Transformer blocks
       → encoder.norm → pool → classifier_head → logits
    5. tape.gradient(logits[:, class_index], fmap) → 유효한 gradient

    Parameters
    ----------
    encoder         : ConvHybridViTBackbone
    classifier_head : Dense head layer
    pool_mode       : "cls" or "gap"
    img_np          : (H, W, 3) float32 in model input range
    class_index     : 0 = Hemangioma (cool colormap), 1 = HCC (warm colormap)
    alpha           : Heatmap blend weight
    input_scale     : "0_255" or "0_1"

    Returns
    -------
    overlay : (H, W, 3) float32 in [0, 1]
    """
    patch_embed = encoder.patch_embed
    blocks      = encoder.blocks
    norm_layer  = encoder.norm

    x = tf.constant(img_np[None], dtype=tf.float32)  # [1, H, W, 3]

    with tf.GradientTape() as tape:
        # ── Step 1: CNN + Conv1x1 proj → fmap (INSIDE tape) ──────────────────
        # patch_embed.call() internally runs:
        #   CNN backbone → Conv1x1 proj → LN → returns (tokens, fmap, gh, gw)
        # fmap [1, gh, gw, d_model] is created here, INSIDE the tape context.
        tokens, fmap, gh, gw = patch_embed(x, training=False)

        # ── Step 2: watch fmap after creation ────────────────────────────────
        tape.watch(fmap)

        # ── Step 3: rebuild sequence from watched fmap ────────────────────────
        B = tf.shape(x)[0]
        d = patch_embed.d_model
        N = gh * gw

        patches = tf.reshape(fmap, [B, N, d])  # [B, N, d]

        # CLS initialisation (mirrors ConvPatchEmbedding.call)
        attn_mode = getattr(patch_embed, "token_attention_mode", "sa")
        if attn_mode == "ca":
            # CA mode: per-sample GAP initialisation
            cls = tf.reduce_mean(patches, axis=1, keepdims=True)  # [B,1,d]
        else:
            # SA mode: learnable CLS token
            cls = tf.cast(
                tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype
            )  # [B,1,d]

        seq = tf.concat([cls, patches], axis=1)  # [B, 1+N, d]

        # PE (sinusoidal / learnable / none)
        if patch_embed.pos_embed is not None:
            seq = seq + tf.cast(patch_embed.pos_embed, seq.dtype)

        # ── Step 4: Transformer blocks ────────────────────────────────────────
        for block in blocks:
            result = block(seq, training=False, return_attention=False)
            # block may return tensor or (tensor, attn_weights) tuple
            seq = result if not isinstance(result, tuple) else result[0]

        # ── Step 5: norm → pool → classifier head → logits ───────────────────
        seq = norm_layer(seq, training=False)

        if pool_mode == "cls":
            embedding = seq[:, 0, :]                     # CLS token [B, d]
        else:
            embedding = tf.reduce_mean(seq[:, 1:, :], axis=1)  # GAP [B, d]

        logits       = classifier_head(embedding, training=False)  # [B, num_cls]
        target_score = logits[:, class_index]                      # [B]

    # ── Step 6: compute gradient ──────────────────────────────────────────────
    grads = tape.gradient(target_score, fmap)  # [1, gh, gw, d]

    if grads is None:
        warnings.warn(
            f"[gradcam_table] GradCAM gradients are None for class_index={class_index}. "
            "fmap may not be connected to logits in the gradient graph. "
            "Falling back to original image.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _to_display_np(img_np, input_scale)

    # ── Step 7: global average pool over channels → CAM ──────────────────────
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # [1,1,1,d]
    cam     = tf.reduce_sum(weights * fmap, axis=-1)              # [1, gh, gw]
    cam     = tf.nn.relu(cam)
    cam     = cam / (tf.reduce_max(cam) + 1e-8)

    # Upsample CAM to original input spatial resolution
    H, W   = img_np.shape[:2]
    cam_up = tf.image.resize(cam[..., None], [H, W]).numpy()[0, :, :, 0]  # (H, W)

    # ── Step 8: colormap ──────────────────────────────────────────────────────
    # Warm (red channel high) for HCC (class=1)
    # Cool (blue channel high) for Hemangioma (class=0)
    zero = np.zeros_like(cam_up)
    if class_index == 1:
        heat = np.stack([cam_up, zero, 1.0 - cam_up], axis=-1)  # warm (R→B)
    else:
        heat = np.stack([1.0 - cam_up, zero, cam_up], axis=-1)  # cool (B→R)

    # Blend with original image
    img_display = _to_display_np(img_np, input_scale)
    overlay = np.clip((1.0 - alpha) * img_display + alpha * heat, 0.0, 1.0)
    return overlay.astype(np.float32)


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
        [2] f"GradCAM ({class_names[0]})"    -- overlay targeting negative class (cool)
        [3] f"GradCAM ({class_names[1]})"    -- overlay targeting positive class (warm)

    Parameters
    ----------
    stage2_model : ClassifierTrainer or PureClassifier wrapper.
    cfg          : GradCAMTableConfig

    Returns
    -------
    wandb.Table or None if WandB run is not active.

    Usage (Cell 13 in notebook, or score_probe integration)
    --------------------------------------------------------
    from validation.gradcam_table import GradCAMTableConfig, run_gradcam_table

    gradcam_cfg = GradCAMTableConfig(
        test_clean_dir = "/kaggle/input/.../test_clean",
        n_per_class    = 8,
        image_size     = (384, 384),
        input_scale    = "0_255",
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

    # ── Unwrap model components ───────────────────────────────────────────────
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
        f"[gradcam_table] {len(records)} images found. "
        f"pool_mode={pool_mode!r}  attn_mode={attn_mode!r}  "
        f"input_scale={cfg.input_scale!r}"
    )
    print("[gradcam_table] Generating GradCAM overlays ...")

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
            img_np = _load_image_np(img_path, cfg.image_size, cfg.input_scale)

            overlay_neg = _gradcam_overlay_np(
                encoder, classifier_head, pool_mode,
                img_np, class_index=0, alpha=cfg.overlay_alpha,
                input_scale=cfg.input_scale,
            )
            overlay_pos = _gradcam_overlay_np(
                encoder, classifier_head, pool_mode,
                img_np, class_index=1, alpha=cfg.overlay_alpha,
                input_scale=cfg.input_scale,
            )

            img_display = _to_display_np(img_np, cfg.input_scale)
            w_orig    = _np_to_wandb_image(
                img_display, caption=f"{cls_name} | {img_path.name}"
            )
            w_neg_cam = _np_to_wandb_image(
                overlay_neg, caption=f"GradCAM\u2192{cfg.class_names[0]}"
            )
            w_pos_cam = _np_to_wandb_image(
                overlay_pos, caption=f"GradCAM\u2192{cfg.class_names[1]}"
            )

            table.add_data(w_orig, cls_name, w_neg_cam, w_pos_cam)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            table.add_data(None, cls_name, None, None)

    _wandb.log({cfg.table_key: table})
    print(
        f"[gradcam_table] WandB table logged -> "
        f"key: '{cfg.table_key}'  rows={len(records)}"
    )
    return table
