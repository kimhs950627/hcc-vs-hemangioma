# validation/gradcam_table.py
"""GradCAM WandB table for test_clean directory images.

Pipeline
--------
1. test_clean/HCC/ 와 test_clean/Hemangioma/ 에서 파일명 순서대로 n_per_class 개씩 선택
2. 각 이미지에 대해 **3회 독립 forward pass** 수행:
   Pass 1: tape 없음  → pred_label (argmax logits)
   Pass 2: GradientTape(persistent=False) → GradCAM for class=0 (Hemangioma)
   Pass 3: GradientTape(persistent=False) → GradCAM for class=1 (HCC)
3. wandb.Table 5-column 생성:
   Col 0: Original Image
   Col 1: True Label  (str)
   Col 2: Pred Label  (str, ✓/✗ mark)
   Col 3: GradCAM overlay for class=0 (Hemangioma)
   Col 4: GradCAM overlay for class=1 (HCC)
4. WandB에 logging

Why 3 separate passes (not persistent tape)
--------------------------------------------
persistent=True tape + tape.watch(fmap) 방식은 Keras layer 내부에서
fmap이 eager tensor로 즉시 materialized되어 tape 그래프와 연결이 끊기는
문제(gradient=None)가 발생했다. 이를 근본적으로 방지하기 위해 각 pass마다
독립적인 GradientTape를 사용하고 tape.watch(x_var)로 입력을 watch하는
표준 패턴으로 변경한다.

Colormap
--------
matplotlib 'jet' colormap 사용:
  - cam=0.0 (low activation)  → 파랑(blue)
  - cam=0.5 (mid activation)  → 초록(green)
  - cam=1.0 (high activation) → 빨강(red)
  - 두 클래스 모두 동일한 jet colormap 적용 (class별 구분은 caption으로 표시)
  - RGBA → RGB slicing으로 alpha 채널 제거

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

    Returns
    -------
    heat : (H, W, 3) float32, range [0, 1]  — RGBA alpha channel stripped
    """
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap_name)
    except Exception:
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
    """Unwrap stage2_model to get (encoder, classifier_head, pool_mode)."""
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
# Core: encoder forward pass (공통 헬퍼)
# ─────────────────────────────────────────────────────────────

def _encoder_forward(
    encoder,
    classifier_head,
    pool_mode: str,
    x: "tf.Tensor",
    fmap_override: "tf.Tensor | None" = None,
) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
    """Run encoder+head forward pass. Returns (logits, fmap, seq).

    Parameters
    ----------
    x             : [1, H, W, 3] float32 input tensor
    fmap_override : 외부에서 이미 계산된 fmap을 주입할 경우 사용 (GradientTape pass에서 활용).
                    None이면 patch_embed(x)로 새로 계산.

    Returns
    -------
    logits : [1, num_cls]
    fmap   : [1, gh, gw, d_model]
    seq    : [1, 1+N, d_model]  (Transformer 출력, norm 후)
    """
    patch_embed = encoder.patch_embed
    blocks      = encoder.blocks
    norm_layer  = encoder.norm

    if fmap_override is None:
        tokens, fmap, gh, gw = patch_embed(x, training=False)
    else:
        fmap = fmap_override
        gh   = tf.shape(fmap)[1]
        gw   = tf.shape(fmap)[2]

    B = tf.shape(x)[0]
    d = patch_embed.d_model
    N = gh * gw
    patches = tf.reshape(fmap, [B, N, d])  # [B, N, d]

    attn_mode = getattr(patch_embed, "token_attention_mode", "sa")
    if attn_mode == "ca":
        cls = tf.reduce_mean(patches, axis=1, keepdims=True)  # [B,1,d]
    else:
        cls = tf.cast(
            tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype
        )  # [B,1,d]

    seq = tf.concat([cls, patches], axis=1)  # [B, 1+N, d]

    if patch_embed.pos_embed is not None:
        seq = seq + tf.cast(patch_embed.pos_embed, seq.dtype)

    for block in blocks:
        result = block(seq, training=False, return_attention=False)
        seq = result if not isinstance(result, tuple) else result[0]

    seq       = norm_layer(seq, training=False)
    embedding = (seq[:, 0, :] if pool_mode == "cls"
                 else tf.reduce_mean(seq[:, 1:, :], axis=1))
    logits    = classifier_head(embedding, training=False)  # [1, num_cls]

    return logits, fmap, seq


# ─────────────────────────────────────────────────────────────
# Pass 1: prediction (no tape)
# ─────────────────────────────────────────────────────────────

def _predict_label(
    encoder,
    classifier_head,
    pool_mode: str,
    img_np: np.ndarray,
    class_names: tuple,
) -> str:
    """Forward pass without tape. Returns predicted class name."""
    x = tf.constant(img_np[None], dtype=tf.float32)  # [1, H, W, 3]
    logits, _, _ = _encoder_forward(encoder, classifier_head, pool_mode, x)
    pred_idx = int(tf.argmax(logits, axis=-1).numpy()[0])
    return class_names[pred_idx]


# ─────────────────────────────────────────────────────────────
# Pass 2 / 3: GradCAM overlay (독립 GradientTape, persistent=False)
# ─────────────────────────────────────────────────────────────

def _gradcam_overlay_np(
    encoder,
    classifier_head,
    pool_mode: str,
    img_np: np.ndarray,
    class_index: int,
    alpha: float = 0.45,
    input_scale: str = "0_255",
    colormap: str = "jet",
) -> np.ndarray:
    """Single GradientTape forward pass → GradCAM overlay for one class.

    Strategy
    --------
    - tf.Variable을 입력으로 사용해 tape.watch() 없이도 gradient 추적 보장.
    - persistent=False (단일 .gradient() 호출이므로 persistent 불필요).
    - fmap을 watch 대상으로 삼지 않고, x_var → patch_embed → fmap의 전체
      계산 경로가 tape 내부에서 이루어지도록 구성한다.

    Parameters
    ----------
    class_index : 0 → Hemangioma gradient, 1 → HCC gradient

    Returns
    -------
    overlay : (H, W, 3) float32 [0, 1]
    """
    patch_embed = encoder.patch_embed
    H, W        = img_np.shape[:2]
    img_display = _to_display_np(img_np, input_scale)

    # tf.Variable로 감싸면 자동으로 tape에 등록되어 watch() 불필요
    x_var = tf.Variable(img_np[None], dtype=tf.float32, trainable=True)  # [1,H,W,3]

    with tf.GradientTape() as tape:  # persistent=False (기본값)
        # x_var → patch_embed → fmap → Transformer → logits
        tokens, fmap, gh, gw = patch_embed(x_var, training=False)

        B = tf.shape(x_var)[0]
        d = patch_embed.d_model
        N = gh * gw
        patches = tf.reshape(fmap, [B, N, d])

        attn_mode = getattr(patch_embed, "token_attention_mode", "sa")
        if attn_mode == "ca":
            cls = tf.reduce_mean(patches, axis=1, keepdims=True)
        else:
            cls = tf.cast(
                tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype
            )

        seq = tf.concat([cls, patches], axis=1)
        if patch_embed.pos_embed is not None:
            seq = seq + tf.cast(patch_embed.pos_embed, seq.dtype)

        for block in encoder.blocks:
            result = block(seq, training=False, return_attention=False)
            seq = result if not isinstance(result, tuple) else result[0]

        seq       = encoder.norm(seq, training=False)
        embedding = (seq[:, 0, :] if pool_mode == "cls"
                     else tf.reduce_mean(seq[:, 1:, :], axis=1))
        logits    = classifier_head(embedding, training=False)  # [1, num_cls]
        score     = logits[:, class_index]  # scalar-ish [1]

    # fmap에 대한 gradient (x_var → fmap 경로는 tape가 추적)
    grads = tape.gradient(score, fmap)  # [1, gh, gw, d]

    if grads is None:
        warnings.warn(
            f"[gradcam_table] GradCAM gradients are None for "
            f"class_index={class_index}. Falling back to original image.",
            RuntimeWarning, stacklevel=2,
        )
        return img_display.copy()

    # Global Average Pool over channel dim → CAM
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # [1,1,1,d]
    cam     = tf.reduce_sum(weights * fmap, axis=-1)               # [1,gh,gw]
    cam     = tf.nn.relu(cam)
    cam     = cam / (tf.reduce_max(cam) + 1e-8)
    cam_up  = tf.image.resize(cam[..., None], [H, W]).numpy()[0, :, :, 0]  # (H,W)

    heat    = _cam_to_rgb(cam_up, colormap_name=colormap)  # (H,W,3) [0,1]
    overlay = np.clip(
        (1.0 - alpha) * img_display + alpha * heat, 0.0, 1.0
    ).astype(np.float32)
    return overlay


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_gradcam_table(
    stage2_model,
    cfg: GradCAMTableConfig,
) -> "_wandb.Table | None":
    """Build and log a GradCAM WandB table to the active WandB run.

    Forward pass schedule per image (3 passes, all independent):
        Pass 1: no tape     → pred_label
        Pass 2: GradientTape → GradCAM overlay for class_index=0 (Hemangioma)
        Pass 3: GradientTape → GradCAM overlay for class_index=1 (HCC)

    Table columns:
        [0] "Original"                       -- wandb.Image of original US frame
        [1] "True Label"                     -- string label
        [2] "Pred Label"                     -- argmax prediction + ✓/✗ mark
        [3] f"GradCAM ({class_names[0]})"    -- jet heatmap targeting negative class
        [4] f"GradCAM ({class_names[1]})"    -- jet heatmap targeting positive class

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
        f"colormap={cfg.colormap!r}  forward_pass=3/image (3x independent GradientTape)"
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

            # ── Pass 1: prediction (no tape) ──────────────────────────────
            pred_name = _predict_label(
                encoder, classifier_head, pool_mode, img_np, cfg.class_names
            )

            # ── Pass 2: GradCAM for class=0 (Hemangioma) ─────────────────
            overlay_neg = _gradcam_overlay_np(
                encoder, classifier_head, pool_mode, img_np,
                class_index=0,
                alpha=cfg.overlay_alpha,
                input_scale=cfg.input_scale,
                colormap=cfg.colormap,
            )

            # ── Pass 3: GradCAM for class=1 (HCC) ───────────────────────
            overlay_pos = _gradcam_overlay_np(
                encoder, classifier_head, pool_mode, img_np,
                class_index=1,
                alpha=cfg.overlay_alpha,
                input_scale=cfg.input_scale,
                colormap=cfg.colormap,
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
