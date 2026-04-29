"""
SMC-LUD Dataloader for HCC vs. Hemangioma Classification
=========================================================
Dataset: Samsung Medical Center - Liver Ultrasound Dataset (SMC-LUD)
  Reference: Nature Scientific Data, 2026-03-10
  DOI: https://doi.org/10.6084/m9.figshare.31112716

Expected directory structure (pre-split, flat layout)
------------------------------------------------------
data_root/                          ← e.g. "clean_ver_for_train/"
    train_clean/
        HCC/
            img001.png
            img002.png
            ...
        Hemangioma/
            img001.png
            ...
    val_clean/
        HCC/
        Hemangioma/
    test_clean/
        HCC/
        Hemangioma/

Label convention
----------------
    Hemangioma → 0
    HCC        → 1

Usage
-----
    from dataloader import build_dataset, MaskedMultiViewDataset

    # 1. Vanilla supervised loader
    ds_train, ds_val, ds_test = build_dataset(
        data_root="./clean_ver_for_train",
        img_size=(224, 224),
        batch_size=32,
    )

    # 2. 권장 B+ SSL loader (DINO + SimMIM hybrid)
    ds = MaskedMultiViewDataset(
        data_root="./clean_ver_for_train",
        split="train",
        img_size=(224, 224),
        batch_size=16,
        local_views=4,
        mask_ratio=0.75,
    )
    for batch in ds:
        original_clean, masked_clean, patch_mask, aug_global, *locals_ = batch
        # original_clean : [B, 224, 224, 3]  teacher input + SimMIM target
        # masked_clean   : [B, 224, 224, 3]  SimMIM student input
        # patch_mask     : [B, 196]           1=masked, 0=visible
        # aug_global     : [B, 224, 224, 3]  student global (solarization p=0.2)
        # locals_        : list of [B, 112, 112, 3]

Design rationale (권장 B+ 구조)
--------------------------------
  Teacher : original_clean만 사용 (solarization 없음 → stable target)
            검증: D_KL(teacher_raw‖teacher_solar)=12.06 vs noise=0.06
            → solarized teacher는 target distribution을 193× 불안정하게 만듦

  Student : augmented_global (solarization p=0.2) + locals (solarization p=0.2)
            → shortcut 억제는 student에서만 수행 (DINO 원논문 설계 원칙)

  SimMIM  : masked_clean → encoder → pixel_pred_head → L1(original_clean)
            → patch-level structural signal (pixel reconstruction)

  DINO    : CE(cls_aug_global, cls_teacher)  global-global 1쌍
          + CE(cls_local_i,   cls_teacher) × N_local
            → global-global loss 제거 시 student가 224px full-res를 never 처리
              → fine-tuning distribution shift 문제 발생 (검증 완료)

References
----------
  Caron et al. (2021) DINO — solarization on student only, global-global loss
  Xie et al.  (2022) SimMIM — masked patch reconstruction, L1 loss
  Bassi et al. (MICCAI best paper) — medical SSL augmentation
"""

from __future__ import annotations

import os
import pathlib
import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# ---------------------------------------------------------------------------
# 0. Constants & Seed
# ---------------------------------------------------------------------------

_LABEL_MAP: dict[str, int] = {
    "Hemangioma": 0,
    "HCC": 1,
}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

_SPLIT_DIR: dict[str, str] = {
    "train": "train_clean",
    "val":   "val_clean",
    "test":  "test_clean",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# 1. Path Collection
# ---------------------------------------------------------------------------

def _collect_split(data_root: str, split: str) -> list[tuple[str, int]]:
    """Return [(abs_path, label), ...] for one split."""
    if split not in _SPLIT_DIR:
        raise ValueError(f"split must be one of {list(_SPLIT_DIR)}, got {split!r}")

    split_dir = pathlib.Path(data_root) / _SPLIT_DIR[split]
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    samples: list[tuple[str, int]] = []
    for cls_name, label in _LABEL_MAP.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                samples.append((str(p), label))

    if not samples:
        raise RuntimeError(f"No images found in {split_dir}")

    n_hema = sum(1 for _, l in samples if l == 0)
    n_hcc  = sum(1 for _, l in samples if l == 1)
    print(f"  [{split:5s}] Hemangioma={n_hema:5d}  HCC={n_hcc:5d}  total={len(samples):5d}")
    return samples


# ---------------------------------------------------------------------------
# 2. Keras Augmentation Layers
# ---------------------------------------------------------------------------


class RandomGamma(layers.Layer):
    """Per-sample random gamma correction.

    Args:
        gamma_range : (min_gamma, max_gamma).
        p           : Probability of applying per sample.
    Input  : (B, H, W, C) float32 or uint8  [0, 255]
    Output : (B, H, W, C) float32           [0, 255]
    """

    def __init__(self, gamma_range: tuple[float, float] = (0.7, 1.5), p: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma_range = gamma_range
        self.p = p

    def call(self, inputs, training: bool = False):
        x = tf.cast(inputs, tf.float32)
        if not training:
            return x
        batch = tf.shape(x)[0]
        apply_mask = tf.cast(tf.random.uniform([batch, 1, 1, 1]) < self.p, tf.float32)
        gamma = tf.random.uniform([batch, 1, 1, 1], self.gamma_range[0], self.gamma_range[1], dtype=tf.float32)
        x01 = tf.clip_by_value(x / 255.0, 0.0, 1.0)
        x_gamma = tf.pow(x01 + 1e-6, gamma)
        x_gamma = tf.clip_by_value(x_gamma * 255.0, 0.0, 255.0)
        return apply_mask * x_gamma + (1.0 - apply_mask) * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'gamma_range': self.gamma_range, 'p': self.p})
        return cfg


class RandomSolarization(layers.Layer):
    """Probabilistic solarization (student views only).

    Applied ONLY to student branches.
    Teacher must never receive solarized input — validated:
      D_KL(teacher_raw ‖ teacher_solar) = 12.06  vs  tiny_noise = 0.06
      → 193× more unstable target distribution at teacher_temp=0.04

    Args:
        p               : Per-sample solarization probability. DINO uses p=0.2.
        threshold_factor: Pixels above (threshold_factor × 255) are inverted.
        value_range     : (min, max) of input pixel values.

    Input  : (B, H, W, C) uint8  [0, 255]
    Output : (B, H, W, C) float32 [0, 255]

    References:
        Caron et al. (2021) DINO, Sec 3.3 — solarization on view 2 student only
    """

    def __init__(
        self,
        p: float = 0.2,
        threshold_factor: float = 0.5,
        addition_factor: float = 0.0,
        value_range: tuple[float, float] = (0, 255),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.threshold_factor = threshold_factor
        self.addition_factor = addition_factor
        self.value_range = value_range
        self._solar_layer = layers.Solarization(
            addition_factor=addition_factor,
            threshold_factor=threshold_factor,
            value_range=value_range,
        )

    def call(self, inputs, training: bool = False):
        x = tf.cast(inputs, tf.float32)
        if not training:
            return x
        batch = tf.shape(x)[0]
        apply_mask = tf.cast(tf.random.uniform([batch, 1, 1, 1]) < self.p, tf.float32)
        x_solar = tf.cast(self._solar_layer(tf.cast(x, tf.uint8)), tf.float32)
        return apply_mask * x_solar + (1.0 - apply_mask) * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'p': self.p,
            'threshold_factor': self.threshold_factor,
            'addition_factor': self.addition_factor,
            'value_range': self.value_range,
        })
        return cfg


class RandomMaskGenerator(layers.Layer):
    """Random patch masking for SimMIM-style masked image modeling.

    Randomly masks `mask_ratio` fraction of ViT patches per sample.
    Masked patches are replaced with a constant mask_token value.
    Returns (masked_image, patch_mask) as a tuple.

    Design notes:
      - Independent random mask per sample in batch (not shared mask)
      - mask_token = 0.5 (mid-gray) — neutral for B-mode US images
      - patch_mask: 1.0 = masked (to predict), 0.0 = visible
      - Input must be float32 in [0, 1] (after Rescaling)

    Args:
        img_size    : (H, W) of input image.
        patch_size  : ViT patch size (must divide H and W evenly).
        mask_ratio  : Fraction of patches to mask. SimMIM uses 0.75.
        mask_token  : Float value used to fill masked patches (default 0.5).

    Input  : (B, H, W, C) float32 [0, 1]
    Output : tuple of
        masked_image : (B, H, W, C) float32 [0, 1]  — mask_token at masked patches
        patch_mask   : (B, N)       float32          — 1=masked, 0=visible
                       where N = (H // patch_size) * (W // patch_size)

    References:
        Xie et al. (2022) SimMIM, https://arxiv.org/abs/2111.09886
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (224, 224),
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        mask_token: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size   = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"img_size {img_size} must be divisible by patch_size {patch_size}"
        )
        self.n_h = H // patch_size
        self.n_w = W // patch_size
        self.n_patches = self.n_h * self.n_w
        self.n_masked  = int(self.n_patches * mask_ratio)

    def call(self, inputs, training: bool = False):
        """
        Args:
            inputs : (B, H, W, C) float32 [0, 1]
        Returns:
            masked_image : (B, H, W, C) float32
            patch_mask   : (B, N)       float32  (1=masked)
        """
        x = tf.cast(inputs, tf.float32)
        B = tf.shape(x)[0]
        P = self.patch_size
        n_h, n_w = self.n_h, self.n_w
        N = self.n_patches
        n_mask = self.n_masked

        # Build patch_mask [B, N]: 1=masked, 0=visible
        # Strategy: sort random noise per row → top-k are masked positions
        noise = tf.random.uniform([B, N])                        # [B, N]
        sorted_idx  = tf.argsort(noise, axis=-1)                 # [B, N] ascending
        # mask = 1 for first n_mask indices (smallest noise)
        mask_flat = tf.cast(sorted_idx < n_mask, tf.float32)     # [B, N] bool-like

        # Reshape mask to spatial: [B, n_h, n_w, 1]
        mask_spatial = tf.reshape(mask_flat, [B, n_h, n_w])      # [B, n_h, n_w]
        # Tile to patch pixels: [B, H, W]
        mask_pixel = tf.repeat(tf.repeat(mask_spatial, P, axis=1), P, axis=2)  # [B, H, W]
        mask_pixel = tf.expand_dims(mask_pixel, axis=-1)          # [B, H, W, 1]

        # Apply: masked positions → mask_token, visible → original
        masked_image = x * (1.0 - mask_pixel) + self.mask_token * mask_pixel

        return masked_image, mask_flat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'mask_ratio': self.mask_ratio,
            'mask_token': self.mask_token,
        })
        return cfg


# ---------------------------------------------------------------------------
# 3. Augmentation Pipelines
# ---------------------------------------------------------------------------

# Design rationale:
#   ✓ Teacher (original_clean): Rescaling only — no augmentation, stable target
#     Validated: D_KL(teacher_raw‖teacher_solar)=12.06, EMA center cannot compensate
#   ✓ Student global (augmented_global): solarization p=0.2 — shortcut suppression
#   ✓ Student local: solarization p=0.2 added (was missing before)
#   ✓ global-global DINO loss: student sees 224×224 full-res → no fine-tuning shift
#   ✗ Solarization on teacher: validated harmful (D_KL 193× larger than noise)


def build_teacher_rescale() -> keras.Sequential:
    """Teacher-only pipeline: NO augmentation, only Rescaling to [0,1].

    Teacher must receive clean, unaugmented images to maintain stable
    target distribution (validated: solarization causes D_KL=12.06 shift).

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H, W, C) float32 [0, 1]
    """
    return keras.Sequential(
        [layers.Rescaling(scale=1.0 / 255.0)],
        name="teacher_rescale",
    )


def build_base_augmentation(img_size: tuple[int, int]) -> keras.Sequential:
    """Supervised training augmentation — moderate strength.

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H, W, C) float32 [0, 1]
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.042, fill_mode="reflect"),
            layers.RandomZoom(
                height_factor=(-0.15, 0.15),
                width_factor=(-0.15, 0.15),
                fill_mode="reflect",
            ),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
            ),
            layers.RandomBrightness(factor=0.15),
            layers.RandomContrast(factor=0.25),
            RandomGamma(gamma_range=(0.80, 1.25), p=0.8, name="base_random_gamma"),
            layers.GaussianNoise(stddev=0.025),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="base_augmentation",
    )


def build_strong_augmentation(img_size: tuple[int, int]) -> keras.Sequential:
    """Student global-view augmentation — strong, NO solarization.

    Used only when solarization is not desired on student global view.
    Kept for backward compatibility with existing MultiViewDataset.

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H, W, C) float32 [0, 1]
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.042, fill_mode="reflect"),
            layers.RandomZoom(
                height_factor=(-0.20, 0.20),
                width_factor=(-0.20, 0.20),
                fill_mode="reflect",
            ),
            layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="reflect",
            ),
            layers.RandomBrightness(factor=0.25),
            layers.RandomContrast(factor=0.40),
            RandomGamma(gamma_range=(0.70, 1.40), p=0.9, name="strong_random_gamma"),
            layers.GaussianNoise(stddev=0.04),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="strong_augmentation",
    )


def build_strong_augmentation_with_solarization(img_size: tuple[int, int]) -> keras.Sequential:
    """Student global-view augmentation WITH solarization (p=0.2).

    Used for augmented_global (student branch) in MaskedMultiViewDataset.
    Solarization is applied to STUDENT only — never to teacher.
    This breaks bright-pixel shortcuts without destabilizing the teacher target.

    Validated: solarization on teacher → D_KL=12.06 (193× noise level)
    → teacher target becomes inconsistent → student receives noisy signal

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H, W, C) float32 [0, 1]

    References:
        Caron et al. (2021) DINO, Appendix A — solarization on student view 2 only
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.042, fill_mode="reflect"),
            layers.RandomZoom(
                height_factor=(-0.20, 0.20),
                width_factor=(-0.20, 0.20),
                fill_mode="reflect",
            ),
            layers.RandomTranslation(
                height_factor=0.08,
                width_factor=0.08,
                fill_mode="reflect",
            ),
            layers.RandomBrightness(factor=0.25),
            layers.RandomContrast(factor=0.40),
            RandomGamma(gamma_range=(0.70, 1.40), p=0.9, name="solarized_random_gamma"),
            layers.GaussianNoise(stddev=0.04),
            # Solarization BEFORE Rescaling (input still in [0,255])
            # Student only — teacher pipeline (build_teacher_rescale) never uses this
            RandomSolarization(
                p=0.2,
                threshold_factor=0.5,
                addition_factor=0.0,
                value_range=(0, 255),
                name="student_global_solarization",
            ),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="strong_augmentation_with_solarization",
    )


def build_local_crop_augmentation(
    parent_size: tuple[int, int],
    crop_scale: tuple[float, float] = (0.05, 0.40),
    output_size: tuple[int, int] | None = None,
) -> keras.Sequential:
    """Local-crop augmentation for student small-crop views.

    Now includes RandomSolarization(p=0.2) to suppress bright-spot shortcuts
    in local crops as well (consistent with student global view policy).
    Solarization is applied BEFORE Rescaling.

    Args:
        parent_size : (H, W) of the incoming full image.
        crop_scale  : (min_frac, max_frac) of image AREA to crop.
        output_size : target (H, W) after resize; default = parent_size // 2.

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H//2, W//2, C) float32 [0, 1]
    """
    H, W = parent_size
    min_frac, max_frac = crop_scale
    mid_frac = ((min_frac + max_frac) / 2) ** 0.5
    crop_h = max(16, int(H * mid_frac))
    crop_w = max(16, int(W * mid_frac))
    out_h, out_w = output_size if output_size else (H // 2, W // 2)

    return keras.Sequential(
        [
            layers.RandomCrop(height=crop_h, width=crop_w),
            layers.Resizing(height=out_h, width=out_w, interpolation="bilinear"),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.083, fill_mode="reflect"),
            layers.RandomBrightness(factor=0.22),
            layers.RandomContrast(factor=0.35),
            RandomGamma(gamma_range=(0.75, 1.35), p=0.9, name="local_random_gamma"),
            layers.GaussianNoise(stddev=0.035),
            # Solarization on student local crops (shortcut suppression)
            # Applied BEFORE Rescaling so input is still in [0, 255]
            RandomSolarization(
                p=0.2,
                threshold_factor=0.5,
                addition_factor=0.0,
                value_range=(0, 255),
                name="student_local_solarization",
            ),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="local_crop_augmentation",
    )


# ---------------------------------------------------------------------------
# 4. tf.data I/O Helper
# ---------------------------------------------------------------------------

def _decode_image(
    path: tf.Tensor,
    label: tf.Tensor,
    img_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.uint8)
    return img, label


def _make_tf_dataset(
    samples: list[tuple[str, int]],
    img_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    augment_layer: keras.Sequential | None = None,
    seed: int = 42,
) -> tf.data.Dataset:
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _decode_image(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)

    if augment_layer is not None:
        ds = ds.map(
            lambda imgs, lbls: (augment_layer(imgs, training=True), lbls),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# 5. Vanilla Supervised Dataloader
# ---------------------------------------------------------------------------

def build_dataset(
    data_root: str,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    use_augmentation: bool = True,
    shuffle_train: bool = True,
    shuffle_val: bool = True,
    shuffle_test: bool = True,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train / val / test tf.data.Dataset from SMC-LUD flat structure.

    Returns:
        (ds_train, ds_val, ds_test)
        images : float32 [B, H, W, 3]  in [0, 1]
        labels : int32   [B]            {0=Hemangioma, 1=HCC}
    """
    set_seed(seed)
    print("[build_dataset] Collecting samples ...")
    train_s = _collect_split(data_root, "train")
    val_s   = _collect_split(data_root, "val")
    test_s  = _collect_split(data_root, "test")

    aug = build_base_augmentation(img_size) if use_augmentation else None
    rescale_only = keras.Sequential([layers.Rescaling(scale=1.0 / 255.0)])

    ds_train = _make_tf_dataset(train_s, img_size, batch_size, shuffle=shuffle_train, augment_layer=aug,          seed=seed)
    ds_val   = _make_tf_dataset(val_s,   img_size, batch_size, shuffle=shuffle_val,   augment_layer=rescale_only, seed=seed)
    ds_test  = _make_tf_dataset(test_s,  img_size, batch_size, shuffle=shuffle_test,  augment_layer=rescale_only, seed=seed)

    return ds_train, ds_val, ds_test


# ---------------------------------------------------------------------------
# 6. Masked Multi-View Dataloader  (권장 B+ 구조)
# ---------------------------------------------------------------------------

def _build_masked_multiview_dataset(
    samples: list[tuple[str, int]],
    img_size: tuple[int, int],
    batch_size: int,
    local_views: int,
    local_crop_scale: tuple[float, float],
    local_output_size: tuple[int, int] | None,
    mask_ratio: float,
    patch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Core tf.data pipeline for MaskedMultiViewDataset.

    Returns tuple:
        (original_clean, masked_clean, patch_mask, augmented_global,
         local_1, ..., local_N)

    original_clean  : [B, H, W, 3]  float32 — teacher input + SimMIM target
    masked_clean    : [B, H, W, 3]  float32 — SimMIM student input
    patch_mask      : [B, N]        float32 — 1=masked, 0=visible
    augmented_global: [B, H, W, 3]  float32 — DINO student global (solarization p=0.2)
    local_i         : [B, h, w, 3]  float32 — DINO student local (solarization p=0.2)
    """
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    raw_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        raw_ds = raw_ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    raw_ds = (
        raw_ds
        .map(lambda p, l: _decode_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
    )

    # Pipelines
    teacher_pipeline  = build_teacher_rescale()                          # teacher: clean only
    student_global    = build_strong_augmentation_with_solarization(img_size)  # student global
    mask_generator    = RandomMaskGenerator(
        img_size=img_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        mask_token=0.5,
        name="random_mask_generator",
    )
    local_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)
    local_aug  = None
    if local_views > 0:
        local_aug = build_local_crop_augmentation(
            parent_size=img_size,
            crop_scale=local_crop_scale,
            output_size=local_size,
        )

    def _to_views(imgs_uint8, _labels):
        # ── Teacher / SimMIM target ──────────────────────────────────────
        original_clean = teacher_pipeline(imgs_uint8, training=False)   # [B,H,W,3]

        # ── SimMIM student: mask original_clean ──────────────────────────
        masked_clean, patch_mask = mask_generator(original_clean, training=True)
        # masked_clean: [B,H,W,3],  patch_mask: [B,N]

        # ── Student global (solarization p=0.2) ─────────────────────────
        aug_global = student_global(imgs_uint8, training=True)          # [B,H,W,3]

        # ── Student local crops ──────────────────────────────────────────
        if local_views == 0:
            return (original_clean, masked_clean, patch_mask, aug_global)

        views = [original_clean, masked_clean, patch_mask, aug_global]
        for _ in range(local_views):
            views.append(local_aug(imgs_uint8, training=True))
        return tuple(views)

    return raw_ds.map(_to_views, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


class MaskedMultiViewDataset:
    """Masked multi-view dataloader for DINO + SimMIM hybrid SSL (권장 B+).

    View structure per batch:
        (original_clean, masked_clean, patch_mask, aug_global, local_1..N)

    ┌─────────────────┬──────────────────────┬──────────────────────────┐
    │ Tensor          │ Shape                │ Used for                 │
    ├─────────────────┼──────────────────────┼──────────────────────────┤
    │ original_clean  │ [B, H, W, 3]         │ Teacher CLS + SimMIM tgt │
    │ masked_clean    │ [B, H, W, 3]         │ SimMIM student input     │
    │ patch_mask      │ [B, N]               │ SimMIM loss mask         │
    │ aug_global      │ [B, H, W, 3]         │ DINO student global      │
    │ local_i         │ [B, H//2, W//2, 3]   │ DINO student local       │
    └─────────────────┴──────────────────────┴──────────────────────────┘

    Loss structure in DINOSimMIMModel:
        L_dino  = CE(cls_aug_global, cls_teacher)           global-global
                + mean_i CE(cls_local_i, cls_teacher)       local-global
        L_simmim = mean over masked patches |pred - target|  (mean L1)
        L_total  = alpha * L_dino + (1-alpha) * lambda_mim * L_simmim

    Why original_clean for teacher (no solarization):
        Validated: D_KL(teacher_raw ‖ teacher_solar) = 12.06
        vs tiny-noise baseline = 0.06 → 193× more unstable
        EMA center does NOT compensate (KL increases 17.7 → 18.1 over 500 steps)

    Why global-global loss is kept:
        Without it, student encoder never processes 224×224 full-res
        → distribution shift at fine-tuning time (validated)
        Holistic representation learning (texture + global shape)
        is critical for HCC vs Hemangioma distinction

    Args:
        data_root         : Path to clean_ver_for_train root.
        split             : "train" / "val" / "test".
        img_size          : (H, W) for global views.
        batch_size        : Batch size.
        local_views       : Number of local crops (0 = global views + masked only).
        local_crop_scale  : (min, max) area fraction for local crops.
        local_output_size : (H, W) output for local crops; default = img_size // 2.
        mask_ratio        : Fraction of patches to mask for SimMIM (default 0.75).
        patch_size        : ViT patch size (must match encoder, default 16).
        shuffle           : Shuffle dataset.
        seed              : Random seed.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        local_views: int = 4,
        local_crop_scale: tuple[float, float] = (0.05, 0.40),
        local_output_size: tuple[int, int] | None = None,
        mask_ratio: float = 0.75,
        patch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.data_root         = data_root
        self.split             = split
        self.img_size          = img_size
        self.batch_size        = batch_size
        self.local_views       = local_views
        self.local_crop_scale  = local_crop_scale
        self.local_output_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)
        self.mask_ratio        = mask_ratio
        self.patch_size        = patch_size
        self.shuffle           = shuffle
        self.seed              = seed

        n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        n_masked  = int(n_patches * mask_ratio)

        print(f"[MaskedMultiViewDataset] split={split}")
        self.samples    = _collect_split(data_root, split)
        self.num_samples = len(self.samples)

        self.dataset = _build_masked_multiview_dataset(
            samples=self.samples,
            img_size=img_size,
            batch_size=batch_size,
            local_views=local_views,
            local_crop_scale=local_crop_scale,
            local_output_size=self.local_output_size,
            mask_ratio=mask_ratio,
            patch_size=patch_size,
            shuffle=shuffle,
            seed=seed,
        )

        print(
            f"  [권장 B+ 구조]\n"
            f"  original_clean  : {img_size} (teacher + SimMIM target, NO aug)\n"
            f"  masked_clean    : {img_size} (SimMIM student, mask_ratio={mask_ratio}, "
            f"n_masked={n_masked}/{n_patches})\n"
            f"  aug_global      : {img_size} (student global, solar p=0.2)\n"
            f"  local_views     : {local_views} @ {self.local_output_size} "
            f"(student local, solar p=0.2)\n"
            f"  total images    : {self.num_samples}  |  "
            f"steps/epoch ≈ {self.num_samples // batch_size}"
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def take(self, count: int) -> tf.data.Dataset:
        return self.dataset.take(count)

    def prefetch(self, buffer_size=tf.data.AUTOTUNE) -> tf.data.Dataset:
        return self.dataset.prefetch(buffer_size)

    def as_dataset(self) -> tf.data.Dataset:
        return self.dataset


# ---------------------------------------------------------------------------
# 7. Legacy MultiViewDataset  (backward compatibility)
# ---------------------------------------------------------------------------

def _build_multiview_dataset(
    samples: list[tuple[str, int]],
    img_size: tuple[int, int],
    batch_size: int,
    local_views: int,
    local_crop_scale: tuple[float, float],
    local_output_size: tuple[int, int] | None,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Legacy pipeline without masked branch (kept for compatibility)."""
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    raw_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        raw_ds = raw_ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    raw_ds = (
        raw_ds
        .map(lambda p, l: _decode_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
    )

    global_aug_v1 = build_strong_augmentation(img_size)
    global_aug_v2 = build_strong_augmentation_with_solarization(img_size)

    local_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)
    local_aug  = None
    if local_views > 0:
        local_aug = build_local_crop_augmentation(
            parent_size=img_size,
            crop_scale=local_crop_scale,
            output_size=local_size,
        )

    def _to_views(imgs_uint8, _):
        g1 = global_aug_v1(imgs_uint8, training=True)
        g2 = global_aug_v2(imgs_uint8, training=True)
        if local_views == 0:
            return g1, g2
        views = [g1, g2]
        for _ in range(local_views):
            views.append(local_aug(imgs_uint8, training=True))
        return tuple(views)

    return raw_ds.map(_to_views, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


class MultiViewDataset:
    """Legacy multi-crop dataloader (DINO-only, no SimMIM).

    Kept for backward compatibility with existing Stage1 runs.
    For new training, use MaskedMultiViewDataset (권장 B+).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        local_views: int = 0,
        local_crop_scale: tuple[float, float] = (0.05, 0.40),
        local_output_size: tuple[int, int] | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.data_root         = data_root
        self.split             = split
        self.img_size          = img_size
        self.batch_size        = batch_size
        self.local_views       = local_views
        self.local_crop_scale  = local_crop_scale
        self.local_output_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)
        self.shuffle           = shuffle
        self.seed              = seed

        print(f"[MultiViewDataset] split={split}  (legacy, no SimMIM)")
        self.samples     = _collect_split(data_root, split)
        self.num_samples = len(self.samples)

        self.dataset = _build_multiview_dataset(
            samples=self.samples,
            img_size=img_size,
            batch_size=batch_size,
            local_views=local_views,
            local_crop_scale=local_crop_scale,
            local_output_size=self.local_output_size,
            shuffle=shuffle,
            seed=seed,
        )

        print(
            f"  global_view_1 : {img_size} (strong aug, no solarization)\n"
            f"  global_view_2 : {img_size} (strong aug + solarization p=0.2)\n"
            f"  local_views   : {local_views} @ "
            f"{self.local_output_size if local_views else 'N/A'}  (crop aug, solarization p=0.2)\n"
            f"  total images  : {self.num_samples}  |  steps/epoch ≈ {self.num_samples // batch_size}"
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def take(self, count: int) -> tf.data.Dataset:
        return self.dataset.take(count)

    def prefetch(self, buffer_size=tf.data.AUTOTUNE) -> tf.data.Dataset:
        return self.dataset.prefetch(buffer_size)

    def as_dataset(self) -> tf.data.Dataset:
        return self.dataset


# ---------------------------------------------------------------------------
# 8. Smoke-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./clean_ver_for_train"

    print("=" * 60)
    print("Smoke-test 1: build_dataset (supervised)")
    print("=" * 60)
    try:
        ds_tr, ds_va, ds_te = build_dataset(
            DATA_ROOT, img_size=(224, 224), batch_size=4,
            shuffle_train=True, shuffle_val=True, shuffle_test=True
        )
        for imgs, lbls in ds_tr.take(1):
            print(f"  imgs  : {imgs.shape}  dtype={imgs.dtype}")
            print(f"  labels: {lbls.numpy()}")
            print(f"  pixel range: [{imgs.numpy().min():.3f}, {imgs.numpy().max():.3f}]")
    except Exception as e:
        print(f"  [WARN] {e}")

    print()
    print("=" * 60)
    print("Smoke-test 2: MaskedMultiViewDataset (권장 B+ 구조)")
    print("=" * 60)
    try:
        mmv = MaskedMultiViewDataset(
            DATA_ROOT, split="train", img_size=(224, 224),
            batch_size=4, local_views=2, mask_ratio=0.75, patch_size=16,
        )
        for batch in mmv.take(1):
            original_clean, masked_clean, patch_mask, aug_global, *locals_ = batch
            print(f"  original_clean  : {original_clean.shape}  [{original_clean.numpy().min():.3f}, {original_clean.numpy().max():.3f}]")
            print(f"  masked_clean    : {masked_clean.shape}   [{masked_clean.numpy().min():.3f}, {masked_clean.numpy().max():.3f}]")
            print(f"  patch_mask      : {patch_mask.shape}   sum={patch_mask.numpy().sum(axis=1)}")
            print(f"  aug_global      : {aug_global.shape}")
            for i, lv in enumerate(locals_):
                print(f"  local_{i+1}         : {lv.shape}")
            assert original_clean.shape == masked_clean.shape == aug_global.shape
            assert patch_mask.shape[1] == (224 // 16) ** 2
            print("  PASS: MaskedMultiViewDataset")
            break
    except Exception as e:
        print(f"  [WARN] {e}")

    print()
    print("=" * 60)
    print("Smoke-test 3: MultiViewDataset (legacy, backward compat)")
    print("=" * 60)
    try:
        mvd = MultiViewDataset(
            DATA_ROOT, split="train", img_size=(224, 224),
            batch_size=4, local_views=2,
        )
        for batch in mvd.take(1):
            g1, g2, *locs = batch
            print(f"  g1={g1.shape}  g2={g2.shape}  n_locals={len(locs)}")
            print("  PASS: MultiViewDataset (legacy)")
            break
    except Exception as e:
        print(f"  [WARN] {e}")
