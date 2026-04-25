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
    from dataloader import build_dataset, MultiViewDataset

    # 1. Vanilla supervised loader
    ds_train, ds_val, ds_test = build_dataset(
        data_root="./clean_ver_for_train",
        img_size=(224, 224),
        batch_size=32,
    )
    for imgs, labels in ds_train:
        # imgs : float32 [B, H, W, 3]  in [0, 1]
        # labels: int32  [B]  {0=Hemangioma, 1=HCC}
        ...

    # 2. Self-supervised multi-view — global views only (original resolution preserved)
    multiview_ds = MultiViewDataset(
        data_root="./clean_ver_for_train",
        split="train",
        img_size=(224, 224),
        batch_size=16,
        local_views=0,          # 0 → return only (global1, global2)
    )
    for g1, g2 in multiview_ds:
        # g1, g2: float32 [B, 224, 224, 3]
        ...

    # 3. Multi-view — global + local views
    multiview_ds = MultiViewDataset(
        data_root="./clean_ver_for_train",
        split="train",
        img_size=(224, 224),
        batch_size=16,
        local_views=6,
        local_crop_scale=(0.05, 0.40),
    )
    for views in multiview_ds:
        # views: list of tensors, len = 2 + local_views
        g1, g2, *locals_ = views
        # g1, g2  : [B, 224, 224, 3]
        # locals_ : [B, 112, 112, 3]  each (img_size // 2 by default)
        ...
"""

from __future__ import annotations

import os
import pathlib
import random
import numpy as np
import tensorflow as tf          # used ONLY for tf.data I/O pipeline
import keras
from keras import layers

# ---------------------------------------------------------------------------
# 0. Constants & Seed
# ---------------------------------------------------------------------------

# Class folders use capital-case in SMC-LUD
_LABEL_MAP: dict[str, int] = {
    "Hemangioma": 0,
    "HCC": 1,
}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Split folder names as shipped in SMC-LUD clean_ver_for_train
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
# 1. Path Collection  (flat layout — no patient sub-dirs)
# ---------------------------------------------------------------------------

def _collect_split(data_root: str, split: str) -> list[tuple[str, int]]:
    """Return [(abs_path, label), ...] for one split.

    Args:
        data_root : Root of clean_ver_for_train (or any compatible root).
        split     : One of "train" / "val" / "test".
    """
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
# 2. Keras Data-Augmentation Layers  (Medical-Domain, Keras 3 only)
# ---------------------------------------------------------------------------
# Design rationale for B-mode liver US images:
#   ✓ H/V flip — no anatomical handedness constraint in axial US views
#   ✓ Rotation ≤ 15° — probe tilt variability; heavier rotation risks artefact
#   ✓ Zoom / Translation — lesion size and position variability
#   ✓ Brightness / Contrast — US gain, TGC, and depth attenuation variation
#   ✓ GaussianNoise — speckle noise simulation (Keras-native; no scipy/cv2)
#   ✗ Color jitter — meaningless for grayscale-to-RGB B-mode images
#   ✗ Heavy elastic deformation — risks corrupting anatomical landmarks
#
# References:
#   - Bassi et al. (Johns Hopkins, MICCAI best-paper) — medical SSL aug
#   - Sowrirajan et al. (2021) MoCo chest X-ray
#   - Caron et al. (2021) multi-crop self-supervised strategy


def build_base_augmentation(img_size: tuple[int, int]) -> keras.Sequential:
    """Supervised training augmentation — moderate strength.

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H, W, C) float32 [0, 1]
    """
    return keras.Sequential(
        [
            # ── Geometric ─────────────────────────────────────────────────
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(
                factor=0.042,           # ±15°  (15/360 ≈ 0.042)
                fill_mode="reflect",
            ),
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
            # ── Photometric (US gain / TGC simulation) ────────────────────
            layers.RandomBrightness(factor=0.15),
            layers.RandomContrast(factor=0.20),
            # ── Speckle noise simulation ──────────────────────────────────
            layers.GaussianNoise(stddev=0.025),
            # ── Normalise to [0, 1] ───────────────────────────────────────
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="base_augmentation",
    )


def build_strong_augmentation(img_size: tuple[int, int]) -> keras.Sequential:
    """Global-view augmentation for self-supervised / contrastive-style SSL.

    Stronger photometric distortion than supervised baseline;
    Inspired by multi-crop self-supervised learning (Caron et al., 2021), adapted for medical US.

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
            layers.RandomContrast(factor=0.35),
            layers.GaussianNoise(stddev=0.04),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="strong_augmentation",
    )


def build_local_crop_augmentation(
    parent_size: tuple[int, int],
    crop_scale: tuple[float, float] = (0.05, 0.40),
    output_size: tuple[int, int] | None = None,
) -> keras.Sequential:
    """Local-crop augmentation for small multi-view crops.

    Crops a random small region of the image, then resizes to output_size.
    Crop height/width is computed from the geometric mean of crop_scale bounds.

    Args:
        parent_size : (H, W) of the incoming full image.
        crop_scale  : (min_frac, max_frac) of image AREA to crop.
        output_size : target (H, W) after resize; default = parent_size // 2.

    Input  : (B, H, W, C) uint8 [0, 255]
    Output : (B, H//2, W//2, C) float32 [0, 1]  (or output_size)
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
            layers.RandomRotation(factor=0.083, fill_mode="reflect"),  # ±30°
            layers.RandomBrightness(factor=0.20),
            layers.RandomContrast(factor=0.30),
            layers.GaussianNoise(stddev=0.035),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="local_crop_augmentation",
    )


# ---------------------------------------------------------------------------
# 3. tf.data I/O Helper
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
# 4. Vanilla Supervised Dataloader
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

    The function reads the pre-split directories:
        All three splits can be shuffled independently.
        data_root/train_clean/{HCC,Hemangioma}/
        data_root/val_clean/{HCC,Hemangioma}/
        data_root/test_clean/{HCC,Hemangioma}/

    Args:
        data_root       : Path to clean_ver_for_train (or equivalent root).
        img_size        : (H, W) to resize all images.
        batch_size      : Batch size for all splits.
        use_augmentation: If True, applies base_augmentation to train split only.
        shuffle_train   : If True, shuffle train samples.
        shuffle_val     : If True, shuffle validation samples.
        shuffle_test    : If True, shuffle test samples.
        seed            : Random seed.

    Returns:
        (ds_train, ds_val, ds_test)
        Each yields: (images, labels)
            images : float32 [B, H, W, 3]  in [0, 1]
            labels : int32   [B]            {0=Hemangioma, 1=HCC}
    """
    set_seed(seed)
    print("[build_dataset] Collecting samples ...")
    train_s = _collect_split(data_root, "train")
    val_s   = _collect_split(data_root, "val")
    test_s  = _collect_split(data_root, "test")

    aug = build_base_augmentation(img_size) if use_augmentation else None

    # Val / Test: only rescale (no augmentation), keep uint8 → float mapping
    rescale_only = keras.Sequential([layers.Rescaling(scale=1.0 / 255.0)])

    ds_train = _make_tf_dataset(train_s, img_size, batch_size, shuffle=shuffle_train, augment_layer=aug,          seed=seed)
    ds_val   = _make_tf_dataset(val_s,   img_size, batch_size, shuffle=shuffle_val,   augment_layer=rescale_only, seed=seed)
    ds_test  = _make_tf_dataset(test_s,  img_size, batch_size, shuffle=shuffle_test,  augment_layer=rescale_only, seed=seed)

    return ds_train, ds_val, ds_test


# ---------------------------------------------------------------------------
# 5. Multi-View Dataloader
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
    """Create a fit-ready tf.data.Dataset for multi-view learning.

    Returns:
        local_views == 0:
            dataset yielding (g1, g2)
        local_views > 0:
            dataset yielding tuple(g1, g2, l1, ..., lN)
    """
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    raw_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        raw_ds = raw_ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    raw_ds = (
        raw_ds
        .map(lambda p, l: _decode_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
    )

    global_aug = build_strong_augmentation(img_size)
    local_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)
    local_aug = None
    if local_views > 0:
        local_aug = build_local_crop_augmentation(
            parent_size=img_size,
            crop_scale=local_crop_scale,
            output_size=local_size,
        )

    def _to_views(imgs_uint8, _):
        g1 = global_aug(imgs_uint8, training=True)
        g2 = global_aug(imgs_uint8, training=True)
        if local_views == 0:
            return g1, g2
        views = [g1, g2]
        for _ in range(local_views):
            views.append(local_aug(imgs_uint8, training=True))
        return tuple(views)

    return raw_ds.map(_to_views, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

class MultiViewDataset:
    """Multi-crop dataloader for self-supervised multi-view learning.

    This class wraps an internal ``tf.data.Dataset`` and exposes a small
    tf.data-like interface (``take()``, ``prefetch()``, ``as_dataset()``),
    so it can be used directly with ``model.fit(...)``.

    Reads ONE split from the SMC-LUD flat structure.
    Labels are NOT yielded (unsupervised pre-training mode).
    For linear probing / fine-tuning use build_dataset().

    Global views (always 2):
        - Resolution = img_size  (original input resolution is PRESERVED)
        - Augmentation : build_strong_augmentation()

    Local views (optional, N = local_views):
        - Created only when local_views > 0
        - Resolution = img_size // 2  (override via local_output_size)
        - Augmentation : build_local_crop_augmentation()

    Args:
        data_root        : Path to clean_ver_for_train root.
        split            : "train" / "val" / "test"
        img_size         : (H, W) for global views.
        batch_size       : Batch size.
        local_views      : Number of local crops.
                           0  → yield (g1, g2)   — tuple of 2 tensors
                           N  → yield (g1, g2, l1, ..., lN)  — tuple
        local_crop_scale : (min, max) area fraction for local crops.
        local_output_size: (H, W) of local view output; default = img_size // 2.
        shuffle          : Shuffle the dataset (applies to train / val / test).
        seed             : Random seed.

    Iteration:
        local_views == 0:
            yields (g1, g2)
            g1, g2 : float32 [B, H, W, 3]

        local_views > 0:
            yields (g1, g2, l1, ..., lN)
            tuple length = 2 + local_views
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
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.batch_size = batch_size
        self.local_views = local_views
        self.local_crop_scale = local_crop_scale
        self.local_output_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)
        self.shuffle = shuffle
        self.seed = seed

        print(f"[MultiViewDataset] split={split}")
        self.samples = _collect_split(data_root, split)
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
            f"  global_views : 2 @ {img_size}  (strong aug)\n"
            f"  local_views  : {local_views} @ "
            f"{self.local_output_size if local_views else 'N/A'}  (crop aug)\n"
            f"  total images : {self.num_samples}  |  steps/epoch ≈ {self.num_samples // batch_size}"
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
# 6. Smoke-Test  (python dataloader.py <data_root>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./clean_ver_for_train"

    # ── Test 1: Supervised ────────────────────────────────────────────────
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
            print(f"  labels: {lbls.numpy()}  (0=Hemangioma, 1=HCC)")
            print(f"  pixel range: [{imgs.numpy().min():.3f}, {imgs.numpy().max():.3f}]")
    except Exception as e:
        print(f"  [WARN] {e}")

    # ── Test 2: global-only multi-view ───────────────────────────────────────
    print()
    print("=" * 60)
    print("Smoke-test 2: MultiViewDataset (local_views=0, global-only)")
    print("=" * 60)
    try:
        multiview_ds = MultiViewDataset(DATA_ROOT, split="train", img_size=(224, 224),
                                        batch_size=4, local_views=0)
        for g1, g2 in multiview_ds.take(1):
            print(f"  global_view_1 : {g1.shape}  dtype={g1.dtype}")
            print(f"  global_view_2 : {g2.shape}  dtype={g2.dtype}")
            assert g1.shape == g2.shape, "Shape mismatch!"
            break
        print("  PASS: global-only mode")
    except Exception as e:
        print(f"  [WARN] {e}")

    # ── Test 3: multi-view + local crops ─────────────────────────────────────
    print()
    print("=" * 60)
    print("Smoke-test 3: MultiViewDataset (local_views=6, with local crops)")
    print("=" * 60)
    try:
        multiview_ds6 = MultiViewDataset(DATA_ROOT, split="train", img_size=(224, 224),
                                         batch_size=4, local_views=6)
        for views in multiview_ds6.take(1):
            print(f"  Total views returned : {len(views)}  (expected 8)")
            for i, v in enumerate(views):
                tag = "global" if i < 2 else "local "
                print(f"    [{tag} view {i}] shape={v.shape}  dtype={v.dtype}")
            assert len(views) == 8
            break
        print("  PASS: multi-crop mode")
    except Exception as e:
        print(f"  [WARN] {e}")
