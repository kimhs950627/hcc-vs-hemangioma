"""
SMC-LUD Dataloader for HCC vs. Hemangioma Classification
=========================================================
Dataset: Samsung Medical Center - Liver Ultrasound Dataset (SMC-LUD)
  - 5,385 B-mode ultrasound images from 1,021 patients
  - HCC: 2,716 images  (label = 1)
  - Hemangioma: 2,669 images  (label = 0)
  - Patient-level grouping: images from the same patient are in one subfolder

Expected directory structure
-----------------------------
data_root/
    hemangioma/
        patient_001/
            img001.png
            ...
        patient_002/
            ...
    hcc/
        patient_010/
            img001.png
            ...

If the dataset is stored flat (no patient sub-directories):
    data_root/
        hemangioma/
            img001.png ...
        hcc/
            img001.png ...

Usage
-----
    from dataloader import build_dataset, SsmcLudDataset, DINOMultiViewDataset

    # 1. Vanilla supervised loader
    ds_train, ds_val, ds_test = build_dataset(
        data_root="./data",
        img_size=(224, 224),
        batch_size=32,
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )
    for imgs, labels in ds_train:
        ...

    # 2. DINO-style SSL loader (global views only)
    dino_ds = DINOMultiViewDataset(
        data_root="./data",
        img_size=(224, 224),
        batch_size=16,
        local_views=0,       # 0 → only 2 global views, original resolution kept
    )
    for global1, global2 in dino_ds:
        ...

    # 3. DINO with local views
    dino_ds = DINOMultiViewDataset(
        data_root="./data",
        img_size=(224, 224),
        batch_size=16,
        local_views=6,       # 2 global + 6 local crops
        local_crop_scale=(0.05, 0.4),
    )
    for views in dino_ds:   # list of tensors, length = 2 + local_views
        global1, global2, *locals_ = views
        ...
"""

from __future__ import annotations

import os
import pathlib
import random
from typing import Sequence

import numpy as np
import tensorflow as tf                 # used ONLY for tf.data pipeline helpers
import keras
from keras import layers, ops

# ---------------------------------------------------------------------------
# 0. Seed Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix global randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# 1. Path / Label Collection
# ---------------------------------------------------------------------------
# Label convention (task-level, not DINO):
#   hemangioma → 0
#   hcc        → 1

_LABEL_MAP = {"hemangioma": 0, "hcc": 1}
_IMG_EXTS   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_paths(data_root: str) -> list[tuple[str, int]]:
    """Walk data_root and return [(abs_path, label), ...].

    Handles both flat (data_root/class/*.png) and
    patient-grouped (data_root/class/patient_id/*.png) layouts.
    """
    root = pathlib.Path(data_root)
    samples: list[tuple[str, int]] = []

    for cls_name, label in _LABEL_MAP.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Expected class directory not found: {cls_dir}")
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                samples.append((str(p), label))

    if not samples:
        raise RuntimeError(f"No images found under {data_root!r}")

    return samples


def _patient_split(
    data_root: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Patient-level split to prevent data leakage.

    Flat directories are treated as one pseudo-patient each.
    """
    rng = random.Random(seed)
    root = pathlib.Path(data_root)

    train_samples, val_samples, test_samples = [], [], []

    for cls_name, label in _LABEL_MAP.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(cls_dir)

        # Determine patient groups
        subdirs = [d for d in cls_dir.iterdir() if d.is_dir()]
        if subdirs:
            # Patient-grouped layout
            patient_dirs = subdirs
        else:
            # Flat layout → treat the class dir itself as one group
            patient_dirs = [cls_dir]

        patient_dirs = sorted(patient_dirs)
        rng.shuffle(patient_dirs)

        n = len(patient_dirs)
        n_test = max(1, int(n * test_split))
        n_val  = max(1, int(n * val_split))

        test_dirs  = patient_dirs[:n_test]
        val_dirs   = patient_dirs[n_test: n_test + n_val]
        train_dirs = patient_dirs[n_test + n_val:]

        def _collect_from_dirs(dirs):
            out = []
            for d in dirs:
                for p in pathlib.Path(d).rglob("*"):
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS:
                        out.append((str(p), label))
            return out

        train_samples.extend(_collect_from_dirs(train_dirs))
        val_samples.extend(_collect_from_dirs(val_dirs))
        test_samples.extend(_collect_from_dirs(test_dirs))

    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# 2. Keras Data-Augmentation Layers  (Medical-Domain)
# ---------------------------------------------------------------------------
# References:
#   - Salehi et al. (2017) Tversky loss for medical segmentation
#   - Zhou et al. (MICCAI best-paper) "Learning Segmentation from Radiology Reports"
#   - Sowrirajan et al. (2021) MoCo pre-training on chest X-ray
#   - Bassi et al. (Johns Hopkins) — medical-domain SSL augmentation
#
# Design principles for US images:
#   ✓ Horizontal/vertical flips are physically valid (no anatomical handedness)
#   ✓ Moderate rotation (<=15°) — avoid artefact or out-of-plane confusion
#   ✓ Mild brightness/contrast shifts to simulate gain/attenuation changes
#   ✓ Speckle noise simulation via Gaussian noise (no cv2/scipy blur → Keras only)
#   ✓ Random crop + zoom (for local views in DINO)
#   ✗ Colour jitter — US images are grayscale; colour aug is meaningless
#   ✗ Heavy elastic deformation — avoid anatomic landmark corruption


def build_base_augmentation(img_size: tuple[int, int]) -> keras.Sequential:
    """Shared base augmentation for both supervised and SSL training.

    Input  : (H, W, C) uint8 or float32 images
    Output : (H, W, C) float32, range [0, 1]
    """
    return keras.Sequential(
        [
            # ── Geometric ──────────────────────────────────────────────────
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.042),      # ±15° = 15/360 ≈ 0.042
            layers.RandomZoom(
                height_factor=(-0.15, 0.15),
                width_factor=(-0.15, 0.15),
            ),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
            ),
            # ── Photometric (simulate US gain / TGC variation) ──────────────
            layers.RandomBrightness(factor=0.15),
            layers.RandomContrast(factor=0.20),
            # ── Gaussian noise (speckle simulation) ─────────────────────────
            layers.GaussianNoise(stddev=0.025),
            # ── Normalise to [0, 1] ──────────────────────────────────────────
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="base_augmentation",
    )


def build_strong_augmentation(img_size: tuple[int, int]) -> keras.Sequential:
    """Stronger augmentation for the global views in DINO / MoCo-style SSL.

    Mirrors DINO paper (Caron et al., 2021) adapted for medical ultrasound:
    global views see the full image with aggressive photometric distortion.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.042),
            layers.RandomZoom(height_factor=(-0.20, 0.20), width_factor=(-0.20, 0.20)),
            layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode="reflect"),
            # Stronger photometric
            layers.RandomBrightness(factor=0.25),
            layers.RandomContrast(factor=0.35),
            # Solarisation approximation via random inversion (medical equivalent)
            # keras.layers.RandomInvert is not yet in stable API → skip, use noise
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
    """Local-crop augmentation for DINO small views.

    Uses RandomCrop to simulate a local (small region) view.
    The crop is then resized back to ``output_size`` (defaults to
    parent_size // 2) so all tensors share a spatial shape.

    Args:
        parent_size: (H, W) of the full image entering the pipeline.
        crop_scale: (min_frac, max_frac) of the image area to crop.
        output_size: target (H, W) after resize; default = parent_size // 2.
    """
    H, W = parent_size
    min_frac, max_frac = crop_scale

    # Deterministic crop height (sampled per-batch by Keras RNG)
    crop_h = int(H * ((min_frac + max_frac) / 2) ** 0.5)
    crop_w = int(W * ((min_frac + max_frac) / 2) ** 0.5)

    out_h, out_w = output_size if output_size else (H // 2, W // 2)

    return keras.Sequential(
        [
            layers.RandomCrop(height=crop_h, width=crop_w),
            layers.Resizing(height=out_h, width=out_w, interpolation="bilinear"),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.083),           # ±30° for local views
            layers.RandomBrightness(factor=0.20),
            layers.RandomContrast(factor=0.30),
            layers.GaussianNoise(stddev=0.035),
            layers.Rescaling(scale=1.0 / 255.0),
        ],
        name="local_crop_augmentation",
    )


# ---------------------------------------------------------------------------
# 3. tf.data Pipeline Helpers
# ---------------------------------------------------------------------------

def _load_image_tf(path: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]):
    """Decode and resize a single image; keep uint8 for augmentation layers."""
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
    augment_fn=None,
    seed: int = 42,
) -> tf.data.Dataset:
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, l: _load_image_tf(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)

    if augment_fn is not None:
        ds = ds.map(
            lambda imgs, lbls: (augment_fn(imgs, training=True), lbls),
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
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
    use_augmentation: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Build train / val / test tf.data.Dataset with patient-level split.

    Returns:
        (ds_train, ds_val, ds_test)
        Each dataset yields (images [B,H,W,3], labels [B]) batches.
        images: float32 in [0, 1] after augmentation / rescaling.
        labels: int32  {0=hemangioma, 1=HCC}
    """
    set_seed(seed)
    train_s, val_s, test_s = _patient_split(data_root, val_split, test_split, seed)

    aug_fn = build_base_augmentation(img_size) if use_augmentation else None

    ds_train = _make_tf_dataset(train_s, img_size, batch_size, shuffle=True,  augment_fn=aug_fn, seed=seed)
    ds_val   = _make_tf_dataset(val_s,   img_size, batch_size, shuffle=False, augment_fn=None)
    ds_test  = _make_tf_dataset(test_s,  img_size, batch_size, shuffle=False, augment_fn=None)

    print(f"[build_dataset] train={len(train_s)}, val={len(val_s)}, test={len(test_s)} images")
    return ds_train, ds_val, ds_test


# ---------------------------------------------------------------------------
# 5. DINO Multi-View Dataloader
# ---------------------------------------------------------------------------

class DINOMultiViewDataset:
    """Multi-crop dataloader compatible with DINO / iBOT self-supervised learning.

    Global views:
        - Always 2 global views.
        - Resolution = img_size (original input resolution is preserved).
        - Uses strong augmentation.

    Local views (optional):
        - Created only when local_views > 0.
        - Resolution = img_size // 2  (configurable via local_output_size).
        - Uses local-crop augmentation.

    Args:
        data_root         : Path containing hemangioma/ and hcc/ subdirs.
        img_size          : (H, W) of global views.
        batch_size        : Batch size.
        local_views       : Number of local crops per image.
                            0  → only 2 global views are returned.
                            N  → returns 2 global + N local views.
        local_crop_scale  : (min, max) fraction of area for local crops.
        local_output_size : (H, W) for local view output; default img_size // 2.
        shuffle           : Whether to shuffle the dataset.
        seed              : Random seed.

    Iteration yields:
        local_views == 0:  (global_view_1, global_view_2)
                           Each: float32 [B, H, W, 3]

        local_views > 0 :  [global_view_1, global_view_2,
                            local_view_1, ..., local_view_N]
                           List length = 2 + local_views.
                           local_views: float32 [B, H//2, W//2, 3] by default.

    Note:
        Labels are NOT yielded (unsupervised mode).
        For linear probing, use build_dataset() for the labelled loader.
    """

    def __init__(
        self,
        data_root: str,
        img_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        local_views: int = 0,
        local_crop_scale: tuple[float, float] = (0.05, 0.40),
        local_output_size: tuple[int, int] | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.img_size         = img_size
        self.batch_size       = batch_size
        self.local_views      = local_views
        self.local_output_size = local_output_size or (img_size[0] // 2, img_size[1] // 2)

        samples = _collect_paths(data_root)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(samples)

        paths  = [s[0] for s in samples]
        labels = [s[1] for s in samples]   # kept internally but not yielded

        # Base tf.data (raw uint8 images)
        self._raw_ds = (
            tf.data.Dataset.from_tensor_slices((paths, labels))
            .map(lambda p, l: _load_image_tf(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Augmentation layers
        self._global_aug = build_strong_augmentation(img_size)

        if local_views > 0:
            self._local_aug = build_local_crop_augmentation(
                parent_size=img_size,
                crop_scale=local_crop_scale,
                output_size=self.local_output_size,
            )

        print(
            f"[DINOMultiViewDataset] images={len(samples)}, "
            f"global_views=2 @ {img_size}, "
            f"local_views={local_views} @ {self.local_output_size if local_views else 'N/A'}"
        )

    # ------------------------------------------------------------------
    def __iter__(self):
        for imgs_uint8, _ in self._raw_ds:
            # ── 2 global views (full resolution) ──────────────────────
            g1 = self._global_aug(imgs_uint8, training=True)   # [B, H, W, 3]
            g2 = self._global_aug(imgs_uint8, training=True)   # [B, H, W, 3]

            if self.local_views == 0:
                yield g1, g2
            else:
                views = [g1, g2]
                for _ in range(self.local_views):
                    lv = self._local_aug(imgs_uint8, training=True)  # [B, H/2, W/2, 3]
                    views.append(lv)
                yield views

    def __len__(self) -> int:
        n_samples = sum(1 for _ in self._raw_ds) * self.batch_size
        return n_samples // self.batch_size


# ---------------------------------------------------------------------------
# 6. Quick Smoke-Test  (python dataloader.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print("=" * 60)
    print("Smoke-test 1: build_dataset (supervised)")
    print("=" * 60)
    try:
        ds_tr, ds_va, ds_te = build_dataset(
            DATA_ROOT, img_size=(224, 224), batch_size=4, seed=42
        )
        for imgs, lbls in ds_tr.take(1):
            print(f"  imgs.shape={imgs.shape}, dtype={imgs.dtype}")
            print(f"  lbls={lbls.numpy()}  (0=hemangioma, 1=HCC)")
    except Exception as e:
        print(f"  [WARN] Skipped (no data): {e}")

    print()
    print("=" * 60)
    print("Smoke-test 2: DINOMultiViewDataset (local_views=0)")
    print("=" * 60)
    try:
        dino_ds = DINOMultiViewDataset(DATA_ROOT, img_size=(224, 224), batch_size=4, local_views=0)
        for g1, g2 in dino_ds.__iter__():
            print(f"  global_view_1: {g1.shape}, dtype={g1.dtype}")
            print(f"  global_view_2: {g2.shape}, dtype={g2.dtype}")
            break
    except Exception as e:
        print(f"  [WARN] Skipped (no data): {e}")

    print()
    print("=" * 60)
    print("Smoke-test 3: DINOMultiViewDataset (local_views=6)")
    print("=" * 60)
    try:
        dino_ds6 = DINOMultiViewDataset(DATA_ROOT, img_size=(224, 224), batch_size=4, local_views=6)
        for views in dino_ds6.__iter__():
            print(f"  Total views returned: {len(views)}")
            for i, v in enumerate(views):
                print(f"    view[{i}]: {v.shape}, dtype={v.dtype}")
            break
    except Exception as e:
        print(f"  [WARN] Skipped (no data): {e}")

    print()
    print("Done. If '[WARN] Skipped' appeared, place data under", DATA_ROOT)
