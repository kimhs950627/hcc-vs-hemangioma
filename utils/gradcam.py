"""
utils/gradcam.py
Grad-CAM heatmap generation for both CNN and ViT encoders.

Public API
----------
make_gradcam_heatmap(model, images, last_conv_layer_name, class_index)  -> np.ndarray (N,H,W)
apply_gradcam_overlay(image_uint8, heatmap)                              -> np.ndarray (H,W,3)
save_gradcam_grid(images, heatmaps, path, titles)                        -> None

Notes
-----
- Follows the official Keras Grad-CAM recipe:
  https://keras.io/examples/vision/grad_cam/
- Supports batch of images (N, H, W, C) for efficiency.
- For ViT encoders that do NOT expose a convolutional feature map,
  the function falls back to a gradient-based input saliency map (GradInputs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(
    model,
    images: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmaps for a batch of images.

    Args:
        model: A Keras model whose output is class logits.
        images: Float32 array of shape (N, H, W, C), values in [0, 1].
        last_conv_layer_name: Name of the last convolutional layer to
            differentiate with respect to.  If None or the layer is not
            found, falls back to gradient × input saliency.
        class_index: Target class index.  If None, the highest-scoring class
            for each image is used.

    Returns:
        heatmaps: Float32 array of shape (N, H, W), values in [0, 1].
    """
    import tensorflow as tf
    import keras

    images_tf = tf.cast(images, tf.float32)

    # ── Try to build a Grad-CAM feature extractor ──────────────────────────────
    grad_model = None
    if last_conv_layer_name is not None:
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[conv_layer.output, model.output],
            )
        except (ValueError, AttributeError):
            grad_model = None

    if grad_model is not None:
        heatmaps = _gradcam_from_conv(
            grad_model, images_tf, class_index
        )
    else:
        # Fallback: gradient × input saliency
        heatmaps = _gradient_input_saliency(model, images_tf, class_index)

    return heatmaps  # shape (N, H, W), float32 in [0,1]


def _gradcam_from_conv(
    grad_model,
    images_tf,
    class_index: Optional[int],
) -> np.ndarray:
    """Standard Grad-CAM using the last convolutional feature map."""
    import tensorflow as tf

    with tf.GradientTape() as tape:
        inputs = tf.cast(images_tf, tf.float32)
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs, training=False)
        if class_index is None:
            # Per-image argmax
            idx = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            batch_size = tf.shape(predictions)[0]
            gather_idx = tf.stack(
                [tf.range(batch_size), idx], axis=1
            )
            class_scores = tf.gather_nd(predictions, gather_idx)
        else:
            class_scores = predictions[:, class_index]

    grads = tape.gradient(class_scores, conv_outputs)  # (N, fh, fw, C)
    # Global average pooling of gradients -> channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))   # (N, C)
    # Weighted sum of feature channels
    pooled_grads_4d = pooled_grads[:, tf.newaxis, tf.newaxis, :]  # (N,1,1,C)
    cam = tf.reduce_sum(conv_outputs * pooled_grads_4d, axis=-1)  # (N, fh, fw)
    cam = tf.nn.relu(cam).numpy()  # ReLU: keep positive activations

    # Resize to input spatial size and normalise per image
    h, w = int(images_tf.shape[1]), int(images_tf.shape[2])
    heatmaps = []
    for i in range(cam.shape[0]):
        c = cam[i]  # (fh, fw)
        # Upscale via bilinear interpolation
        c_resized = _resize_array(c, h, w)
        c_min, c_max = c_resized.min(), c_resized.max()
        c_norm = (c_resized - c_min) / max(c_max - c_min, 1e-8)
        heatmaps.append(c_norm.astype(np.float32))
    return np.stack(heatmaps, axis=0)  # (N, H, W)


def _gradient_input_saliency(
    model,
    images_tf,
    class_index: Optional[int],
) -> np.ndarray:
    """Fallback: gradient × input saliency (ViT / no conv layer)."""
    import tensorflow as tf

    with tf.GradientTape() as tape:
        inputs = tf.cast(images_tf, tf.float32)
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        if class_index is None:
            idx = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            batch_size = tf.shape(predictions)[0]
            gather_idx = tf.stack(
                [tf.range(batch_size), idx], axis=1
            )
            class_scores = tf.gather_nd(predictions, gather_idx)
        else:
            class_scores = predictions[:, class_index]

    grads = tape.gradient(class_scores, inputs)  # (N, H, W, C)
    saliency = tf.reduce_max(tf.abs(grads * inputs), axis=-1)  # (N, H, W)
    saliency = saliency.numpy()

    heatmaps = []
    for i in range(saliency.shape[0]):
        s = saliency[i]
        s_min, s_max = s.min(), s.max()
        heatmaps.append(((s - s_min) / max(s_max - s_min, 1e-8)).astype(np.float32))
    return np.stack(heatmaps, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_gradcam_overlay(
    image_float: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a Grad-CAM heatmap on top of the original image.

    Args:
        image_float: float32 (H, W, 3) in [0, 1].
        heatmap: float32 (H, W) in [0, 1].
        alpha: Heatmap blend factor (0 = no overlay, 1 = heatmap only).
        colormap: Matplotlib colormap name for the heatmap.

    Returns:
        uint8 RGB array (H, W, 3).
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[..., :3].astype(np.float32)  # (H, W, 3)
    image_float = np.clip(image_float, 0, 1)
    overlay = (1 - alpha) * image_float + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay


def save_gradcam_grid(
    images: np.ndarray,
    heatmaps: np.ndarray,
    path: str | Path,
    titles: Optional[list[str]] = None,
    n_cols: int = 4,
) -> None:
    """Save a grid of (image | overlay) pairs as a PNG.

    Args:
        images: (N, H, W, C) float32 in [0, 1].
        heatmaps: (N, H, W) float32 in [0, 1].
        path: Output PNG file path.
        titles: Optional list of N title strings (e.g., 'GT:HCC | Pred:0.83').
        n_cols: Number of image pairs per row.
    """
    import matplotlib.pyplot as plt

    n = images.shape[0]
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols * 2,
        figsize=(n_cols * 4, n_rows * 2.5),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n):
        row = i // n_cols
        col_base = (i % n_cols) * 2

        # Left: original image
        axes[row][col_base].imshow(
            np.clip(images[i], 0, 1),
            cmap="gray" if images[i].shape[-1] == 1 else None,
        )
        if titles and i < len(titles):
            axes[row][col_base].set_title(titles[i], fontsize=7)

        # Right: overlay
        img_rgb = images[i]
        if img_rgb.shape[-1] == 1:
            img_rgb = np.concatenate([img_rgb] * 3, axis=-1)
        overlay = apply_gradcam_overlay(img_rgb, heatmaps[i])
        axes[row][col_base + 1].imshow(overlay)
        axes[row][col_base + 1].set_title("Grad-CAM", fontsize=7)

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Internal resize (no cv2 / PIL dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _resize_array(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear resize via TensorFlow (no cv2 / PIL needed)."""
    import tensorflow as tf

    t = tf.convert_to_tensor(arr[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
    t = tf.image.resize(t, (target_h, target_w), method="bilinear")
    return t.numpy()[0, :, :, 0]
