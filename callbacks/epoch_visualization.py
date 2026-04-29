"""Epoch-wise metrics and visualization callback for HCC vs. Hemangioma.

Every epoch (or every N epochs):

1. **Classification metrics** (full validation set)
   - Accuracy, ROC-AUC, Sensitivity, Specificity, PPV, NPV
   - Saved as ``metrics/metrics_epoch_{epoch:03d}.json``
   - Appended to ``metrics/metrics_all.csv``
   - ROC curve PNG saved to ``roc/roc_epoch_{epoch:03d}.png``

2. **Grad-CAM heatmaps** (CNN encoders: ConvNeXt, EfficientNet)
   - Overlaid on sample images
   - Saved to ``gradcam/epoch_{epoch:03d}_sample_{i:02d}.png``

3. **Attention map visualizations** (ViT / Swin encoders)
   - CLS-to-patch attention weight averaged over heads
   - Resized to image dimensions and overlaid
   - Saved to ``attention/epoch_{epoch:03d}_sample_{i:02d}.png``

"""

from __future__ import annotations

import csv
import pathlib
from typing import Any

import numpy as np
import tensorflow as tf
import keras

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from utils.metrics import binary_classification_metrics, save_metrics_json, plot_roc_curve


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

def _make_gradcam_heatmap(
    grad_model: keras.Model,
    image: np.ndarray,
) -> np.ndarray | None:
    """Compute single-image Grad-CAM heatmap [H_cam, W_cam].

    Args:
        grad_model : A Keras model with TWO outputs:
                     [last_conv_feature_map (B,H,W,C), logit (B,1)].
        image      : Float32 array (H, W, 3), values in [0, 1].

    Returns:
        Normalised heatmap (H_cam, W_cam) in [0, 1], or None on failure.
    """
    try:
        img_t = tf.cast(tf.expand_dims(image, 0), tf.float32)  # (1,H,W,3)
        with tf.GradientTape() as tape:
            inputs = tf.cast(img_t, tf.float32)
            tape.watch(inputs)
            conv_out, logit = grad_model(inputs, training=False)
            loss = tf.squeeze(logit)                             # scalar
        grads = tape.gradient(loss, conv_out)                    # (1,H,W,C)
        # Pool spatial dimensions → channel weights.
        pooled = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # (1,1,1,C)
        cam    = tf.reduce_sum(conv_out * pooled, axis=-1)           # (1,H,W)
        cam    = tf.squeeze(cam).numpy()                             # (H,W)
        cam    = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[Grad-CAM] Failed: {exc}")
        return None


def _build_grad_model(
    core_model: keras.Model,
    last_conv_layer_name: str | None,
) -> keras.Model | None:
    """Build a gradient model for Grad-CAM.

    Searches `core_model.enc` (if present) or `core_model` for a Conv2D layer
    matching `last_conv_layer_name`. Returns None if not found.

    Returns:
        keras.Model with outputs [conv_feature_map, logit] or None.
    """
    if last_conv_layer_name is None:
        return None

    # Resolve the backbone model.
    backbone = getattr(core_model, 'enc', core_model)
    inner    = getattr(backbone,   'base_model', backbone)

    try:
        conv_layer = inner.get_layer(last_conv_layer_name)
    except (ValueError, AttributeError):
        # Fall back: iterate submodels.
        conv_layer = None
        for layer in inner.layers:
            if layer.name == last_conv_layer_name:
                conv_layer = layer
                break
        if conv_layer is None:
            return None

    try:
        # Build functional model: input → [conv_out, logit]
        grad_model = keras.Model(
            inputs  = inner.input,
            outputs = [conv_layer.output, inner.output],
        )
        return grad_model
    except Exception as exc:  # noqa: BLE001
        print(f"[Grad-CAM] Could not build gradient model: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ViT / Swin attention visualization
# ─────────────────────────────────────────────────────────────────────────────

def _extract_attention_map(
    core_model: keras.Model,
    image: np.ndarray,
    patch_size: int = 16,
) -> np.ndarray | None:
    """Extract CLS-to-patch attention map from ViT / Swin encoder.

    Returns:
        Float32 array (H_img, W_img) normalised to [0, 1], or None.
    """
    try:
        enc = getattr(core_model, 'enc', core_model)
        img_t = tf.cast(tf.expand_dims(image, 0), tf.float32)  # (1,H,W,3)
        enc_out = enc(img_t, training=False, return_attention=True)
        attn = enc_out.get('last_encoder_layer_attentional_weights')
        if attn is None:
            return None
        # attn shape: (B, n_heads, N_tokens, N_tokens)
        # where N_tokens = n_patches + 1 (CLS token at index 0)
        attn_np = attn.numpy()                    # (1, H, N, N)
        attn_np = attn_np[0]                      # (n_heads, N, N)
        # Average over heads; take CLS row (index 0); skip CLS column.
        cls_attn = attn_np[:, 0, 1:]              # (n_heads, n_patches)
        cls_attn = cls_attn.mean(axis=0)          # (n_patches,)
        n_patches = cls_attn.shape[0]
        grid_size = int(np.sqrt(n_patches))
        if grid_size * grid_size != n_patches:
            return None
        cam = cls_attn.reshape(grid_size, grid_size)
        # Upsample to original image size.
        h, w = image.shape[:2]
        cam_resized = np.array(
            tf.image.resize(
                cam[..., np.newaxis],
                (h, w),
                method='bilinear',
            )
        ).squeeze(-1)
        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()
        return cam_resized.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[Attention] Failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Image saving helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    save_path: str | pathlib.Path,
    label: int | None = None,
    prob: float | None = None,
    epoch: int | None = None,
    title_prefix: str = "",
    alpha: float = 0.45,
) -> None:
    """Save side-by-side [original | heatmap overlay] PNG.

    Args:
        image     : (H, W, 3) float32 in [0, 1]
        heatmap   : (H, W) float32 in [0, 1]
        save_path : Output file path.
        label     : Ground-truth label (0/1).
        prob      : Predicted probability of class 1.
        epoch     : Epoch index (for title).
        title_prefix : Extra prefix for subplot title.
        alpha     : Heatmap blending strength.
    """
    if not _MPL_OK:
        return
    label_str = {
        0: 'Hemangioma', 1: 'HCC', None: '?'
    }.get(label, str(label))
    prob_str  = f"P(HCC)={prob:.3f}" if prob is not None else ""

    colormap   = cm.get_cmap('jet')
    heatmap_rgb = colormap(heatmap)[..., :3].astype(np.float32)
    overlay     = (1 - alpha) * image + alpha * heatmap_rgb
    overlay     = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"{title_prefix}  GT: {label_str}", fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(overlay)
    ep_str = f" [Epoch {epoch}]" if epoch is not None else ""
    axes[1].set_title(f"Heatmap{ep_str}  {prob_str}", fontsize=10)
    axes[1].axis('off')

    fig.tight_layout()
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main callback
# ─────────────────────────────────────────────────────────────────────────────

class EpochMetricsAndVisualizationCallback(keras.callbacks.Callback):
    """Compute full classification metrics and save visualizations every epoch.

    This callback operates on the *raw* core model (not the functional fit_model)
    so that it can access encoder-specific outputs (attention, feature_map).

    Args:
        val_dataset          : Validation ``tf.data.Dataset`` yielding (imgs, labels).
        output_dir           : Root output directory.
        last_conv_layer_name : Last Conv2D layer name for Grad-CAM (CNN only).
                               Set to None to skip Grad-CAM.
        core_model           : The raw model with call() → dict output.
                               If None, falls back to ``self.model``.
        sample_images        : Fixed (N, H, W, 3) float32 array for visualization.
        sample_labels        : Corresponding int labels (N,).
        viz_every_n_epochs   : Save visualizations every N epochs (default 1).
        max_viz_samples      : Max samples to visualize per epoch.
        threshold            : Classification threshold for metric computation.
        csv_header_written   : Internal flag — do not set manually.
    """

    _CSV_FIELDS = [
        'epoch', 'accuracy', 'roc_auc',
        'sensitivity', 'specificity', 'ppv', 'npv', 'f1',
        'threshold', 'youden_threshold',
        'youden_sensitivity', 'youden_specificity',
        'tp', 'tn', 'fp', 'fn',
    ]

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        output_dir: str,
        last_conv_layer_name: str | None = None,
        core_model: keras.Model | None = None,
        sample_images: np.ndarray | None = None,
        sample_labels: np.ndarray | None = None,
        viz_every_n_epochs: int = 1,
        max_viz_samples: int = 8,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.val_dataset          = val_dataset
        self.outdir               = pathlib.Path(output_dir)
        self.last_conv_layer_name = last_conv_layer_name
        self.core_model           = core_model
        self.sample_images        = sample_images
        self.sample_labels        = sample_labels if sample_labels is not None else []
        self.viz_every_n_epochs   = viz_every_n_epochs
        self.max_viz_samples      = max_viz_samples
        self.threshold            = threshold
        self._csv_written         = False
        self._grad_model          = None   # built lazily on first epoch

    # ── Keras hook ─────────────────────────────────────────────────────────
    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        core = self.core_model or self.model
        epoch_1 = epoch + 1   # 1-indexed for filenames

        # ── 1. Full-validation metrics ──────────────────────────────────────
        y_true, y_prob = self._predict_val(core)
        if len(y_true) == 0:
            return

        metrics = binary_classification_metrics(
            y_true, y_prob,
            threshold      = self.threshold,
            compute_youden = True,
        )
        metrics['epoch'] = epoch_1

        # Save per-epoch JSON.
        json_path = self.outdir / 'metrics' / f'metrics_epoch_{epoch_1:03d}.json'
        save_metrics_json(metrics, json_path)

        # Append to CSV.
        self._append_csv(metrics, epoch_1)

        # Log to Keras progress bar.
        if logs is not None:
            for k in ('accuracy', 'roc_auc', 'sensitivity', 'specificity', 'ppv', 'npv'):
                logs[f'val_full_{k}'] = metrics.get(k, float('nan'))

        _m = metrics
        print(
            f"  [EpochViz E{epoch_1:03d}] "
            f"acc={_m['accuracy']:.4f}  auc={_m['roc_auc']:.4f}  "
            f"sens={_m['sensitivity']:.4f}  spec={_m['specificity']:.4f}  "
            f"PPV={_m['ppv']:.4f}  NPV={_m['npv']:.4f}"
        )

        # ROC curve PNG.
        roc_path = self.outdir / 'roc' / f'roc_epoch_{epoch_1:03d}.png'
        plot_roc_curve(y_true, y_prob, epoch=epoch_1, save_path=roc_path)

        # ── 2. Visualizations (every N epochs) ──────────────────────────
        if (epoch_1 % self.viz_every_n_epochs != 0) and (epoch_1 != 1):
            return
        if self.sample_images is None or len(self.sample_images) == 0:
            return

        n = min(self.max_viz_samples, len(self.sample_images))
        imgs   = self.sample_images[:n]
        labels = np.array(self.sample_labels[:n], dtype=int) if len(self.sample_labels) else np.zeros(n, int)

        # Get per-sample probabilities from core model.
        probs = self._batch_predict(core, imgs)

        # Lazily build grad model.
        if self._grad_model is None and self.last_conv_layer_name is not None:
            self._grad_model = _build_grad_model(core, self.last_conv_layer_name)
            if self._grad_model is None:
                print("[EpochViz] Grad-CAM model could not be built; skipping Grad-CAM.")

        for i in range(n):
            img   = imgs[i]           # (H, W, 3)
            lbl   = int(labels[i])
            prob  = float(probs[i])

            # ---- Grad-CAM ----
            if self._grad_model is not None:
                heatmap = _make_gradcam_heatmap(self._grad_model, img)
                if heatmap is not None:
                    h, w = img.shape[:2]
                    heatmap_up = np.array(
                        tf.image.resize(heatmap[..., np.newaxis], (h, w))
                    ).squeeze(-1)
                    gcam_path = (
                        self.outdir / 'gradcam'
                        / f'epoch_{epoch_1:03d}_sample_{i:02d}.png'
                    )
                    _save_overlay(
                        image=img, heatmap=heatmap_up,
                        save_path=gcam_path,
                        label=lbl, prob=prob, epoch=epoch_1,
                        title_prefix='GradCAM',
                    )

            # ---- Attention map (ViT/Swin) ----
            attn_map = _extract_attention_map(core, img)
            if attn_map is not None:
                attn_path = (
                    self.outdir / 'attention'
                    / f'epoch_{epoch_1:03d}_sample_{i:02d}.png'
                )
                _save_overlay(
                    image=img, heatmap=attn_map,
                    save_path=attn_path,
                    label=lbl, prob=prob, epoch=epoch_1,
                    title_prefix='Attention',
                )

    # ── Internal helpers ────────────────────────────────────────────────
    def _predict_val(
        self,
        core: keras.Model,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on full val dataset; return (y_true, y_prob)."""
        all_true, all_prob = [], []
        for imgs, lbls in self.val_dataset:
            out = core(imgs, training=False)
            # Support both 'probability' (benchmark) and 'probabilities' (Classifier).
            if 'probability' in out:
                probs = tf.squeeze(out['probability'], axis=-1)   # (B,)
            elif 'probabilities' in out:
                probs = out['probabilities'][:, 1]                # (B,)  class-1
            elif 'logit' in out:
                probs = tf.squeeze(tf.sigmoid(out['logit']), -1)
            elif 'logits' in out:
                probs = tf.sigmoid(out['logits'][:, 1] - out['logits'][:, 0])
            else:
                continue
            all_true.append(lbls.numpy())
            all_prob.append(probs.numpy())
        if not all_true:
            return np.array([]), np.array([])
        return np.concatenate(all_true), np.concatenate(all_prob)

    def _batch_predict(
        self,
        core: keras.Model,
        images: np.ndarray,
    ) -> np.ndarray:
        """Return (N,) probability array for sample images."""
        imgs_t = tf.cast(images, tf.float32)
        out    = core(imgs_t, training=False)
        if 'probability' in out:
            return tf.squeeze(out['probability'], -1).numpy()
        if 'probabilities' in out:
            return out['probabilities'][:, 1].numpy()
        if 'logit' in out:
            return tf.squeeze(tf.sigmoid(out['logit']), -1).numpy()
        if 'logits' in out:
            return tf.sigmoid(out['logits'][:, 1] - out['logits'][:, 0]).numpy()
        return np.full(len(images), 0.5)

    def _append_csv(self, metrics: dict[str, Any], epoch_1: int) -> None:
        """Append one row to metrics_all.csv."""
        csv_path = self.outdir / 'metrics' / 'metrics_all.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._csv_written or not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS, extrasaction='ignore')
            if write_header:
                writer.writeheader()
                self._csv_written = True
            row = {k: metrics.get(k, '') for k in self._CSV_FIELDS}
            row['epoch'] = epoch_1
            writer.writerow(row)
