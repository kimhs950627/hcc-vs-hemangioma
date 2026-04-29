"""
callbacks/epoch_visualization.py

Keras callback that runs at the end of every epoch and saves:
  1. Full validation metrics (accuracy, ROC-AUC, sensitivity, specificity, PPV, NPV)
     -> outputs/<run>/metrics/metrics_epoch_{epoch:03d}.json
     -> outputs/<run>/metrics/metrics_all.csv  (append every epoch)
  2. ROC curve PNG
     -> outputs/<run>/roc/roc_epoch_{epoch:03d}.png
  3. Grad-CAM / attention overlay grid for sample images
     -> outputs/<run>/heatmap/heatmap_epoch_{epoch:03d}.png

Design decisions
----------------
- No sklearn dependency: metrics are computed with utils.metrics.
- No cv2 / PIL dependency: resizing is done via tf.image.resize.
- Attention extraction is best-effort:
    - ViT family: tries to call model with return_attention=True.
    - CNN family: falls back to Grad-CAM via utils.gradcam.
- Callback is safe to use with both benchmark_supervised and stage2 models.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


class EpochMetricsAndVisualizationCallback:
    """Per-epoch validation metrics + Grad-CAM / attention visualizations.

    Parameters
    ----------
    val_dataset :
        A ``tf.data.Dataset`` yielding ``(images, labels)`` batches.
        Labels should be integer scalars (0 or 1).
    output_dir :
        Root directory for all saved outputs.
    last_conv_layer_name :
        Name of the last convolutional layer for Grad-CAM.  Pass ``None``
        for pure Transformer models (gradient × input saliency is used instead).
    sample_images :
        Optional float32 array (M, H, W, C) of *fixed* sample images to
        visualize each epoch.  If ``None``, the first batch of val_dataset
        is used.
    sample_labels :
        Optional int array (M,) of ground-truth labels for sample_images.
    threshold :
        Hard decision threshold for confusion-matrix metrics.
    max_visualizations :
        Maximum number of images in the per-epoch heatmap grid.
    log_to_console :
        Print a one-line metric summary after each epoch.
    """

    def __init__(
        self,
        val_dataset,
        output_dir: str | Path,
        last_conv_layer_name: Optional[str] = None,
        sample_images: Optional[np.ndarray] = None,
        sample_labels: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        max_visualizations: int = 8,
        log_to_console: bool = True,
    ):
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.last_conv_layer_name = last_conv_layer_name
        self.sample_images = sample_images
        self.sample_labels = sample_labels
        self.threshold = threshold
        self.max_vis = max_visualizations
        self.log_to_console = log_to_console
        self._csv_path: Optional[Path] = None
        self._csv_header_written = False

    # ───────────────────────────────────────────────────────────────────
    # Keras callback interface
    # ───────────────────────────────────────────────────────────────────

    def set_model(self, model):
        """Called automatically by Keras before training starts."""
        self.model = model

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Run after every epoch."""
        # ── 1. Collect validation predictions ─────────────────────────────
        all_probs, all_labels = self._collect_predictions()

        # ── 2. Compute metrics ────────────────────────────────────────────
        from utils.metrics import (
            binary_classification_metrics,
            save_metrics_json,
            format_metrics_string,
        )
        metrics = binary_classification_metrics(
            all_labels, all_probs, threshold=self.threshold, compute_youden=True
        )
        metrics["epoch"] = epoch + 1

        # ── 3. Save JSON + CSV ────────────────────────────────────────────
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        json_path = metrics_dir / f"metrics_epoch_{epoch + 1:03d}.json"
        save_metrics_json(metrics, json_path)
        self._append_csv(metrics, metrics_dir / "metrics_all.csv")

        # ── 4. ROC curve PNG ───────────────────────────────────────────────
        roc_dir = self.output_dir / "roc"
        roc_dir.mkdir(parents=True, exist_ok=True)
        self._save_roc_curve(
            metrics,
            roc_dir / f"roc_epoch_{epoch + 1:03d}.png",
            epoch=epoch + 1,
        )

        # ── 5. Heatmap visualization ──────────────────────────────────────
        heatmap_dir = self.output_dir / "heatmap"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        self._save_heatmap_grid(
            heatmap_dir / f"heatmap_epoch_{epoch + 1:03d}.png",
            all_probs,
        )

        # ── 6. Console log ─────────────────────────────────────────────────
        if self.log_to_console:
            print(f"\n[EpochViz] epoch={epoch+1:03d}  {format_metrics_string(metrics)}")

        # Propagate to Keras logs dict so ModelCheckpoint can monitor them
        if logs is not None:
            for key in ("accuracy", "roc_auc", "sensitivity", "specificity", "ppv", "npv"):
                logs[f"val_{key}"] = metrics.get(key, 0.0)

    # ───────────────────────────────────────────────────────────────────
    # Private helpers
    # ───────────────────────────────────────────────────────────────────

    def _collect_predictions(self):
        import tensorflow as tf

        all_probs, all_labels = [], []
        for batch in self.val_dataset:
            imgs, labels = batch
            logits = self.model(imgs, training=False)
            if hasattr(logits, "numpy"):
                logits_np = logits.numpy()
            else:
                logits_np = np.array(logits)
            # Handle both single-logit (shape N,1 or N,) and two-class (shape N,2) outputs
            if logits_np.ndim == 1 or logits_np.shape[-1] == 1:
                probs = _sigmoid(logits_np.ravel())
            else:
                probs = _softmax(logits_np)[:, 1]
            all_probs.append(probs)
            all_labels.append(np.array(labels).ravel())

        return np.concatenate(all_probs), np.concatenate(all_labels)

    def _save_roc_curve(
        self, metrics: dict, path: Path, epoch: int
    ) -> None:
        import matplotlib.pyplot as plt

        fpr = metrics.get("roc_fpr", [])
        tpr = metrics.get("roc_tpr", [])
        auc = metrics.get("roc_auc", 0.0)
        best_thresh = metrics.get("youden_threshold", self.threshold)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

        # Mark Youden-optimal point
        if fpr and tpr:
            youden_vals = [t - f for t, f in zip(tpr, fpr)]
            best_idx = int(np.argmax(youden_vals))
            ax.scatter(
                fpr[best_idx], tpr[best_idx],
                color="red", zorder=5,
                label=f"Youden thresh={best_thresh:.3f}",
            )

        ax.set_xlabel("FPR (1 - Specificity)")
        ax.set_ylabel("TPR (Sensitivity)")
        ax.set_title(f"ROC Curve  Epoch {epoch}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(path), dpi=120)
        plt.close(fig)

    def _save_heatmap_grid(self, path: Path, all_probs: np.ndarray) -> None:
        from utils.gradcam import make_gradcam_heatmap, save_gradcam_grid

        # Pick sample images
        if self.sample_images is not None:
            imgs = self.sample_images[: self.max_vis]
            labels = (
                self.sample_labels[: self.max_vis]
                if self.sample_labels is not None
                else None
            )
        else:
            imgs, labels = self._take_sample_batch()

        if imgs is None or len(imgs) == 0:
            return

        # Compute per-sample probs from model
        import tensorflow as tf

        logits = self.model(tf.cast(imgs, tf.float32), training=False)
        logits_np = logits.numpy() if hasattr(logits, "numpy") else np.array(logits)
        if logits_np.ndim == 1 or logits_np.shape[-1] == 1:
            probs = _sigmoid(logits_np.ravel())
        else:
            probs = _softmax(logits_np)[:, 1]

        # Build title strings
        class_names = ["Hemangioma", "HCC"]
        titles = []
        for i in range(len(imgs)):
            pred_class = int(probs[i] >= self.threshold)
            gt_str = class_names[int(labels[i])] if labels is not None else "?"
            titles.append(f"GT:{gt_str} | P(HCC)={probs[i]:.2f}")

        # Grad-CAM
        heatmaps = make_gradcam_heatmap(
            self.model, imgs, self.last_conv_layer_name, class_index=1
        )
        save_gradcam_grid(imgs, heatmaps, path, titles=titles)

    def _take_sample_batch(self):
        """Grab the first batch from val_dataset as sample images."""
        try:
            for batch in self.val_dataset.take(1):
                imgs, labels = batch
                imgs_np = np.array(imgs)[: self.max_vis]
                labels_np = np.array(labels)[: self.max_vis]
                return imgs_np, labels_np
        except Exception:
            pass
        return None, None

    def _append_csv(self, metrics: dict, csv_path: Path) -> None:
        """Append one row to the running metrics CSV."""
        skip_keys = {"roc_fpr", "roc_tpr"}
        row = {k: v for k, v in metrics.items() if k not in skip_keys}
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Keras callback adapter
# (wraps the above class so Keras can call it via fit(callbacks=[...]))
# ─────────────────────────────────────────────────────────────────────────────

def build_epoch_visualization_callback(
    val_dataset,
    output_dir: str | Path,
    last_conv_layer_name: Optional[str] = None,
    sample_images: Optional[np.ndarray] = None,
    sample_labels: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    max_visualizations: int = 8,
    log_to_console: bool = True,
):
    """Factory that returns a Keras-compatible callback.

    Internally imports keras and subclasses keras.callbacks.Callback so that
    the heavy import is deferred until training actually starts.

    Returns
    -------
    A ``keras.callbacks.Callback`` instance ready to be passed to
    ``model.fit(callbacks=[...])``.
    """
    import keras

    _inner = EpochMetricsAndVisualizationCallback(
        val_dataset=val_dataset,
        output_dir=output_dir,
        last_conv_layer_name=last_conv_layer_name,
        sample_images=sample_images,
        sample_labels=sample_labels,
        threshold=threshold,
        max_visualizations=max_visualizations,
        log_to_console=log_to_console,
    )

    class _KerasAdapter(keras.callbacks.Callback):
        def set_model(self, model):
            super().set_model(model)
            _inner.set_model(model)

        def on_epoch_end(self, epoch, logs=None):
            _inner.on_epoch_end(epoch, logs)

    return _KerasAdapter()


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers (no sklearn dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
