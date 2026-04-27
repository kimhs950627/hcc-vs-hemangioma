
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf
import keras

try:
    import wandb
except Exception:
    wandb = None


@dataclass
class WandbVisualizationConfig:
    test_dir: str
    num_images: int = 8
    class_names: tuple[str, str] = ("hemangioma", "hcc")
    image_size: tuple[int, int] = (224, 224)
    log_every_n_epochs: int = 1
    overlay_alpha: float = 0.45
    normalize_from_minus1: bool = False
    gradcam_layer_name: str | None = None
    stage: str = "stage1"


def _require_wandb():
    if wandb is None:
        raise ImportError("wandb is required. Install with `pip install wandb`.")


def init_wandb(project: str, run_name: str | None = None, config: dict[str, Any] | None = None, tags: list[str] | None = None):
    _require_wandb()
    if wandb.run is None:
        wandb.init(project=project, name=run_name, config=config or {}, tags=tags or [])
    return wandb.run


def get_wandb_callbacks(log_freq: str = "epoch"):
    _require_wandb()
    from wandb.integration.keras import WandbMetricsLogger
    return [WandbMetricsLogger(log_freq=log_freq)]


def _list_test_images(test_dir: str, num_images: int) -> list[tuple[str, int | None]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    root = Path(test_dir)
    items: list[tuple[str, int | None]] = []
    hemi = root / "hemangioma"
    hcc = root / "hcc"
    if hemi.exists() or hcc.exists():
        for class_name, label in (("hemangioma", 0), ("hcc", 1)):
            cdir = root / class_name
            if not cdir.exists():
                continue
            for p in sorted(cdir.rglob("*")):
                if p.suffix.lower() in exts:
                    items.append((str(p), label))
    else:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in exts:
                items.append((str(p), None))
    return items[:num_images]


def _load_raw_image(path: str, image_size: tuple[int, int]) -> tuple[np.ndarray, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    original = img.numpy()
    resized = tf.image.resize(img, image_size, method="bilinear")
    resized = tf.expand_dims(resized, axis=0)
    return original, resized


def _prepare_display_image(image: np.ndarray, normalize_from_minus1: bool = False) -> np.ndarray:
    x = image.astype("float32")
    if normalize_from_minus1:
        x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype("uint8")


def _normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.asarray(hm, dtype="float32")
    hm = np.maximum(hm, 0.0)
    m = hm.max()
    if m > 0:
        hm = hm / m
    return hm


def _resize_heatmap(hm: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    hm_tf = tf.convert_to_tensor(hm[..., None], dtype=tf.float32)
    hm_tf = tf.image.resize(hm_tf, target_hw, method="bilinear")
    return tf.squeeze(hm_tf, axis=-1).numpy()


def _colormap_heatmap(hm: np.ndarray) -> np.ndarray:
    hm_uint8 = np.uint8(np.clip(hm, 0.0, 1.0) * 255.0)
    cmap = keras.src.utils.image_utils.array_to_img(np.zeros((1,1,3), dtype='uint8'))  # noop to keep keras imported
    import matplotlib.cm as cm
    colored = cm.get_cmap("jet")(hm_uint8)[..., :3]
    return (colored * 255.0).astype("uint8")


def _overlay_image(image_uint8: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_rgb = _colormap_heatmap(hm)
    out = image_uint8.astype("float32") * (1.0 - alpha) + heat_rgb.astype("float32") * alpha
    return np.clip(out, 0, 255).astype("uint8")


def _infer_patch_grid(attn: np.ndarray) -> tuple[int, int]:
    n_tokens = attn.shape[-1]
    n_patches = n_tokens - 1
    side = int(np.sqrt(n_patches))
    if side * side == n_patches:
        return side, side
    for h in range(side, 0, -1):
        if n_patches % h == 0:
            return h, n_patches // h
    return 1, n_patches


def _extract_cls_attention_map(attn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if attn.ndim != 4:
        raise ValueError(f"Expected attention shape [B,H,T,T], got {attn.shape}")
    cls_to_patch = attn[:, :, 0, 1:]
    gh, gw = _infer_patch_grid(attn[0])
    merged = np.mean(cls_to_patch, axis=1)
    merged = merged.reshape(attn.shape[0], gh, gw)
    headwise = cls_to_patch.reshape(attn.shape[0], attn.shape[1], gh, gw)
    return merged, headwise


def _get_attention_from_outputs(outputs: Any) -> np.ndarray | None:
    if isinstance(outputs, dict):
        attn = outputs.get("last_encoder_layer_attentional_weights")
        if attn is None and outputs.get("attention_weights"):
            attn = outputs["attention_weights"][-1]
    else:
        attn = None
    if attn is None:
        return None
    return tf.convert_to_tensor(attn).numpy()


def _get_predictions(outputs: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(outputs, dict):
        return None, None
    probs = outputs.get("probabilities")
    logits = outputs.get("logits")
    probs_np = tf.convert_to_tensor(probs).numpy() if probs is not None else None
    logits_np = tf.convert_to_tensor(logits).numpy() if logits is not None else None
    return probs_np, logits_np


def _find_last_conv_layer(model: keras.Model) -> str | None:
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output.shape
        except Exception:
            continue
        if len(out_shape) == 4:
            return layer.name
        if isinstance(layer, keras.Model):
            nested = _find_last_conv_layer(layer)
            if nested is not None:
                return nested
    return None


def _resolve_gradcam_layer(model: keras.Model, explicit_name: str | None = None) -> str:
    if explicit_name is not None:
        return explicit_name
    name = _find_last_conv_layer(model)
    if name is None:
        raise ValueError("No 4D convolutional feature layer found for Grad-CAM.")
    return name


def make_gradcam_heatmap(model: keras.Model, image_batch: tf.Tensor, class_index: int, layer_name: str) -> np.ndarray:
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output["logits"] if isinstance(model.output, dict) else model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image_batch, training=False)
        class_channel = preds[:, class_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_out = conv_out[0]
    pooled = pooled_grads[0]
    heatmap = tf.reduce_sum(conv_out * pooled[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap)
    heatmap = tf.where(denom > 0, heatmap / denom, heatmap)
    return heatmap.numpy()


class WandbAttentionVisualizer(keras.callbacks.Callback):
    def __init__(self, vis_cfg: WandbVisualizationConfig):
        super().__init__()
        self.cfg = vis_cfg
        self.samples = _list_test_images(vis_cfg.test_dir, vis_cfg.num_images)

    def _forward_for_attention(self, image_batch: tf.Tensor):
        model = self.model
        if hasattr(model, "teacher_encoder"):
            return model.teacher_encoder(image_batch, training=False)
        if hasattr(model, "online_encoder"):
            return model.online_encoder(image_batch, training=False)
        if hasattr(model, "encoder"):
            enc = getattr(model, "encoder")
            if callable(enc):
                return enc(image_batch, training=False)
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            return model.model.encoder(image_batch, training=False)
        return model(image_batch, training=False)

    def _build_stage1_row(self, path: str, true_label: int | None):
        original, resized = _load_raw_image(path, self.cfg.image_size)
        outputs = self._forward_for_attention(resized)
        attn = _get_attention_from_outputs(outputs)
        if attn is None:
            return None
        merged, headwise = _extract_cls_attention_map(attn)
        img_uint8 = _prepare_display_image(original, self.cfg.normalize_from_minus1)
        target_hw = img_uint8.shape[:2]
        merged_up = _resize_heatmap(merged[0], target_hw)
        row = {
            "path": path,
            "label": true_label,
            "raw_image": wandb.Image(img_uint8),
            "merged_attention": wandb.Image(_colormap_heatmap(merged_up)),
            "overlay_merged": wandb.Image(_overlay_image(img_uint8, merged_up, self.cfg.overlay_alpha)),
        }
        for h in range(headwise.shape[1]):
            hm_up = _resize_heatmap(headwise[0, h], target_hw)
            row[f"overlay_head_{h+1}"] = wandb.Image(_overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha))
        return row

    def on_epoch_end(self, epoch, logs=None):
        if wandb is None or wandb.run is None:
            return
        if (epoch + 1) % self.cfg.log_every_n_epochs != 0:
            return
        rows = []
        for path, true_label in self.samples:
            row = self._build_stage1_row(path, true_label)
            if row is not None:
                rows.append(row)
        if not rows:
            return
        columns = list(rows[0].keys())
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[c] for c in columns])
        wandb.log({"stage1_attention_table": table}, commit=False)


class WandbStage2Visualizer(keras.callbacks.Callback):
    def __init__(self, vis_cfg: WandbVisualizationConfig):
        super().__init__()
        self.cfg = vis_cfg
        self.samples = _list_test_images(vis_cfg.test_dir, vis_cfg.num_images)

    def _forward_outputs(self, image_batch: tf.Tensor):
        model = self.model
        out = model(image_batch, training=False)
        if isinstance(out, dict):
            return out
        raise ValueError("Stage2 visualizer expects model outputs as a dict containing logits/probabilities.")

    def _attention_outputs(self, image_batch: tf.Tensor):
        model = self.model
        if hasattr(model, "encoder"):
            return model.encoder(image_batch, training=False)
        if hasattr(model, "model") and hasattr(model.model, "encoder"):
            return model.model.encoder(image_batch, training=False)
        return self._forward_outputs(image_batch)

    def _resolve_model_for_gradcam(self):
        if hasattr(self.model, "model"):
            return self.model.model
        return self.model

    def _build_stage2_row(self, path: str, true_label: int | None):
        original, resized = _load_raw_image(path, self.cfg.image_size)
        img_uint8 = _prepare_display_image(original, self.cfg.normalize_from_minus1)
        outputs = self._forward_outputs(resized)
        probs, logits = _get_predictions(outputs)
        pred_label = int(np.argmax(probs[0])) if probs is not None else None
        attn_outputs = self._attention_outputs(resized)
        attn = _get_attention_from_outputs(attn_outputs)
        row = {
            "path": path,
            "true_label": true_label,
            "pred_label": pred_label,
            "raw_image": wandb.Image(img_uint8),
        }
        if attn is not None:
            merged, headwise = _extract_cls_attention_map(attn)
            merged_up = _resize_heatmap(merged[0], img_uint8.shape[:2])
            row["merged_attention"] = wandb.Image(_colormap_heatmap(merged_up))
            row["overlay_merged"] = wandb.Image(_overlay_image(img_uint8, merged_up, self.cfg.overlay_alpha))
            for h in range(headwise.shape[1]):
                hm_up = _resize_heatmap(headwise[0, h], img_uint8.shape[:2])
                row[f"overlay_head_{h+1}"] = wandb.Image(_overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha))
        grad_model = self._resolve_model_for_gradcam()
        layer_name = _resolve_gradcam_layer(grad_model, self.cfg.gradcam_layer_name)
        for class_idx in (0, 1):
            heatmap = make_gradcam_heatmap(grad_model, resized, class_idx, layer_name)
            hm_up = _resize_heatmap(heatmap, img_uint8.shape[:2])
            row[f"gradcam_class_{class_idx}"] = wandb.Image(_overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha))
        return row

    def on_epoch_end(self, epoch, logs=None):
        if wandb is None or wandb.run is None:
            return
        if (epoch + 1) % self.cfg.log_every_n_epochs != 0:
            return
        rows = []
        for path, true_label in self.samples:
            row = self._build_stage2_row(path, true_label)
            rows.append(row)
        columns = list(rows[0].keys()) if rows else []
        if not columns:
            return
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*[row[c] for c in columns])
        wandb.log({"stage2_visualization_table": table}, commit=False)
