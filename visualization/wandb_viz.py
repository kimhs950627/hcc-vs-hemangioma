from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
import tensorflow as tf
import keras

try:
    import wandb
except Exception:
    wandb = None


AttentionMode = Literal["last_layer", "rollout", "last"]


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
    table_key: str | None = None
    attention_mode: AttentionMode = "last_layer"
    """attention 시각화 모드.

    - ``"last_layer"`` (기본값): 마지막 transformer block의 attention만 사용.
    - ``"rollout"``: 모든 block attention을 residual-augmented matrix product로
      합산하는 Attention Rollout (Abnar & Zuidema, 2020).

    예시::

        cfg = WandbVisualizationConfig(
            test_dir="/data/test",
            attention_mode="rollout",   # rollout 사용
        )
    """
    rollout_head_reduction: Literal["mean", "max"] = "mean"
    rollout_discard_ratio: float = 0.0
    """Rollout 전용 옵션.

    rollout_head_reduction : head 축 집계 방법 ('mean' or 'max')
    rollout_discard_ratio  : 0.0~1.0. 각 layer에서 하위 비율 attention을 0으로 버림
                             (noise 제거). 0.0이면 vanilla rollout.
    """
    enable_eigen: bool = False
    """EigenAttention 시각화 활성화 여부.

    True 로 설정하면 WandB table에 k=1,2,3 eigenattention map 및 overlay columns 이
    기존 rollout/last_layer columns 뒤에 추가로 logging됨.
    선언하지 않으면 False(비활성) 로 간주. 기존 호출 코드 수정 불필요.
    """


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
    hm_min = float(np.min(hm))
    hm_max = float(np.max(hm))
    if hm_max > hm_min:
        hm = (hm - hm_min) / (hm_max - hm_min)
    else:
        hm = np.zeros_like(hm, dtype="float32")
    hm = np.clip(hm, 0.0, 1.0)
    return hm


def _resize_heatmap(hm: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    hm = _normalize_heatmap(hm)
    hm_tf = tf.convert_to_tensor(hm[..., None], dtype=tf.float32)
    hm_tf = tf.image.resize(hm_tf, target_hw, method="bilinear")
    hm_np = tf.squeeze(hm_tf, axis=-1).numpy()
    return _normalize_heatmap(hm_np)


def _colormap_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = _normalize_heatmap(hm)
    hm_uint8 = np.uint8(hm * 255.0)
    import matplotlib.cm as cm
    colored = cm.get_cmap("jet")(hm_uint8.astype(np.float32) / 255.0)[..., :3]
    return np.uint8(np.clip(colored * 255.0, 0.0, 255.0))


def _overlay_image(image_uint8: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hm = _normalize_heatmap(hm)
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
    """Last-layer attention: CLS row → patch grid.

    Args:
        attn: [B, H, T, T] — single layer attention (last block)
    Returns:
        merged   : [B, gh, gw]    — head-mean CLS→patch map
        headwise : [B, H, gh, gw] — per-head CLS→patch map
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected attention shape [B,H,T,T], got {attn.shape}")
    cls_to_patch = attn[:, :, 0, 1:]  # [B, H, N]
    gh, gw = _infer_patch_grid(attn[0])
    merged = np.mean(cls_to_patch, axis=1).reshape(attn.shape[0], gh, gw)
    headwise = cls_to_patch.reshape(attn.shape[0], attn.shape[1], gh, gw)
    return merged, headwise


# ── Attention Rollout (Abnar & Zuidema, 2020) ─────────────────────────────

def compute_attention_rollout(
    attn_list: list[np.ndarray],
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Attention Rollout across all transformer blocks.

    Args:
        attn_list      : List[[B, H, N+1, N+1]] — 모든 block의 attention weight list.
                         encoder output dict의 'attention_weights' 를 그대로 전달.
        head_reduction : 'mean' | 'max' — head 축 집계 방법.
        discard_ratio  : 0.0~1.0 — 각 layer에서 하위 비율 attention을 0으로 버림
                         (noise 제거용). 0.0이면 vanilla rollout.
    Returns:
        rollout : [B, N+1, N+1] — residual-augmented layer-product rollout map.
                  CLS row (rollout[:, 0, 1:]) 가 최종 시각화에 사용됨.
    """
    if head_reduction == "mean":
        mats = [np.mean(a, axis=1) for a in attn_list]   # [B, T, T] per layer
    else:
        mats = [np.max(a, axis=1) for a in attn_list]

    if discard_ratio > 0.0:
        cleaned = []
        for mat in mats:
            flat = mat.reshape(mat.shape[0], -1)
            thresh = np.quantile(flat, discard_ratio, axis=-1, keepdims=True)
            thresh = thresh.reshape(mat.shape[0], 1, 1)
            cleaned.append(np.where(mat >= thresh, mat, 0.0))
        mats = cleaned

    n_tokens = mats[0].shape[-1]
    eye = np.eye(n_tokens, dtype=np.float32)
    augmented = []
    for mat in mats:
        aug = 0.5 * mat + 0.5 * eye[None]           # residual connection 반영
        aug = aug / (aug.sum(axis=-1, keepdims=True) + 1e-8)  # row-normalize
        augmented.append(aug)

    rollout = augmented[0]
    for mat in augmented[1:]:
        rollout = np.einsum("bij,bjk->bik", mat, rollout)  # [B,T,T]

    return rollout


# ── EigenAttention ────────────────────────────────────────────────────────

_EIGEN_K = 3  # hardcoded


def compute_eigen_attention(
    attn_list: list[np.ndarray],
    k: int = _EIGEN_K,
    head_reduction: str = "mean",
) -> np.ndarray:
    """EigenAttention: patch-to-patch rollout matrix의 top-k eigenvector 반환.

    Args:
        attn_list      : List[[B, H, N+1, N+1]] — 모든 block attention list
        k              : 반환할 eigenvector 수 (기본값 = 3, 하드코딩)
        head_reduction : 'mean' | 'max'

    Returns:
        eigenmaps : [B, k, N] — reshape(gh, gw) 가능한 patch-level eigenmap.
                   부호는 max absolute value 원소가 양수가 되도록 flip됨.
    """
    if head_reduction == "mean":
        mats = [np.mean(a, axis=1) for a in attn_list]
    else:
        mats = [np.max(a, axis=1) for a in attn_list]

    n_tokens = mats[0].shape[-1]
    eye = np.eye(n_tokens, dtype=np.float32)
    rollout = eye[None].repeat(mats[0].shape[0], axis=0)
    for mat in mats:
        aug = 0.5 * mat + 0.5 * eye[None]
        aug = aug / (aug.sum(axis=-1, keepdims=True) + 1e-8)
        rollout = np.einsum("bij,bjk->bik", aug, rollout)

    # patch-to-patch sub-matrix [B, N, N] (CLS row/col 제거)
    patch_attn = rollout[:, 1:, 1:]
    B, N, _ = patch_attn.shape

    # symmetrize for numerically stable eigh
    sym = 0.5 * (patch_attn + patch_attn.transpose(0, 2, 1))  # [B, N, N]

    eigenmaps = np.zeros((B, k, N), dtype=np.float32)
    for b in range(B):
        vals, vecs = np.linalg.eigh(sym[b])   # ascending eigenvalue order
        # top-k: last k (largest eigenvalues) → reverse to descending
        topk_vecs = vecs[:, -k:][:, ::-1]     # [N, k]
        for ki in range(k):
            ev = topk_vecs[:, ki]
            # sign convention: max-abs element 이 양수가 되도록 flip
            if ev[np.argmax(np.abs(ev))] < 0:
                ev = -ev
            eigenmaps[b, ki] = ev.astype(np.float32)

    return eigenmaps   # [B, k, N]


def _append_eigen_to_row(
    row: dict,
    outputs: Any,
    img_uint8: np.ndarray,
    head_reduction: str,
    alpha: float,
) -> dict:
    """outputs에서 eigenattention을 계산해 row dict에 컬럼을 추가하고 반환.

    enable_eigen=True 인 경우에만 호출됨. 실패 시 row를 그대로 반환.
    """
    attn_list = _get_attention_list_from_outputs(outputs)
    if attn_list is None:
        return row
    try:
        eigenmaps = compute_eigen_attention(attn_list, k=_EIGEN_K, head_reduction=head_reduction)
        gh, gw = _infer_patch_grid(attn_list[-1][0])
        target_hw = img_uint8.shape[:2]
        for ki in range(_EIGEN_K):
            ev = eigenmaps[0, ki].reshape(gh, gw)   # [gh, gw]
            ev_up = _resize_heatmap(ev, target_hw)
            row[f"eigenattn_k{ki + 1}"] = wandb.Image(_colormap_heatmap(ev_up))
            row[f"eigen_overlay_k{ki + 1}"] = wandb.Image(
                _overlay_image(img_uint8, ev_up, alpha)
            )
    except Exception as e:
        print(f"[EigenAttention] 계산 실패 (skip): {e}")
    return row


def _get_attention_list_from_outputs(outputs: Any) -> list[np.ndarray] | None:
    """encoder output dict에서 모든 layer attention list를 추출.

    'attention_weights' (full list) 우선, 없으면 'last_encoder_layer_attentional_weights'
    를 1-element list로 포장해서 반환.
    """
    if not isinstance(outputs, dict):
        return None
    attn_list = outputs.get("attention_weights")
    if attn_list and isinstance(attn_list, (list, tuple)) and len(attn_list) > 0:
        return [tf.convert_to_tensor(a).numpy() for a in attn_list]
    attn_last = outputs.get("last_encoder_layer_attentional_weights")
    if attn_last is not None:
        return [tf.convert_to_tensor(attn_last).numpy()]
    return None


def _get_attention_from_outputs(outputs: Any) -> np.ndarray | None:
    """Last-layer attention 단일 array 반환 (기존 호환)."""
    if isinstance(outputs, dict):
        attn = outputs.get("last_encoder_layer_attentional_weights")
        if attn is None and outputs.get("attention_weights"):
            attn = outputs["attention_weights"][-1]
    else:
        attn = None
    if attn is None:
        return None
    return tf.convert_to_tensor(attn).numpy()


def _extract_attention_maps(
    outputs: Any,
    mode: AttentionMode = "last_layer",
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """attention_mode에 따라 (merged [B,gh,gw], headwise [B,H,gh,gw]) 반환.

    Args:
        outputs        : encoder forward output dict
        mode           : 'last_layer' (기본값) 또는 'rollout'
        head_reduction : rollout 전용 — head 축 집계 방법
        discard_ratio  : rollout 전용 — 하위 noise 비율 제거
    Returns:
        (merged, headwise) 또는 None (attention 없을 때)
    """
    normalized_mode = "last_layer" if mode == "last" else mode

    if normalized_mode == "rollout":
        attn_list = _get_attention_list_from_outputs(outputs)
        if attn_list is None:
            return None
        # rollout: all layers
        rollout = compute_attention_rollout(attn_list, head_reduction, discard_ratio)
        cls_rollout = rollout[:, 0, 1:]               # [B, N]
        gh, gw = _infer_patch_grid(attn_list[-1][0])
        merged = cls_rollout.reshape(rollout.shape[0], gh, gw)
        # headwise: last layer 그대로 (per-head column은 last layer 기준 유지)
        last = attn_list[-1]                           # [B, H, T, T]
        cls_last = last[:, :, 0, 1:]                  # [B, H, N]
        headwise = cls_last.reshape(last.shape[0], last.shape[1], gh, gw)
        return merged, headwise

    if normalized_mode == "last_layer":
        attn = _get_attention_from_outputs(outputs)
        if attn is None:
            return None
        return _extract_cls_attention_map(attn)

    raise ValueError(
        f"Unsupported attention_mode: {mode!r}. "
        "Use 'rollout' or 'last_layer' (alias: 'last')."
    )


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


def _render_classification_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
) -> np.ndarray:
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=20, ha='right')
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Classification heatmap')
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def _classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return report_text, report_dict


_DEFAULT_ATTENTION_TABLE_KEY = "stage1_attention_table"


class WandbAttentionVisualizer(keras.callbacks.Callback):
    """Stage1 attention visualization callback.

    attention_mode 옵션:
        - ``"last_layer"`` (기본값): 마지막 block attention만 사용.
        - ``"rollout"``: Attention Rollout (모든 block 합산).

    enable_eigen=True 시 k=1,2,3 EigenAttention map 및 overlay가
    기존 columns 뒤에 추가 logging됨. 선언하지 않으면 False로 간주.

    사용 예시::

        # last_layer (기본값, eigen 없음)
        cb = WandbAttentionVisualizer(WandbVisualizationConfig(
            test_dir="/data/test",
        ))

        # rollout + eigen 활성화
        cb = WandbAttentionVisualizer(WandbVisualizationConfig(
            test_dir="/data/test",
            attention_mode="rollout",
            enable_eigen=True,
        ))
    """

    def __init__(self, vis_cfg: WandbVisualizationConfig):
        super().__init__()
        self.cfg = vis_cfg
        self.samples = _list_test_images(vis_cfg.test_dir, vis_cfg.num_images)
        self._table_key: str = vis_cfg.table_key if vis_cfg.table_key is not None else _DEFAULT_ATTENTION_TABLE_KEY

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

        result = _extract_attention_maps(
            outputs,
            mode=self.cfg.attention_mode,
            head_reduction=self.cfg.rollout_head_reduction,
            discard_ratio=self.cfg.rollout_discard_ratio,
        )
        if result is None:
            return None
        merged, headwise = result

        img_uint8 = _prepare_display_image(original, self.cfg.normalize_from_minus1)
        target_hw = img_uint8.shape[:2]
        merged_up = _resize_heatmap(merged[0], target_hw)

        mode_label = self.cfg.attention_mode  # 'last_layer' or 'rollout'
        row = {
            "path": path,
            "label": true_label,
            "raw_image": wandb.Image(img_uint8),
            f"attention_map_{mode_label}": wandb.Image(_colormap_heatmap(merged_up)),
            f"overlay_{mode_label}": wandb.Image(_overlay_image(img_uint8, merged_up, self.cfg.overlay_alpha)),
        }
        for h in range(headwise.shape[1]):
            hm_up = _resize_heatmap(headwise[0, h], target_hw)
            row[f"overlay_head_{h+1}"] = wandb.Image(_overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha))

        # ── EigenAttention k=1,2,3 (enable_eigen=True 시에만) ─────────────
        if getattr(self.cfg, "enable_eigen", False):
            row = _append_eigen_to_row(
                row, outputs, img_uint8,
                head_reduction=self.cfg.rollout_head_reduction,
                alpha=self.cfg.overlay_alpha,
            )

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
        wandb.log({self._table_key: table}, commit=False)


class WandbStage2Visualizer(keras.callbacks.Callback):
    """Stage2 classification visualization callback.

    attention_mode 옵션:
        - ``"last_layer"`` (기본값): 마지막 block attention만 사용.
        - ``"rollout"``: Attention Rollout (모든 block 합산).

    enable_eigen=True 시 k=1,2,3 EigenAttention map 및 overlay가
    기존 columns 뒤에 추가 logging됨. 선언하지 않으면 False로 간주.
    """

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

        row = {
            "path": path,
            "true_label": true_label,
            "pred_label": pred_label,
            "raw_image": wandb.Image(img_uint8),
        }

        result = _extract_attention_maps(
            attn_outputs,
            mode=self.cfg.attention_mode,
            head_reduction=self.cfg.rollout_head_reduction,
            discard_ratio=self.cfg.rollout_discard_ratio,
        )
        if result is not None:
            merged, headwise = result
            mode_label = self.cfg.attention_mode
            merged_up = _resize_heatmap(merged[0], img_uint8.shape[:2])
            row[f"attention_map_{mode_label}"] = wandb.Image(_colormap_heatmap(merged_up))
            row[f"overlay_{mode_label}"] = wandb.Image(_overlay_image(img_uint8, merged_up, self.cfg.overlay_alpha))
            for h in range(headwise.shape[1]):
                hm_up = _resize_heatmap(headwise[0, h], img_uint8.shape[:2])
                row[f"overlay_head_{h+1}"] = wandb.Image(_overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha))

        # ── EigenAttention k=1,2,3 (enable_eigen=True 시에만) ─────────────
        if getattr(self.cfg, "enable_eigen", False):
            row = _append_eigen_to_row(
                row, attn_outputs, img_uint8,
                head_reduction=self.cfg.rollout_head_reduction,
                alpha=self.cfg.overlay_alpha,
            )

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


class WandbBenchmarkVisualizer(keras.callbacks.Callback):
    def __init__(self, vis_cfg: WandbVisualizationConfig, test_dataset: tf.data.Dataset):
        super().__init__()
        self.cfg = vis_cfg
        self.test_dataset = test_dataset

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        if wandb is None or wandb.run is None:
            return
        if (epoch + 1) % self.cfg.log_every_n_epochs != 0:
            return

        y_true_parts: list[np.ndarray] = []
        y_prob_parts: list[np.ndarray] = []
        for images, labels in self.test_dataset:
            outputs = self.model(images, training=False)
            if isinstance(outputs, dict):
                probs = outputs.get('probability', outputs.get('probabilities'))
            else:
                probs = outputs
            probs_np = tf.reshape(tf.convert_to_tensor(probs), (-1,)).numpy()
            y_true_parts.append(tf.reshape(labels, (-1,)).numpy())
            y_prob_parts.append(probs_np)

        if not y_true_parts:
            return

        y_true_np = np.concatenate(y_true_parts).astype(np.int32)
        y_prob_np = np.concatenate(y_prob_parts).astype(np.float32)
        y_pred_np = (y_prob_np >= 0.5).astype(np.int32)

        heatmap_img = _render_classification_heatmap(y_true_np, y_pred_np, self.cfg.class_names)
        report_text, report_dict = _classification_summary(y_true_np, y_pred_np, self.cfg.class_names)

        flat_metrics: dict[str, float] = {}
        for key, value in report_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_metrics[f'benchmark_report/{key}/{sub_key}'] = float(sub_val)
            elif isinstance(value, (int, float)):
                flat_metrics[f'benchmark_report/{key}'] = float(value)

        wandb.log({
            'benchmark/classification_heatmap': wandb.Image(heatmap_img, caption=f'Epoch {epoch + 1}'),
            'benchmark/classification_report_text': wandb.Html(f'<pre>{report_text}</pre>'),
            **flat_metrics,
        }, commit=False)
