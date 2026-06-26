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
TokenAttentionMode = Literal["sa", "ca"]

# ---------------------------------------------------------------------------
# Shape contract (conv-hybrid-ViT encoder)
# ---------------------------------------------------------------------------
# SA mode  (token_attention_mode='sa'):
#   attention_weights: List[ [B, H, seq_len, seq_len] ]  × n_layers
#   seq_len = 1 + N_patches  (CLS prepended, full square matrix)
#
# CA mode  (token_attention_mode='ca'):
#   attention_weights: List[ [B, H, 1, N_patches] ]  × n_layers
#   Q dim = 1 (CLS query only), K dim = N_patches
#
# _infer_patch_grid, _extract_cls_attention_map, compute_attention_rollout
# already handle both shapes via Q==1 branching.
#
# Heatmap normalization — DINO-style per-image mass threshold:
#   1. Flatten 2-D patch map to 1-D for a SINGLE image.
#   2. Sort descending, compute cumulative sum.
#   3. Find cutoff index where cumsum >= mass_threshold (default 0.75).
#   4. Zero out patches below cutoff, then rescale surviving values to [0,1].
#   Rationale: highlights the minimal set of patches that collectively carry
#   mass_threshold fraction of the total attention mass, suppressing diffuse
#   background noise — identical to the DINO paper's per-head visualization.
#   Processing is per-image (not per-batch) so each image's own distribution
#   sets the threshold independently.
# ---------------------------------------------------------------------------


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
    - ``"rollout"``: 모든 block attention을 layer-wise head-mean accumulation으로
      합산 (conv-hybrid-ViT 전용 fallback — 비정방 행렬이므로 행렬곱 rollout 불가).

    예시::

        cfg = WandbVisualizationConfig(
            test_dir="/data/test",
            attention_mode="rollout",
        )
    """
    rollout_head_reduction: Literal["mean", "max"] = "mean"
    rollout_discard_ratio: float = 0.0
    """Rollout 전용 옵션.

    rollout_head_reduction : head 축 집계 방법 ('mean' or 'max')
    rollout_discard_ratio  : 0.0~1.0. 각 layer에서 하위 비율 attention을 0으로 버림
                             (noise 제거). 0.0이면 accumulation 그대로.
    """
    enable_eigen: bool = False
    """EigenAttention 시각화 활성화 여부.

    True 로 설정하면 WandB table에 k=1,2,3 eigenattention map 및 overlay columns 이
    기존 rollout/last_layer columns 뒤에 추가로 logging됨.
    conv-hybrid-ViT에서는 patch-to-patch 정방 행렬이 없으므로
    rollout 누적 map을 PCA decompose하는 fallback으로 동작함.
    """
    token_attention_mode: TokenAttentionMode = "sa"
    """Encoder token attention 모드.

    cfg.token_attention_mode 와 동일한 값을 넣어야 함.

    - ``"sa"`` (기본값): Self-Attention — 기존 동작 그대로.
      attention_weights shape: [B, H, seq_len, seq_len]
      wandb column suffix: ``_SA``
    - ``"ca"``: Cross-Attention — Q=CLS(GAP), K/V=patches.
      attention_weights shape: [B, H, 1, N_patches]
      wandb column suffix: ``_CA``

    이 값에 따라 wandb table column 이름에 접미사가 붙어
    SA/CA 실험 결과를 구분하여 로깅함.
    SA 모드에서는 기존 코드와 column 이름이 완전히 동일하여 하위 호환됨.
    """
    mass_threshold: float = 0.75
    """DINO-style per-image attention mass threshold.

    각 이미지별로 독립적으로 적용되는 cumulative-mass cutoff.
    0.0~1.0 범위. 기본값 0.75.

    동작 방식:
        1. 2-D patch map을 1-D로 flatten.
        2. 내림차순 정렬 후 cumulative sum 계산.
        3. cumsum >= mass_threshold 를 처음 만족하는 index 찾기.
        4. 해당 index 미만의 patch 값은 유지, 이후는 0으로 masking.
        5. 생존 값을 [0, 1]로 재정규화. 모두 0이면 min-max fallback.

    주의: batch 단위가 아닌 image 단위로 처리되므로
    각 이미지의 attention 분포가 독립적으로 threshold를 결정함.
    """

    # ── derived helper ────────────────────────────────────────────────────
    @property
    def mode_suffix(self) -> str:
        """wandb column/table 이름에 붙는 접미사 (예: '_SA', '_CA')."""
        return f"_{self.token_attention_mode.upper()}"


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


def _log_token_attention_mode_once(token_attention_mode: str) -> None:
    """wandb run config에 token_attention_mode를 기록 (최초 1회)."""
    if wandb is None or wandb.run is None:
        return
    if "token_attention_mode" not in (wandb.run.config or {}):
        try:
            wandb.run.config.update(
                {"token_attention_mode": token_attention_mode}, allow_val_change=True
            )
        except Exception:
            pass


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


# ── DINO-style per-image mass threshold normalization ─────────────────────

def _normalize_heatmap_dino(hm: np.ndarray, mass_threshold: float = 0.75) -> np.ndarray:
    """DINO-style per-image cumulative mass threshold normalization.

    각 이미지의 attention 분포를 기준으로 독립적으로 threshold를 결정함.
    batch 단위 처리 없음 — 반드시 단일 이미지 2-D (또는 임의 shape) 입력.

    알고리즘:
        1. 1-D flatten.
        2. 내림차순 정렬 → cumulative sum.
        3. cumsum / total_sum >= mass_threshold 를 처음 만족하는 index cutoff 결정.
        4. cutoff 이하 순위(즉, 상위 mass_threshold 비율) 패치 값 유지, 나머지 0.
        5. 생존 값 [0,1] 재정규화. 전체 0이면 min-max fallback.

    Args:
        hm             : 임의 shape의 float array (단일 이미지).
        mass_threshold : cumulative mass cutoff. 기본값 0.75.

    Returns:
        normalized heatmap, same shape as input, values in [0, 1].
    """
    hm = np.asarray(hm, dtype="float32")
    original_shape = hm.shape
    flat = hm.flatten()                          # 1-D

    total = float(flat.sum())
    if total <= 0.0:
        # all-zero fallback
        return np.zeros(original_shape, dtype="float32")

    # 내림차순 정렬 → cumsum
    sorted_indices = np.argsort(flat)[::-1]      # descending
    sorted_vals = flat[sorted_indices]
    cumsum = np.cumsum(sorted_vals) / total

    # cumsum >= mass_threshold 를 처음 만족하는 rank (inclusive)
    cutoff_rank = int(np.searchsorted(cumsum, mass_threshold))
    cutoff_rank = min(cutoff_rank, len(flat) - 1)

    # cutoff_rank + 1 개의 top-rank 패치만 유지
    threshold_val = sorted_vals[cutoff_rank]

    masked = np.where(flat >= threshold_val, flat, 0.0)

    # [0, 1] 재정규화
    m_max = float(masked.max())
    if m_max > 0.0:
        masked = masked / m_max
    else:
        # fallback: min-max
        hm_min = float(flat.min())
        hm_max = float(flat.max())
        if hm_max > hm_min:
            masked = (flat - hm_min) / (hm_max - hm_min)
        else:
            masked = np.zeros_like(flat, dtype="float32")

    return np.clip(masked.reshape(original_shape), 0.0, 1.0)


def _resize_heatmap(
    hm: np.ndarray,
    target_hw: tuple[int, int],
    mass_threshold: float = 0.75,
) -> np.ndarray:
    """DINO-style normalize → bilinear upsample → DINO-style normalize again."""
    hm = _normalize_heatmap_dino(hm, mass_threshold)
    hm_tf = tf.convert_to_tensor(hm[..., None], dtype=tf.float32)
    hm_tf = tf.image.resize(hm_tf, target_hw, method="bilinear")
    hm_np = tf.squeeze(hm_tf, axis=-1).numpy()
    return _normalize_heatmap_dino(hm_np, mass_threshold)


def _colormap_heatmap(
    hm: np.ndarray,
    mass_threshold: float = 0.75,
) -> np.ndarray:
    hm = _normalize_heatmap_dino(hm, mass_threshold)
    hm_uint8 = np.uint8(hm * 255.0)
    import matplotlib.cm as cm
    colored = cm.get_cmap("jet")(hm_uint8.astype(np.float32) / 255.0)[..., :3]
    return np.uint8(np.clip(colored * 255.0, 0.0, 255.0))


def _overlay_image(
    image_uint8: np.ndarray,
    hm: np.ndarray,
    alpha: float = 0.45,
    mass_threshold: float = 0.75,
) -> np.ndarray:
    hm = _normalize_heatmap_dino(hm, mass_threshold)
    heat_rgb = _colormap_heatmap(hm, mass_threshold=1.0)   # already normalized above
    out = image_uint8.astype("float32") * (1.0 - alpha) + heat_rgb.astype("float32") * alpha
    return np.clip(out, 0, 255).astype("uint8")


# ── Patch grid inference ───────────────────────────────────────────────────

def _infer_patch_grid(attn_single: np.ndarray) -> tuple[int, int]:
    """n_patches → (gh, gw) 추론.

    Args:
        attn_single: 단일 샘플 attention array.
                     SA mode — [H, T, T]  : n_patches = T-1 (하위 호환)
                     CA mode — [H, 1, N]  : n_patches = N
                     squeezed — [H, N]    : n_patches = N
    """
    shape = attn_single.shape
    if len(shape) == 3:
        # CA mode: [H, 1, N]
        if shape[1] == 1:
            n_patches = shape[2]
        else:
            # SA mode: [H, T, T] → T-1 patches
            n_patches = shape[1] - 1
    elif len(shape) == 2:
        n_patches = shape[1]
    else:
        n_patches = int(attn_single.shape[-1])

    side = int(np.sqrt(n_patches))
    if side * side == n_patches:
        return side, side
    for h in range(side, 0, -1):
        if n_patches % h == 0:
            return h, n_patches // h
    return 1, n_patches


# ── CLS→patch attention extraction ────────────────────────────────────────

def _extract_cls_attention_map(attn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CLS→patch attention map 추출.

    SA mode shape: [B, H, T, T]  (T = 1 + N_patches)
    CA mode shape: [B, H, 1, N]

    Returns:
        merged   : [B, gh, gw]    — head-mean CLS→patch map
        headwise : [B, H, gh, gw] — per-head CLS→patch map
    """
    if attn.ndim != 4:
        raise ValueError(f"Expected 4-D attention, got shape {attn.shape}")

    B, H, Q, K = attn.shape

    if Q == 1:
        # CA mode: [B, H, 1, N] → [B, H, N]
        cls_to_patch = attn[:, :, 0, :]
    else:
        # SA mode: [B, H, T, T] → CLS row, skip CLS self-attn → [B, H, N]
        cls_to_patch = attn[:, :, 0, 1:]

    gh, gw = _infer_patch_grid(attn[0])

    merged   = np.mean(cls_to_patch, axis=1).reshape(B, gh, gw)
    headwise = cls_to_patch.reshape(B, H, gh, gw)
    return merged, headwise


# ── Attention Rollout — SA/CA unified ─────────────────────────────────────

def compute_attention_rollout(
    attn_list: list[np.ndarray],
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Layer-wise attention accumulation.

    SA mode: [B, H, T, T] — cls row (idx 0, skip col 0) used
    CA mode: [B, H, 1, N] — cls row (idx 0) used directly

    Returns:
        accumulated : [B, N]
    """
    per_layer: list[np.ndarray] = []

    for attn in attn_list:
        B, H, Q, K = attn.shape

        if Q == 1:
            # CA mode
            patch_attn = attn[:, :, 0, :]       # [B, H, N]
        else:
            # SA mode
            patch_attn = attn[:, :, 0, 1:]      # [B, H, N]

        if head_reduction == "mean":
            layer_map = np.mean(patch_attn, axis=1)
        else:
            layer_map = np.max(patch_attn, axis=1)

        if discard_ratio > 0.0:
            thresh = np.quantile(layer_map, discard_ratio, axis=-1, keepdims=True)
            layer_map = np.where(layer_map >= thresh, layer_map, 0.0)

        row_sum = layer_map.sum(axis=-1, keepdims=True) + 1e-8
        layer_map = layer_map / row_sum
        per_layer.append(layer_map)

    accumulated = per_layer[0]
    for i, lm in enumerate(per_layer[1:], start=1):
        alpha = i / (i + 1)
        accumulated = alpha * accumulated + (1 - alpha) * lm

    return accumulated  # [B, N]


# ── EigenAttention ─────────────────────────────────────────────────────────

_EIGEN_K = 3


def compute_eigen_attention(
    attn_list: list[np.ndarray],
    k: int = _EIGEN_K,
    head_reduction: str = "mean",
) -> np.ndarray:
    """EigenAttention fallback for [B, H, 1, N] / [B, H, T, T].

    Returns:
        eigenmaps : [B, k, N]
    """
    accumulated = compute_attention_rollout(
        attn_list, head_reduction=head_reduction, discard_ratio=0.0
    )  # [B, N]

    B, N = accumulated.shape
    eigenmaps = np.zeros((B, k, N), dtype=np.float32)

    freq_x = np.arange(N, dtype=np.float32)
    for ki in range(k):
        if ki == 0:
            ev = accumulated.copy()
        else:
            modulator = np.cos(ki * np.pi * freq_x / max(N - 1, 1))
            ev = accumulated * modulator[None, :]

        for b in range(B):
            if ev[b, np.argmax(np.abs(ev[b]))] < 0:
                ev[b] = -ev[b]
        eigenmaps[:, ki, :] = ev.astype(np.float32)

    return eigenmaps  # [B, k, N]


def _append_eigen_to_row(
    row: dict,
    outputs: Any,
    img_uint8: np.ndarray,
    head_reduction: str,
    alpha: float,
    mass_threshold: float = 0.75,
) -> dict:
    attn_list = _get_attention_list_from_outputs(outputs)
    if attn_list is None:
        return row
    try:
        eigenmaps = compute_eigen_attention(attn_list, k=_EIGEN_K, head_reduction=head_reduction)
        last_attn = attn_list[-1]
        gh, gw = _infer_patch_grid(last_attn[0])
        target_hw = img_uint8.shape[:2]
        for ki in range(_EIGEN_K):
            ev = eigenmaps[0, ki].reshape(gh, gw)
            ev_up = _resize_heatmap(ev, target_hw, mass_threshold=mass_threshold)
            row[f"eigenattn_k{ki + 1}"] = wandb.Image(_colormap_heatmap(ev_up, mass_threshold=mass_threshold))
            row[f"eigen_overlay_k{ki + 1}"] = wandb.Image(
                _overlay_image(img_uint8, ev_up, alpha, mass_threshold=mass_threshold)
            )
    except Exception as e:
        print(f"[EigenAttention] 계산 실패 (skip): {e}")
    return row


def _get_attention_list_from_outputs(outputs: Any) -> list[np.ndarray] | None:
    if not isinstance(outputs, dict):
        return None
    attn_list = outputs.get("attention_weights")
    if attn_list and isinstance(attn_list, (list, tuple)) and len(attn_list) > 0:
        return [np.array(tf.convert_to_tensor(a)) for a in attn_list]
    attn_last = outputs.get("last_encoder_layer_attentional_weights")
    if attn_last is not None:
        return [np.array(tf.convert_to_tensor(attn_last))]
    return None


def _get_attention_from_outputs(outputs: Any) -> np.ndarray | None:
    if isinstance(outputs, dict):
        attn = outputs.get("last_encoder_layer_attentional_weights")
        if attn is None and outputs.get("attention_weights"):
            attn = outputs["attention_weights"][-1]
    else:
        attn = None
    if attn is None:
        return None
    return np.array(tf.convert_to_tensor(attn))


def _extract_attention_maps(
    outputs: Any,
    mode: AttentionMode = "last_layer",
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
    token_attention_mode: TokenAttentionMode = "sa",
) -> tuple[np.ndarray, np.ndarray] | None:
    """attention_mode + token_attention_mode 에 따라 (merged, headwise) 반환.

    SA mode (token_attention_mode='sa'):
        attn shape: [B, H, T, T]  →  _extract_cls_attention_map uses Q=T branch
    CA mode (token_attention_mode='ca'):
        attn shape: [B, H, 1, N]  →  _extract_cls_attention_map uses Q==1 branch

    두 경우 모두 _extract_cls_attention_map 이 Q dim 을 자동 감지하므로
    추가 분기 없이 동일한 경로를 통과함.

    Returns:
        (merged [B, gh, gw], headwise [B, H, gh, gw])  or  None
    """
    normalized_mode = "last_layer" if mode == "last" else mode

    if normalized_mode == "rollout":
        attn_list = _get_attention_list_from_outputs(outputs)
        if attn_list is None:
            return None
        accumulated = compute_attention_rollout(attn_list, head_reduction, discard_ratio)
        B = accumulated.shape[0]
        last_attn = attn_list[-1]
        gh, gw = _infer_patch_grid(last_attn[0])
        merged = accumulated.reshape(B, gh, gw)
        _, headwise = _extract_cls_attention_map(last_attn)
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
    probs_np = np.array(tf.convert_to_tensor(probs)) if probs is not None else None
    logits_np = np.array(tf.convert_to_tensor(logits)) if logits is not None else None
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
        y_true, y_pred, target_names=list(class_names), digits=4, zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=list(class_names), digits=4, zero_division=0, output_dict=True,
    )
    return report_text, report_dict


# ── build_attention_wandb_table ───────────────────────────────────────────

def build_attention_wandb_table(
    model: keras.Model,
    image_paths: list[tuple[str, int | None]],
    vis_cfg: WandbVisualizationConfig,
    encoder_accessor: str = "encoder",
) -> "wandb.Table | None":
    """attention map wandb.Table을 반환하는 독립 유틸리티 함수.

    token_attention_mode='ca' 일 때:
        column names에 '_CA' suffix 가 붙어 SA 실험과 구분됨.
    token_attention_mode='sa' 일 때 (기본값):
        column names 변화 없음 (하위 호환).

    heatmap 정규화는 DINO-style per-image mass threshold (vis_cfg.mass_threshold).
    """
    _require_wandb()
    _log_token_attention_mode_once(vis_cfg.token_attention_mode)

    mode_sfx = vis_cfg.mode_suffix           # '_SA' or '_CA'
    attn_vis_label = vis_cfg.attention_mode  # e.g. 'last_layer'
    mt = vis_cfg.mass_threshold              # per-image DINO threshold

    def _get_encoder_outputs(m, img_batch):
        enc = getattr(m, encoder_accessor, None)
        if enc is not None and callable(enc):
            return enc(img_batch, training=False)
        return m(img_batch, training=False)

    rows = []
    for path, true_label in image_paths:
        original, resized = _load_raw_image(path, vis_cfg.image_size)
        outputs = _get_encoder_outputs(model, resized)
        result = _extract_attention_maps(
            outputs,
            mode=vis_cfg.attention_mode,
            head_reduction=vis_cfg.rollout_head_reduction,
            discard_ratio=vis_cfg.rollout_discard_ratio,
            token_attention_mode=vis_cfg.token_attention_mode,
        )
        if result is None:
            continue
        merged, headwise = result

        img_uint8 = _prepare_display_image(original, vis_cfg.normalize_from_minus1)
        target_hw = img_uint8.shape[:2]
        merged_up = _resize_heatmap(merged[0], target_hw, mass_threshold=mt)

        row: dict[str, Any] = {
            "path": path,
            "label": true_label,
            "raw_image": wandb.Image(img_uint8),
            f"attention_map_{attn_vis_label}{mode_sfx}": wandb.Image(
                _colormap_heatmap(merged_up, mass_threshold=mt)
            ),
            f"overlay_{attn_vis_label}{mode_sfx}": wandb.Image(
                _overlay_image(img_uint8, merged_up, vis_cfg.overlay_alpha, mass_threshold=mt)
            ),
        }
        for h in range(headwise.shape[1]):
            hm_up = _resize_heatmap(headwise[0, h], target_hw, mass_threshold=mt)
            row[f"overlay_head_{h + 1}{mode_sfx}"] = wandb.Image(
                _overlay_image(img_uint8, hm_up, vis_cfg.overlay_alpha, mass_threshold=mt)
            )

        if getattr(vis_cfg, "enable_eigen", False):
            row = _append_eigen_to_row(
                row, outputs, img_uint8,
                head_reduction=vis_cfg.rollout_head_reduction,
                alpha=vis_cfg.overlay_alpha,
                mass_threshold=mt,
            )
        rows.append(row)

    if not rows:
        return None

    columns = list(rows[0].keys())
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row[c] for c in columns])
    return table


# ── ExtValConfig ──────────────────────────────────────────────────────────

@dataclass
class ExtValConfig:
    """외부 검증 데이터셋에 대한 wandb logging 설정.

    token_attention_mode 옵션:
        - ``"sa"`` (기본값): SA 모드 — 기존 동작.
        - ``"ca"``: CA 모드 — attn [B,H,1,N], column suffix _CA.

    attention_mode 옵션:
        - ``"last_layer"`` (기본값)
        - ``"rollout"``: 모든 block attention 누적

    Attributes:
        val_dir               : 외부 검증 이미지 루트 경로
        num_images            : logging할 이미지 수
        table_key             : wandb table 키 이름
        image_size            : 추론 시 resize 크기
        overlay_alpha         : attention overlay 투명도
        attention_mode        : 'last_layer' | 'rollout'
        token_attention_mode  : 'sa' | 'ca'
        rollout_head_reduction: 'mean' | 'max'
        rollout_discard_ratio : 0.0~1.0
        enable_eigen          : EigenAttention 시각화 여부
        encoder_accessor      : model 내 encoder attribute 이름
        mass_threshold        : DINO-style per-image cumulative mass cutoff (기본 0.75)
    """
    val_dir: str
    num_images: int = 8
    table_key: str = "ext_val_attention_table"
    image_size: tuple[int, int] = (224, 224)
    overlay_alpha: float = 0.45
    normalize_from_minus1: bool = False
    attention_mode: AttentionMode = "last_layer"
    token_attention_mode: TokenAttentionMode = "sa"
    rollout_head_reduction: Literal["mean", "max"] = "mean"
    rollout_discard_ratio: float = 0.0
    enable_eigen: bool = False
    encoder_accessor: str = "encoder"
    mass_threshold: float = 0.75

    def to_vis_cfg(self, class_names: tuple[str, str] = ("hemangioma", "hcc")) -> WandbVisualizationConfig:
        """ExtValConfig → WandbVisualizationConfig 변환 헬퍼."""
        return WandbVisualizationConfig(
            test_dir=self.val_dir,
            num_images=self.num_images,
            class_names=class_names,
            image_size=self.image_size,
            overlay_alpha=self.overlay_alpha,
            normalize_from_minus1=self.normalize_from_minus1,
            attention_mode=self.attention_mode,
            token_attention_mode=self.token_attention_mode,
            rollout_head_reduction=self.rollout_head_reduction,
            rollout_discard_ratio=self.rollout_discard_ratio,
            enable_eigen=self.enable_eigen,
            table_key=self.table_key,
            mass_threshold=self.mass_threshold,
        )

    def log_to_wandb(self, model: keras.Model) -> None:
        """외부 검증셋 attention을 즉시 wandb에 logging."""
        _require_wandb()
        vis_cfg = self.to_vis_cfg()
        image_paths = _list_test_images(self.val_dir, self.num_images)
        table = build_attention_wandb_table(
            model=model,
            image_paths=image_paths,
            vis_cfg=vis_cfg,
            encoder_accessor=self.encoder_accessor,
        )
        if table is not None:
            wandb.log({self.table_key: table}, commit=False)
            print(f"[ExtValConfig] logged {len(image_paths)} images → wandb table '{self.table_key}'")
        else:
            print(f"[ExtValConfig] no attention maps obtained from {self.val_dir}")


_DEFAULT_ATTENTION_TABLE_KEY = "stage1_attention_table"


class WandbAttentionVisualizer(keras.callbacks.Callback):
    """Stage1 attention visualization callback.

    token_attention_mode 분기:
        SA mode ('sa', 기본값):
            attn shape: [B, H, T, T]  (T = 1 + N_patches)
            column suffix: _SA  (e.g. 'attention_map_last_layer_SA')
            ※ SA는 기존 실험과의 연속성을 위해 suffix를 붙임.

        CA mode ('ca'):
            attn shape: [B, H, 1, N]
            column suffix: _CA  (e.g. 'attention_map_last_layer_CA')

    wandb run.config['token_attention_mode'] 에 모드를 최초 1회 기록함.

    enable_eigen=True 시 k=1,2,3 EigenAttention map 및 overlay가
    기존 columns 뒤에 추가 logging됨.

    heatmap 정규화: DINO-style per-image mass threshold (vis_cfg.mass_threshold, 기본 0.75).
    """

    def __init__(self, vis_cfg: WandbVisualizationConfig):
        super().__init__()
        self.cfg = vis_cfg
        self.samples = _list_test_images(vis_cfg.test_dir, vis_cfg.num_images)
        self._table_key: str = (
            vis_cfg.table_key if vis_cfg.table_key is not None
            else _DEFAULT_ATTENTION_TABLE_KEY
        )
        self._mode_logged = False

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
            token_attention_mode=self.cfg.token_attention_mode,
        )
        if result is None:
            return None
        merged, headwise = result

        img_uint8 = _prepare_display_image(original, self.cfg.normalize_from_minus1)
        target_hw = img_uint8.shape[:2]
        mt = self.cfg.mass_threshold
        merged_up = _resize_heatmap(merged[0], target_hw, mass_threshold=mt)

        mode_sfx = self.cfg.mode_suffix           # '_SA' or '_CA'
        attn_vis_label = self.cfg.attention_mode  # e.g. 'last_layer'

        row = {
            "path": path,
            "label": true_label,
            "raw_image": wandb.Image(img_uint8),
            f"attention_map_{attn_vis_label}{mode_sfx}": wandb.Image(
                _colormap_heatmap(merged_up, mass_threshold=mt)
            ),
            f"overlay_{attn_vis_label}{mode_sfx}": wandb.Image(
                _overlay_image(img_uint8, merged_up, self.cfg.overlay_alpha, mass_threshold=mt)
            ),
        }
        for h in range(headwise.shape[1]):
            hm_up = _resize_heatmap(headwise[0, h], target_hw, mass_threshold=mt)
            row[f"overlay_head_{h + 1}{mode_sfx}"] = wandb.Image(
                _overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha, mass_threshold=mt)
            )

        if getattr(self.cfg, "enable_eigen", False):
            row = _append_eigen_to_row(
                row, outputs, img_uint8,
                head_reduction=self.cfg.rollout_head_reduction,
                alpha=self.cfg.overlay_alpha,
                mass_threshold=mt,
            )

        return row

    def on_epoch_end(self, epoch, logs=None):
        if wandb is None or wandb.run is None:
            return
        if (epoch + 1) % self.cfg.log_every_n_epochs != 0:
            return

        if not self._mode_logged:
            _log_token_attention_mode_once(self.cfg.token_attention_mode)
            self._mode_logged = True

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

    token_attention_mode 분기:
        SA mode ('sa', 기본값): column suffix _SA
        CA mode ('ca'):         column suffix _CA

    attention_mode 옵션:
        - ``"last_layer"`` (기본값)
        - ``"rollout"``

    enable_eigen=True 시 k=1,2,3 EigenAttention map 및 overlay 추가 logging.

    heatmap 정규화: DINO-style per-image mass threshold (vis_cfg.mass_threshold, 기본 0.75).
    """

    def __init__(self, vis_cfg: WandbVisualizationConfig):
        super().__init__()
        self.cfg = vis_cfg
        self.samples = _list_test_images(vis_cfg.test_dir, vis_cfg.num_images)
        self._mode_logged = False

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

        mode_sfx = self.cfg.mode_suffix
        attn_vis_label = self.cfg.attention_mode
        mt = self.cfg.mass_threshold

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
            token_attention_mode=self.cfg.token_attention_mode,
        )
        if result is not None:
            merged, headwise = result
            merged_up = _resize_heatmap(merged[0], img_uint8.shape[:2], mass_threshold=mt)
            row[f"attention_map_{attn_vis_label}{mode_sfx}"] = wandb.Image(
                _colormap_heatmap(merged_up, mass_threshold=mt)
            )
            row[f"overlay_{attn_vis_label}{mode_sfx}"] = wandb.Image(
                _overlay_image(img_uint8, merged_up, self.cfg.overlay_alpha, mass_threshold=mt)
            )
            for h in range(headwise.shape[1]):
                hm_up = _resize_heatmap(headwise[0, h], img_uint8.shape[:2], mass_threshold=mt)
                row[f"overlay_head_{h + 1}{mode_sfx}"] = wandb.Image(
                    _overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha, mass_threshold=mt)
                )

        if getattr(self.cfg, "enable_eigen", False):
            row = _append_eigen_to_row(
                row, attn_outputs, img_uint8,
                head_reduction=self.cfg.rollout_head_reduction,
                alpha=self.cfg.overlay_alpha,
                mass_threshold=mt,
            )

        grad_model = self._resolve_model_for_gradcam()
        layer_name = _resolve_gradcam_layer(grad_model, self.cfg.gradcam_layer_name)
        for class_idx in (0, 1):
            heatmap = make_gradcam_heatmap(grad_model, resized, class_idx, layer_name)
            hm_up = _resize_heatmap(heatmap, img_uint8.shape[:2], mass_threshold=mt)
            row[f"gradcam_class_{class_idx}"] = wandb.Image(
                _overlay_image(img_uint8, hm_up, self.cfg.overlay_alpha, mass_threshold=mt)
            )
        return row

    def on_epoch_end(self, epoch, logs=None):
        if wandb is None or wandb.run is None:
            return
        if (epoch + 1) % self.cfg.log_every_n_epochs != 0:
            return

        if not self._mode_logged:
            _log_token_attention_mode_once(self.cfg.token_attention_mode)
            self._mode_logged = True

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
    """Benchmark dataset classification summary callback.

    token_attention_mode를 wandb run.config에 기록하여
    SA/CA 실험을 dashboard에서 필터링할 수 있도록 함.
    """

    def __init__(
        self,
        vis_cfg: WandbVisualizationConfig,
        test_dataset: tf.data.Dataset,
    ):
        super().__init__()
        self.cfg = vis_cfg
        self.test_dataset = test_dataset
        self._mode_logged = False

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        if wandb is None or wandb.run is None:
            return
        if (epoch + 1) % self.cfg.log_every_n_epochs != 0:
            return

        if not self._mode_logged:
            _log_token_attention_mode_once(self.cfg.token_attention_mode)
            self._mode_logged = True

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

        mode_sfx = self.cfg.mode_suffix  # '_SA' or '_CA'
        flat_metrics: dict[str, float] = {}
        for key, value in report_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat_metrics[f'benchmark_report{mode_sfx}/{key}/{sub_key}'] = float(sub_val)
            elif isinstance(value, (int, float)):
                flat_metrics[f'benchmark_report{mode_sfx}/{key}'] = float(value)

        wandb.log({
            f'benchmark{mode_sfx}/classification_heatmap': wandb.Image(
                heatmap_img, caption=f'Epoch {epoch + 1} [{self.cfg.token_attention_mode.upper()}]'
            ),
            f'benchmark{mode_sfx}/classification_report_text': wandb.Html(
                f'<pre>{report_text}</pre>'
            ),
            **flat_metrics,
        }, commit=False)
