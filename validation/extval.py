from __future__ import annotations

"""External validation utilities.

Group A  (cfg.split_names)     : prototype-based quantitative eval (K-Means)
Ext-Val  (cfg.ext_val_dir)     : *정성 평가* — per-image score + attention overlay, no K-Means
                                  hcc_score, hemangioma_score, margin = hcc - hem,
                                  attention overlay (last_layer or rollout)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from PIL import Image
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import wandb
except Exception:
    wandb = None

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
CLASS_DIR_CANDIDATES = {
    'hcc': ('HCC', 'hcc'),
    'hemangioma': ('Hemangioma', 'hemangioma'),
}
SPLIT_DIRS = {
    'val': 'val_clean',
    'test': 'test_clean',
}

AttentionMode = Literal['last_layer', 'rollout']


@dataclass
class ExtValConfig:
    """External validation configuration.

    Parameters
    ----------
    data_root :
        학습 데이터 루트. split_names 에 해당하는 {split}_clean/HCC 등이 여기에 있다.
    ext_val_dir :
        완전히 별도의 external-validation 디렉토리 (예: Atlas dataset).
        None 이면 ext-val 단계를 건너뛴다.
        이 경로 아래에도 HCC/ Hemangioma/ 서브폴더가 있다고 가정한다.
        **Ext-val 은 정성 평가만 수행한다 — K-Means prototype 없음.**
    work_dir         : 결과물 저장 루트.
    model_id         : 실험 식별자 (파일명/WandB prefix 에 사용).
    ext_val_attention_mode :
        Ext-val attention overlay 방법.
        'last_layer' (기본값) 또는 'rollout'.
    ext_val_rollout_discard_ratio :
        rollout 전용 noise 제거 비율 (0.0 ~ 1.0).
    ext_val_max_vis :
        Ext-val 에서 attention overlay를 저장할 최대 이미지 수 (per class).
        None 이면 전체 저장.
    image_size       : 리사이즈 목표 (H, W).
    batch_size       : 추론 배치 크기.
    n_proto          : Group A prototype K-means cluster 수.
    split_names      : data_root 기반 evaluation 에 사용할 split 이름들.
    stage_name       : 로그/파일명 prefix.
    wandb_prefix     : WandB log key prefix.
    random_state     : 재현성 seed.
    """
    data_root: str
    work_dir: str
    model_id: str
    ext_val_dir: Optional[str] = None
    ext_val_attention_mode: AttentionMode = 'last_layer'
    ext_val_rollout_discard_ratio: float = 0.0
    ext_val_max_vis: Optional[int] = 20
    image_size: tuple[int, int] = (384, 384)
    batch_size: int = 32
    n_proto: int = 8
    split_names: tuple[str, ...] = ('val', 'test')
    stage_name: str = 'stage2'
    wandb_prefix: str = 'extval'
    random_state: int = 42


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────

def _find_class_dir(split_dir: Path, class_key: str) -> Path:
    for name in CLASS_DIR_CANDIDATES[class_key]:
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f'class directory for {class_key!r} not found under {split_dir}'
    )


def _collect_image_paths(data_root: str, split: str) -> dict[str, list[Path]]:
    split_dir = Path(data_root) / SPLIT_DIRS[split]
    if not split_dir.exists():
        raise FileNotFoundError(f'split directory not found: {split_dir}')
    out: dict[str, list[Path]] = {}
    for class_key in ('hcc', 'hemangioma'):
        cdir = _find_class_dir(split_dir, class_key)
        paths = sorted(
            [p for p in cdir.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]
        )
        out[class_key] = paths
    return out


def _collect_extval_paths(ext_val_dir: str) -> dict[str, list[Path]]:
    """ext_val_dir/HCC/ + ext_val_dir/Hemangioma/ flat 구조를 가정."""
    root = Path(ext_val_dir)
    out: dict[str, list[Path]] = {}
    for class_key in ('hcc', 'hemangioma'):
        cdir = _find_class_dir(root, class_key)
        paths = sorted(
            [p for p in cdir.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]
        )
        out[class_key] = paths
        print(f'[ext_val] {class_key}: {len(paths)} images from {cdir}')
    return out


def _load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize(image_size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _chunked(seq: Sequence[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


# ──────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────

def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def _fit_prototypes(rep: np.ndarray, n_proto: int, random_state: int) -> np.ndarray:
    n_clusters = min(n_proto, len(rep))
    if n_clusters < 1:
        raise ValueError('empty representation matrix')
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(rep)
    return km.cluster_centers_.astype(np.float32)


def _mean_similarity(rep: np.ndarray, proto: np.ndarray) -> np.ndarray:
    rep_n   = _l2_normalize(rep)
    proto_n = _l2_normalize(proto)
    return (rep_n @ proto_n.T).mean(axis=-1)


def _sem(x: np.ndarray) -> float:
    if len(x) == 0:
        return float('nan')
    if len(x) == 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


def _ttest(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if len(a) == 0 or len(b) == 0:
        return {'t_value': float('nan'), 'p_value': float('nan')}
    stat, p = ttest_ind(a, b, equal_var=False)
    return {'t_value': float(stat), 'p_value': float(p)}


# ──────────────────────────────────────────────────────────────
# Encoder / representation extraction
# ──────────────────────────────────────────────────────────────

def _extract_encoder_output(stage2_model, batch: np.ndarray) -> np.ndarray:
    if hasattr(stage2_model, 'model') and hasattr(stage2_model.model, 'encoder'):
        encoder = stage2_model.model.encoder
    elif hasattr(stage2_model, 'encoder'):
        encoder = stage2_model.encoder
    else:
        encoder = stage2_model
    out = encoder(tf.convert_to_tensor(batch, dtype=tf.float32), training=False)
    if isinstance(out, dict):
        emb = out.get('embedding', out.get('cls_token'))
        if emb is None:
            raise KeyError('encoder output dict must contain "embedding" or "cls_token"')
    else:
        emb = out
    emb = tf.convert_to_tensor(emb)
    if len(emb.shape) > 2:
        emb = emb[:, 0]
    return emb.numpy().astype(np.float32)


def extract_representations(
    stage2_model,
    image_paths: Sequence[Path],
    image_size: tuple[int, int],
    batch_size: int,
) -> np.ndarray:
    reps: list[np.ndarray] = []
    for chunk in _chunked(image_paths, batch_size):
        batch = np.stack([_load_image(p, image_size) for p in chunk], axis=0)
        reps.append(_extract_encoder_output(stage2_model, batch))
    if not reps:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(reps, axis=0)


# ──────────────────────────────────────────────────────────────
# Attention overlay (wraps visualization/interpret.py)
# ──────────────────────────────────────────────────────────────

def _is_cnn_encoder(stage2_model) -> bool:
    """CNN vs ViT encoder 판별 — 이름 하드코딩 대신 구조 기반 탐지.

    판별 우선순위
    ─────────────
    1. encoder 객체 내 MultiHeadAttention layer 존재 → ViT (False)
    2. encoder 객체 내 GlobalAveragePooling2D 존재   → CNN  (True)
    3. encoder.name 에 'vit' / 'transformer' 포함    → ViT  (False)
    4. 위 셋 모두 해당 없으면 → CNN 간주             (True)

    지원 backbone 예시
    ──────────────────
    CNN  : EfficientNetV2S/M, ResNet50V2, ResNet101V2, DenseNet121,
           ConvNeXtSmall/Tiny/Base, MobileNetV3 등 GAP 기반 모델
    ViT  : DINOv2-ViT-S/B, ViT-B/L, Swin-Transformer 등 MHA 기반 모델
    """
    try:
        encoder = (
            stage2_model.model.encoder
            if hasattr(stage2_model, 'model') and hasattr(stage2_model.model, 'encoder')
            else stage2_model.encoder
        )
    except AttributeError:
        return False

    try:
        layer_types = {type(l).__name__ for l in encoder.layers}
    except AttributeError:
        layer_types = set()

    # 1. MHA 가 있으면 ViT 계열
    if 'MultiHeadAttention' in layer_types:
        return False

    # 2. GAP 가 있으면 CNN 계열
    if 'GlobalAveragePooling2D' in layer_types:
        return True

    # 3. 이름 기반 보조 판별 (ViT 쪽만 — CNN 이름은 다양해서 양성 판별 불가)
    try:
        name = encoder.name.lower()
        if any(k in name for k in ('vit', 'transformer', 'dino', 'swin')):
            return False
    except AttributeError:
        pass

    # 4. 판별 불가 → CNN 간주 (GAP 없어도 Dense 연결 CNN 일 수 있음)
    return True


def _attention_overlay_single(
    stage2_model,
    image_arr: np.ndarray,
    attention_mode: AttentionMode,
    rollout_discard_ratio: float,
) -> np.ndarray:
    """단일 이미지 (H,W,3) → attention overlay (H,W,3) float32.

    visualization/interpret.py 의 attention_overlay / gradcam_overlay 를 재사용.
    """
    from visualization.interpret import attention_overlay, gradcam_overlay

    if _is_cnn_encoder(stage2_model):
        return gradcam_overlay(stage2_model, image=image_arr, class_index=1)
    return attention_overlay(
        stage2_model,
        image=image_arr,
        head_reduction='mean',
        mode=attention_mode,
        rollout_discard_ratio=rollout_discard_ratio,
    )


# ──────────────────────────────────────────────────────────────
# Per-image score computation (no K-Means)
# ──────────────────────────────────────────────────────────────

def _score_single_image(
    stage2_model,
    image_arr: np.ndarray,
    hcc_prototypes: np.ndarray,
    hem_prototypes: np.ndarray,
) -> dict[str, float]:
    """단일 이미지 forward → hcc_score, hemangioma_score, margin.

    Group A 의 fit된 prototype bank를 넘겨받아 그대로 사용한다.
    K-Means 없음.
    """
    from inference.scorer import dual_bank_scores

    x = tf.convert_to_tensor(image_arr[None], dtype=tf.float32)  # [1,H,W,3]

    # 모델 forward — classifier wrapper 또는 bare encoder 모두 지원
    if hasattr(stage2_model, '__call__'):
        out = stage2_model(x, training=False)
    else:
        raise TypeError('stage2_model must be callable')

    embedding       = out.get('embedding', None)
    encoded_patches = out.get('encoded_patches', None)

    # embedding이 출력에 없으면 encoder를 직접 호출
    if embedding is None:
        enc = (
            stage2_model.model.encoder
            if hasattr(stage2_model, 'model')
            else stage2_model.encoder
        )
        enc_out = enc(x, training=False)
        if isinstance(enc_out, dict):
            embedding       = enc_out.get('embedding', enc_out.get('cls_token'))
            encoded_patches = enc_out.get('encoded_patches', encoded_patches)
        else:
            embedding = enc_out
        if len(tf.shape(embedding)) > 2:
            embedding = embedding[:, 0]

    scores = dual_bank_scores(
        embedding       = embedding,
        encoded_patches = encoded_patches,
        hemangioma_prototypes = tf.convert_to_tensor(hem_prototypes, dtype=tf.float32),
        hcc_prototypes        = tf.convert_to_tensor(hcc_prototypes, dtype=tf.float32),
        alpha=0.5,
    )
    return {
        'hcc_score'        : float(scores['hcc_score'].numpy()[0]),
        'hemangioma_score' : float(scores['hemangioma_score'].numpy()[0]),
        'margin'           : float(
            scores['hcc_score'].numpy()[0] - scores['hemangioma_score'].numpy()[0]
        ),
    }


# ──────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────

def _save_matrix(path: Path, arr: np.ndarray, header_prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), arr)
    if arr.ndim == 2:
        cols = [f'{header_prefix}_{i}' for i in range(arr.shape[1])]
        pd.DataFrame(arr, columns=cols).to_csv(path.with_suffix('.csv'), index=False)
    else:
        pd.DataFrame(arr).to_csv(path.with_suffix('.csv'), index=False)


def _save_paths(path: Path, paths: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path'])
        for p in paths:
            writer.writerow([str(p)])


def _save_overlay(save_path: Path, original: np.ndarray, overlay: np.ndarray) -> None:
    """원본 | overlay 를 나란히 붙여 PNG로 저장."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # overlay는 float32 [0,1], original은 float32 [0,1]
    side = np.concatenate([original, overlay], axis=1)  # 좌우 병합
    img_uint8 = (np.clip(side, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path)


# ──────────────────────────────────────────────────────────────
# Visualisation helpers (Group A)
# ──────────────────────────────────────────────────────────────

def _pca_3d(
    hcc_rep: np.ndarray, hem_rep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_rep = np.concatenate([hcc_rep, hem_rep], axis=0)
    pca     = PCA(n_components=3)
    all_3d  = pca.fit_transform(all_rep)
    n_hcc   = len(hcc_rep)
    return all_3d[:n_hcc], all_3d[n_hcc:], pca.explained_variance_ratio_


def _log_pca_scatter(
    model_id: str,
    prefix: str,
    hcc_3d: np.ndarray,
    hem_3d: np.ndarray,
    hcc_paths: Sequence[Path],
    hem_paths: Sequence[Path],
    pca_var: np.ndarray,
    group_tag: str = 'groupA',
) -> None:
    if wandb is None or wandb.run is None:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=hcc_3d[:, 0], y=hcc_3d[:, 1], z=hcc_3d[:, 2],
        mode='markers', name='HCC',
        text=[str(p) for p in hcc_paths],
        hovertemplate='HCC<br>%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
        marker=dict(size=4, color='#C0504D', opacity=0.8),
    ))
    fig.add_trace(go.Scatter3d(
        x=hem_3d[:, 0], y=hem_3d[:, 1], z=hem_3d[:, 2],
        mode='markers', name='Hemangioma',
        text=[str(p) for p in hem_paths],
        hovertemplate='Hemangioma<br>%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>',
        marker=dict(size=4, color='#4F81BD', opacity=0.8),
    ))
    fig.update_layout(
        title=f'{group_tag} PCA 3D scatter',
        scene=dict(
            xaxis_title=f'PC1 ({pca_var[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({pca_var[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({pca_var[2]*100:.1f}%)',
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    wandb.log({f'{prefix}/{model_id}/{group_tag}_pca3d_figure': fig})
    wandb.log({
        f'{prefix}/{model_id}/{group_tag}_pca_pc1': float(pca_var[0]),
        f'{prefix}/{model_id}/{group_tag}_pca_pc2': float(pca_var[1]),
        f'{prefix}/{model_id}/{group_tag}_pca_pc3': float(pca_var[2]),
    })


def _make_boxplot(
    title: str,
    ylabel: str,
    arrays: Sequence[np.ndarray],
    labels: Sequence[str],
) -> plt.Figure:
    if len(arrays) != len(labels):
        raise ValueError('arrays and labels must have the same length')
    _palette   = ['#C0504D', '#4F81BD', '#9BBB59', '#8064A2']
    box_colors = [_palette[i % len(_palette)] for i in range(len(labels))]
    fig, ax    = plt.subplots(figsize=(6, 5), dpi=150,
                              facecolor="white")
    ax.set_facecolor("white")
    bp = ax.boxplot(
        list(arrays), patch_artist=True, notch=False, widths=0.50,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, linestyle='none', markeredgewidth=0.6),
        boxprops=dict(linewidth=1.0),
    )
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for flier, color in zip(bp['fliers'], box_colors):
        flier.set_markerfacecolor(color); flier.set_markeredgecolor(color)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.40)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    return fig


def _log_summary_plots(
    model_id: str,
    prefix: str,
    stats: dict,
    ttest_df: pd.DataFrame,
    sim_A: np.ndarray,
    sim_B: np.ndarray,
    sim_C: np.ndarray,
    sim_D: np.ndarray,
    group_tag: str = 'groupA',
) -> None:
    if wandb is None or wandb.run is None:
        return
    labels = ['HCC group', 'Hemangioma group']
    fig_hcc    = _make_boxplot('HCC similarity',       'Similarity (a.u.)', [sim_A, sim_C], labels)
    fig_hem    = _make_boxplot('Hemangioma similarity','Similarity (a.u.)', [sim_B, sim_D], labels)
    fig_margin = _make_boxplot('HCC sim - Hem sim',    'Similarity margin (a.u.)', [sim_A - sim_B, sim_C - sim_D], labels)
    wandb.log({
        f'{prefix}/{model_id}/{group_tag}_hcc_similarity_box'      : wandb.Image(fig_hcc),
        f'{prefix}/{model_id}/{group_tag}_hemangioma_similarity_box': wandb.Image(fig_hem),
        f'{prefix}/{model_id}/{group_tag}_margin_similarity_box'   : wandb.Image(fig_margin),
        f'{prefix}/{model_id}/{group_tag}_ttest_table'             : wandb.Table(dataframe=ttest_df),
    })
    plt.close(fig_hcc); plt.close(fig_hem); plt.close(fig_margin)


# ──────────────────────────────────────────────────────────────
# Group A: quantitative eval block (internal val/test)
# ──────────────────────────────────────────────────────────────

def _eval_block(
    stage2_model,
    hcc_paths: list[Path],
    hem_paths: list[Path],
    cfg: ExtValConfig,
    out_dir: Path,
    group_tag: str,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """K-Means prototype eval → stats, ttest_df, hcc_proto, hem_proto 반환.

    반환된 prototype bank 는 ext-val score 계산에 재사용된다.
    """
    hcc_rep = extract_representations(stage2_model, hcc_paths, cfg.image_size, cfg.batch_size)
    hem_rep = extract_representations(stage2_model, hem_paths, cfg.image_size, cfg.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_matrix(out_dir / 'hcc_rep',        hcc_rep, 'emb')
    _save_matrix(out_dir / 'hemangioma_rep', hem_rep, 'emb')
    _save_paths(out_dir / 'hcc_paths.csv',        hcc_paths)
    _save_paths(out_dir / 'hemangioma_paths.csv', hem_paths)

    hcc_3d, hem_3d, pca_var = _pca_3d(hcc_rep, hem_rep)
    _save_matrix(out_dir / 'hcc_3d_rep',        hcc_3d, 'pc')
    _save_matrix(out_dir / 'hemangioma_3d_rep', hem_3d, 'pc')
    pd.DataFrame([{
        'pc1': float(pca_var[0]),
        'pc2': float(pca_var[1]),
        'pc3': float(pca_var[2]),
    }]).to_csv(out_dir / 'pca_explained_variance_ratio.csv', index=False)

    hcc_proto = _fit_prototypes(hcc_rep, cfg.n_proto, cfg.random_state)
    hem_proto = _fit_prototypes(hem_rep, cfg.n_proto, cfg.random_state)
    _save_matrix(out_dir / 'hcc_prototype',        hcc_proto, 'proto')
    _save_matrix(out_dir / 'hemangioma_prototype', hem_proto, 'proto')

    A = _mean_similarity(hcc_rep, hcc_proto)
    B = _mean_similarity(hcc_rep, hem_proto)
    C = _mean_similarity(hem_rep, hcc_proto)
    D = _mean_similarity(hem_rep, hem_proto)
    margin_hcc = A - B
    margin_hem = C - D

    stats = {
        'model_id'               : cfg.model_id,
        'group'                  : group_tag,
        'n_hcc'                  : int(len(hcc_rep)),
        'n_hemangioma'           : int(len(hem_rep)),
        'embed_dim'              : int(hcc_rep.shape[1]),
        'n_hcc_prototype'        : int(hcc_proto.shape[0]),
        'n_hemangioma_prototype' : int(hem_proto.shape[0]),
        'A_hcc_to_hccproto_mean' : float(np.mean(A)),
        'A_hcc_to_hccproto_sem'  : _sem(A),
        'B_hcc_to_hemproto_mean' : float(np.mean(B)),
        'B_hcc_to_hemproto_sem'  : _sem(B),
        'C_hem_to_hccproto_mean' : float(np.mean(C)),
        'C_hem_to_hccproto_sem'  : _sem(C),
        'D_hem_to_hemproto_mean' : float(np.mean(D)),
        'D_hem_to_hemproto_sem'  : _sem(D),
        'A_minus_B_mean'         : float(np.mean(margin_hcc)),
        'A_minus_B_sem'          : _sem(margin_hcc),
        'C_minus_D_mean'         : float(np.mean(margin_hem)),
        'C_minus_D_sem'          : _sem(margin_hem),
    }

    ttest_hcc    = _ttest(A, C)
    ttest_margin = _ttest(margin_hcc, margin_hem)
    ttest_df     = pd.DataFrame([
        {'metric': 'hcc_similarity',           **ttest_hcc},
        {'metric': 'hcc_minus_hem_similarity', **ttest_margin},
    ])

    pd.DataFrame([stats]).to_csv(out_dir / 'summary_stats.csv', index=False)
    ttest_df.to_csv(out_dir / 'ttest.csv', index=False)

    if wandb is not None and wandb.run is not None:
        wandb.log({
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_n_hcc'                  : stats['n_hcc'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_n_hemangioma'           : stats['n_hemangioma'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_A_hcc_to_hccproto_mean': stats['A_hcc_to_hccproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_B_hcc_to_hemproto_mean': stats['B_hcc_to_hemproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_C_hem_to_hccproto_mean': stats['C_hem_to_hccproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_D_hem_to_hemproto_mean': stats['D_hem_to_hemproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_A_minus_B_mean'        : stats['A_minus_B_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_C_minus_D_mean'        : stats['C_minus_D_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_ttest_hcc_t'           : ttest_hcc['t_value'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_ttest_hcc_p'           : ttest_hcc['p_value'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_ttest_margin_t'        : ttest_margin['t_value'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{group_tag}_ttest_margin_p'        : ttest_margin['p_value'],
        })
        _log_pca_scatter(
            cfg.model_id, cfg.wandb_prefix,
            hcc_3d, hem_3d, hcc_paths, hem_paths, pca_var,
            group_tag=group_tag,
        )
        _log_summary_plots(
            cfg.model_id, cfg.wandb_prefix, stats, ttest_df,
            sim_A=A, sim_B=B, sim_C=C, sim_D=D,
            group_tag=group_tag,
        )

    return ttest_df, stats, hcc_proto, hem_proto


# ──────────────────────────────────────────────────────────────
# Ext-Val: qualitative block (score + attention overlay, no K-Means)
# ──────────────────────────────────────────────────────────────

def _extval_qualitative_block(
    stage2_model,
    hcc_paths: list[Path],
    hem_paths: list[Path],
    hcc_prototypes: np.ndarray,    # Group A 에서 fit된 bank 재사용
    hem_prototypes: np.ndarray,
    cfg: ExtValConfig,
    out_dir: Path,
) -> pd.DataFrame:
    """External dataset 정성 평가.

    수행 내용
    ---------
    1. 이미지별 hcc_score / hemangioma_score / margin 산출
       → extval_scores.csv 저장
    2. attention overlay (last_layer or rollout)
       → out_dir/overlays/{hcc,hemangioma}/{stem}_overlay.png 저장
    3. WandB 에 score table + overlay 이미지 panel 로깅 (run 있을 때만)

    Parameters
    ----------
    hcc_prototypes, hem_prototypes :
        Group A _eval_block() 이 반환한 K-Means prototype bank.
        ext-val 은 이를 그대로 사용해 score를 산출한다.
    cfg.ext_val_max_vis :
        overlay 저장 최대 수 (None = 전체). 가장 앞 max_vis 개 처리.

    Returns
    -------
    DataFrame  (path, true_label, hcc_score, hemangioma_score, margin)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / 'overlays'
    records: list[dict] = []
    wandb_images: dict[str, list] = {'hcc': [], 'hemangioma': []}

    for class_key, paths, true_label in [
        ('hcc',        hcc_paths, 'HCC'),
        ('hemangioma', hem_paths, 'Hemangioma'),
    ]:
        cls_overlay_dir = overlay_dir / class_key
        cls_overlay_dir.mkdir(parents=True, exist_ok=True)

        max_vis = cfg.ext_val_max_vis
        vis_paths  = paths[:max_vis] if max_vis is not None else paths
        # score는 전체 이미지 계산 (overlay만 max_vis 제한)
        score_paths = paths

        print(f'[ext_val] {class_key}: scoring {len(score_paths)} images …')
        for path in score_paths:
            img = _load_image(path, cfg.image_size)   # float32 [H,W,3]
            score = _score_single_image(
                stage2_model, img, hcc_prototypes, hem_prototypes
            )
            records.append({
                'path'             : str(path),
                'true_label'       : true_label,
                'hcc_score'        : score['hcc_score'],
                'hemangioma_score' : score['hemangioma_score'],
                'margin'           : score['margin'],
            })

        print(f'[ext_val] {class_key}: generating attention overlays for {len(vis_paths)} images …')
        for path in vis_paths:
            img = _load_image(path, cfg.image_size)
            try:
                overlay = _attention_overlay_single(
                    stage2_model, img,
                    cfg.ext_val_attention_mode,
                    cfg.ext_val_rollout_discard_ratio,
                )
            except Exception as e:
                print(f'  [WARN] attention failed for {path.name}: {e}')
                continue

            save_path = cls_overlay_dir / f'{path.stem}_overlay.png'
            _save_overlay(save_path, img, overlay)

            if wandb is not None and wandb.run is not None:
                wandb_images[class_key].append(
                    wandb.Image(
                        str(save_path),
                        caption=f'{path.name}',
                    )
                )

    # --- CSV 저장 ---
    score_df = pd.DataFrame(records)
    score_df.to_csv(out_dir / 'extval_scores.csv', index=False)
    print(f'[ext_val] scores saved → {out_dir / "extval_scores.csv"}')

    # --- 집계 stats ---
    for cls, label in [('hcc', 'HCC'), ('hemangioma', 'Hemangioma')]:
        sub = score_df[score_df['true_label'] == label]
        if len(sub):
            print(
                f'  {label}  n={len(sub)} '
                f'hcc_score={sub["hcc_score"].mean():.4f}±{sub["hcc_score"].std():.4f}  '
                f'margin={sub["margin"].mean():.4f}±{sub["margin"].std():.4f}'
            )

    # --- WandB 로깅 ---
    if wandb is not None and wandb.run is not None:
        prefix  = cfg.wandb_prefix
        mid     = cfg.model_id
        if wandb_images['hcc']:
            wandb.log({f'{prefix}/{mid}/extval_hcc_overlays': wandb_images['hcc']})
        if wandb_images['hemangioma']:
            wandb.log({f'{prefix}/{mid}/extval_hemangioma_overlays': wandb_images['hemangioma']})
        wandb.log({f'{prefix}/{mid}/extval_score_table': wandb.Table(dataframe=score_df)})

        # boxplot: hcc_score distribution per true label
        hcc_scores_hcc = score_df[score_df['true_label'] == 'HCC']['hcc_score'].values
        hcc_scores_hem = score_df[score_df['true_label'] == 'Hemangioma']['hcc_score'].values
        margin_hcc     = score_df[score_df['true_label'] == 'HCC']['margin'].values
        margin_hem     = score_df[score_df['true_label'] == 'Hemangioma']['margin'].values
        if len(hcc_scores_hcc) and len(hcc_scores_hem):
            fig_score  = _make_boxplot(
                'Ext-Val: HCC score by true label',
                'HCC score (a.u.)',
                [hcc_scores_hcc, hcc_scores_hem],
                ['HCC', 'Hemangioma'],
            )
            fig_margin = _make_boxplot(
                'Ext-Val: Score margin (HCC - Hem) by true label',
                'Margin (a.u.)',
                [margin_hcc, margin_hem],
                ['HCC', 'Hemangioma'],
            )
            wandb.log({
                f'{prefix}/{mid}/extval_hcc_score_box'    : wandb.Image(fig_score),
                f'{prefix}/{mid}/extval_margin_score_box' : wandb.Image(fig_margin),
            })
            plt.close(fig_score)
            plt.close(fig_margin)

    return score_df


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_extval(stage2_model, cfg: ExtValConfig) -> pd.DataFrame:
    """Prototype-based external validation.

    Pipeline
    --------
    1. Group A (cfg.split_names, data_root)
       → K-Means prototype fit + quantitative eval (similarity, ttest, PCA)
       → prototype bank 저장
    2. Ext-Val (cfg.ext_val_dir, optional)
       → Group A prototype bank 재사용
       → 정성 평가: per-image hcc_score / hemangioma_score / margin
       → attention overlay 저장 (PNG + WandB)
       → K-Means 없음

    Returns
    -------
    ext-val score DataFrame (ext_val_dir 없으면 빈 DataFrame).
    """
    base_out = Path(cfg.work_dir) / 'output' / 'extval' / cfg.model_id

    # ── Group A ─────────────────────────────────────────────
    all_hcc_paths_A: list[Path] = []
    all_hem_paths_A: list[Path] = []
    for split in cfg.split_names:
        paths = _collect_image_paths(cfg.data_root, split)
        all_hcc_paths_A.extend(paths['hcc'])
        all_hem_paths_A.extend(paths['hemangioma'])

    print(f'[groupA] hcc={len(all_hcc_paths_A)}  hemangioma={len(all_hem_paths_A)}')
    _, _, hcc_proto, hem_proto = _eval_block(
        stage2_model,
        hcc_paths = all_hcc_paths_A,
        hem_paths = all_hem_paths_A,
        cfg       = cfg,
        out_dir   = base_out / 'groupA',
        group_tag = 'groupA',
    )

    # ── Ext-Val (정성 평가) ───────────────────────────────────
    if cfg.ext_val_dir is None:
        print('[ext_val] ext_val_dir is None — skipping external validation.')
        return pd.DataFrame()

    print(f'[ext_val] loading from {cfg.ext_val_dir}')
    ext_paths = _collect_extval_paths(cfg.ext_val_dir)

    score_df = _extval_qualitative_block(
        stage2_model,
        hcc_paths      = ext_paths['hcc'],
        hem_paths      = ext_paths['hemangioma'],
        hcc_prototypes = hcc_proto,
        hem_prototypes = hem_proto,
        cfg            = cfg,
        out_dir        = base_out / 'extval',
    )
    return score_df
