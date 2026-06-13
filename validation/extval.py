from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, Optional

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


@dataclass
class ExtValConfig:
    """External validation configuration.

    Parameters
    ----------
    data_root:
        학습 데이터 루트. split_names 에 해당하는 {split}_clean/HCC 등이 여기에 있다.
    ext_val_dir:
        완전히 별도의 external-validation 디렉토리 (예: Atlas dataset).
        None 이면 ext-val 단계를 건너뛴다.
        이 경로 아래에도 HCC/ Hemangioma/ 서브폴더가 있다고 가정한다.
    work_dir:         결과물 저장 루트.
    model_id:         실험 식별자 (파일명/WandB prefix 에 사용).
    image_size:       리사이즈 목표 (H, W).
    batch_size:       추론 배치 크기.
    n_proto:          prototype K-means cluster 수.
    split_names:      data_root 기반 evaluation 에 사용할 split 이름들.
    stage_name:       로그/파일명 prefix.
    wandb_prefix:     WandB log key prefix.
    random_state:     재현성 seed.
    """
    data_root: str
    work_dir: str
    model_id: str
    ext_val_dir: Optional[str] = None          # <-- 추가된 필드
    image_size: tuple[int, int] = (384, 384)
    batch_size: int = 32
    n_proto: int = 8
    split_names: tuple[str, ...] = ('val', 'test')
    stage_name: str = 'stage2'
    wandb_prefix: str = 'extval'
    random_state: int = 42


# ──────────────────────────────────────────────────────────────
# Internal helpers
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
    """data_root/{split}_clean/{HCC,Hemangioma} 에서 경로 수집."""
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
    """ext_val_dir 아래의 HCC/ Hemangioma/ 폴더에서 경로를 수집한다.

    data_root 기반과 달리 split 서브폴더 없이 flat 구조를 가정한다.
    즉 ext_val_dir/HCC/*.png, ext_val_dir/Hemangioma/*.png.
    """
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


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _fit_prototypes(
    rep: np.ndarray, n_proto: int, random_state: int
) -> np.ndarray:
    """K-Means cluster centers as prototypes."""
    n_clusters = min(n_proto, len(rep))
    if n_clusters < 1:
        raise ValueError('empty representation matrix')
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(rep)
    return km.cluster_centers_.astype(np.float32)


def _mean_similarity(rep: np.ndarray, proto: np.ndarray) -> np.ndarray:
    """Per-sample mean cosine similarity to prototype set."""
    rep_n   = _l2_normalize(rep)
    proto_n = _l2_normalize(proto)
    sim     = rep_n @ proto_n.T          # [N, K]
    return sim.mean(axis=-1)             # [N]


def _sem(x: np.ndarray) -> float:
    """Standard Error of the Mean  (ddof=1)."""
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


def _extract_encoder_output(stage2_model, batch: np.ndarray) -> np.ndarray:
    """Stage2TrainWrapper / SupervisedClassifier / bare encoder 모두 지원."""
    # Stage2TrainWrapper → .model.encoder
    if hasattr(stage2_model, 'model') and hasattr(stage2_model.model, 'encoder'):
        encoder = stage2_model.model.encoder
    # SupervisedClassifier → .encoder
    elif hasattr(stage2_model, 'encoder'):
        encoder = stage2_model.encoder
    else:
        encoder = stage2_model

    out = encoder(tf.convert_to_tensor(batch, dtype=tf.float32), training=False)
    if isinstance(out, dict):
        if 'embedding' in out:
            emb = out['embedding']
        elif 'cls_token' in out:
            emb = out['cls_token']
        else:
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
# Save utilities
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


# ──────────────────────────────────────────────────────────────
# Visualisation helpers
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
    fig, ax    = plt.subplots(figsize=(6, 5), dpi=150)
    bp = ax.boxplot(
        list(arrays),
        patch_artist=True,
        notch=False,
        widths=0.50,
        medianprops=dict(color='black', linewidth=2.0),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, linestyle='none', markeredgewidth=0.6),
        boxprops=dict(linewidth=1.0),
    )
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for flier, color in zip(bp['fliers'], box_colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.40)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
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
    plt.close(fig_hcc)
    plt.close(fig_hem)
    plt.close(fig_margin)


# ──────────────────────────────────────────────────────────────
# Core eval block (reused for groupA and ext_val)
# ──────────────────────────────────────────────────────────────

def _eval_block(
    stage2_model,
    hcc_paths: list[Path],
    hem_paths: list[Path],
    cfg: ExtValConfig,
    out_dir: Path,
    group_tag: str,
) -> tuple[pd.DataFrame, dict]:
    """주어진 경로 목록으로 prototype eval을 실행하고 stats + ttest_df를 반환.

    group_tag: 'groupA' (internal val/test) 또는 'extval' (Atlas 등 외부)
    """
    hcc_rep = extract_representations(stage2_model, hcc_paths, cfg.image_size, cfg.batch_size)
    hem_rep = extract_representations(stage2_model, hem_paths, cfg.image_size, cfg.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_matrix(out_dir / f'{group_tag}_hcc_rep',        hcc_rep, 'emb')
    _save_matrix(out_dir / f'{group_tag}_hemangioma_rep', hem_rep, 'emb')
    _save_paths(out_dir / f'{group_tag}_hcc_paths.csv',        hcc_paths)
    _save_paths(out_dir / f'{group_tag}_hemangioma_paths.csv', hem_paths)

    hcc_3d, hem_3d, pca_var = _pca_3d(hcc_rep, hem_rep)
    _save_matrix(out_dir / f'{group_tag}_hcc_3d_rep',        hcc_3d, 'pc')
    _save_matrix(out_dir / f'{group_tag}_hemangioma_3d_rep', hem_3d, 'pc')
    pd.DataFrame([{
        'pc1': float(pca_var[0]),
        'pc2': float(pca_var[1]),
        'pc3': float(pca_var[2]),
    }]).to_csv(out_dir / f'{group_tag}_pca_explained_variance_ratio.csv', index=False)

    hcc_proto = _fit_prototypes(hcc_rep, cfg.n_proto, cfg.random_state)
    hem_proto = _fit_prototypes(hem_rep, cfg.n_proto, cfg.random_state)
    _save_matrix(out_dir / f'{group_tag}_hcc_prototype',        hcc_proto, 'proto')
    _save_matrix(out_dir / f'{group_tag}_hemangioma_prototype', hem_proto, 'proto')

    A = _mean_similarity(hcc_rep, hcc_proto)
    B = _mean_similarity(hcc_rep, hem_proto)
    C = _mean_similarity(hem_rep, hcc_proto)
    D = _mean_similarity(hem_rep, hem_proto)
    margin_hcc = A - B
    margin_hem = C - D

    stats = {
        'model_id'                 : cfg.model_id,
        'group'                    : group_tag,
        'n_hcc'                    : int(len(hcc_rep)),
        'n_hemangioma'             : int(len(hem_rep)),
        'embed_dim'                : int(hcc_rep.shape[1]),
        'n_hcc_prototype'          : int(hcc_proto.shape[0]),
        'n_hemangioma_prototype'   : int(hem_proto.shape[0]),
        'A_hcc_to_hccproto_mean'   : float(np.mean(A)),
        'A_hcc_to_hccproto_sem'    : _sem(A),
        'B_hcc_to_hemproto_mean'   : float(np.mean(B)),
        'B_hcc_to_hemproto_sem'    : _sem(B),
        'C_hem_to_hccproto_mean'   : float(np.mean(C)),
        'C_hem_to_hccproto_sem'    : _sem(C),
        'D_hem_to_hemproto_mean'   : float(np.mean(D)),
        'D_hem_to_hemproto_sem'    : _sem(D),
        'A_minus_B_mean'           : float(np.mean(margin_hcc)),
        'A_minus_B_sem'            : _sem(margin_hcc),
        'C_minus_D_mean'           : float(np.mean(margin_hem)),
        'C_minus_D_sem'            : _sem(margin_hem),
    }

    ttest_hcc    = _ttest(A, C)
    ttest_margin = _ttest(margin_hcc, margin_hem)
    ttest_df     = pd.DataFrame([
        {'metric': 'hcc_similarity',         **ttest_hcc},
        {'metric': 'hcc_minus_hem_similarity', **ttest_margin},
    ])

    pd.DataFrame([stats]).to_csv(out_dir / f'{group_tag}_summary_stats.csv', index=False)
    ttest_df.to_csv(out_dir / f'{group_tag}_ttest.csv', index=False)

    # WandB logging
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

    return ttest_df, stats


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_extval(stage2_model, cfg: ExtValConfig) -> pd.DataFrame:
    """Prototype-based external validation.

    1. groupA  : cfg.data_root 의 split_names (val/test)
    2. extval  : cfg.ext_val_dir  (None 이면 skip)

    Returns a combined ttest_df (both groups stacked).
    """
    base_out = Path(cfg.work_dir) / 'output' / 'extval' / cfg.model_id
    all_results: list[pd.DataFrame] = []

    # ── Group A: internal val/test splits ────────────────────
    all_hcc_paths_A: list[Path] = []
    all_hem_paths_A: list[Path] = []
    for split in cfg.split_names:
        paths = _collect_image_paths(cfg.data_root, split)
        all_hcc_paths_A.extend(paths['hcc'])
        all_hem_paths_A.extend(paths['hemangioma'])

    print(f'[groupA] hcc={len(all_hcc_paths_A)}  hemangioma={len(all_hem_paths_A)}')
    ttest_A, _ = _eval_block(
        stage2_model,
        hcc_paths  = all_hcc_paths_A,
        hem_paths  = all_hem_paths_A,
        cfg        = cfg,
        out_dir    = base_out / 'groupA',
        group_tag  = 'groupA',
    )
    ttest_A.insert(0, 'group', 'groupA')
    all_results.append(ttest_A)

    # ── Ext-Val: Atlas (or any separate dataset) ─────────────
    if cfg.ext_val_dir is not None:
        print(f'[extval] loading from {cfg.ext_val_dir}')
        ext_paths  = _collect_extval_paths(cfg.ext_val_dir)
        ttest_ext, _ = _eval_block(
            stage2_model,
            hcc_paths  = ext_paths['hcc'],
            hem_paths  = ext_paths['hemangioma'],
            cfg        = cfg,
            out_dir    = base_out / 'extval',
            group_tag  = 'extval',
        )
        ttest_ext.insert(0, 'group', 'extval')
        all_results.append(ttest_ext)
    else:
        print('[extval] ext_val_dir is None — skipping external validation.')

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(base_out / 'combined_ttest.csv', index=False)
    return combined
