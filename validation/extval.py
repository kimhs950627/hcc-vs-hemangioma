from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans

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
    data_root: str
    work_dir: str
    model_id: str
    image_size: tuple[int, int] = (384, 384)
    batch_size: int = 32
    n_proto: int = 8
    split_names: tuple[str, ...] = ('val', 'test')
    stage_name: str = 'stage2'
    wandb_prefix: str = 'extval'
    random_state: int = 42


def _find_class_dir(split_dir: Path, class_key: str) -> Path:
    for name in CLASS_DIR_CANDIDATES[class_key]:
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'class directory for {class_key!r} not found under {split_dir}')


def _collect_image_paths(data_root: str, split: str) -> dict[str, list[Path]]:
    split_dir = Path(data_root) / SPLIT_DIRS[split]
    if not split_dir.exists():
        raise FileNotFoundError(f'split directory not found: {split_dir}')
    out: dict[str, list[Path]] = {}
    for class_key in ('hcc', 'hemangioma'):
        cdir = _find_class_dir(split_dir, class_key)
        paths = sorted([p for p in cdir.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS])
        out[class_key] = paths
    return out


def _load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _chunked(seq: Sequence[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def _extract_encoder_output(stage2_model, batch: np.ndarray):
    encoder = stage2_model.model.encoder if hasattr(stage2_model, 'model') and hasattr(stage2_model.model, 'encoder') else stage2_model.encoder
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


def extract_representations(stage2_model, image_paths: Sequence[Path], image_size: tuple[int, int], batch_size: int) -> np.ndarray:
    reps: list[np.ndarray] = []
    for chunk in _chunked(image_paths, batch_size):
        batch = np.stack([_load_image(p, image_size) for p in chunk], axis=0)
        reps.append(_extract_encoder_output(stage2_model, batch))
    if not reps:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(reps, axis=0)


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


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _fit_prototypes(rep: np.ndarray, n_proto: int, random_state: int) -> np.ndarray:
    n_clusters = min(n_proto, len(rep))
    if n_clusters < 1:
        raise ValueError('empty representation matrix')
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    km.fit(rep)
    return km.cluster_centers_.astype(np.float32)


def _mean_similarity(rep: np.ndarray, proto: np.ndarray) -> np.ndarray:
    rep_n = _l2_normalize(rep)
    proto_n = _l2_normalize(proto)
    sim = rep_n @ proto_n.T
    return sim.mean(axis=-1)


def _ttest(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    test = ttest_ind(a, b, equal_var=False)
    return {
        't_value': float(test.statistic),
        'p_value': float(test.pvalue),
    }


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


def _make_barplot(title: str, ylabel: str, means: list[float], errors: list[float], labels: list[str]):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    colors = ['#4F81BD', '#C0504D']
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, color=colors[:len(labels)], capsize=6, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def _log_summary_plots(model_id: str, prefix: str, stats: dict[str, float], ttest_df: pd.DataFrame):
    if wandb is None or wandb.run is None:
        return
    labels = ['HCC group', 'Hemangioma group']

    fig_hcc = _make_barplot(
        title='HCC similarity',
        ylabel='Mean similarity (a.u.)',
        means=[stats['A_hcc_to_hccproto_mean'], stats['C_hem_to_hccproto_mean']],
        errors=[stats['A_hcc_to_hccproto_sem'], stats['C_hem_to_hccproto_sem']],
        labels=labels,
    )
    fig_hem = _make_barplot(
        title='Hemangioma similarity',
        ylabel='Mean similarity (a.u.)',
        means=[stats['B_hcc_to_hemproto_mean'], stats['D_hem_to_hemproto_mean']],
        errors=[stats['B_hcc_to_hemproto_sem'], stats['D_hem_to_hemproto_sem']],
        labels=labels,
    )
    fig_margin = _make_barplot(
        title='HCC sim - Hemangioma sim',
        ylabel='Similarity margin (a.u.)',
        means=[stats['A_minus_B_mean'], stats['C_minus_D_mean']],
        errors=[stats['A_minus_B_sem'], stats['C_minus_D_sem']],
        labels=labels,
    )

    wandb.log({
        f'{prefix}/{model_id}/hcc_similarity_bar': wandb.Image(fig_hcc),
        f'{prefix}/{model_id}/hemangioma_similarity_bar': wandb.Image(fig_hem),
        f'{prefix}/{model_id}/margin_similarity_bar': wandb.Image(fig_margin),
        f'{prefix}/{model_id}/ttest_table': wandb.Table(dataframe=ttest_df),
    })
    plt.close(fig_hcc)
    plt.close(fig_hem)
    plt.close(fig_margin)


def run_extval(stage2_model, cfg: ExtValConfig) -> pd.DataFrame:
    all_hcc_paths: list[Path] = []
    all_hem_paths: list[Path] = []
    all_hcc_rep: list[np.ndarray] = []
    all_hem_rep: list[np.ndarray] = []

    for split in cfg.split_names:
        paths = _collect_image_paths(cfg.data_root, split)
        hcc_paths = paths['hcc']
        hem_paths = paths['hemangioma']
        hcc_rep = extract_representations(stage2_model, hcc_paths, cfg.image_size, cfg.batch_size)
        hem_rep = extract_representations(stage2_model, hem_paths, cfg.image_size, cfg.batch_size)
        all_hcc_paths.extend(hcc_paths)
        all_hem_paths.extend(hem_paths)
        all_hcc_rep.append(hcc_rep)
        all_hem_rep.append(hem_rep)

    hcc_rep = np.concatenate(all_hcc_rep, axis=0)
    hem_rep = np.concatenate(all_hem_rep, axis=0)

    out_dir = Path(cfg.work_dir) / 'output' / 'extval' / cfg.model_id / 'groupA'
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_matrix(out_dir / 'groupA_hcc_rep', hcc_rep, 'emb')
    _save_matrix(out_dir / 'groupA_hemangioma_rep', hem_rep, 'emb')
    _save_paths(out_dir / 'groupA_hcc_paths.csv', all_hcc_paths)
    _save_paths(out_dir / 'groupA_hemangioma_paths.csv', all_hem_paths)

    hcc_proto = _fit_prototypes(hcc_rep, cfg.n_proto, cfg.random_state)
    hem_proto = _fit_prototypes(hem_rep, cfg.n_proto, cfg.random_state)
    _save_matrix(out_dir / 'groupA_hcc_prototype', hcc_proto, 'proto')
    _save_matrix(out_dir / 'groupA_hemangioma_prototype', hem_proto, 'proto')

    A = _mean_similarity(hcc_rep, hcc_proto)
    B = _mean_similarity(hcc_rep, hem_proto)
    C = _mean_similarity(hem_rep, hcc_proto)
    D = _mean_similarity(hem_rep, hem_proto)
    margin_hcc = A - B
    margin_hem = C - D

    stats = {
        'model_id': cfg.model_id,
        'n_hcc': int(len(hcc_rep)),
        'n_hemangioma': int(len(hem_rep)),
        'embed_dim': int(hcc_rep.shape[1]),
        'n_hcc_prototype': int(hcc_proto.shape[0]),
        'n_hemangioma_prototype': int(hem_proto.shape[0]),
        'A_hcc_to_hccproto_mean': float(np.mean(A)),
        'A_hcc_to_hccproto_sem': _sem(A),
        'B_hcc_to_hemproto_mean': float(np.mean(B)),
        'B_hcc_to_hemproto_sem': _sem(B),
        'C_hem_to_hccproto_mean': float(np.mean(C)),
        'C_hem_to_hccproto_sem': _sem(C),
        'D_hem_to_hemproto_mean': float(np.mean(D)),
        'D_hem_to_hemproto_sem': _sem(D),
        'A_minus_B_mean': float(np.mean(margin_hcc)),
        'A_minus_B_sem': _sem(margin_hcc),
        'C_minus_D_mean': float(np.mean(margin_hem)),
        'C_minus_D_sem': _sem(margin_hem),
    }

    ttest_hcc = _ttest(A, C)
    ttest_margin = _ttest(margin_hcc, margin_hem)
    ttest_df = pd.DataFrame([
        {'metric': 'hcc_similarity', 't_value': ttest_hcc['t_value'], 'p_value': ttest_hcc['p_value']},
        {'metric': 'hcc_minus_hem_similarity', 't_value': ttest_margin['t_value'], 'p_value': ttest_margin['p_value']},
    ])

    pd.DataFrame([stats]).to_csv(out_dir / 'groupA_summary_stats.csv', index=False)
    ttest_df.to_csv(out_dir / 'groupA_ttest.csv', index=False)

    if wandb is not None and wandb.run is not None:
        wandb.log({
            f'{cfg.wandb_prefix}/{cfg.model_id}/n_hcc': stats['n_hcc'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/n_hemangioma': stats['n_hemangioma'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/A_hcc_to_hccproto_mean': stats['A_hcc_to_hccproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/B_hcc_to_hemproto_mean': stats['B_hcc_to_hemproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/C_hem_to_hccproto_mean': stats['C_hem_to_hccproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/D_hem_to_hemproto_mean': stats['D_hem_to_hemproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/A_minus_B_mean': stats['A_minus_B_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/C_minus_D_mean': stats['C_minus_D_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/ttest_hcc_similarity_t': ttest_hcc['t_value'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/ttest_hcc_similarity_p': ttest_hcc['p_value'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/ttest_margin_t': ttest_margin['t_value'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/ttest_margin_p': ttest_margin['p_value'],
        })
        _log_summary_plots(cfg.model_id, cfg.wandb_prefix, stats, ttest_df)

    return ttest_df
