from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import math

import numpy as np
import pandas as pd
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


def _require_wandb():
    if wandb is None:
        raise ImportError('wandb is required. Install with `pip install wandb`.')


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


def _pca_3d(hcc_rep: np.ndarray, hem_rep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_rep = np.concatenate([hcc_rep, hem_rep], axis=0)
    pca = PCA(n_components=3)
    all_3d = pca.fit_transform(all_rep)
    n_hcc = len(hcc_rep)
    return all_3d[:n_hcc], all_3d[n_hcc:], pca.explained_variance_ratio_


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
        'mean_a': float(np.mean(a)),
        'std_a': float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        'mean_b': float(np.mean(b)),
        'std_b': float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        't_stat': float(test.statistic),
        'p_value': float(test.pvalue),
        'n_a': int(len(a)),
        'n_b': int(len(b)),
    }


def _log_scatter(split: str, model_id: str, hcc_3d: np.ndarray, hem_3d: np.ndarray, hcc_paths: Sequence[Path], hem_paths: Sequence[Path], prefix: str) -> None:
    if wandb is None or wandb.run is None:
        return
    table = wandb.Table(columns=['x', 'y', 'z', 'label', 'path', 'split', 'model_id'])
    for xyz, p in zip(hcc_3d, hcc_paths):
        table.add_data(float(xyz[0]), float(xyz[1]), float(xyz[2]), 'HCC', str(p), split, model_id)
    for xyz, p in zip(hem_3d, hem_paths):
        table.add_data(float(xyz[0]), float(xyz[1]), float(xyz[2]), 'Hemangioma', str(p), split, model_id)
    wandb.log({f'{prefix}/{model_id}/{split}/pca3d_table': table})


def _log_similarity_table(split: str, model_id: str, hcc_paths: Sequence[Path], hem_paths: Sequence[Path], A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, prefix: str) -> None:
    if wandb is None or wandb.run is None:
        return
    table = wandb.Table(columns=['path', 'label', 'sim_to_hcc_proto', 'sim_to_hem_proto', 'margin', 'split', 'model_id'])
    for p, a, b in zip(hcc_paths, A, B):
        table.add_data(str(p), 'HCC', float(a), float(b), float(a - b), split, model_id)
    for p, c, d in zip(hem_paths, C, D):
        table.add_data(str(p), 'Hemangioma', float(c), float(d), float(c - d), split, model_id)
    wandb.log({f'{prefix}/{model_id}/{split}/similarity_table': table})


def _save_stats(path: Path, stats: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(path, index=False)


def run_split_extval(stage2_model, split: str, cfg: ExtValConfig) -> dict[str, float]:
    paths = _collect_image_paths(cfg.data_root, split)
    hcc_paths = paths['hcc']
    hem_paths = paths['hemangioma']

    hcc_rep = extract_representations(stage2_model, hcc_paths, cfg.image_size, cfg.batch_size)
    hem_rep = extract_representations(stage2_model, hem_paths, cfg.image_size, cfg.batch_size)

    out_dir = Path(cfg.work_dir) / 'output' / 'extval' / cfg.model_id / split
    _save_matrix(out_dir / 'hcc_rep', hcc_rep, 'emb')
    _save_matrix(out_dir / 'hemangioma_rep', hem_rep, 'emb')
    _save_paths(out_dir / 'hcc_paths.csv', hcc_paths)
    _save_paths(out_dir / 'hemangioma_paths.csv', hem_paths)

    hcc_3d, hem_3d, pca_var = _pca_3d(hcc_rep, hem_rep)
    _save_matrix(out_dir / 'hcc_3d_rep', hcc_3d, 'pc')
    _save_matrix(out_dir / 'hemangioma_3d_rep', hem_3d, 'pc')
    pd.DataFrame([{'pc1': float(pca_var[0]), 'pc2': float(pca_var[1]), 'pc3': float(pca_var[2])}]).to_csv(out_dir / 'pca_explained_variance_ratio.csv', index=False)
    _log_scatter(split, cfg.model_id, hcc_3d, hem_3d, hcc_paths, hem_paths, cfg.wandb_prefix)

    hcc_proto = _fit_prototypes(hcc_rep, cfg.n_proto, cfg.random_state)
    hem_proto = _fit_prototypes(hem_rep, cfg.n_proto, cfg.random_state)
    _save_matrix(out_dir / 'hcc_prototype', hcc_proto, 'proto')
    _save_matrix(out_dir / 'hemangioma_prototype', hem_proto, 'proto')

    A = _mean_similarity(hcc_rep, hcc_proto)
    B = _mean_similarity(hcc_rep, hem_proto)
    C = _mean_similarity(hem_rep, hcc_proto)
    D = _mean_similarity(hem_rep, hem_proto)

    pd.DataFrame({
        'path': [str(p) for p in hcc_paths],
        'label': ['HCC'] * len(hcc_paths),
        'sim_to_hcc_proto': A,
        'sim_to_hem_proto': B,
        'margin': A - B,
    }).to_csv(out_dir / 'hcc_similarity.csv', index=False)
    pd.DataFrame({
        'path': [str(p) for p in hem_paths],
        'label': ['Hemangioma'] * len(hem_paths),
        'sim_to_hcc_proto': C,
        'sim_to_hem_proto': D,
        'margin': C - D,
    }).to_csv(out_dir / 'hemangioma_similarity.csv', index=False)
    _log_similarity_table(split, cfg.model_id, hcc_paths, hem_paths, A, B, C, D, cfg.wandb_prefix)

    ttest_ac = _ttest(A, C)
    ttest_margin = _ttest(A - B, C - D)
    stats = {
        'split': split,
        'model_id': cfg.model_id,
        'n_hcc': int(len(hcc_paths)),
        'n_hemangioma': int(len(hem_paths)),
        'embed_dim': int(hcc_rep.shape[1]),
        'n_hcc_prototype': int(hcc_proto.shape[0]),
        'n_hemangioma_prototype': int(hem_proto.shape[0]),
        'A_hcc_to_hccproto_mean': float(np.mean(A)),
        'B_hcc_to_hemproto_mean': float(np.mean(B)),
        'C_hem_to_hccproto_mean': float(np.mean(C)),
        'D_hem_to_hemproto_mean': float(np.mean(D)),
        'A_minus_B_mean': float(np.mean(A - B)),
        'C_minus_D_mean': float(np.mean(C - D)),
        'ttest_AC_t': ttest_ac['t_stat'],
        'ttest_AC_p': ttest_ac['p_value'],
        'ttest_margin_t': ttest_margin['t_stat'],
        'ttest_margin_p': ttest_margin['p_value'],
        'pca_pc1': float(pca_var[0]),
        'pca_pc2': float(pca_var[1]),
        'pca_pc3': float(pca_var[2]),
    }
    _save_stats(out_dir / 'summary_stats.csv', stats)
    _save_stats(out_dir / 'ttest_ac.csv', {'split': split, 'model_id': cfg.model_id, **ttest_ac})
    _save_stats(out_dir / 'ttest_margin.csv', {'split': split, 'model_id': cfg.model_id, **ttest_margin})

    if wandb is not None and wandb.run is not None:
        wandb.log({
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/A_hcc_to_hccproto_mean': stats['A_hcc_to_hccproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/B_hcc_to_hemproto_mean': stats['B_hcc_to_hemproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/C_hem_to_hccproto_mean': stats['C_hem_to_hccproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/D_hem_to_hemproto_mean': stats['D_hem_to_hemproto_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/A_minus_B_mean': stats['A_minus_B_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/C_minus_D_mean': stats['C_minus_D_mean'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/ttest_AC_t': stats['ttest_AC_t'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/ttest_AC_p': stats['ttest_AC_p'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/ttest_margin_t': stats['ttest_margin_t'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/ttest_margin_p': stats['ttest_margin_p'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/n_hcc': stats['n_hcc'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/n_hemangioma': stats['n_hemangioma'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/n_hcc_prototype': stats['n_hcc_prototype'],
            f'{cfg.wandb_prefix}/{cfg.model_id}/{split}/n_hemangioma_prototype': stats['n_hemangioma_prototype'],
        })
    return stats


def run_extval(stage2_model, cfg: ExtValConfig) -> pd.DataFrame:
    rows = [run_split_extval(stage2_model, split, cfg) for split in cfg.split_names]
    summary = pd.DataFrame(rows)
    out_path = Path(cfg.work_dir) / 'output' / 'extval' / cfg.model_id / 'summary_all_splits.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    if wandb is not None and wandb.run is not None:
        wandb.log({f'{cfg.wandb_prefix}/{cfg.model_id}/summary_all_splits': wandb.Table(dataframe=summary)})
    return summary
