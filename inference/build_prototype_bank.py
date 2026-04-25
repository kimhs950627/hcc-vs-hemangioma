
from __future__ import annotations

import argparse
import pathlib
import numpy as np
import tensorflow as tf

from dataloader import build_dataset
from models.encoder import build_classifier


class SimpleKMeans:
    def __init__(self, n_clusters: int = 8, n_iter: int = 50, seed: int = 42):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.seed = seed
        self.cluster_centers_ = None

    def fit(self, x: np.ndarray):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(x), size=min(self.n_clusters, len(x)), replace=False)
        centers = x[idx].copy()
        for _ in range(self.n_iter):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
            labels = dist.argmin(axis=1)
            new_centers = []
            for k in range(len(centers)):
                members = x[labels == k]
                if len(members) == 0:
                    new_centers.append(centers[k])
                else:
                    new_centers.append(members.mean(axis=0))
            new_centers = np.stack(new_centers, axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        self.cluster_centers_ = centers
        return self


def collect_embeddings(model, dataset, target_label: int) -> np.ndarray:
    embs = []
    for images, labels in dataset:
        mask = tf.equal(labels, target_label)
        if not tf.reduce_any(mask):
            continue
        out = model(images, training=False)
        emb = tf.boolean_mask(out['embedding'], mask)
        embs.append(emb.numpy())
    if not embs:
        raise RuntimeError(f'No samples found for target label {target_label}.')
    return np.concatenate(embs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='vit')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--n_prototypes', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='output/prototype_bank')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    ds_train, _, _ = build_dataset(
        data_root=args.data_root,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        use_augmentation=False,
        shuffle_train=False,
        shuffle_val=False,
        shuffle_test=False,
    )
    model = build_classifier(args.encoder, input_shape=(args.img_size, args.img_size, 3))
    if args.weights:
        model.load_weights(args.weights)

    hema_embs = collect_embeddings(model, ds_train, target_label=0)
    hcc_embs = collect_embeddings(model, ds_train, target_label=1)

    hema_bank = SimpleKMeans(n_clusters=args.n_prototypes).fit(hema_embs).cluster_centers_.astype(np.float32)
    hcc_bank = SimpleKMeans(n_clusters=args.n_prototypes).fit(hcc_embs).cluster_centers_.astype(np.float32)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'hemangioma_prototypes.npy', hema_bank)
    np.save(out_dir / 'hcc_prototypes.npy', hcc_bank)

    print({
        'split': 'train',
        'n_hemangioma_embeddings': int(len(hema_embs)),
        'n_hcc_embeddings': int(len(hcc_embs)),
        'n_prototypes_per_class': int(args.n_prototypes),
        'output_dir': str(out_dir),
    })


if __name__ == '__main__':
    main()
