
from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from training.losses import l2_normalize


def cosine_similarity_matrix(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = l2_normalize(x)
    y = l2_normalize(y)
    return tf.matmul(x, y, transpose_b=True)


def global_prototype_score(embedding: tf.Tensor, prototypes: tf.Tensor, reduction: str = 'max') -> tf.Tensor:
    sim = cosine_similarity_matrix(embedding, prototypes)
    if reduction == 'max':
        return tf.reduce_max(sim, axis=-1)
    if reduction == 'mean':
        return tf.reduce_mean(sim, axis=-1)
    return sim


def patch_prototype_similarity(encoded_patches: tf.Tensor, prototypes: tf.Tensor) -> tf.Tensor:
    b = tf.shape(encoded_patches)[0]
    n = tf.shape(encoded_patches)[1]
    d = tf.shape(encoded_patches)[2]
    flat = tf.reshape(encoded_patches, [b * n, d])
    sim = cosine_similarity_matrix(flat, prototypes)
    return tf.reshape(sim, [b, n, tf.shape(prototypes)[0]])


def patch_level_score(encoded_patches: tf.Tensor, prototypes: tf.Tensor, patch_reduction: str = 'max', image_reduction: str = 'mean') -> tf.Tensor:
    sim = patch_prototype_similarity(encoded_patches, prototypes)
    if patch_reduction == 'max':
        per_patch = tf.reduce_max(sim, axis=-1)
    elif patch_reduction == 'mean':
        per_patch = tf.reduce_mean(sim, axis=-1)
    else:
        per_patch = sim
    if image_reduction == 'mean':
        return tf.reduce_mean(per_patch, axis=-1)
    if image_reduction == 'max':
        return tf.reduce_max(per_patch, axis=-1)
    return per_patch


def malignancy_score(
    embedding: tf.Tensor,
    encoded_patches: tf.Tensor | None,
    prototypes: tf.Tensor,
    alpha: float = 0.5,
) -> dict[str, tf.Tensor]:
    g = global_prototype_score(embedding, prototypes, reduction='max')
    if encoded_patches is None:
        return {'global_score': g, 'patch_score': g, 'malignancy_score': g}
    p = patch_level_score(encoded_patches, prototypes, patch_reduction='max', image_reduction='mean')
    score = alpha * g + (1.0 - alpha) * p
    return {'global_score': g, 'patch_score': p, 'malignancy_score': score}


def prototype_heatmap(encoded_patches: tf.Tensor, prototypes: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    sim = patch_prototype_similarity(encoded_patches, prototypes)
    patch_max = tf.reduce_max(sim, axis=-1)
    n = tf.shape(patch_max)[1]
    side = tf.cast(tf.math.sqrt(tf.cast(n, tf.float32)), tf.int32)
    heat = tf.reshape(patch_max, [tf.shape(encoded_patches)[0], side, side, 1])
    heat = heat / (tf.reduce_max(heat, axis=(1, 2, 3), keepdims=True) + 1e-8)
    heat = tf.image.resize(heat, image_size)
    return heat


def infer_scores_from_outputs(outputs: dict[str, Any], prototypes: np.ndarray | tf.Tensor, alpha: float = 0.5) -> dict[str, Any]:
    proto = tf.convert_to_tensor(prototypes, dtype=tf.float32)
    scores = malignancy_score(
        embedding=outputs['embedding'],
        encoded_patches=outputs.get('encoded_patches', None),
        prototypes=proto,
        alpha=alpha,
    )
    return {k: v.numpy() for k, v in scores.items()}
