from __future__ import annotations

import math

import keras
import tensorflow as tf
from keras import layers


class SelfPatch(layers.Layer):
    """Simple SelfPatch auxiliary module.

    Shapes
    ------
    student_patches : (B, N, C)
    teacher_patches : (B, N, C)
    output loss     : scalar

    Notes
    -----
    - Assumes CLS token is already removed.
    - Uses 8-neighborhood on the patch grid.
    - For each query patch, selects top-k similar neighboring teacher patches,
      aggregates them, and matches the student query patch to that target.
    """

    def __init__(
        self,
        proj_dim: int = 256,
        top_k: int = 4,
        temperature: float = 0.07,
        name: str = 'selfpatch',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.proj_dim = int(proj_dim)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.proj = keras.Sequential([
            layers.Dense(self.proj_dim, activation='gelu'),
            layers.Dense(self.proj_dim),
        ], name=f'{name}_proj')
        self._neighbor_index = None
        self._cached_n = None

    def _build_neighbor_index(self, n_patches: int) -> tf.Tensor:
        side = int(math.sqrt(n_patches))
        if side * side != n_patches:
            raise ValueError(
                f'SelfPatch expects square patch grid. Got n_patches={n_patches}.'
            )
        neighbors = []
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        for r in range(side):
            for c in range(side):
                idxs = []
                for dr, dc in offsets:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < side and 0 <= cc < side:
                        idxs.append(rr * side + cc)
                if not idxs:
                    idxs = [r * side + c]
                while len(idxs) < len(offsets):
                    idxs.append(idxs[-1])
                neighbors.append(idxs)
        return tf.constant(neighbors, dtype=tf.int32)

    def _get_neighbor_index(self, n_patches: int) -> tf.Tensor:
        if self._neighbor_index is None or self._cached_n != n_patches:
            self._neighbor_index = self._build_neighbor_index(n_patches)
            self._cached_n = n_patches
        return self._neighbor_index

    def call(self, student_patches: tf.Tensor, teacher_patches: tf.Tensor, training: bool = False) -> tf.Tensor:
        student_proj = tf.math.l2_normalize(self.proj(student_patches, training=training), axis=-1)  # (B,N,D)
        teacher_proj = tf.math.l2_normalize(self.proj(teacher_patches, training=False), axis=-1)     # (B,N,D)

        n_patches = int(student_proj.shape[1])
        if n_patches is None:
            raise ValueError('SelfPatch requires static patch dimension N.')
        neighbor_index = self._get_neighbor_index(n_patches)                                           # (N,Kc)

        teacher_neighbors = tf.gather(teacher_proj, neighbor_index, axis=1)                           # (B,N,Kc,D)
        sim = tf.reduce_sum(student_proj[:, :, None, :] * teacher_neighbors, axis=-1)                 # (B,N,Kc)

        k = min(self.top_k, teacher_neighbors.shape[2])
        top_vals, top_idx = tf.math.top_k(sim, k=k, sorted=False)                                     # (B,N,K)
        top_neighbors = tf.gather(teacher_neighbors, top_idx, batch_dims=2)                           # (B,N,K,D)
        weights = tf.nn.softmax(top_vals / max(self.temperature, 1e-6), axis=-1)                      # (B,N,K)
        target = tf.reduce_sum(top_neighbors * weights[..., None], axis=2)                            # (B,N,D)
        target = tf.stop_gradient(target)

        return tf.reduce_mean(tf.square(student_proj - target))
