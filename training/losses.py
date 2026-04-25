
from __future__ import annotations

import tensorflow as tf
import keras


def l2_normalize(x: tf.Tensor, axis: int = -1, eps: float = 1e-6) -> tf.Tensor:
    return tf.math.l2_normalize(x, axis=axis, epsilon=eps)


def info_nce_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    z1 = l2_normalize(z1)
    z2 = l2_normalize(z2)
    logits = tf.matmul(z1, z2, transpose_b=True) / temperature
    labels = tf.range(tf.shape(z1)[0])
    loss_12 = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss_21 = keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    return 0.5 * (tf.reduce_mean(loss_12) + tf.reduce_mean(loss_21))


def supervised_contrastive_loss(labels: tf.Tensor, features: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    labels = tf.reshape(labels, [-1])
    features = l2_normalize(features)
    logits = tf.matmul(features, features, transpose_b=True) / temperature
    logits_mask = 1.0 - tf.eye(tf.shape(features)[0])
    mask = tf.cast(tf.equal(labels[:, None], labels[None, :]), tf.float32) * logits_mask
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-9)
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-9)
    return -tf.reduce_mean(mean_log_prob_pos)


def prototype_similarity_score(embedding: tf.Tensor, prototypes: tf.Tensor, reduction: str = 'max') -> tf.Tensor:
    embedding = l2_normalize(embedding)
    prototypes = l2_normalize(prototypes)
    sim = tf.matmul(embedding, prototypes, transpose_b=True)
    if reduction == 'max':
        return tf.reduce_max(sim, axis=-1)
    if reduction == 'mean':
        return tf.reduce_mean(sim, axis=-1)
    return sim
