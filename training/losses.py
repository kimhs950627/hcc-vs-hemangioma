
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


def negative_cosine_similarity(p: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
    p = tf.math.l2_normalize(p, axis=-1)
    z = tf.math.l2_normalize(z, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.stop_gradient(z), axis=-1))


def dino_cross_entropy(student_logits: tf.Tensor, teacher_logits: tf.Tensor, student_temp: float = 0.1, teacher_temp: float = 0.04) -> tf.Tensor:
    s = student_logits / student_temp
    t = tf.stop_gradient(tf.nn.softmax(teacher_logits / teacher_temp, axis=-1))
    logp = tf.nn.log_softmax(s, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(t * logp, axis=-1))



def patchify_images(images: tf.Tensor, patch_size: int = 16) -> tf.Tensor:
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )
    patch_dim = tf.shape(patches)[-1]
    return tf.reshape(patches, [tf.shape(images)[0], -1, patch_dim])


def normalize_patch_targets(patches: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    mean = tf.reduce_mean(patches, axis=-1, keepdims=True)
    var = tf.reduce_mean(tf.square(patches - mean), axis=-1, keepdims=True)
    return (patches - mean) / tf.sqrt(var + eps)


def masked_patch_l1_loss(
    pred_patches: tf.Tensor,
    target_patches: tf.Tensor,
    patch_mask: tf.Tensor,
    eps: float = 1e-6,
) -> tf.Tensor:
    patch_mask = tf.cast(patch_mask, pred_patches.dtype)
    per_patch_l1 = tf.reduce_mean(tf.abs(pred_patches - target_patches), axis=-1)
    weighted = per_patch_l1 * patch_mask
    denom = tf.reduce_sum(patch_mask) + eps
    return tf.reduce_sum(weighted) / denom


# ---------------------------------------------------------------------------
# Head Disagreement Loss
# ---------------------------------------------------------------------------

def head_disagreement_loss(attn_scores: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    """Penalise cosine similarity between attention maps of different heads.

    Args:
        attn_scores: (B, H, T, T)  – raw attention weights after softmax.
                     Typically the last-layer scores returned by the encoder.
    Returns:
        scalar diversity penalty (lower = more diverse heads).
    """
    # average over batch → (H, T, T)
    a = tf.reduce_mean(attn_scores, axis=0)           # (H, T, T)
    # flatten token-token dimension → (H, T*T)
    h = tf.shape(a)[0]
    tt = tf.shape(a)[1] * tf.shape(a)[2]
    a_flat = tf.reshape(a, [h, tt])                   # (H, T*T)

    # L2-normalise each head's attention map
    norm = tf.math.l2_normalize(a_flat, axis=-1)      # (H, T*T)

    # pairwise cosine similarity matrix → (H, H)
    sim = tf.matmul(norm, norm, transpose_b=True)     # (H, H)

    # mask diagonal (self-similarity = 1 always)
    h_int = attn_scores.shape[1] or tf.shape(attn_scores)[1]
    mask = 1.0 - tf.eye(h, dtype=attn_scores.dtype)

    # mean off-diagonal cosine sim
    n_pairs = tf.cast(h * (h - 1), sim.dtype) + tf.cast(eps, sim.dtype)
    return tf.reduce_sum(sim * mask) / n_pairs



def extract_cls_patch_attn(
    attn_scores: tf.Tensor,
    exclude_cls_col: bool = True,
) -> tf.Tensor:
    cls_row = attn_scores[:, :, 0, :]
    if exclude_cls_col:
        cls_row = cls_row[:, :, 1:]
    return cls_row


def head_cls_diversity_loss(
    attn_scores: tf.Tensor,
    exclude_cls_col: bool = True,
    eps: float = 1e-6,
) -> tf.Tensor:
    a = extract_cls_patch_attn(attn_scores, exclude_cls_col=exclude_cls_col)
    a = tf.math.l2_normalize(a, axis=-1, epsilon=eps)
    sim = tf.matmul(a, a, transpose_b=True)
    b = tf.shape(a)[0]
    h = tf.shape(a)[1]
    mask = 1.0 - tf.eye(h, batch_shape=[b], dtype=sim.dtype)
    n_pairs = tf.cast(h * (h - 1), sim.dtype) + tf.cast(eps, sim.dtype)
    per_sample = tf.reduce_sum(sim * mask, axis=[1, 2]) / n_pairs
    return tf.reduce_mean(per_sample)


def attention_entropy_floor_loss(
    attn_scores: tf.Tensor,
    exclude_cls_col: bool = True,
    entropy_min: float = 2.5,
    eps: float = 1e-8,
) -> tf.Tensor:
    a = extract_cls_patch_attn(attn_scores, exclude_cls_col=exclude_cls_col)
    a = a / (tf.reduce_sum(a, axis=-1, keepdims=True) + eps)
    ent = -tf.reduce_sum(a * tf.math.log(a + eps), axis=-1)
    return tf.reduce_mean(tf.nn.relu(tf.cast(entropy_min, ent.dtype) - ent))


def multilayer_cls_diversity_with_entropy(
    attention_list,
    layer_indices: list[int] | None = None,
    exclude_cls_col: bool = True,
    entropy_min: float = 2.5,
    entropy_weight: float = 1.0,
):
    if not attention_list:
        zero = tf.constant(0.0, dtype=tf.float32)
        return zero, zero, zero

    if layer_indices is None:
        layer_indices = list(range(len(attention_list)))

    div_terms = []
    ent_terms = []
    for idx in layer_indices:
        attn = attention_list[idx]
        div_terms.append(head_cls_diversity_loss(attn, exclude_cls_col=exclude_cls_col))
        ent_terms.append(attention_entropy_floor_loss(attn, exclude_cls_col=exclude_cls_col, entropy_min=entropy_min))

    div_loss = tf.add_n(div_terms) / tf.cast(len(div_terms), tf.float32)
    ent_loss = tf.add_n(ent_terms) / tf.cast(len(ent_terms), tf.float32)
    total = div_loss + tf.cast(entropy_weight, tf.float32) * ent_loss
    return total, div_loss, ent_loss
