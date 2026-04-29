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


def dino_cross_entropy(
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
) -> tf.Tensor:
    """DINO cross-entropy loss.

    H(t, s) = -sum_k  softmax(t/teacher_temp)_k * log_softmax(s/student_temp)_k

    Args:
        student_logits : [B, D]  raw student projector output
        teacher_logits : [B, D]  raw teacher projector output (will be stop_gradient'd)
        student_temp   : student softmax temperature
        teacher_temp   : teacher softmax temperature (small → sharper target)
    Returns:
        scalar mean cross-entropy loss
    """
    s = student_logits / student_temp
    t = tf.stop_gradient(tf.nn.softmax(teacher_logits / teacher_temp, axis=-1))
    logp = tf.nn.log_softmax(s, axis=-1)
    return -tf.reduce_mean(tf.reduce_sum(t * logp, axis=-1))


# ---------------------------------------------------------------------------
# SimMIM helpers
# ---------------------------------------------------------------------------

def patchify_images(
    images: tf.Tensor,
    patch_size: int = 16,
) -> tf.Tensor:
    """Convert pixel images to sequence of flattened patches.

    Used to build the SimMIM reconstruction target.

    Args:
        images     : [B, H, W, C]  float32  (values in [0, 1])
        patch_size : ViT patch size P (must evenly divide H and W)

    Returns:
        patches    : [B, N, D]  float32
                     N = (H // P) * (W // P)
                     D = P * P * C
    """
    x = tf.cast(images, tf.float32)
    B = tf.shape(x)[0]
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    C = tf.shape(x)[3]
    P = patch_size

    n_h = H // P
    n_w = W // P

    # [B, n_h, P, n_w, P, C]
    x = tf.reshape(x, [B, n_h, P, n_w, P, C])
    # [B, n_h, n_w, P, P, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    # [B, N, D]
    x = tf.reshape(x, [B, n_h * n_w, P * P * C])
    return x


def simmim_l1_loss(
    pred: tf.Tensor,
    target: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """SimMIM pixel-level L1 reconstruction loss over masked patches only.

    Loss = (1 / |\u03a9_M|) * \u03a3_{p \u2208 \u03a9_M}  |x_p - x_p_hat|_1

    where \u03a9_M = set of masked patch positions.
    Only masked positions (mask==1) contribute to the loss;
    visible patches are explicitly excluded.

    Args:
        pred   : [B, N, D]  float32  \u2014 pixel_pred_head output
                 D = patch_size * patch_size * C
        target : [B, N, D]  float32  \u2014 patchified original_clean
                 computed with patchify_images(original_clean, patch_size)
        mask   : [B, N]     float32  \u2014 1 = masked (predict), 0 = visible (ignore)
                 output of RandomMaskGenerator in dataloader.py

    Returns:
        loss   : scalar float32  \u2014 mean L1 over masked positions
                 returns 0.0 if mask is all-zero (edge case guard)

    Design notes:
      - L1 is used instead of L2 (RMSE) because US images have sparse
        high-frequency structures (vessel walls, calcifications).
        L1 penalizes large errors less aggressively and is more robust
        to intensity outliers typical in B-mode ultrasound.
      - Normalization by |\u03a9_M| ensures loss is independent of mask_ratio,
        so lambda_mim scale doesn't need to be re-tuned when mask_ratio changes.
      - stop_gradient on target is NOT applied here because the target
        (original_clean) is already detached from any trainable path
        (teacher encoder is EMA, not backpropagated through).

    References:
        Xie et al. (2022) SimMIM, https://arxiv.org/abs/2111.09886
        \u00a7 3.1: L = (1/|\u03a9_M|) \u03a3 |x_i - \u0302x_i|_1 for i \u2208 \u03a9_M
    """
    pred   = tf.cast(pred,   tf.float32)   # [B, N, D]
    target = tf.cast(target, tf.float32)   # [B, N, D]
    mask   = tf.cast(mask,   tf.float32)   # [B, N]

    # Per-patch mean-absolute-error: [B, N]
    per_patch_l1 = tf.reduce_mean(tf.abs(pred - target), axis=-1)

    # Apply mask: sum over masked patches, normalize by |\u03a9_M|
    # mask_sum: [B]
    mask_sum  = tf.reduce_sum(mask, axis=-1)            # number of masked patches per sample
    loss_per_sample = tf.reduce_sum(per_patch_l1 * mask, axis=-1) / (
        mask_sum + 1e-8
    )                                                    # guard: empty mask -> 0

    return tf.reduce_mean(loss_per_sample)              # scalar
