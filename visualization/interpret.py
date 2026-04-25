
from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf

from training.losses import prototype_similarity_score
from inference.scorer import prototype_heatmap, malignancy_score


def preprocess_image_array(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> tf.Tensor:
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if tf.rank(image) == 3:
        image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, target_size)
    if tf.reduce_max(image) > 1.0:
        image = image / 255.0
    return image


def gradcam_overlay(classifier_model, image: np.ndarray, class_index: int = 1) -> np.ndarray:
    x = preprocess_image_array(image)
    with tf.GradientTape() as tape:
        out = classifier_model(x, training=False)
        fmap = out.get('feature_map', None)
        if fmap is None:
            raise ValueError('GradCAM requires feature_map output.')
        tape.watch(fmap)
        score = out['logits'][:, class_index]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * fmap, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], x.shape[1:3]).numpy()[0, ..., 0]
    img = x.numpy()[0]
    heat = np.stack([cam, np.zeros_like(cam), 1.0 - cam], axis=-1)
    return np.clip(0.55 * img + 0.45 * heat, 0.0, 1.0)


def attention_overlay(classifier_model, image: np.ndarray, head_reduction: str = 'mean') -> np.ndarray:
    x = preprocess_image_array(image)
    out = classifier_model(x, training=False)
    attn = out.get('last_encoder_layer_attentional_weights', None)
    if attn is None:
        raise ValueError('Attention overlay requires last_encoder_layer_attentional_weights output.')
    if head_reduction == 'mean':
        attn = tf.reduce_mean(attn, axis=1)
    else:
        attn = attn[:, 0, :, :]
    if len(attn.shape) == 3:
        cls_to_patch = attn[:, 0, 1:]
    else:
        cls_to_patch = attn[:, 1:]
    n = tf.shape(cls_to_patch)[-1]
    side = tf.cast(tf.math.sqrt(tf.cast(n, tf.float32)), tf.int32)
    grid = tf.reshape(cls_to_patch, [1, side, side, 1])
    grid = grid / (tf.reduce_max(grid) + 1e-8)
    grid = tf.image.resize(grid, x.shape[1:3]).numpy()[0, ..., 0]
    img = x.numpy()[0]
    heat = np.stack([grid, 1.0 - grid, np.zeros_like(grid)], axis=-1)
    return np.clip(0.55 * img + 0.45 * heat, 0.0, 1.0)


def infer_with_prototypes(
    classifier_model,
    image: np.ndarray,
    hemangioma_prototypes: np.ndarray | tf.Tensor,
    hcc_prototypes: np.ndarray | tf.Tensor,
) -> dict[str, Any]:
    x = preprocess_image_array(image)
    out = classifier_model(x, training=False)
    probs = out['probabilities'].numpy()[0]
    embedding = out['embedding']
    from inference.scorer import dual_bank_scores
    score_dict = dual_bank_scores(
        embedding,
        out.get('encoded_patches', None),
        tf.convert_to_tensor(hemangioma_prototypes, dtype=tf.float32),
        tf.convert_to_tensor(hcc_prototypes, dtype=tf.float32),
        alpha=0.5,
    )
    malignancy = score_dict['malignancy_score'].numpy()[0]
    encoder_name = classifier_model.model.encoder.name if hasattr(classifier_model, 'model') else classifier_model.encoder.name
    if 'convnext' in encoder_name or 'efficientnet' in encoder_name or 'cnn' in encoder_name:
        overlay = gradcam_overlay(classifier_model, image=image, class_index=1)
    else:
        overlay = attention_overlay(classifier_model, image=image, head_reduction='mean')
    proto_overlay = None
    if out.get('encoded_patches', None) is not None:
        proto_overlay = prototype_overlay(image=image, encoded_patches=out['encoded_patches'], prototypes=hcc_prototypes)
    return {
        'p_hemangioma': float(probs[0]),
        'p_hcc': float(probs[1]),
        'pred_label': int(np.argmax(probs)),
        'hemangioma_global_score': float(score_dict['hemangioma_global_score'].numpy()[0]),
        'hemangioma_patch_score': float(score_dict['hemangioma_patch_score'].numpy()[0]),
        'hemangioma_score': float(score_dict['hemangioma_score'].numpy()[0]),
        'hcc_global_score': float(score_dict['hcc_global_score'].numpy()[0]),
        'hcc_patch_score': float(score_dict['hcc_patch_score'].numpy()[0]),
        'hcc_score': float(score_dict['hcc_score'].numpy()[0]),
        'malignancy_score': float(malignancy),
        'overlay_image': overlay,
        'prototype_overlay_image': proto_overlay,
    }



def prototype_overlay(image: np.ndarray, encoded_patches: tf.Tensor, prototypes: np.ndarray | tf.Tensor) -> np.ndarray:
    x = preprocess_image_array(image)
    heat = prototype_heatmap(encoded_patches, tf.convert_to_tensor(prototypes, dtype=tf.float32), image_size=(x.shape[1], x.shape[2]))
    heat = heat.numpy()[0, ..., 0]
    img = x.numpy()[0]
    color = np.stack([heat, np.zeros_like(heat), 1.0 - heat], axis=-1)
    return np.clip(0.55 * img + 0.45 * color, 0.0, 1.0)
