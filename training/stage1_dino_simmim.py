from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import (
    dino_cross_entropy,
    masked_patch_l1_loss,
    normalize_patch_targets,
    patchify_images,
)
from training.ssl_schedules import LrInput


class DINOSimMIMPretrainModel(keras.Model):
    """DINO + SimMIM hybrid SSL trainer.

    Expected batch structure from MaskedMultiViewDataset:
        (original_clean, masked_clean, patch_mask, aug_global, local_1, ..., local_N)

    - original_clean : teacher input and SimMIM target source
    - masked_clean   : student input for SimMIM branch
    - patch_mask     : [B, N] binary mask, 1=masked, 0=visible
    - aug_global     : student global view for DINO
    - local_i        : student local views for DINO
    """

    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
        n_local: int = 0,
        lambda_simmim: float = 0.3,
        patch_size: int = 16,
        simmim_norm_target: bool = True,
        use_pe: bool = False,
        encoder_embed_dim: int = 384,
        encoder_depth: int | None = None,
        encoder_num_heads: int | None = None,
        encoder_mlp_dim: int | None = None,
        encoder_patch_size: int | None = None,
        encoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum = ema_momentum
        self.n_local = n_local
        self.lambda_simmim = float(lambda_simmim)
        self.patch_size = int(patch_size)
        self.simmim_norm_target = bool(simmim_norm_target)
        self.use_pe = bool(use_pe)

        if input_shape[0] % self.patch_size != 0 or input_shape[1] % self.patch_size != 0:
            raise ValueError(
                f"input_shape spatial dims must be divisible by patch_size. "
                f"Got input_shape={input_shape}, patch_size={patch_size}"
            )

        self.online_encoder = build_encoder(
            encoder_name,
            input_shape=input_shape,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            patch_size=encoder_patch_size,
            dropout=encoder_dropout,
            use_pe=self.use_pe,
        )
        self.teacher_encoder = build_encoder(
            encoder_name,
            input_shape=input_shape,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            patch_size=encoder_patch_size,
            dropout=encoder_dropout,
            use_pe=self.use_pe,
        )
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

        patch_dim = self.patch_size * self.patch_size * int(input_shape[-1])
        self.simmim_head = keras.Sequential([
            layers.Dense(patch_dim, name='simmim_reconstruction_head'),
        ], name='simmim_head')

        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False, name='dino_center'
        )
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var'
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    def _student_logits(self, x, training=True):
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def _teacher_logits(self, x):
        out = self.teacher_encoder(x, training=False)
        return self.teacher_projector(out['embedding'], training=False)

    def _student_patch_predictions(self, x, training=True):
        out = self.online_encoder(x, training=training)
        encoded_patches = out['encoded_patches']
        return self.simmim_head(encoded_patches, training=training)

    def _init_teacher(self):
        if self._teacher_initialized:
            return
        for t_w, s_w in zip(self.teacher_encoder.weights, self.online_encoder.weights):
            t_w.assign(s_w)
        for t_w, s_w in zip(self.teacher_projector.weights, self.projector.weights):
            t_w.assign(s_w)
        self._teacher_initialized = True

    def _ema_update(self):
        m = self.ema_momentum
        for t_w, s_w in zip(self.teacher_encoder.weights, self.online_encoder.weights):
            t_w.assign(m * t_w + (1.0 - m) * s_w)
        for t_w, s_w in zip(self.teacher_projector.weights, self.projector.weights):
            t_w.assign(m * t_w + (1.0 - m) * s_w)

    def _update_center(self, t1_logits, t2_logits):
        batch_center = tf.reduce_mean(
            tf.concat([t1_logits, t2_logits], axis=0), axis=0, keepdims=True
        )
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _compute_dino_loss(self, original_clean, aug_global, local_views):
        t1 = tf.stop_gradient(
            tf.nn.softmax(
                (self._teacher_logits(original_clean) - self.center) / self.teacher_temp_var,
                axis=-1,
            )
        )
        t2 = tf.stop_gradient(
            tf.nn.softmax(
                (self._teacher_logits(aug_global) - self.center) / self.teacher_temp_var,
                axis=-1,
            )
        )

        s_all = [self._student_logits(original_clean), self._student_logits(aug_global)]
        for i in range(self.n_local):
            s_all.append(self._student_logits(local_views[i]))

        loss = tf.constant(0.0, dtype=tf.float32)
        n_pairs = 0
        for i, s in enumerate(s_all):
            for j, t in enumerate([t1, t2]):
                if i == j and i < 2:
                    continue
                loss = loss + dino_cross_entropy(
                    s,
                    t,
                    student_temp=self.student_temp,
                    teacher_temp=1.0,
                )
                n_pairs += 1
        return loss / tf.cast(n_pairs, tf.float32), t1, t2

    def _compute_simmim_loss(self, original_clean, masked_clean, patch_mask):
        pred_patches = self._student_patch_predictions(masked_clean, training=True)
        target_patches = patchify_images(original_clean, patch_size=self.patch_size)
        if self.simmim_norm_target:
            target_patches = normalize_patch_targets(target_patches)
        return masked_patch_l1_loss(pred_patches, target_patches, patch_mask)

    def train_step(self, data):
        views = tf.nest.flatten(data)
        original_clean = views[0]
        masked_clean = views[1]
        patch_mask = views[2]
        aug_global = views[3]
        local_views = tuple(views[4 + i] for i in range(self.n_local))

        self._init_teacher()

        with tf.GradientTape() as tape:
            dino_loss, t1, t2 = self._compute_dino_loss(original_clean, aug_global, local_views)
            simmim_loss = self._compute_simmim_loss(original_clean, masked_clean, patch_mask)
            lambda_s = tf.cast(self.lambda_simmim, tf.float32)
            total_loss = (1.0 - lambda_s) * dino_loss + lambda_s * simmim_loss

        trainable_vars = (
            list(self.online_encoder.trainable_variables)
            + list(self.projector.trainable_variables)
            + list(self.simmim_head.trainable_variables)
        )
        grads = tape.gradient(total_loss, trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        if not grads_and_vars:
            raise ValueError('No gradients found for online encoder/projector/simmim_head variables.')
        self.optimizer.apply_gradients(grads_and_vars)
        self._ema_update()
        self._update_center(self._teacher_logits(original_clean), self._teacher_logits(aug_global))
        return {
            'loss': total_loss,
            'dino_loss': dino_loss,
            'simmim_loss': simmim_loss,
            'lambda_simmim': tf.cast(self.lambda_simmim, tf.float32),
        }

    def test_step(self, data):
        views = tf.nest.flatten(data)
        original_clean = views[0]
        masked_clean = views[1]
        patch_mask = views[2]
        aug_global = views[3]
        local_views = tuple(views[4 + i] for i in range(self.n_local))

        dino_loss, _, _ = self._compute_dino_loss(original_clean, aug_global, local_views)
        simmim_loss = self._compute_simmim_loss(original_clean, masked_clean, patch_mask)
        lambda_s = tf.cast(self.lambda_simmim, tf.float32)
        total_loss = (1.0 - lambda_s) * dino_loss + lambda_s * simmim_loss
        return {
            'loss': total_loss,
            'dino_loss': dino_loss,
            'simmim_loss': simmim_loss,
            'lambda_simmim': tf.cast(self.lambda_simmim, tf.float32),
        }

    def get_stage2_encoder(self, use_teacher: bool = True) -> keras.Model:
        return self.teacher_encoder if use_teacher else self.online_encoder



def build_stage1_dino_simmim_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    n_local: int = 0,
    lambda_simmim: float = 0.3,
    patch_size: int = 16,
    simmim_norm_target: bool = True,
    use_pe: bool = False,
    encoder_embed_dim: int = 384,
    encoder_depth: int | None = None,
    encoder_num_heads: int | None = None,
    encoder_mlp_dim: int | None = None,
    encoder_patch_size: int | None = None,
    encoder_dropout: float = 0.1,
    lr: LrInput = 1e-4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
) -> DINOSimMIMPretrainModel:
    model = DINOSimMIMPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
        lambda_simmim=lambda_simmim,
        patch_size=patch_size,
        simmim_norm_target=simmim_norm_target,
        use_pe=use_pe,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_mlp_dim=encoder_mlp_dim,
        encoder_patch_size=encoder_patch_size,
        encoder_dropout=encoder_dropout,
    )
    opt_kwargs: dict = dict(learning_rate=lr)
    if clipnorm is not None:
        opt_kwargs['clipnorm'] = clipnorm
    if clipvalue is not None:
        opt_kwargs['clipvalue'] = clipvalue
    optimizer = keras.optimizers.AdamW(weight_decay=weight_decay, **opt_kwargs)
    model.compile(optimizer=optimizer)
    return model
