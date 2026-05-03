from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder, PixelReconstructionHead, patchify
from training.losses import dino_cross_entropy, simmim_reconstruction_loss
from training.ssl_schedules import LrInput


# ===========================================================================
# DINOPretrainModel  (original — unchanged)
# ===========================================================================

class DINOPretrainModel(keras.Model):
    """DINO self-supervised pretraining model.

    Paired with MultiViewDataset (legacy).
    views order: (global_1, global_2, local_1, ..., local_N)

    teacher_temp_var is a plain tf.Variable updated externally by
    TeacherTempWarmupCallback. The model only reads it.
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
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum    = ema_momentum
        self.n_local         = n_local

        self.online_encoder  = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

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

    def _init_teacher(self):
        if self._teacher_initialized:
            return
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(sw)
        self._teacher_initialized = True

    def _ema_update(self):
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(self.ema_momentum * tw + (1.0 - self.ema_momentum) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(self.ema_momentum * tw + (1.0 - self.ema_momentum) * sw)

    def _update_center(self, teacher_logits_list):
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _teacher_probs(self, logits):
        centered = logits - self.center
        return tf.nn.softmax(centered / tf.maximum(self.teacher_temp_var, 1e-6), axis=-1)

    def train_step(self, data):
        views   = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]

        with tf.GradientTape() as tape:
            t1_logits = self._teacher_logits(global1)
            t2_logits = self._teacher_logits(global2)
            t1 = tf.stop_gradient(self._teacher_probs(t1_logits))
            t2 = tf.stop_gradient(self._teacher_probs(t2_logits))

            s1 = self._student_logits(global1, training=True)
            s2 = self._student_logits(global2, training=True)

            loss_terms = [
                dino_cross_entropy(s1, tf.math.log(t1 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s1, tf.math.log(t2 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s2, tf.math.log(t1 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s2, tf.math.log(t2 + 1e-8), self.student_temp, 1.0),
            ]
            for local in views[2:2 + self.n_local]:
                s_local = self._student_logits(local, training=True)
                loss_terms.append(dino_cross_entropy(s_local, tf.math.log(t1 + 1e-8), self.student_temp, 1.0))
                loss_terms.append(dino_cross_entropy(s_local, tf.math.log(t2 + 1e-8), self.student_temp, 1.0))

            loss = tf.add_n(loss_terms) / tf.cast(len(loss_terms), tf.float32)

        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))

        self._init_teacher()
        self._ema_update()
        self._update_center([t1_logits, t2_logits])

        return {'loss': loss, 'teacher_temp': self.teacher_temp_var}

    def test_step(self, data):
        views   = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]
        t1_logits = self._teacher_logits(global1)
        t1 = self._teacher_probs(t1_logits)
        s2 = self._student_logits(global2, training=False)
        loss = dino_cross_entropy(s2, tf.math.log(t1 + 1e-8), self.student_temp, 1.0)
        return {'loss': loss, 'teacher_temp': self.teacher_temp_var}

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_dino_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    n_local: int = 0,
    lr: LrInput = 1e-4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
):
    model = DINOPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
        )
    )
    return model


# ===========================================================================
# DINOSimMIMModel  (new — DINO + SimMIM joint training)
# ===========================================================================

class DINOSimMIMModel(keras.Model):
    """Joint DINO + SimMIM self-supervised pretraining model.

    MUST be paired with MaskedMultiViewDataset.
    views order from MaskedMultiViewDataset:
        views[0] = original_clean  (B, H, W, 3)  teacher input + SimMIM pixel target
        views[1] = masked_clean    (B, H, W, 3)  SimMIM student input
        views[2] = patch_mask      (B, N)         1=masked, 0=visible
        views[3] = aug_global      (B, H, W, 3)  DINO student global
        views[4..] = local_i       (B, h, w, 3)  DINO student locals

    Loss:
        L_total = L_dino + lambda_mim * L_mim

        L_dino = CE(s_aug,     teacher_probs)        # aug_global vs teacher
               + CE(s_mim_cls, teacher_probs)        # masked_clean CLS vs teacher
               + sum_i CE(s_local_i, teacher_probs)  # locals vs teacher

        L_mim  = mean L1 over masked patches only
                 (pred from mim_head vs patchify(original_clean))

    Why s_mim_cls in DINO loss:
        Ensures encoder maintains global representation under masked input,
        consistent with SimMIM design (CLS token not used for reconstruction).

    Why original_clean for teacher:
        Validated: D_KL(teacher_raw || teacher_solar) = 12.06 (193x noise level).
        Solarization on teacher destabilises the target distribution.

    Args:
        lambda_mim  : Weight for SimMIM loss term. Default 0.1.
        patch_size  : ViT patch size — must match encoder. Default 16.
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
        lambda_mim: float = 0.1,
        patch_size: int = 16,
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum    = ema_momentum
        self.n_local         = n_local
        self.lambda_mim      = lambda_mim
        self.patch_size      = patch_size

        # Encoders
        self.online_encoder  = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)

        # DINO projection heads
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

        # SimMIM pixel reconstruction head (linear decoder)
        in_channels = int(input_shape[-1])
        self.mim_head = PixelReconstructionHead(
            patch_size=patch_size,
            in_channels=in_channels,
            name='mim_pixel_head',
        )

        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False, name='dino_center'
        )
        # teacher_temp_var: written only by TeacherTempWarmupCallback
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var'
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    # ── internal helpers ──────────────────────────────────────────────────

    def _init_teacher(self):
        if self._teacher_initialized:
            return
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(sw)
        self._teacher_initialized = True

    def _ema_update(self):
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(self.ema_momentum * tw + (1.0 - self.ema_momentum) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(self.ema_momentum * tw + (1.0 - self.ema_momentum) * sw)

    def _update_center(self, teacher_logits_list):
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _teacher_probs(self, logits):
        centered = logits - self.center
        return tf.nn.softmax(centered / tf.maximum(self.teacher_temp_var, 1e-6), axis=-1)

    # ── train / test step ─────────────────────────────────────────────────

    def train_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]              # (B, H, W, 3) teacher + SimMIM target
        masked_clean   = views[1]              # (B, H, W, 3) SimMIM student input
        patch_mask     = views[2]              # (B, N)       1=masked
        aug_global     = views[3]              # (B, H, W, 3) DINO student global
        locals_        = views[4:4 + self.n_local]

        with tf.GradientTape() as tape:
            # ── SimMIM student forward ─────────────────────────────────────
            mim_out      = self.online_encoder(masked_clean, training=True)
            s_mim_logits = self.projector(mim_out['embedding'], training=True)
            pixel_pred   = self.mim_head(mim_out['encoded_patches'], training=True)
            # pixel_pred: (B, N, patch_size^2 * C)

            # ── DINO student global forward ────────────────────────────────
            aug_out      = self.online_encoder(aug_global, training=True)
            s_aug_logits = self.projector(aug_out['embedding'], training=True)

            # ── DINO student local forwards ────────────────────────────────
            s_local_logits_list = []
            for local in locals_:
                lo = self.online_encoder(local, training=True)
                s_local_logits_list.append(
                    self.projector(lo['embedding'], training=True)
                )

            # ── Teacher forward (no grad, original_clean only) ─────────────
            t_out    = self.teacher_encoder(original_clean, training=False)
            t_logits = self.teacher_projector(t_out['embedding'], training=False)
            t_probs  = tf.stop_gradient(self._teacher_probs(t_logits))
            t_log    = tf.math.log(t_probs + 1e-8)

            # ── SimMIM loss ────────────────────────────────────────────────
            target_patches = patchify(original_clean, self.patch_size)
            l_mim = simmim_reconstruction_loss(pixel_pred, target_patches, patch_mask)

            # ── DINO loss ──────────────────────────────────────────────────
            # aug_global vs teacher
            dino_terms = [
                dino_cross_entropy(s_aug_logits,  t_log, self.student_temp, 1.0),
                # masked_clean CLS vs teacher: enforces global repr under masking
                dino_cross_entropy(s_mim_logits,  t_log, self.student_temp, 1.0),
            ]
            for sl in s_local_logits_list:
                dino_terms.append(
                    dino_cross_entropy(sl, t_log, self.student_temp, 1.0)
                )
            l_dino = tf.add_n(dino_terms) / tf.cast(len(dino_terms), tf.float32)

            loss = l_dino + self.lambda_mim * l_mim

        # ── Gradient update ────────────────────────────────────────────────
        vars_ = (
            self.online_encoder.trainable_weights
            + self.projector.trainable_weights
            + self.mim_head.trainable_weights
        )
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))

        self._init_teacher()
        self._ema_update()
        self._update_center([t_logits])

        return {
            'loss':         loss,
            'l_dino':       l_dino,
            'l_mim':        l_mim,
            'teacher_temp': self.teacher_temp_var,
        }

    def test_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]
        masked_clean   = views[1]
        patch_mask     = views[2]
        aug_global     = views[3]

        t_out    = self.teacher_encoder(original_clean, training=False)
        t_logits = self.teacher_projector(t_out['embedding'], training=False)
        t_probs  = self._teacher_probs(t_logits)
        t_log    = tf.math.log(t_probs + 1e-8)

        aug_out      = self.online_encoder(aug_global, training=False)
        s_aug_logits = self.projector(aug_out['embedding'], training=False)

        mim_out    = self.online_encoder(masked_clean, training=False)
        s_mim_logits = self.projector(mim_out['embedding'], training=False)
        pixel_pred = self.mim_head(mim_out['encoded_patches'], training=False)

        target_patches = patchify(original_clean, self.patch_size)
        l_mim  = simmim_reconstruction_loss(pixel_pred, target_patches, patch_mask)
        l_dino = 0.5 * (
            dino_cross_entropy(s_aug_logits,  t_log, self.student_temp, 1.0)
            + dino_cross_entropy(s_mim_logits, t_log, self.student_temp, 1.0)
        )
        loss = l_dino + self.lambda_mim * l_mim

        return {
            'loss':         loss,
            'l_dino':       l_dino,
            'l_mim':        l_mim,
            'teacher_temp': self.teacher_temp_var,
        }

    def get_stage2_encoder(self, use_teacher: bool = True):
        """Return encoder for Stage 2 fine-tuning (compatible with DINOPretrainModel)."""
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
    lambda_mim: float = 0.1,
    patch_size: int = 16,
    lr: LrInput = 1e-4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
):
    """Build DINOSimMIMModel trainer.

    MUST be paired with MaskedMultiViewDataset in the training loop.
    Using MultiViewDataset will cause a shape mismatch crash at step 1.
    """
    model = DINOSimMIMModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
        lambda_mim=lambda_mim,
        patch_size=patch_size,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
        )
    )
    return model
