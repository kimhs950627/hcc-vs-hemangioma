from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import dino_cross_entropy, simmim_l1_loss, patchify_images
from training.ssl_schedules import LrInput


# ===========================================================================
# Legacy DINO-only model  (backward compat)
# ===========================================================================

class DINOPretrainModel(keras.Model):
    """DINO self-supervised pretraining model (legacy, DINO-only).

    teacher_temp_var is a plain tf.Variable that external callbacks
    (TeacherTempWarmupCallback) update on_train_batch_end.  The model
    itself never touches teacher_temp_var except to read it.

    For new training, prefer DINOSimMIMModel (\uad8c\uc7a5 B+ \uc2e4\ud5d8 \ud22c\uc601).
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
    """Build legacy DINO-only trainer (MultiViewDataset compatible)."""
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
# DINO + SimMIM hybrid model  (\uad8c\uc7a5 B+ \uad6c\uc870)
# ===========================================================================

class DINOSimMIMModel(keras.Model):
    """DINO + SimMIM hybrid SSL pretraining model.

    Views (from MaskedMultiViewDataset):
        views[0] = original_clean   : teacher input + SimMIM target
        views[1] = masked_clean     : SimMIM student input
        views[2] = patch_mask       : [B, N]  1=masked, 0=visible
        views[3] = aug_global       : DINO student global view
        views[4:] = local_i         : DINO student local crops

    Loss:
        L = alpha * L_dino + (1 - alpha) * lambda_mim * L_simmim

    Alpha warm-up:
        alpha: 1.0 -> alpha_final  (linear over alpha_warmup_epochs)
    """

    def __init__(
        self,
        encoder_name: str,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        projection_dim: int = 256,
        patch_size: int = 16,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
        n_local: int = 4,
        lambda_mim: float = 1.0,
        alpha_final: float = 0.7,
        alpha_warmup_epochs: int = 20,
    ):
        super().__init__()
        self.student_temp         = student_temp
        self.center_momentum      = center_momentum
        self.ema_momentum         = ema_momentum
        self.n_local              = n_local
        self.lambda_mim           = lambda_mim
        self.alpha_final          = alpha_final
        self.alpha_warmup_epochs  = alpha_warmup_epochs
        self.patch_size           = patch_size

        H, W, C = input_shape
        self._n_patches = (H // patch_size) * (W // patch_size)
        self._patch_dim = patch_size * patch_size * C

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

        self.pixel_pred_head = layers.Dense(
            self._patch_dim, use_bias=True, name='pixel_pred_head',
        )

        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False, name='dino_center',
        )
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var',
        )
        self.alpha_var = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name='alpha_var',
        )
        self.epoch_counter = tf.Variable(
            0, trainable=False, dtype=tf.int32, name='epoch_counter',
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    def _init_teacher(self):
        if self._teacher_initialized:
            return
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(sw)
        self._teacher_initialized = True

    def _ema_update(self):
        m = self.ema_momentum
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(m * tw + (1.0 - m) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(m * tw + (1.0 - m) * sw)

    def _update_center(self, teacher_logits_list: list[tf.Tensor]):
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _teacher_probs(self, logits: tf.Tensor) -> tf.Tensor:
        centered = logits - self.center
        return tf.nn.softmax(centered / tf.maximum(self.teacher_temp_var, 1e-6), axis=-1)

    def _student_cls(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def _teacher_cls(self, x: tf.Tensor) -> tf.Tensor:
        out = self.teacher_encoder(x, training=False)
        return self.teacher_projector(out['embedding'], training=False)

    def _student_patch_tokens(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        out = self.online_encoder(x, training=training)
        if 'last_hidden_state' in out:
            return out['last_hidden_state'][:, 1:, :]
        feat = out.get('feature_map', None)
        if feat is None:
            raise KeyError("Encoder must output 'last_hidden_state' or 'feature_map'.")
        B = tf.shape(feat)[0]
        H = tf.shape(feat)[1]
        W = tf.shape(feat)[2]
        D = feat.shape[-1]
        return tf.reshape(feat, [B, H * W, D])

    def update_alpha(self, epoch: int) -> None:
        if self.alpha_warmup_epochs <= 0:
            self.alpha_var.assign(self.alpha_final)
            return
        progress = min(1.0, epoch / self.alpha_warmup_epochs)
        self.alpha_var.assign(1.0 + progress * (self.alpha_final - 1.0))
        self.epoch_counter.assign(epoch)

    def train_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]
        masked_clean   = views[1]
        patch_mask     = views[2]
        aug_global     = views[3]
        locals_        = views[4:4 + self.n_local]

        with tf.GradientTape() as tape:
            t_logits = self._teacher_cls(original_clean)
            t_probs  = tf.stop_gradient(self._teacher_probs(t_logits))

            s_global_cls  = self._student_cls(aug_global, training=True)
            l_dino_global = dino_cross_entropy(
                s_global_cls, tf.math.log(t_probs + 1e-8), self.student_temp, 1.0,
            )

            l_dino_local_terms: list[tf.Tensor] = []
            for local_v in locals_:
                s_local_cls = self._student_cls(local_v, training=True)
                l_dino_local_terms.append(
                    dino_cross_entropy(
                        s_local_cls, tf.math.log(t_probs + 1e-8), self.student_temp, 1.0,
                    )
                )

            l_dino = l_dino_global
            if l_dino_local_terms:
                l_dino = l_dino + tf.add_n(l_dino_local_terms) / tf.cast(
                    len(l_dino_local_terms), tf.float32
                )

            patch_tokens   = self._student_patch_tokens(masked_clean, training=True)
            pred_pixels    = self.pixel_pred_head(patch_tokens)
            target_patches = patchify_images(
                tf.stop_gradient(original_clean), self.patch_size
            )
            l_simmim = simmim_l1_loss(pred_pixels, target_patches, patch_mask)

            alpha = self.alpha_var
            loss  = alpha * l_dino + (1.0 - alpha) * self.lambda_mim * l_simmim

        trainable = (
            self.online_encoder.trainable_weights
            + self.projector.trainable_weights
            + self.pixel_pred_head.trainable_weights
        )
        grads = tape.gradient(loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable))

        self._init_teacher()
        self._ema_update()
        self._update_center([t_logits])

        return {
            'loss': loss, 'l_dino': l_dino, 'l_simmim': l_simmim,
            'alpha': alpha, 'teacher_temp': self.teacher_temp_var,
        }

    def test_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]
        masked_clean   = views[1]
        patch_mask     = views[2]
        aug_global     = views[3]

        t_logits  = self._teacher_cls(original_clean)
        t_probs   = self._teacher_probs(t_logits)
        s_global  = self._student_cls(aug_global, training=False)
        l_dino    = dino_cross_entropy(
            s_global, tf.math.log(t_probs + 1e-8), self.student_temp, 1.0
        )
        patch_tokens   = self._student_patch_tokens(masked_clean, training=False)
        pred_pixels    = self.pixel_pred_head(patch_tokens)
        target_patches = patchify_images(tf.stop_gradient(original_clean), self.patch_size)
        l_simmim       = simmim_l1_loss(pred_pixels, target_patches, patch_mask)

        alpha = self.alpha_var
        loss  = alpha * l_dino + (1.0 - alpha) * self.lambda_mim * l_simmim
        return {
            'loss': loss, 'l_dino': l_dino, 'l_simmim': l_simmim,
            'alpha': alpha, 'teacher_temp': self.teacher_temp_var,
        }

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_dino_simmim_trainer(
    encoder_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    projection_dim: int = 256,
    patch_size: int = 16,
    temperature: float = 0.1,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    n_local: int = 4,
    lambda_mim: float = 1.0,
    alpha_final: float = 0.7,
    alpha_warmup_epochs: int = 20,
    lr: LrInput = 1e-4,
    clipnorm: float | None = 1.0,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
) -> DINOSimMIMModel:
    """Build DINOSimMIMModel. encoder.py handles dynamic pos_embed interpolation."""
    model = DINOSimMIMModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        patch_size=patch_size,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
        lambda_mim=lambda_mim,
        alpha_final=alpha_final,
        alpha_warmup_epochs=alpha_warmup_epochs,
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


class AlphaWarmupCallback(keras.callbacks.Callback):
    """Linearly warm-up the DINO/SimMIM alpha blend ratio."""

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'update_alpha'):
            self.model.update_alpha(epoch)
            alpha = float(self.model.alpha_var.numpy())
            print(f"  [AlphaWarmup] epoch={epoch}  alpha={alpha:.4f}")
