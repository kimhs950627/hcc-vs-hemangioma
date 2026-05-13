from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import dino_cross_entropy
from training.ssl_schedules import LrInput


class DINOPretrainModel(keras.Model):
    """DINO self-supervised pretraining model.

    teacher_temp_var is a plain tf.Variable that external callbacks
    (TeacherTempWarmupCallback) update on_train_batch_end.  The model
    itself never touches teacher_temp_var except to read it.
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
        encoder_embed_dim: int = 384,
        encoder_depth: int | None = None,
        encoder_num_heads: int | None = None,
        encoder_mlp_dim: int | None = None,
        encoder_patch_size: int | None = None,
        encoder_dropout: float = 0.1,
        use_pe: bool = False,
    ):
        super().__init__()
        self.student_temp  = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum    = ema_momentum
        self.n_local         = n_local

        self.online_encoder  = build_encoder(
            encoder_name,
            input_shape=input_shape,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            patch_size=encoder_patch_size,
            dropout=encoder_dropout,
            use_pe=use_pe,
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
            use_pe=use_pe,
        )
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
        # ── teacher_temp_var: ONLY written by TeacherTempWarmupCallback ──
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var'
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    # ── internal helpers ──────────────────────────────────────────────────

    def _student_logits(self, x, training=True):
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def _teacher_logits(self, x):
        out = self.teacher_encoder(x, training=False)
        return self.teacher_projector(out['embedding'], training=False)

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

    # ── train_step ────────────────────────────────────────────────────────

    def train_step(self, data):
        views = tf.nest.flatten(data)
        g1 = views[0]
        g2 = views[1]
        locals_ = tuple(views[2 + i] for i in range(self.n_local)) if self.n_local > 0 else ()

        self._init_teacher()

        with tf.GradientTape() as tape:
            t1 = tf.stop_gradient(
                tf.nn.softmax(
                    (self._teacher_logits(g1) - self.center) / self.teacher_temp_var,
                    axis=-1,
                )
            )
            t2 = tf.stop_gradient(
                tf.nn.softmax(
                    (self._teacher_logits(g2) - self.center) / self.teacher_temp_var,
                    axis=-1,
                )
            )

            s_all = [self._student_logits(g1), self._student_logits(g2)]
            for lv in locals_:
                s_all.append(self._student_logits(lv))

            loss = tf.constant(0.0)
            n_pairs = 0
            for i, s in enumerate(s_all):
                for j, t in enumerate([t1, t2]):
                    if i == j and i < 2:
                        continue
                    loss = loss + dino_cross_entropy(
                        s, t,
                        student_temp=self.student_temp,
                        teacher_temp=1.0,
                    )
                    n_pairs += 1
            loss = loss / tf.cast(n_pairs, tf.float32)

        trainable_vars = (
            list(self.online_encoder.trainable_variables)
            + list(self.projector.trainable_variables)
        )
        grads = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        if not grads_and_vars:
            raise ValueError('No gradients found for online encoder/projector variables.')
        self.optimizer.apply_gradients(grads_and_vars)
        self._ema_update()
        self._update_center(
            self._teacher_logits(g1),
            self._teacher_logits(g2),
        )
        return {'loss': loss}

    def get_stage2_encoder(self, use_teacher: bool = True) -> keras.Model:
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
    encoder_embed_dim: int = 384,
    encoder_depth: int | None = None,
    encoder_num_heads: int | None = None,
    encoder_mlp_dim: int | None = None,
    encoder_patch_size: int | None = None,
    encoder_dropout: float = 0.1,
    use_pe: bool = False,
    lr: LrInput = 1e-4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
) -> DINOPretrainModel:
    model = DINOPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_mlp_dim=encoder_mlp_dim,
        encoder_patch_size=encoder_patch_size,
        encoder_dropout=encoder_dropout,
        use_pe=use_pe,
    )
    opt_kwargs: dict = dict(learning_rate=lr)
    if clipnorm  is not None: opt_kwargs['clipnorm']  = clipnorm
    if clipvalue is not None: opt_kwargs['clipvalue'] = clipvalue
    optimizer = keras.optimizers.AdamW(weight_decay=weight_decay, **opt_kwargs)
    model.compile(optimizer=optimizer)
    return model
