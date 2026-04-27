
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import dino_cross_entropy


class DINOPretrainModel(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.ema_momentum = ema_momentum
        self.online_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
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

    def train_step(self, data):
        views = data
        if isinstance(data, tuple):
            views = data[0]
        global1, global2 = views[:2]
        local_views = views[2:] if len(views) > 2 else []
        with tf.GradientTape() as tape:
            t1 = tf.stop_gradient(self._teacher_logits(global1))
            t2 = tf.stop_gradient(self._teacher_logits(global2))
            loss_terms = []
            for v in [global1, global2] + list(local_views):
                s = self._student_logits(v, training=True)
                loss_terms.append(dino_cross_entropy(s, t1, self.student_temp, self.teacher_temp))
                loss_terms.append(dino_cross_entropy(s, t2, self.student_temp, self.teacher_temp))
            loss = tf.add_n(loss_terms) / tf.cast(len(loss_terms), tf.float32)
        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        self._init_teacher()
        self._ema_update()
        return {'loss': loss}

    def test_step(self, data):
        views = data
        if isinstance(data, tuple):
            views = data[0]
        global1, global2 = views[:2]
        t1 = self._teacher_logits(global1)
        s2 = self._student_logits(global2, training=False)
        loss = dino_cross_entropy(s2, t1, self.student_temp, self.teacher_temp)
        return {'loss': loss}

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_dino_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    teacher_temp: float = 0.04,
    ema_momentum: float = 0.996,
    lr: float = 1e-4,
):
    model = DINOPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        ema_momentum=ema_momentum,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model
