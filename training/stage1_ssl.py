
from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras

from models.encoder import build_encoder
from training.losses import info_nce_loss


class SSLPretrainModel(keras.Model):
    """Stage 1 SSL with online student + EMA teacher.

    Use the EMA teacher weights as initialization for Stage 2 by default.
    Rationale: teacher is temporally ensembled and typically more stable for downstream transfer.
    """

    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        temperature: float = 0.1,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.online_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projector = keras.Sequential([
            keras.layers.Dense(512, activation='gelu'),
            keras.layers.Dense(projection_dim),
        ], name='ssl_projector')
        self.teacher_projector = keras.Sequential([
            keras.layers.Dense(512, activation='gelu'),
            keras.layers.Dense(projection_dim),
        ], name='ssl_teacher_projector')
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self._teacher_initialized = False

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def _embed_online(self, x, training: bool):
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def _embed_teacher(self, x):
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
        m = self.ema_momentum
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(m * tw + (1.0 - m) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(m * tw + (1.0 - m) * sw)

    def train_step(self, data: Any):
        views = data
        if not isinstance(views, (tuple, list)) or len(views) < 2:
            raise ValueError('SSLPretrainModel expects at least two views.')
        v1, v2 = views[0], views[1]
        with tf.GradientTape() as tape:
            z1 = self._embed_online(v1, training=True)
            z2_teacher = tf.stop_gradient(self._embed_teacher(v2))
            loss = info_nce_loss(z1, z2_teacher, temperature=self.temperature)
        vars_ = self.online_encoder.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        self._init_teacher()
        self._ema_update()
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data: Any):
        views = data
        v1, v2 = views[0], views[1]
        z1 = self._embed_online(v1, training=False)
        z2 = self._embed_teacher(v2)
        loss = info_nce_loss(z1, z2, temperature=self.temperature)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    ema_momentum: float = 0.996,
    lr: float = 1e-4,
):
    model = SSLPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        temperature=temperature,
        ema_momentum=ema_momentum,
    )
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr))
    return model
