
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import info_nce_loss


class MoCoPretrainModel(keras.Model):
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
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='moco_projector')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='moco_teacher_projector')
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def _embed_online(self, x, training=True):
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
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(self.ema_momentum * tw + (1.0 - self.ema_momentum) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(self.ema_momentum * tw + (1.0 - self.ema_momentum) * sw)

    def train_step(self, data):
        views = data
        if isinstance(data, tuple):
            views = data[0]
        v1, v2 = views[:2]
        with tf.GradientTape() as tape:
            z1 = self._embed_online(v1, training=True)
            z2_teacher = tf.stop_gradient(self._embed_teacher(v2))
            loss = info_nce_loss(z1, z2_teacher, temperature=self.temperature)
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
        v1, v2 = views[:2]
        z1 = self._embed_online(v1, training=False)
        z2 = self._embed_teacher(v2)
        loss = info_nce_loss(z1, z2, temperature=self.temperature)
        return {'loss': loss}

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_moco_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    ema_momentum: float = 0.996,
    lr: float = 1e-4,
):
    model = MoCoPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        temperature=temperature,
        ema_momentum=ema_momentum,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model
