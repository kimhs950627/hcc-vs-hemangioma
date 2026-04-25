
from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras

from models.encoder import build_encoder
from training.losses import info_nce_loss


class SSLPretrainModel(keras.Model):
    def __init__(self, encoder_name: str, input_shape=(224, 224, 3), projection_dim: int = 256, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.online_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projector = keras.Sequential([
            keras.layers.Dense(512, activation='gelu'),
            keras.layers.Dense(projection_dim),
        ], name='ssl_projector')
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def _embed(self, x, training: bool):
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def train_step(self, data: Any):
        views = data
        if not isinstance(views, (tuple, list)) or len(views) < 2:
            raise ValueError('SSLPretrainModel expects at least two views.')
        v1, v2 = views[0], views[1]
        with tf.GradientTape() as tape:
            z1 = self._embed(v1, training=True)
            z2 = self._embed(v2, training=True)
            loss = info_nce_loss(z1, z2, temperature=self.temperature)
        vars_ = self.online_encoder.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}

    def test_step(self, data: Any):
        views = data
        v1, v2 = views[0], views[1]
        z1 = self._embed(v1, training=False)
        z2 = self._embed(v2, training=False)
        loss = info_nce_loss(z1, z2, temperature=self.temperature)
        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}


def build_stage1_trainer(encoder_name: str, input_shape=(224, 224, 3), projection_dim: int = 256, temperature: float = 0.1, lr: float = 1e-4):
    model = SSLPretrainModel(encoder_name=encoder_name, input_shape=input_shape, projection_dim=projection_dim, temperature=temperature)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr))
    return model
