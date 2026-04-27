
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import negative_cosine_similarity


class BYOLPretrainModel(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        predictor_dim: int = 256,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.online_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='byol_projector')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='byol_teacher_projector')
        self.predictor = keras.Sequential([
            layers.Dense(predictor_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='byol_predictor')
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(run_eagerly=True, **kwargs)
        self.optimizer = optimizer

    def _online_proj(self, x, training=True):
        out = self.online_encoder(x, training=training)
        z = self.projector(out['embedding'], training=training)
        p = self.predictor(z, training=training)
        return p

    def _teacher_proj(self, x):
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
        views = data if isinstance(data, (tuple, list)) else (data,)
        v1 = views[0]
        v2 = views[1]
        with tf.GradientTape() as tape:
            p1 = self._online_proj(v1, training=True)
            p2 = self._online_proj(v2, training=True)
            t1 = tf.stop_gradient(self._teacher_proj(v1))
            t2 = tf.stop_gradient(self._teacher_proj(v2))
            loss = 0.5 * (negative_cosine_similarity(p1, t2) + negative_cosine_similarity(p2, t1))
        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights + self.predictor.trainable_weights
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        self._init_teacher()
        self._ema_update()
        return {'loss': loss}

    def test_step(self, data):
        views = data if isinstance(data, (tuple, list)) else (data,)
        v1 = views[0]
        v2 = views[1]
        p1 = self._online_proj(v1, training=False)
        p2 = self._online_proj(v2, training=False)
        t1 = self._teacher_proj(v1)
        t2 = self._teacher_proj(v2)
        loss = 0.5 * (negative_cosine_similarity(p1, t2) + negative_cosine_similarity(p2, t1))
        return {'loss': loss}

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_byol_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    predictor_dim: int = 256,
    ema_momentum: float = 0.996,
    lr: float = 1e-4,
):
    model = BYOLPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        predictor_dim=predictor_dim,
        ema_momentum=ema_momentum,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model
