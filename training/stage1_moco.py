
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import info_nce_loss
from training.ssl_schedules import LrInput, TemperatureInput, resolve_schedule_value


class MoCoPretrainModel(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        temperature: float = 0.1,
        ema_momentum: float = 0.996,
        teacher_temperature: TemperatureInput = 0.04,
    ):
        super().__init__()
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        self.teacher_temperature = teacher_temperature
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
        self.teacher_temp_var = tf.Variable(0.04, trainable=False, dtype=tf.float32, name='moco_teacher_temp_var')
        self._teacher_initialized = False
        self.ssl_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='ssl_step')

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    def _update_teacher_temp(self):
        self.teacher_temp_var.assign(resolve_schedule_value(self.teacher_temperature, self.ssl_step))

    def _embed_online(self, x, training=True):
        out = self.online_encoder(x, training=training)
        z = self.projector(out['embedding'], training=training)
        return tf.math.l2_normalize(z, axis=-1)

    def _embed_teacher(self, x):
        out = self.teacher_encoder(x, training=False)
        z = self.teacher_projector(out['embedding'], training=False)
        temp = tf.maximum(self.teacher_temp_var, 1e-6)
        return tf.math.l2_normalize(z / temp, axis=-1)

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
        views = tf.nest.flatten(data)
        v1 = views[0]
        v2 = views[1]
        self._update_teacher_temp()
        with tf.GradientTape() as tape:
            z1 = self._embed_online(v1, training=True)
            z2_teacher = tf.stop_gradient(self._embed_teacher(v2))
            loss = info_nce_loss(z1, z2_teacher, temperature=self.temperature)
        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        self._init_teacher()
        self._ema_update()
        self.ssl_step.assign_add(1)
        return {'loss': loss, 'teacher_temp': self.teacher_temp_var}

    def test_step(self, data):
        views = tf.nest.flatten(data)
        v1 = views[0]
        v2 = views[1]
        self._update_teacher_temp()
        z1 = self._embed_online(v1, training=False)
        z2 = self._embed_teacher(v2)
        loss = info_nce_loss(z1, z2, temperature=self.temperature)
        return {'loss': loss, 'teacher_temp': self.teacher_temp_var}

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_moco_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    ema_momentum: float = 0.996,
    lr: LrInput = 1e-4,
    teacher_temperature: TemperatureInput = 0.04,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
):
    model = MoCoPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        temperature=temperature,
        ema_momentum=ema_momentum,
        teacher_temperature=teacher_temperature,
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
