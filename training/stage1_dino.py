
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import dino_cross_entropy
from training.ssl_schedules import LrInput, TemperatureInput, resolve_schedule_value


class DINOPretrainModel(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        teacher_temperature: TemperatureInput | None = None,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
        n_local: int = 0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_temperature = teacher_temperature if teacher_temperature is not None else teacher_temp
        self.center_momentum = center_momentum
        self.ema_momentum = ema_momentum
        self.n_local = n_local
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
        self.center = tf.Variable(tf.zeros([1, projection_dim], dtype=tf.float32), trainable=False, name='dino_center')
        self.teacher_temp_var = tf.Variable(float(teacher_temp), trainable=False, dtype=tf.float32, name='teacher_temp_var')
        self._teacher_initialized = False
        self.ssl_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='ssl_step')

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
        concat_logits = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat_logits, axis=0, keepdims=True)
        self.center.assign(self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center)

    def _update_teacher_temp(self):
        self.teacher_temp_var.assign(resolve_schedule_value(self.teacher_temperature, self.ssl_step))

    def _teacher_probs(self, logits):
        centered = logits - self.center
        return tf.nn.softmax(centered / tf.maximum(self.teacher_temp_var, 1e-6), axis=-1)

    def train_step(self, data):
        views = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]
        self._update_teacher_temp()
        with tf.GradientTape() as tape:
            t1_logits = self._teacher_logits(global1)
            t2_logits = self._teacher_logits(global2)
            t1 = tf.stop_gradient(self._teacher_probs(t1_logits))
            t2 = tf.stop_gradient(self._teacher_probs(t2_logits))
            s_global1 = self._student_logits(global1, training=True)
            s_global2 = self._student_logits(global2, training=True)
            loss_terms = [
                dino_cross_entropy(s_global1, tf.math.log(t1 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s_global1, tf.math.log(t2 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s_global2, tf.math.log(t1 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s_global2, tf.math.log(t2 + 1e-8), self.student_temp, 1.0),
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
        views = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]
        self._update_teacher_temp()
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
    teacher_temperature: TemperatureInput | None = None,
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
        teacher_temperature=teacher_temperature,
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
