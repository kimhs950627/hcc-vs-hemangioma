
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
        teacher_temp_warmup_start: float = 0.04,
        teacher_temp_target: float = 0.04,
        warmup_epochs: int = 10,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_warmup_start = teacher_temp_warmup_start
        self.teacher_temp_target = teacher_temp_target
        self.warmup_epochs = warmup_epochs
        self.center_momentum = center_momentum
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
        self.center = tf.Variable(tf.zeros([1, projection_dim], dtype=tf.float32), trainable=False, name='dino_center')
        self.teacher_temp_var = tf.Variable(float(teacher_temp), trainable=False, dtype=tf.float32, name='teacher_temp_var')
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

    def _update_center(self, teacher_logits_list):
        concat_logits = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat_logits, axis=0, keepdims=True)
        self.center.assign(self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center)

    def _update_teacher_temp(self):
        if not hasattr(self, '_train_counter'):
            return
        step = tf.cast(self._train_counter, tf.float32)
        warmup = tf.cast(max(self.warmup_epochs, 1), tf.float32)
        ratio = tf.minimum(step / warmup, 1.0)
        value = self.teacher_temp_warmup_start + ratio * (self.teacher_temp_target - self.teacher_temp_warmup_start)
        self.teacher_temp_var.assign(value)

    def _teacher_probs(self, logits):
        centered = logits - self.center
        return tf.nn.softmax(centered / self.teacher_temp_var, axis=-1)

    def train_step(self, data):
        views = data
        if isinstance(data, tuple):
            views = data[0]
        global1, global2 = views[:2]
        local_views = views[2:] if len(views) > 2 else []
        self._update_teacher_temp()
        with tf.GradientTape() as tape:
            t1_logits = self._teacher_logits(global1)
            t2_logits = self._teacher_logits(global2)
            t1 = tf.stop_gradient(self._teacher_probs(t1_logits))
            t2 = tf.stop_gradient(self._teacher_probs(t2_logits))
            loss_terms = []
            for v in [global1, global2] + list(local_views):
                s = self._student_logits(v, training=True)
                loss_terms.append(dino_cross_entropy(s, tf.math.log(t1 + 1e-8), self.student_temp, 1.0))
                loss_terms.append(dino_cross_entropy(s, tf.math.log(t2 + 1e-8), self.student_temp, 1.0))
            loss = tf.add_n(loss_terms) / tf.cast(len(loss_terms), tf.float32)
        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))
        self._init_teacher()
        self._ema_update()
        self._update_center([t1_logits, t2_logits])
        return {'loss': loss, 'teacher_temp': self.teacher_temp_var}

    def test_step(self, data):
        views = data
        if isinstance(data, tuple):
            views = data[0]
        global1, global2 = views[:2]
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
    teacher_temp_warmup_start: float = 0.04,
    teacher_temp_target: float = 0.04,
    warmup_epochs: int = 10,
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    lr: float = 1e-4,
):
    model = DINOPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        teacher_temp_warmup_start=teacher_temp_warmup_start,
        teacher_temp_target=teacher_temp_target,
        warmup_epochs=warmup_epochs,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model
