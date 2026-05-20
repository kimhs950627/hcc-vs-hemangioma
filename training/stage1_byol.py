from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import negative_cosine_similarity, head_disagreement_loss
from training.ssl_schedules import LrInput, TemperatureInput, resolve_schedule_value


class BYOLPretrainModel(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        predictor_dim: int = 256,
        ema_momentum: float = 0.996,
        teacher_temperature: TemperatureInput = 0.04,
        encoder_attn_drop: bool = False,
        encoder_attn_drop_rate: float = 0.0,
        encoder_attn_drop_top_k: int = 40,
        lambda_diversity: float = 0.0,
    ):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.teacher_temperature = teacher_temperature
        self.lambda_diversity = float(lambda_diversity)
        self.online_encoder = build_encoder(
            encoder_name,
            input_shape=input_shape,
            attn_drop=encoder_attn_drop,
            attn_drop_rate=encoder_attn_drop_rate,
            attn_drop_top_k=encoder_attn_drop_top_k,
        )
        self.teacher_encoder = build_encoder(
            encoder_name,
            input_shape=input_shape,
            attn_drop=encoder_attn_drop,
            attn_drop_rate=encoder_attn_drop_rate,
            attn_drop_top_k=encoder_attn_drop_top_k,
        )
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
        self.teacher_temp_var = tf.Variable(0.04, trainable=False, dtype=tf.float32, name='byol_teacher_temp_var')
        self._teacher_initialized = False
        self.ssl_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='ssl_step')

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    def _update_teacher_temp(self):
        self.teacher_temp_var.assign(resolve_schedule_value(self.teacher_temperature, self.ssl_step))

    def _online_proj(self, x, training=True):
        out = self.online_encoder(x, training=training)
        z = self.projector(out['embedding'], training=training)
        p = self.predictor(z, training=training)
        return tf.math.l2_normalize(p, axis=-1)

    def _teacher_proj(self, x):
        out = self.teacher_encoder(x, training=False)
        z = self.teacher_projector(out['embedding'], training=False)
        temp = tf.maximum(self.teacher_temp_var, 1e-6)
        return tf.math.l2_normalize(z / temp, axis=-1)

    def _compute_diversity_loss(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        out = self.online_encoder(x, training=training, return_attention=True)
        attn = out.get('last_encoder_layer_attentional_weights', None)
        if attn is None:
            attn = out.get('last_attn_scores', None)
        if attn is None:
            return tf.constant(0.0, dtype=tf.float32)
        return head_disagreement_loss(attn)

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
            p1 = self._online_proj(v1, training=True)
            p2 = self._online_proj(v2, training=True)
            t1 = tf.stop_gradient(self._teacher_proj(v1))
            t2 = tf.stop_gradient(self._teacher_proj(v2))
            byol_loss = 0.5 * (negative_cosine_similarity(p1, t2) + negative_cosine_similarity(p2, t1))
            diversity_loss = tf.constant(0.0, dtype=tf.float32)
            if self.lambda_diversity > 0.0:
                diversity_loss = self._compute_diversity_loss(v1, training=True)
            loss = byol_loss + self.lambda_diversity * diversity_loss
        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights + self.predictor.trainable_weights
        grads = tape.gradient(loss, vars_)
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if not grads_and_vars:
            raise ValueError('No gradients found for online encoder/projector/predictor variables.')
        self.optimizer.apply_gradients(grads_and_vars)
        self._init_teacher()
        self._ema_update()
        self.ssl_step.assign_add(1)
        return {
            'loss': loss,
            'byol_loss': byol_loss,
            'diversity_loss': diversity_loss,
            'lambda_diversity': tf.cast(self.lambda_diversity, tf.float32),
            'teacher_temp': self.teacher_temp_var,
        }

    def test_step(self, data):
        views = tf.nest.flatten(data)
        v1 = views[0]
        v2 = views[1]
        self._update_teacher_temp()
        p1 = self._online_proj(v1, training=False)
        p2 = self._online_proj(v2, training=False)
        t1 = self._teacher_proj(v1)
        t2 = self._teacher_proj(v2)
        byol_loss = 0.5 * (negative_cosine_similarity(p1, t2) + negative_cosine_similarity(p2, t1))
        diversity_loss = tf.constant(0.0, dtype=tf.float32)
        if self.lambda_diversity > 0.0:
            diversity_loss = self._compute_diversity_loss(v1, training=False)
        loss = byol_loss + self.lambda_diversity * diversity_loss
        return {
            'loss': loss,
            'byol_loss': byol_loss,
            'diversity_loss': diversity_loss,
            'lambda_diversity': tf.cast(self.lambda_diversity, tf.float32),
            'teacher_temp': self.teacher_temp_var,
        }

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_byol_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    predictor_dim: int = 256,
    ema_momentum: float = 0.996,
    lr: LrInput = 1e-4,
    teacher_temperature: TemperatureInput = 0.04,
    encoder_attn_drop: bool = False,
    encoder_attn_drop_rate: float = 0.0,
    encoder_attn_drop_top_k: int = 40,
    lambda_diversity: float = 0.0,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
):
    model = BYOLPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        predictor_dim=predictor_dim,
        ema_momentum=ema_momentum,
        teacher_temperature=teacher_temperature,
        encoder_attn_drop=encoder_attn_drop,
        encoder_attn_drop_rate=encoder_attn_drop_rate,
        encoder_attn_drop_top_k=encoder_attn_drop_top_k,
        lambda_diversity=lambda_diversity,
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
