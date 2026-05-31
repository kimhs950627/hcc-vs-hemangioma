from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras

from models.encoder import Classifier
from training.losses import supervised_contrastive_loss


class SupConClassifier(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        num_classes: int = 2,
        supcon_weight: float = 0.3,
        projection_dim: int = 128,
        classifier_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        encoder_init_weights: str | None = None,
    ):
        super().__init__()
        self.supcon_weight = supcon_weight
        self.model = Classifier(
            encoder_name=encoder_name,
            input_shape=input_shape,
            num_classes=num_classes,
            projection_dim=projection_dim,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.encoder_init_weights = encoder_init_weights
        if encoder_init_weights:
            self.model.encoder.load_weights(encoder_init_weights)
        self.ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.ce_loss_tracker = keras.metrics.Mean(name='ce_loss')
        self.supcon_loss_tracker = keras.metrics.Mean(name='supcon_loss')
        self.acc = keras.metrics.SparseCategoricalAccuracy(name='acc')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_loss_tracker,
            self.supcon_loss_tracker,
            self.acc,
        ]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def train_step(self, data: Any):
        x, y = data
        with tf.GradientTape() as tape:
            out = self.model(x, training=True)
            ce = self.ce_loss(y, out['logits'])
            scl = supervised_contrastive_loss(y, out['projection'])
            loss = ce if self.supcon_weight <= 0.0 else 0.5 * (ce + scl)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.ce_loss_tracker.update_state(ce)
        self.supcon_loss_tracker.update_state(scl)
        self.acc.update_state(y, out['probabilities'])
        return {
            'loss': self.loss_tracker.result(),
            'ce_loss': self.ce_loss_tracker.result(),
            'supcon_loss': self.supcon_loss_tracker.result(),
            'acc': self.acc.result(),
        }

    def test_step(self, data: Any):
        x, y = data
        out = self.model(x, training=False)
        ce = self.ce_loss(y, out['logits'])
        scl = supervised_contrastive_loss(y, out['projection'])
        loss = ce if self.supcon_weight <= 0.0 else 0.5 * (ce + scl)
        self.loss_tracker.update_state(loss)
        self.ce_loss_tracker.update_state(ce)
        self.supcon_loss_tracker.update_state(scl)
        self.acc.update_state(y, out['probabilities'])
        return {
            'loss': self.loss_tracker.result(),
            'ce_loss': self.ce_loss_tracker.result(),
            'supcon_loss': self.supcon_loss_tracker.result(),
            'acc': self.acc.result(),
        }


def build_stage2_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    supcon_weight: float = 0.3,
    projection_dim: int = 128,
    classifier_hidden_dim: int = 256,
    dropout_rate: float = 0.2,
    lr: float = 1e-4,
    teacher_encoder_weights: str | None = None,
):
    model = SupConClassifier(
        encoder_name=encoder_name,
        input_shape=input_shape,
        num_classes=num_classes,
        supcon_weight=supcon_weight,
        projection_dim=projection_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout_rate=dropout_rate,
        encoder_init_weights=teacher_encoder_weights,
    )
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr))
    return model



def stage2_encoder_recommendation() -> str:
    return (
        'Use the EMA teacher encoder from Stage 1 as the default initializer for Stage 2. '
        'The student/online encoder can be used for ablation, but the teacher is generally '
        'more stable because it is a temporal ensemble of online weights.'
    )
