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
        supcon_weight: float = 0.1,
        projection_dim: int = 128,
        classifier_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        encoder_init_weights: str | None = None,
    ):
        super().__init__()
        self.supcon_weight = supcon_weight   # L_total = CE + supcon_weight * SC
        self.model = Classifier(
            encoder_name=encoder_name,
            input_shape=input_shape,
            num_classes=num_classes,
            projection_dim=projection_dim,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.encoder_init_weights = encoder_init_weights
        dummy_x = tf.zeros((1, *input_shape), dtype=tf.float32)
        _ = self.model(dummy_x, training=False)
        if encoder_init_weights:
            self.model.encoder.load_weights(encoder_init_weights)
        self.ce_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.ce_loss_tracker = keras.metrics.Mean(name='ce_loss')
        self.supcon_loss_tracker = keras.metrics.Mean(name='supcon_loss')
        self.acc = keras.metrics.SparseCategoricalAccuracy(name='acc')
        self.auc = keras.metrics.AUC(name='auc')
        self.precision = keras.metrics.Precision(name='ppv')
        self.recall = keras.metrics.Recall(name='sensitivity')
        self.tp = keras.metrics.TruePositives(name='tp')
        self.tn = keras.metrics.TrueNegatives(name='tn')
        self.fp = keras.metrics.FalsePositives(name='fp')
        self.fn = keras.metrics.FalseNegatives(name='fn')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.ce_loss_tracker,
            self.supcon_loss_tracker,
            self.acc,
            self.auc,
            self.precision,
            self.recall,
            self.tp,
            self.tn,
            self.fp,
            self.fn,
        ]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def _update_classification_metrics(self, y, probabilities):
        if probabilities.shape[-1] == 2:
            pos_scores = probabilities[:, 1]
            pos_pred = pos_scores
        else:
            pos_scores = probabilities
            pos_pred = probabilities
        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        pos_scores = tf.cast(tf.reshape(pos_scores, (-1, 1)), tf.float32)
        pos_pred = tf.cast(tf.reshape(pos_pred, (-1, 1)), tf.float32)
        self.auc.update_state(y, pos_scores)
        self.precision.update_state(y, pos_pred)
        self.recall.update_state(y, pos_pred)
        self.tp.update_state(y, pos_pred)
        self.tn.update_state(y, pos_pred)
        self.fp.update_state(y, pos_pred)
        self.fn.update_state(y, pos_pred)

    def _collect_metric_results(self):
        tp = self.tp.result()
        tn = self.tn.result()
        fp = self.fp.result()
        fn = self.fn.result()
        precision = self.precision.result()
        recall = self.recall.result()
        specificity = tf.math.divide_no_nan(tn, tn + fp)
        npv = tf.math.divide_no_nan(tn, tn + fn)
        f1 = tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)
        return {
            'loss': self.loss_tracker.result(),
            'ce_loss': self.ce_loss_tracker.result(),
            'supcon_loss': self.supcon_loss_tracker.result(),
            'acc': self.acc.result(),
            'auc': self.auc.result(),
            'f1': f1,
            'sensitivity': recall,
            'specificity': specificity,
            'ppv': precision,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
        }

    def _compute_loss(self, y, out, training: bool) -> tuple:
        """
        L_total = CE + supcon_weight * SC

        - supcon_weight == 0.0 → CE only (SC는 logging만)
        - supcon_weight  > 0.0 → CE + w * SC
          w=0.1 권장: SC 초기값(~2.7) vs CE 초기값(~0.69), ratio ≈ 4x
          w=0.1이면 weighted SC ≈ 0.27로 CE와 비슷한 scale

        Returns:
            total_loss, ce_loss, sc_loss
        """
        ce = self.ce_loss(y, out['logits'])
        sc = supervised_contrastive_loss(y, out['projection'])
        if self.supcon_weight <= 0.0:
            total = ce
        else:
            total = ce + self.supcon_weight * sc
        return total, ce, sc

    def train_step(self, data: Any):
        x, y = data
        with tf.GradientTape() as tape:
            out = self.model(x, training=True)
            loss, ce, scl = self._compute_loss(y, out, training=True)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.ce_loss_tracker.update_state(ce)
        self.supcon_loss_tracker.update_state(scl)
        self.acc.update_state(y, out['probabilities'])
        self._update_classification_metrics(y, out['probabilities'])
        return self._collect_metric_results()

    def test_step(self, data: Any):
        x, y = data
        out = self.model(x, training=False)
        loss, ce, scl = self._compute_loss(y, out, training=False)
        self.loss_tracker.update_state(loss)
        self.ce_loss_tracker.update_state(ce)
        self.supcon_loss_tracker.update_state(scl)
        self.acc.update_state(y, out['probabilities'])
        self._update_classification_metrics(y, out['probabilities'])
        return self._collect_metric_results()


def build_stage2_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    supcon_weight: float = 0.1,
    projection_dim: int = 128,
    classifier_hidden_dim: int = 256,
    dropout_rate: float = 0.2,
    lr: float = 1e-4,
    teacher_encoder_weights: str | None = None,
):
    """
    Stage 2 trainer factory.

    Loss: L_total = CE + supcon_weight * SC
      supcon_weight=0.0  → CE only
      supcon_weight=0.1  → 권장 (SC scale ~2.7, CE scale ~0.69; w=0.1이면 균형)
      supcon_weight=0.5  → SC dominant (class collapse 위험, 비권장)
    """
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
