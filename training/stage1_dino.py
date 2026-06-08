from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import (
    dino_cross_entropy,
    diversity_loss as compute_diversity_loss,
)
from training.ssl_schedules import LrInput
from training.selfpatch import SelfPatch


class DINOPretrainModel(keras.Model):
    """DINO self-supervised pretraining model.

    teacher_temp_var is a plain tf.Variable that external callbacks
    (TeacherTempWarmupCallback) update on_train_batch_end.

    ortho_alpha_var is a tf.Variable controlled by OrthoAlphaScheduleCallback:
      - starts at 0.0 (pure entropy phase)
      - linearly warms up to target_alpha over warmup_steps
    """

    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        projection_dim: int = 256,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
        n_local: int = 0,
        encoder_embed_dim: int = 384,
        encoder_depth: int | None = None,
        encoder_num_heads: int | None = None,
        encoder_mlp_dim: int | None = None,
        encoder_patch_size: int | None = None,
        encoder_dropout: float = 0.1,
        use_pe: bool = False,
        encoder_attn_drop: bool = False,
        encoder_attn_drop_rate: float = 0.0,
        encoder_attn_drop_top_k: int = 40,
        lambda_selfpatch: float = 0.0,
        lambda_diversity: float = 0.0,
        selfpatch_proj_dim: int = 256,
        selfpatch_top_k: int = 4,
        selfpatch_temperature: float = 0.07,
        diversity_mode: str = 'cls_entropy',
        diversity_ortho_alpha: float = 0.5,
        diversity_layer_mode: str = 'top_half',
        diversity_start_layer: int | None = None,
        diversity_end_layer: int | None = None,
        diversity_exclude_cls_col: bool = True,
        diversity_entropy_weight: float = 1.0,
        diversity_entropy_min: float = 2.5,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum = ema_momentum
        self.n_local = n_local
        self.lambda_selfpatch = float(lambda_selfpatch)
        self.lambda_diversity = float(lambda_diversity)
        self.diversity_mode = diversity_mode
        self.diversity_layer_mode = diversity_layer_mode
        self.diversity_start_layer = diversity_start_layer
        self.diversity_end_layer = diversity_end_layer
        self.diversity_exclude_cls_col = bool(diversity_exclude_cls_col)
        self.diversity_entropy_weight = float(diversity_entropy_weight)
        self.diversity_entropy_min = float(diversity_entropy_min)

        self.online_encoder = build_encoder(
            encoder_name,
            input_shape=input_shape,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            patch_size=encoder_patch_size,
            dropout=encoder_dropout,
            use_pe=use_pe,
            attn_drop=encoder_attn_drop,
            attn_drop_rate=encoder_attn_drop_rate,
            attn_drop_top_k=encoder_attn_drop_top_k,
        )
        self.teacher_encoder = build_encoder(
            encoder_name,
            input_shape=input_shape,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_dim=encoder_mlp_dim,
            patch_size=encoder_patch_size,
            dropout=encoder_dropout,
            use_pe=use_pe,
            attn_drop=encoder_attn_drop,
            attn_drop_rate=encoder_attn_drop_rate,
            attn_drop_top_k=encoder_attn_drop_top_k,
        )
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

        self.selfpatch = None
        if self.lambda_selfpatch > 0.0:
            self.selfpatch = SelfPatch(
                proj_dim=selfpatch_proj_dim,
                top_k=selfpatch_top_k,
                temperature=selfpatch_temperature,
            )

        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False,
            name='dino_center',
        )
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp),
            trainable=False,
            dtype=tf.float32,
            name='teacher_temp_var',
        )
        # ortho_alpha_var: controlled by OrthoAlphaScheduleCallback.
        # Initialised to diversity_ortho_alpha (cfg value).
        # When OrthoAlphaScheduleCallback is used it overrides to 0 at
        # train_begin and warms up to target_alpha — so cfg value acts
        # as the static fallback when callback is NOT attached.
        self.ortho_alpha_var = tf.Variable(
            float(diversity_ortho_alpha),
            trainable=False,
            dtype=tf.float32,
            name='ortho_alpha_var',
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    def _student_logits(self, x, training=True):
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def _teacher_logits(self, x):
        out = self.teacher_encoder(x, training=False)
        return self.teacher_projector(out['embedding'], training=False)

    def _student_patches(self, x, training=True):
        return self.online_encoder(x, training=training)['encoded_patches']

    def _teacher_patches(self, x):
        return self.teacher_encoder(x, training=False)['encoded_patches']

    def _compute_diversity_loss(self, x: tf.Tensor, training: bool = True):
        out = self.online_encoder(x, training=training, return_attention=True)
        attn_list = out.get('attention_weights', None)
        if not attn_list:
            zero = tf.constant(0.0, dtype=tf.float32)
            return zero, zero, zero
        layer_indices = self._resolve_diversity_layer_indices(len(attn_list))
        # Read ortho_alpha from tf.Variable so OrthoAlphaScheduleCallback
        # changes take effect without graph recompilation.
        return compute_diversity_loss(
            attn_list,
            mode=self.diversity_mode,
            ortho_alpha=float(self.ortho_alpha_var.numpy()),
            layer_indices=layer_indices,
            exclude_cls_col=self.diversity_exclude_cls_col,
            entropy_min=self.diversity_entropy_min,
            entropy_weight=self.diversity_entropy_weight,
        )

    def _resolve_diversity_layer_indices(self, n_layers: int) -> list[int]:
        mode = self.diversity_layer_mode
        if mode == 'last':
            return [n_layers - 1]
        if mode == 'all':
            return list(range(n_layers))
        if mode == 'top_half':
            start = n_layers // 2
            return list(range(start, n_layers))
        if mode == 'range':
            start = 0 if self.diversity_start_layer is None else self.diversity_start_layer
            end = n_layers if self.diversity_end_layer is None else self.diversity_end_layer
            return list(range(start, end))
        raise ValueError(f'Unsupported diversity_layer_mode={mode}')

    def _init_teacher(self):
        if self._teacher_initialized:
            return
        for t_w, s_w in zip(self.teacher_encoder.weights, self.online_encoder.weights):
            t_w.assign(s_w)
        for t_w, s_w in zip(self.teacher_projector.weights, self.projector.weights):
            t_w.assign(s_w)
        self._teacher_initialized = True

    def _ema_update(self):
        m = self.ema_momentum
        for t_w, s_w in zip(self.teacher_encoder.weights, self.online_encoder.weights):
            t_w.assign(m * t_w + (1.0 - m) * s_w)
        for t_w, s_w in zip(self.teacher_projector.weights, self.projector.weights):
            t_w.assign(m * t_w + (1.0 - m) * s_w)

    def _update_center(self, t1_logits, t2_logits):
        batch_center = tf.reduce_mean(
            tf.concat([t1_logits, t2_logits], axis=0),
            axis=0,
            keepdims=True,
        )
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def train_step(self, data):
        views = tf.nest.flatten(data)
        g1 = views[0]
        g2 = views[1]
        locals_ = tuple(views[2 + i] for i in range(self.n_local)) if self.n_local > 0 else ()

        self._init_teacher()

        with tf.GradientTape() as tape:
            t1 = tf.stop_gradient(
                tf.nn.softmax((self._teacher_logits(g1) - self.center) / self.teacher_temp_var, axis=-1)
            )
            t2 = tf.stop_gradient(
                tf.nn.softmax((self._teacher_logits(g2) - self.center) / self.teacher_temp_var, axis=-1)
            )

            s_all = [self._student_logits(g1), self._student_logits(g2)]
            for lv in locals_:
                s_all.append(self._student_logits(lv))

            dino_loss = tf.constant(0.0, dtype=tf.float32)
            n_pairs = 0
            for i, s in enumerate(s_all):
                for j, t in enumerate([t1, t2]):
                    if i == j and i < 2:
                        continue
                    dino_loss = dino_loss + dino_cross_entropy(
                        s,
                        t,
                        student_temp=self.student_temp,
                        teacher_temp=1.0,
                    )
                    n_pairs += 1
            dino_loss = dino_loss / tf.cast(n_pairs, tf.float32)

            selfpatch_loss = tf.constant(0.0, dtype=tf.float32)
            if self.selfpatch is not None:
                selfpatch_loss = self.selfpatch(
                    self._student_patches(g1, training=True),
                    self._teacher_patches(g1),
                    training=True,
                )

            div_total = tf.constant(0.0, dtype=tf.float32)
            div_primary = tf.constant(0.0, dtype=tf.float32)
            div_entropy = tf.constant(0.0, dtype=tf.float32)
            if self.lambda_diversity > 0.0:
                div_total, div_primary, div_entropy = self._compute_diversity_loss(g1, training=True)

            loss = dino_loss + self.lambda_selfpatch * selfpatch_loss + self.lambda_diversity * div_total

        trainable_vars = list(self.online_encoder.trainable_variables) + list(self.projector.trainable_variables)
        grads = tape.gradient(loss, trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, trainable_vars) if g is not None]
        if not grads_and_vars:
            raise ValueError('No gradients found for online encoder/projector variables.')
        self.optimizer.apply_gradients(grads_and_vars)
        self._ema_update()
        self._update_center(self._teacher_logits(g1), self._teacher_logits(g2))
        return {
            'loss': loss,
            'dino_loss': dino_loss,
            'selfpatch_loss': selfpatch_loss,
            'lambda_selfpatch': tf.cast(self.lambda_selfpatch, tf.float32),
            'diversity_loss': div_total,
            'diversity_primary_loss': div_primary,
            'diversity_entropy_loss': div_entropy,
            'lambda_diversity': tf.cast(self.lambda_diversity, tf.float32),
            # ortho_alpha_var 추적: wandb에서 schedule 확인용
            'ortho_alpha': self.ortho_alpha_var,
        }

    def get_stage2_encoder(self, use_teacher: bool = True) -> keras.Model:
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_dino_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    n_local: int = 0,
    encoder_embed_dim: int = 384,
    encoder_depth: int | None = None,
    encoder_num_heads: int | None = None,
    encoder_mlp_dim: int | None = None,
    encoder_patch_size: int | None = None,
    encoder_dropout: float = 0.1,
    use_pe: bool = False,
    encoder_attn_drop: bool = False,
    encoder_attn_drop_rate: float = 0.0,
    encoder_attn_drop_top_k: int = 40,
    lambda_selfpatch: float = 0.0,
    lambda_diversity: float = 0.0,
    selfpatch_proj_dim: int = 256,
    selfpatch_top_k: int = 4,
    selfpatch_temperature: float = 0.07,
    diversity_mode: str = 'cls_entropy',
    diversity_ortho_alpha: float = 0.5,
    diversity_layer_mode: str = 'top_half',
    diversity_start_layer: int | None = None,
    diversity_end_layer: int | None = None,
    diversity_exclude_cls_col: bool = True,
    diversity_entropy_weight: float = 1.0,
    diversity_entropy_min: float = 2.5,
    lr: LrInput = 1e-4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
) -> DINOPretrainModel:
    model = DINOPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_mlp_dim=encoder_mlp_dim,
        encoder_patch_size=encoder_patch_size,
        encoder_dropout=encoder_dropout,
        use_pe=use_pe,
        encoder_attn_drop=encoder_attn_drop,
        encoder_attn_drop_rate=encoder_attn_drop_rate,
        encoder_attn_drop_top_k=encoder_attn_drop_top_k,
        lambda_selfpatch=lambda_selfpatch,
        lambda_diversity=lambda_diversity,
        selfpatch_proj_dim=selfpatch_proj_dim,
        selfpatch_top_k=selfpatch_top_k,
        selfpatch_temperature=selfpatch_temperature,
        diversity_mode=diversity_mode,
        diversity_ortho_alpha=diversity_ortho_alpha,
        diversity_layer_mode=diversity_layer_mode,
        diversity_start_layer=diversity_start_layer,
        diversity_end_layer=diversity_end_layer,
        diversity_exclude_cls_col=diversity_exclude_cls_col,
        diversity_entropy_weight=diversity_entropy_weight,
        diversity_entropy_min=diversity_entropy_min,
    )
    opt_kwargs: dict = dict(learning_rate=lr)
    if clipnorm is not None:
        opt_kwargs['clipnorm'] = clipnorm
    if clipvalue is not None:
        opt_kwargs['clipvalue'] = clipvalue
    optimizer = keras.optimizers.AdamW(weight_decay=weight_decay, **opt_kwargs)
    model.compile(optimizer=optimizer)
    return model
