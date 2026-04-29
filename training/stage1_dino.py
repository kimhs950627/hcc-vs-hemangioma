from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import dino_cross_entropy, simmim_l1_loss, patchify_images
from training.ssl_schedules import LrInput


# ===========================================================================
# Shared DINO forward helpers  (DO NOT duplicate – both models use these)
# ===========================================================================

def _compute_teacher_logits(
    teacher_encoder,
    teacher_projector,
    x: tf.Tensor,
) -> tf.Tensor:
    """Teacher forward pass → raw projection logits (no softmax)."""
    out = teacher_encoder(x, training=False)
    return teacher_projector(out['embedding'], training=False)


def _compute_student_cls(
    online_encoder,
    projector,
    x: tf.Tensor,
    training: bool = True,
) -> tf.Tensor:
    """Student forward pass → raw projection logits."""
    out = online_encoder(x, training=training)
    return projector(out['embedding'], training=training)


def _teacher_probs(
    logits: tf.Tensor,
    center: tf.Variable,
    teacher_temp_var: tf.Variable,
) -> tf.Tensor:
    """Center-subtract + temperature softmax → teacher probability."""
    centered = logits - center
    return tf.nn.softmax(centered / tf.maximum(teacher_temp_var, 1e-6), axis=-1)


def _compute_dino_loss(
    online_encoder,
    projector,
    teacher_encoder,
    teacher_projector,
    center: tf.Variable,
    teacher_temp_var: tf.Variable,
    student_temp: float,
    global1: tf.Tensor,
    global2: tf.Tensor,
    locals_: list[tf.Tensor],
    training: bool = True,
) -> tuple[tf.Tensor, list[tf.Tensor]]:
    """Pure DINO loss helper shared by DINOPretrainModel and DINOSimMIMModel.

    Returns:
        loss        : scalar DINO loss (mean over all cross-entropy pairs)
        t_logits_list : raw teacher logits for both global views
                        (used to update the center after gradient step)
    """
    t1_logits = _compute_teacher_logits(teacher_encoder, teacher_projector, global1)
    t2_logits = _compute_teacher_logits(teacher_encoder, teacher_projector, global2)
    t1 = tf.stop_gradient(_teacher_probs(t1_logits, center, teacher_temp_var))
    t2 = tf.stop_gradient(_teacher_probs(t2_logits, center, teacher_temp_var))

    s1 = _compute_student_cls(online_encoder, projector, global1, training=training)
    s2 = _compute_student_cls(online_encoder, projector, global2, training=training)

    loss_terms = [
        dino_cross_entropy(s1, tf.math.log(t1 + 1e-8), student_temp, 1.0),
        dino_cross_entropy(s1, tf.math.log(t2 + 1e-8), student_temp, 1.0),
        dino_cross_entropy(s2, tf.math.log(t1 + 1e-8), student_temp, 1.0),
        dino_cross_entropy(s2, tf.math.log(t2 + 1e-8), student_temp, 1.0),
    ]
    for local_v in locals_:
        # Local views are encoded with the SAME online_encoder call path as globals
        s_local = _compute_student_cls(
            online_encoder, projector, local_v, training=training
        )
        loss_terms.append(
            dino_cross_entropy(s_local, tf.math.log(t1 + 1e-8), student_temp, 1.0)
        )
        loss_terms.append(
            dino_cross_entropy(s_local, tf.math.log(t2 + 1e-8), student_temp, 1.0)
        )

    loss = tf.add_n(loss_terms) / tf.cast(len(loss_terms), tf.float32)
    return loss, [t1_logits, t2_logits]


# ===========================================================================
# Legacy DINO-only model  (backward compat)
# ===========================================================================

class DINOPretrainModel(keras.Model):
    """DINO self-supervised pretraining model (legacy, DINO-only).

    teacher_temp_var is a plain tf.Variable that external callbacks
    (TeacherTempWarmupCallback) update on_train_batch_end.  The model
    itself never touches teacher_temp_var except to read it.

    For new training, prefer DINOSimMIMModel (권장 B+ 실험 투영).

    NOTE: train_step now delegates to the shared _compute_dino_loss helper.
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
    ):
        super().__init__()
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.ema_momentum    = ema_momentum
        self.n_local         = n_local

        self.online_encoder  = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False, name='dino_center'
        )
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var'
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

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
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def train_step(self, data):
        views   = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]
        locals_ = views[2:2 + self.n_local]

        with tf.GradientTape() as tape:
            loss, t_logits_list = _compute_dino_loss(
                self.online_encoder, self.projector,
                self.teacher_encoder, self.teacher_projector,
                self.center, self.teacher_temp_var,
                self.student_temp,
                global1, global2, locals_,
                training=True,
            )

        vars_ = self.online_encoder.trainable_weights + self.projector.trainable_weights
        grads = tape.gradient(loss, vars_)
        self.optimizer.apply_gradients(zip(grads, vars_))

        self._init_teacher()
        self._ema_update()
        self._update_center(t_logits_list)

        return {'loss': loss, 'teacher_temp': self.teacher_temp_var}

    def test_step(self, data):
        views   = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]
        t1_logits = _compute_teacher_logits(
            self.teacher_encoder, self.teacher_projector, global1
        )
        t1 = _teacher_probs(t1_logits, self.center, self.teacher_temp_var)
        s2 = _compute_student_cls(
            self.online_encoder, self.projector, global2, training=False
        )
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
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    n_local: int = 0,
    lr: LrInput = 1e-4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
):
    """Build legacy DINO-only trainer (MultiViewDataset compatible)."""
    model = DINOPretrainModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        student_temp=temperature,
        teacher_temp=teacher_temp,
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


# ===========================================================================
# DINO + SimMIM hybrid model  (권장 B+ 구조)
# ===========================================================================

class DINOSimMIMModel(keras.Model):
    """DINO + SimMIM hybrid SSL pretraining model.

    Views (from MaskedMultiViewDataset):
        views[0] = original_clean   : teacher global1 input + SimMIM target
        views[1] = aug_global2      : teacher/student global2 (DINO)
        views[2] = masked_clean     : SimMIM student input
        views[3] = patch_mask       : [B, N]  1=masked, 0=visible
        views[4:] = local_i         : DINO student local crops
                                      (encoded with SAME path as global views)

    Loss:
        L = alpha * L_dino + (1 - alpha) * lambda_mim * L_simmim

    Alpha warm-up:
        alpha: 1.0 -> alpha_final  (linear over warmup_steps, step-based)
        Updated by AlphaWarmupCallback via update_alpha_by_step().

    Key design invariants:
        - DINO forward path is NOT re-implemented here; _compute_dino_loss() is used.
        - Local crops use the identical online_encoder call path as global views.
        - patch_mask.shape[1] == pred_tokens.shape[1] is asserted at graph build time.
    """

    def __init__(
        self,
        encoder_name: str,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        projection_dim: int = 256,
        patch_size: int = 16,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        ema_momentum: float = 0.996,
        n_local: int = 4,
        lambda_mim: float = 1.0,
        alpha_final: float = 0.7,
        alpha_warmup_epochs: int = 20,
    ):
        super().__init__()
        self.student_temp         = student_temp
        self.center_momentum      = center_momentum
        self.ema_momentum         = ema_momentum
        self.n_local              = n_local
        self.lambda_mim           = lambda_mim
        self.alpha_final          = alpha_final
        self.alpha_warmup_epochs  = alpha_warmup_epochs
        self.patch_size           = patch_size

        H, W, C = input_shape
        self._n_patches = (H // patch_size) * (W // patch_size)
        self._patch_dim = patch_size * patch_size * C

        self.online_encoder  = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)

        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

        self.pixel_pred_head = layers.Dense(
            self._patch_dim, use_bias=True, name='pixel_pred_head',
        )

        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False, name='dino_center',
        )
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var',
        )
        self.alpha_var = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name='alpha_var',
        )
        self.step_counter = tf.Variable(
            0, trainable=False, dtype=tf.int32, name='step_counter',
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    def _init_teacher(self):
        if self._teacher_initialized:
            return
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(sw)
        self._teacher_initialized = True

    def _ema_update(self):
        m = self.ema_momentum
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(m * tw + (1.0 - m) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(m * tw + (1.0 - m) * sw)

    def _update_center(self, teacher_logits_list: list[tf.Tensor]):
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _student_patch_tokens(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Extract per-patch token sequence from the online encoder.

        Supports ViT (last_hidden_state, strips CLS) and CNN (feature_map reshape).
        """
        out = self.online_encoder(x, training=training)
        if 'last_hidden_state' in out:
            tokens = out['last_hidden_state'][:, 1:, :]   # strip CLS token
            return tokens
        feat = out.get('feature_map', None)
        if feat is None:
            raise KeyError("Encoder must output 'last_hidden_state' or 'feature_map'.")
        B = tf.shape(feat)[0]
        H = tf.shape(feat)[1]
        W = tf.shape(feat)[2]
        return tf.reshape(feat, [B, H * W, feat.shape[-1]])

    def _simmim_forward(
        self,
        masked_clean: tf.Tensor,
        original_clean: tf.Tensor,
        patch_mask: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        """SimMIM branch: masked encoder → pixel head → L1 loss.

        Asserts patch_mask token count matches pred_tokens at runtime.
        """
        patch_tokens = self._student_patch_tokens(masked_clean, training=training)
        pred_pixels  = self.pixel_pred_head(patch_tokens)
        target_patches = patchify_images(
            tf.stop_gradient(original_clean), self.patch_size
        )

        # ----- shape assertion: mask dimension must match token count -----
        n_pred   = tf.shape(pred_pixels)[1]
        n_mask   = tf.shape(patch_mask)[1]
        check_op = tf.debugging.assert_equal(
            n_pred, n_mask,
            message=(
                "patch_mask token count mismatch: "
                "pred_pixels.shape[1] != patch_mask.shape[1]. "
                "Check patch_size vs encoder token count."
            ),
        )
        with tf.control_dependencies([check_op]):
            pred_pixels = tf.identity(pred_pixels)
        # ------------------------------------------------------------------

        return simmim_l1_loss(pred_pixels, target_patches, patch_mask)

    def update_alpha_by_step(self, global_step: int, warmup_steps: int) -> None:
        """Step-based alpha warm-up: 1.0 → alpha_final over warmup_steps batches."""
        if warmup_steps <= 0:
            self.alpha_var.assign(self.alpha_final)
            return
        step = max(int(global_step), 0)
        progress = min(1.0, step / max(int(warmup_steps), 1))
        self.alpha_var.assign(1.0 + progress * (self.alpha_final - 1.0))
        self.step_counter.assign(step)

    def train_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]   # teacher global1 + SimMIM target
        aug_global2    = views[1]   # teacher/student global2 (DINO)
        masked_clean   = views[2]   # SimMIM student input
        patch_mask     = views[3]   # [B, N]
        locals_        = views[4:4 + self.n_local]

        with tf.GradientTape() as tape:
            # --- DINO loss: delegated entirely to shared helper ---
            # Local crops are passed in; helper uses identical encoder path for all views
            l_dino, t_logits_list = _compute_dino_loss(
                self.online_encoder, self.projector,
                self.teacher_encoder, self.teacher_projector,
                self.center, self.teacher_temp_var,
                self.student_temp,
                original_clean, aug_global2, locals_,
                training=True,
            )

            # --- SimMIM loss: separate branch with patch_mask shape assert ---
            l_simmim = self._simmim_forward(
                masked_clean, original_clean, patch_mask, training=True
            )

            alpha = self.alpha_var
            loss  = alpha * l_dino + (1.0 - alpha) * self.lambda_mim * l_simmim

        trainable = (
            self.online_encoder.trainable_weights
            + self.projector.trainable_weights
            + self.pixel_pred_head.trainable_weights
        )
        grads = tape.gradient(loss, trainable)
        self.optimizer.apply_gradients(zip(grads, trainable))

        self._init_teacher()
        self._ema_update()
        self._update_center(t_logits_list)

        return {
            'loss': loss, 'l_dino': l_dino, 'l_simmim': l_simmim,
            'alpha': alpha, 'teacher_temp': self.teacher_temp_var,
        }

    def test_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]
        aug_global2    = views[1]
        masked_clean   = views[2]
        patch_mask     = views[3]

        l_dino, _ = _compute_dino_loss(
            self.online_encoder, self.projector,
            self.teacher_encoder, self.teacher_projector,
            self.center, self.teacher_temp_var,
            self.student_temp,
            original_clean, aug_global2, [],   # no locals in val
            training=False,
        )
        l_simmim = self._simmim_forward(
            masked_clean, original_clean, patch_mask, training=False
        )
        alpha = self.alpha_var
        loss  = alpha * l_dino + (1.0 - alpha) * self.lambda_mim * l_simmim
        return {
            'loss': loss, 'l_dino': l_dino, 'l_simmim': l_simmim,
            'alpha': alpha, 'teacher_temp': self.teacher_temp_var,
        }

    def get_stage2_encoder(self, use_teacher: bool = True):
        return self.teacher_encoder if use_teacher else self.online_encoder


def build_stage1_dino_simmim_trainer(
    encoder_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    projection_dim: int = 256,
    patch_size: int = 16,
    temperature: float = 0.1,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    ema_momentum: float = 0.996,
    n_local: int = 4,
    lambda_mim: float = 1.0,
    alpha_final: float = 0.7,
    alpha_warmup_epochs: int = 20,
    lr: LrInput = 1e-4,
    clipnorm: float | None = 1.0,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
) -> DINOSimMIMModel:
    """Build DINOSimMIMModel. encoder.py handles dynamic pos_embed interpolation."""
    model = DINOSimMIMModel(
        encoder_name=encoder_name,
        input_shape=input_shape,
        projection_dim=projection_dim,
        patch_size=patch_size,
        student_temp=temperature,
        teacher_temp=teacher_temp,
        center_momentum=center_momentum,
        ema_momentum=ema_momentum,
        n_local=n_local,
        lambda_mim=lambda_mim,
        alpha_final=alpha_final,
        alpha_warmup_epochs=alpha_warmup_epochs,
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


class AlphaWarmupCallback(keras.callbacks.Callback):
    """Step-based alpha warm-up for DINOSimMIMModel.

    Mirrors TeacherTempWarmupCallback — both operate on on_train_batch_end.
    alpha linearly transitions 1.0 → alpha_final over warmup_steps batches.

    Args:
        warmup_steps: total training steps for warm-up.
                      Typical usage: alpha_warmup_epochs * steps_per_epoch.
        verbose:      print log every 20 steps when True.
    """

    def __init__(self, warmup_steps: int, verbose: bool = False):
        super().__init__()
        self.warmup_steps = max(int(warmup_steps), 1)
        self.verbose = verbose
        self._global_step = 0

    def on_train_begin(self, logs=None):
        self._global_step = 0
        if hasattr(self.model, 'alpha_var'):
            self.model.alpha_var.assign(1.0)
        if self.verbose:
            print(f"\n[AlphaWarmup] init  alpha = {float(self.model.alpha_var.numpy()):.6f}")

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if hasattr(self.model, 'update_alpha_by_step'):
            self.model.update_alpha_by_step(self._global_step, self.warmup_steps)
            if self.verbose and self._global_step % 20 == 0:
                alpha = float(self.model.alpha_var.numpy())
                ratio = min(self._global_step / self.warmup_steps, 1.0)
                print(
                    f"\n[AlphaWarmup] step={self._global_step:5d}  "
                    f"ratio={ratio:.4f}  alpha={alpha:.6f}"
                )
