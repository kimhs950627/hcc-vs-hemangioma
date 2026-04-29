from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from models.encoder import build_encoder
from training.losses import dino_cross_entropy, simmim_l1_loss, patchify_images
from training.ssl_schedules import LrInput


# ===========================================================================
# Legacy DINO-only model  (backward compat)
# ===========================================================================

class DINOPretrainModel(keras.Model):
    """DINO self-supervised pretraining model (legacy, DINO-only).

    teacher_temp_var is a plain tf.Variable that external callbacks
    (TeacherTempWarmupCallback) update on_train_batch_end.  The model
    itself never touches teacher_temp_var except to read it.

    For new training, prefer DINOSimMIMModel (\uad8c\uc7a5 B+ \uc2e4\ud5d8 \ud22c\uc601).
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
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _teacher_probs(self, logits):
        centered = logits - self.center
        return tf.nn.softmax(centered / tf.maximum(self.teacher_temp_var, 1e-6), axis=-1)

    def train_step(self, data):
        views   = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]

        with tf.GradientTape() as tape:
            t1_logits = self._teacher_logits(global1)
            t2_logits = self._teacher_logits(global2)
            t1 = tf.stop_gradient(self._teacher_probs(t1_logits))
            t2 = tf.stop_gradient(self._teacher_probs(t2_logits))

            s1 = self._student_logits(global1, training=True)
            s2 = self._student_logits(global2, training=True)

            loss_terms = [
                dino_cross_entropy(s1, tf.math.log(t1 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s1, tf.math.log(t2 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s2, tf.math.log(t1 + 1e-8), self.student_temp, 1.0),
                dino_cross_entropy(s2, tf.math.log(t2 + 1e-8), self.student_temp, 1.0),
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
        views   = tf.nest.flatten(data)
        global1 = views[0]
        global2 = views[1]
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

    Implements the \uad8c\uc7a5 B+ training structure:

    Views (from MaskedMultiViewDataset):
        views[0] = original_clean   : teacher input + SimMIM reconstruction target
        views[1] = masked_clean     : SimMIM student input (masked pixels)
        views[2] = patch_mask       : [B, N]  1=masked, 0=visible
        views[3] = aug_global       : DINO student global view (solarization p=0.2)
        views[4:] = local_i         : DINO student local crops (solarization p=0.2)

    Loss structure:
        L_dino   = CE(cls_aug_global, cls_teacher)         global-global
                 + mean_i CE(cls_local_i, cls_teacher)     local-global
        L_simmim = mean_{masked} |pixel_pred - original|_1 (mean L1)
        L_total  = alpha(epoch) * L_dino
                 + (1 - alpha(epoch)) * lambda_mim * L_simmim

    Alpha warm-up (SimMIM cold-start):
        alpha starts at 1.0 (pure DINO) and linearly decreases to
        alpha_final over alpha_warmup_epochs.  This prevents SimMIM from
        destabilizing DINO in the first few epochs when patch reconstruction
        is random.  After warmup, both losses contribute at stable ratio.

    Patch token extraction:
        ViT encoder: uses last_hidden_state (patch tokens, shape [B, N, D])
        CNN encoder: spatially pools feature map to [B, n_h*n_w, D]
            n_h = H // patch_size,  n_w = W // patch_size

    pixel_pred_head:
        Linear projection from encoder hidden dim D to P*P*C per patch.
        Simple single linear layer (no decoder blocks) following SimMIM.

    Teacher note:
        Teacher receives original_clean ONLY (no solarization).
        Validated: D_KL(teacher_raw || teacher_solar) = 12.06 (193x noise)
        -> solarized teacher input destabilizes center update + EMA target

    Why global-global DINO loss is kept:
        Without it, student encoder never processes 224x224 full-res input.
        Fine-tuning receives full-res -> distribution shift validated harmful.
        Holistic texture + shape cues (critical for HCC vs Hemangioma) require
        full-resolution context.

    Args:
        encoder_name      : Encoder identifier passed to build_encoder().
        input_shape       : (H, W, C) for global views.
        projection_dim    : CLS projection head output dim.
        patch_size        : ViT patch size (must match encoder and dataloader).
        student_temp      : Student softmax temperature.
        teacher_temp      : Teacher softmax temperature (initial, warmed up by callback).
        center_momentum   : EMA momentum for teacher center update.
        ema_momentum      : EMA momentum for teacher encoder/head update.
        n_local           : Number of local crop views.
        lambda_mim        : SimMIM loss weight (relative to DINO).
        alpha_final       : Final alpha after warmup (default 0.7 -> 30% SimMIM).
        alpha_warmup_epochs: Epochs to linearly reduce alpha 1.0 -> alpha_final.
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
        self._patch_dim = patch_size * patch_size * C  # P*P*C per patch

        # Encoders
        self.online_encoder  = build_encoder(encoder_name, input_shape=input_shape)
        self.teacher_encoder = build_encoder(encoder_name, input_shape=input_shape)

        # DINO projection heads
        self.projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_student_head')
        self.teacher_projector = keras.Sequential([
            layers.Dense(projection_dim, activation='gelu'),
            layers.Dense(projection_dim),
        ], name='dino_teacher_head')

        # SimMIM pixel reconstruction head
        # Input : [B, N, D_enc]  (patch token sequence from encoder)
        # Output: [B, N, P*P*C]  (flattened pixel values per patch)
        self.pixel_pred_head = layers.Dense(
            self._patch_dim,
            use_bias=True,
            name='pixel_pred_head',
        )

        # DINO center (EMA)
        self.center = tf.Variable(
            tf.zeros([1, projection_dim], dtype=tf.float32),
            trainable=False, name='dino_center',
        )
        # Teacher temperature (updated by TeacherTempWarmupCallback)
        self.teacher_temp_var = tf.Variable(
            float(teacher_temp), trainable=False,
            dtype=tf.float32, name='teacher_temp_var',
        )
        # Alpha for loss balancing (updated on epoch end via callback or train_step)
        self.alpha_var = tf.Variable(
            1.0, trainable=False, dtype=tf.float32, name='alpha_var',
        )
        # Epoch counter (updated by AlphaWarmupCallback or train_step)
        self.epoch_counter = tf.Variable(
            0, trainable=False, dtype=tf.int32, name='epoch_counter',
        )
        self._teacher_initialized = False

    def compile(self, optimizer, **kwargs):
        super().compile(jit_compile=False, **kwargs)
        self.optimizer = optimizer

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _init_teacher(self):
        """Copy student weights to teacher on first call."""
        if self._teacher_initialized:
            return
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(sw)
        self._teacher_initialized = True

    def _ema_update(self):
        """EMA update of teacher encoder and head."""
        m = self.ema_momentum
        for sw, tw in zip(self.online_encoder.weights, self.teacher_encoder.weights):
            tw.assign(m * tw + (1.0 - m) * sw)
        for sw, tw in zip(self.projector.weights, self.teacher_projector.weights):
            tw.assign(m * tw + (1.0 - m) * sw)

    def _update_center(self, teacher_logits_list: list[tf.Tensor]):
        """EMA update of DINO center vector."""
        concat = tf.concat(teacher_logits_list, axis=0)
        batch_center = tf.reduce_mean(concat, axis=0, keepdims=True)
        self.center.assign(
            self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_center
        )

    def _teacher_probs(self, logits: tf.Tensor) -> tf.Tensor:
        """Compute sharpened, centered teacher probabilities."""
        centered = logits - self.center
        return tf.nn.softmax(centered / tf.maximum(self.teacher_temp_var, 1e-6), axis=-1)

    def _student_cls(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Student encoder CLS + DINO head."""
        out = self.online_encoder(x, training=training)
        return self.projector(out['embedding'], training=training)

    def _teacher_cls(self, x: tf.Tensor) -> tf.Tensor:
        """Teacher encoder CLS + DINO head (stop_gradient applied externally)."""
        out = self.teacher_encoder(x, training=False)
        return self.teacher_projector(out['embedding'], training=False)

    def _student_patch_tokens(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Extract patch token sequence from student encoder.

        Returns:
            patch_tokens : [B, N, D_enc]  float32

        For ViT:  uses 'last_hidden_state' (shape [B, N+1, D]),
                  drops CLS token at index 0 -> [B, N, D]
        For CNN:  spatially pools feature map -> [B, n_h*n_w, D]
        """
        out = self.online_encoder(x, training=training)

        if 'last_hidden_state' in out:
            # ViT: [B, 1+N, D] -> [B, N, D]  (drop CLS token)
            tokens = out['last_hidden_state'][:, 1:, :]  # [B, N, D]
        else:
            # CNN fallback: adaptive average pool spatial grid
            # feature_map: [B, h_feat, w_feat, D]
            feat = out.get('feature_map', out.get('embedding', None))
            if feat is None:
                raise KeyError(
                    "Encoder output must contain 'last_hidden_state' (ViT) "
                    "or 'feature_map' (CNN) for SimMIM patch token extraction."
                )
            B = tf.shape(feat)[0]
            H = tf.shape(feat)[1]
            W = tf.shape(feat)[2]
            D = tf.shape(feat)[3]
            n_h = tf.shape(feat)[1] // (self.patch_size // 4)  # rough estimate
            # Simpler: reshape feature map to [B, N_spatial, D]
            tokens = tf.reshape(feat, [B, H * W, D])  # [B, H_f*W_f, D]

        return tokens  # [B, N, D_enc]

    # -------------------------------------------------------------------------
    # Loss alpha warm-up
    # -------------------------------------------------------------------------

    def update_alpha(self, epoch: int) -> None:
        """Update alpha for DINO/SimMIM loss balance. Call at epoch start.

        alpha linearly decays from 1.0 (pure DINO) to alpha_final over
        alpha_warmup_epochs, then stays at alpha_final.

        Args:
            epoch : 0-indexed epoch number
        """
        if self.alpha_warmup_epochs <= 0:
            self.alpha_var.assign(self.alpha_final)
            return
        progress = min(1.0, epoch / self.alpha_warmup_epochs)
        alpha = 1.0 + progress * (self.alpha_final - 1.0)
        self.alpha_var.assign(alpha)
        self.epoch_counter.assign(epoch)

    # -------------------------------------------------------------------------
    # Train step
    # -------------------------------------------------------------------------

    def train_step(self, data):
        """DINO + SimMIM joint training step.

        Expects data as a flat tuple:
            (original_clean, masked_clean, patch_mask,
             aug_global, local_1, ..., local_N)

        Loss:
            L = alpha * L_dino + (1 - alpha) * lambda_mim * L_simmim
        """
        views = tf.nest.flatten(data)
        # Unpack views
        original_clean = views[0]   # [B, H, W, 3]
        masked_clean   = views[1]   # [B, H, W, 3]  (SimMIM student)
        patch_mask     = views[2]   # [B, N]
        aug_global     = views[3]   # [B, H, W, 3]  (DINO student global)
        locals_        = views[4:4 + self.n_local]  # list of [B, h, w, 3]

        with tf.GradientTape() as tape:
            # -------- Teacher (no grad, no aug) --------
            t_logits  = self._teacher_cls(original_clean)         # [B, D]
            t_probs   = tf.stop_gradient(self._teacher_probs(t_logits))  # [B, D]

            # -------- DINO loss --------
            # global-global: student aug_global vs teacher original_clean
            s_global_cls = self._student_cls(aug_global, training=True)  # [B, D]
            l_dino_global = dino_cross_entropy(
                s_global_cls,
                tf.math.log(t_probs + 1e-8),
                self.student_temp, 1.0,
            )

            # local-global
            l_dino_local_terms: list[tf.Tensor] = []
            for local_v in locals_:
                s_local_cls = self._student_cls(local_v, training=True)
                l_dino_local_terms.append(
                    dino_cross_entropy(
                        s_local_cls,
                        tf.math.log(t_probs + 1e-8),
                        self.student_temp, 1.0,
                    )
                )

            if l_dino_local_terms:
                l_dino = l_dino_global + tf.add_n(l_dino_local_terms) / tf.cast(
                    len(l_dino_local_terms), tf.float32
                )
            else:
                l_dino = l_dino_global

            # -------- SimMIM loss --------
            # 1) Extract patch tokens from masked image
            patch_tokens = self._student_patch_tokens(masked_clean, training=True)
            # [B, N, D_enc] -> [B, N, P*P*C]
            pred_pixels  = self.pixel_pred_head(patch_tokens)

            # 2) Build patchified target from original_clean
            target_patches = patchify_images(
                tf.stop_gradient(original_clean), self.patch_size
            )  # [B, N, P*P*C]

            l_simmim = simmim_l1_loss(pred_pixels, target_patches, patch_mask)

            # -------- Combined loss --------
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
        self._update_center([t_logits])

        return {
            'loss':         loss,
            'l_dino':       l_dino,
            'l_simmim':     l_simmim,
            'alpha':        alpha,
            'teacher_temp': self.teacher_temp_var,
        }

    def test_step(self, data):
        views          = tf.nest.flatten(data)
        original_clean = views[0]
        masked_clean   = views[1]
        patch_mask     = views[2]
        aug_global     = views[3]

        t_logits  = self._teacher_cls(original_clean)
        t_probs   = self._teacher_probs(t_logits)
        s_global  = self._student_cls(aug_global, training=False)
        l_dino    = dino_cross_entropy(
            s_global, tf.math.log(t_probs + 1e-8), self.student_temp, 1.0
        )

        patch_tokens   = self._student_patch_tokens(masked_clean, training=False)
        pred_pixels    = self.pixel_pred_head(patch_tokens)
        target_patches = patchify_images(tf.stop_gradient(original_clean), self.patch_size)
        l_simmim       = simmim_l1_loss(pred_pixels, target_patches, patch_mask)

        alpha = self.alpha_var
        loss  = alpha * l_dino + (1.0 - alpha) * self.lambda_mim * l_simmim
        return {
            'loss': loss, 'l_dino': l_dino, 'l_simmim': l_simmim,
            'alpha': alpha, 'teacher_temp': self.teacher_temp_var,
        }

    def get_stage2_encoder(self, use_teacher: bool = True):
        """Return encoder for Stage2 fine-tuning."""
        return self.teacher_encoder if use_teacher else self.online_encoder


# ===========================================================================
# Builder functions
# ===========================================================================

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
    """Build DINOSimMIMModel with AdamW optimizer (권장 B+ 구조).

    Pairs with MaskedMultiViewDataset.
    Recommended hyperparameters for HCC vs. Hemangioma:
        n_local=4, lambda_mim=1.0, alpha_final=0.7,
        alpha_warmup_epochs=20, teacher_temp=0.04.

    Example::
        from dataloader import MaskedMultiViewDataset
        from training.stage1_dino import build_stage1_dino_simmim_trainer

        ds = MaskedMultiViewDataset(
            data_root='./clean_ver_for_train',
            split='train',
            batch_size=16,
            local_views=4,
            mask_ratio=0.75,
        )
        model = build_stage1_dino_simmim_trainer(
            encoder_name='vit_small',
            n_local=4,
        )
        # Optional: add AlphaWarmupCallback for alpha decay
        model.fit(ds.as_dataset(), epochs=100)
    """
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


# ---------------------------------------------------------------------------
# AlphaWarmupCallback: decay alpha from 1.0 -> alpha_final over warmup epochs
# ---------------------------------------------------------------------------

class AlphaWarmupCallback(keras.callbacks.Callback):
    """Callback to linearly warm-up the DINO/SimMIM alpha blend ratio.

    Calls model.update_alpha(epoch) at the start of each epoch.
    Must be passed when using DINOSimMIMModel.

    Usage::
        model = build_stage1_dino_simmim_trainer(...)
        callbacks = [
            AlphaWarmupCallback(),
            TeacherTempWarmupCallback(start_temp=0.04, end_temp=0.07, warmup_epochs=30),
            # ...
        ]
        model.fit(ds, epochs=100, callbacks=callbacks)
    """

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model, 'update_alpha'):
            self.model.update_alpha(epoch)
            alpha = float(self.model.alpha_var.numpy())
            print(f"  [AlphaWarmup] epoch={epoch}  alpha={alpha:.4f}")
