from __future__ import annotations

from training.stage1_byol import BYOLPretrainModel, build_stage1_byol_trainer
from training.stage1_dino import (
    DINOPretrainModel,
    DINOSimMIMModel,
    build_stage1_dino_trainer,
    build_stage1_dino_simmim_trainer,
)
from training.stage1_moco import MoCoPretrainModel, build_stage1_moco_trainer
from training.ssl_schedules import LrInput


SSL_MODES = ('moco', 'byol', 'dino', 'dino_simmim')


def build_stage1_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    ema_momentum: float = 0.996,
    lr: LrInput = 1e-4,
    ssl_mode: str = 'moco',
    predictor_dim: int = 256,
    teacher_temp: float = 0.04,
    center_momentum: float = 0.9,
    n_local: int = 4,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
    # --- dino_simmim-specific (ignored for other modes) ---
    patch_size: int = 16,
    lambda_mim: float = 1.0,
    alpha_final: float = 0.7,
    alpha_warmup_epochs: int = 20,
):
    """Unified Stage-1 SSL trainer router.

    Supported ssl_mode values:
        'moco'         — Momentum Contrast (2 global views)
        'byol'         — Bootstrap Your Own Latent (2 global views)
        'dino'         — DINO self-distillation (2 global + N local views)
        'dino_simmim'  — DINO + SimMIM hybrid (2 global + N local + masked view)

    For 'dino_simmim', the dataset MUST be MaskedMultiViewDataset which yields:
        views[0] original_clean   — teacher global1 + SimMIM target
        views[1] aug_global2      — teacher/student global2 (DINO)
        views[2] masked_clean     — SimMIM student input
        views[3] patch_mask       — [B, N_patches]  1=masked, 0=visible
        views[4:] local_i         — DINO student local crops

    Teacher temperature warmup is handled externally via
    ``TeacherTempWarmupCallback`` — do NOT pass warmup args here.
    Pass ``teacher_temp`` as the *initial* scalar value only.
    """
    mode = ssl_mode.lower()

    if mode == 'moco':
        return build_stage1_moco_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            temperature=temperature,
            ema_momentum=ema_momentum,
            lr=lr,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            weight_decay=weight_decay,
        )
    if mode == 'byol':
        return build_stage1_byol_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            predictor_dim=predictor_dim,
            ema_momentum=ema_momentum,
            lr=lr,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            weight_decay=weight_decay,
        )
    if mode == 'dino':
        return build_stage1_dino_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            temperature=temperature,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
            ema_momentum=ema_momentum,
            n_local=n_local,
            lr=lr,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            weight_decay=weight_decay,
        )
    if mode == 'dino_simmim':
        return build_stage1_dino_simmim_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            patch_size=patch_size,
            temperature=temperature,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
            ema_momentum=ema_momentum,
            n_local=n_local,
            lambda_mim=lambda_mim,
            alpha_final=alpha_final,
            alpha_warmup_epochs=alpha_warmup_epochs,
            lr=lr,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            weight_decay=weight_decay,
        )
    raise ValueError(f'Unsupported ssl_mode={ssl_mode!r}. Expected one of: {SSL_MODES}')
