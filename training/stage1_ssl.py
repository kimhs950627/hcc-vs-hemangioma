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
    n_local: int = 0,
    clipnorm: float | None = None,
    clipvalue: float | None = None,
    weight_decay: float = 1e-4,
    # --- dino_simmim only ---
    lambda_mim: float = 0.1,
    patch_size: int = 16,
):
    """Build a stage-1 SSL trainer.

    Supported ssl_mode values: 'moco', 'byol', 'dino', 'dino_simmim'

    Dataloader pairing (REQUIRED):
        moco / byol / dino   ->  MultiViewDataset
        dino_simmim          ->  MaskedMultiViewDataset

    Mixing incompatible dataloader + ssl_mode WILL crash at step 1
    due to shape mismatch in train_step view unpacking.

    Teacher temperature warmup is handled externally via
    TeacherTempWarmupCallback. Pass teacher_temp as the initial value only.

    Args:
        lambda_mim  : SimMIM loss weight. Only used when ssl_mode='dino_simmim'.
        patch_size  : ViT patch size for SimMIM. Only used when ssl_mode='dino_simmim'.
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
            temperature=temperature,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
            ema_momentum=ema_momentum,
            n_local=n_local,
            lambda_mim=lambda_mim,
            patch_size=patch_size,
            lr=lr,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            weight_decay=weight_decay,
        )

    raise ValueError(f'Unsupported ssl_mode={ssl_mode!r}. Expected one of: {SSL_MODES}')
