
from __future__ import annotations

from training.stage1_byol import BYOLPretrainModel, build_stage1_byol_trainer
from training.stage1_dino import DINOPretrainModel, build_stage1_dino_trainer
from training.stage1_moco import MoCoPretrainModel, build_stage1_moco_trainer


SSL_MODES = ('moco', 'byol', 'dino')


def build_stage1_trainer(
    encoder_name: str,
    input_shape=(224, 224, 3),
    projection_dim: int = 256,
    temperature: float = 0.1,
    ema_momentum: float = 0.996,
    lr: float = 1e-4,
    ssl_mode: str = 'moco',
    predictor_dim: int = 256,
    teacher_temp: float = 0.04,
    teacher_temp_warmup_start: float = 0.04,
    teacher_temp_target: float = 0.04,
    warmup_epochs: int = 10,
    center_momentum: float = 0.9,
    n_local: int = 0,
):
    mode = ssl_mode.lower()
    if mode == 'moco':
        return build_stage1_moco_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            temperature=temperature,
            ema_momentum=ema_momentum,
            lr=lr,
        )
    if mode == 'byol':
        return build_stage1_byol_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            predictor_dim=predictor_dim,
            ema_momentum=ema_momentum,
            lr=lr,
        )
    if mode == 'dino':
        return build_stage1_dino_trainer(
            encoder_name=encoder_name,
            input_shape=input_shape,
            projection_dim=projection_dim,
            temperature=temperature,
            teacher_temp=teacher_temp,
            teacher_temp_warmup_start=teacher_temp_warmup_start,
            teacher_temp_target=teacher_temp_target,
            warmup_epochs=warmup_epochs,
            center_momentum=center_momentum,
            ema_momentum=ema_momentum,
            n_local=n_local,
            lr=lr,
        )
    raise ValueError(f'Unsupported ssl_mode={ssl_mode}. Expected one of: {SSL_MODES}')
