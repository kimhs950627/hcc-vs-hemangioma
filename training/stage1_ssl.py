from __future__ import annotations

from training.stage1_byol import BYOLPretrainModel, build_stage1_byol_trainer
from training.stage1_config import Stage1TrainerConfig
from training.stage1_dino import DINOPretrainModel, build_stage1_dino_trainer
from training.stage1_dino_simmim import DINOSimMIMPretrainModel, build_stage1_dino_simmim_trainer
from training.stage1_moco import MoCoPretrainModel, build_stage1_moco_trainer
from training.ssl_schedules import LrInput


SSL_MODES = ('moco', 'byol', 'dino', 'dino_simmim')


def build_stage1_trainer(
    cfg: Stage1TrainerConfig | None = None,
    encoder_name: str = 'vit',
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
    lambda_simmim: float = 0.3,
    patch_size: int = 16,
    simmim_norm_target: bool = True,
    use_pe: bool = False,
):
    """Build a stage-1 SSL trainer.

    Teacher temperature warmup is handled externally via
    ``TeacherTempWarmupCallback`` — do NOT pass warmup args here.
    Pass ``teacher_temp`` as the *initial* scalar value only.
    """
    if cfg is not None:
        mode = cfg.ssl_mode.lower()

        if mode == 'moco':
            return build_stage1_moco_trainer(
                encoder_name=cfg.encoder.name,
                input_shape=cfg.encoder.input_shape,
                projection_dim=cfg.moco.projection_dim,
                temperature=cfg.moco.temperature,
                ema_momentum=cfg.moco.ema_momentum,
                lr=cfg.optim.lr,
                clipnorm=cfg.optim.clipnorm,
                clipvalue=cfg.optim.clipvalue,
                weight_decay=cfg.optim.weight_decay,
                lambda_diversity=cfg.diversity.weight,
            )
        if mode == 'byol':
            return build_stage1_byol_trainer(
                encoder_name=cfg.encoder.name,
                input_shape=cfg.encoder.input_shape,
                projection_dim=cfg.byol.projection_dim,
                predictor_dim=cfg.byol.predictor_dim,
                ema_momentum=cfg.byol.ema_momentum,
                lr=cfg.optim.lr,
                clipnorm=cfg.optim.clipnorm,
                clipvalue=cfg.optim.clipvalue,
                weight_decay=cfg.optim.weight_decay,
                lambda_diversity=cfg.diversity.weight,
            )
        if mode == 'dino':
            return build_stage1_dino_trainer(
                encoder_name=cfg.encoder.name,
                input_shape=cfg.encoder.input_shape,
                projection_dim=cfg.dino.projection_dim,
                temperature=cfg.dino.student_temp,
                teacher_temp=cfg.dino.teacher_temp,
                center_momentum=cfg.dino.center_momentum,
                ema_momentum=cfg.dino.ema_momentum,
                n_local=cfg.dino.n_local,
                encoder_embed_dim=cfg.encoder.embed_dim,
                encoder_depth=cfg.encoder.depth,
                encoder_num_heads=cfg.encoder.num_heads,
                encoder_mlp_dim=cfg.encoder.mlp_dim,
                encoder_patch_size=cfg.encoder.patch_size,
                encoder_dropout=cfg.encoder.dropout,
                use_pe=cfg.encoder.use_pe,
                encoder_attn_drop=cfg.encoder.attn_drop,
                encoder_attn_drop_rate=cfg.encoder.attn_drop_rate,
                encoder_attn_drop_top_k=cfg.encoder.attn_drop_top_k,
                lambda_selfpatch=cfg.selfpatch.weight,
                selfpatch_proj_dim=cfg.selfpatch.proj_dim,
                selfpatch_top_k=cfg.selfpatch.top_k,
                selfpatch_temperature=cfg.selfpatch.temperature,
                lr=cfg.optim.lr,
                clipnorm=cfg.optim.clipnorm,
                clipvalue=cfg.optim.clipvalue,
                weight_decay=cfg.optim.weight_decay,
                lambda_diversity=cfg.diversity.weight,
            )
        if mode == 'dino_simmim':
            return build_stage1_dino_simmim_trainer(
                encoder_name=cfg.encoder.name,
                input_shape=cfg.encoder.input_shape,
                projection_dim=cfg.dino.projection_dim,
                temperature=cfg.dino.student_temp,
                teacher_temp=cfg.dino.teacher_temp,
                center_momentum=cfg.dino.center_momentum,
                ema_momentum=cfg.dino.ema_momentum,
                n_local=cfg.dino.n_local,
                lambda_simmim=cfg.simmim.weight,
                patch_size=cfg.simmim.patch_size,
                simmim_norm_target=cfg.simmim.norm_target,
                encoder_embed_dim=cfg.encoder.embed_dim,
                encoder_depth=cfg.encoder.depth,
                encoder_num_heads=cfg.encoder.num_heads,
                encoder_mlp_dim=cfg.encoder.mlp_dim,
                encoder_patch_size=cfg.encoder.patch_size,
                encoder_dropout=cfg.encoder.dropout,
                use_pe=cfg.encoder.use_pe,
                encoder_attn_drop=cfg.encoder.attn_drop,
                encoder_attn_drop_rate=cfg.encoder.attn_drop_rate,
                encoder_attn_drop_top_k=cfg.encoder.attn_drop_top_k,
                lambda_selfpatch=cfg.selfpatch.weight,
                selfpatch_proj_dim=cfg.selfpatch.proj_dim,
                selfpatch_top_k=cfg.selfpatch.top_k,
                selfpatch_temperature=cfg.selfpatch.temperature,
                lr=cfg.optim.lr,
                clipnorm=cfg.optim.clipnorm,
                clipvalue=cfg.optim.clipvalue,
                weight_decay=cfg.optim.weight_decay,
                lambda_diversity=cfg.diversity.weight,
            )
        raise ValueError(f'Unsupported ssl_mode={cfg.ssl_mode}. Expected one of: {SSL_MODES}')

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
            lambda_diversity=0.0,
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
            lambda_diversity=0.0,
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
            lambda_diversity=0.0,
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
            lambda_simmim=lambda_simmim,
            patch_size=patch_size,
            simmim_norm_target=simmim_norm_target,
            use_pe=use_pe,
            lr=lr,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            weight_decay=weight_decay,
            lambda_diversity=0.0,
        )
    raise ValueError(f'Unsupported ssl_mode={ssl_mode}. Expected one of: {SSL_MODES}')
