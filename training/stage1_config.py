from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from training.ssl_schedules import LrInput


SSLMode = Literal["moco", "byol", "dino", "dino_simmim"]


@dataclass
class EncoderConfig:
    name: str = "vit"
    input_shape: tuple[int, int, int] = (224, 224, 3)
    embed_dim: int = 384
    depth: int | None = None
    num_heads: int | None = None
    mlp_dim: int | None = None
    patch_size: int | None = 16
    dropout: float = 0.1
    use_pe: bool = False
    attn_drop: bool = False
    attn_drop_rate: float = 0.0
    attn_drop_top_k: int = 40


@dataclass
class DINOConfig:
    projection_dim: int = 256
    student_temp: float = 0.1
    teacher_temp: float = 0.04
    center_momentum: float = 0.9
    ema_momentum: float = 0.996
    n_local: int = 0


@dataclass
class BYOLConfig:
    projection_dim: int = 256
    predictor_dim: int = 256
    ema_momentum: float = 0.996


@dataclass
class MoCoConfig:
    projection_dim: int = 256
    temperature: float = 0.1
    ema_momentum: float = 0.996


@dataclass
class SimMIMConfig:
    weight: float = 0.0
    patch_size: int = 16
    norm_target: bool = True


@dataclass
class SelfPatchConfig:
    weight: float = 0.0
    proj_dim: int = 256
    top_k: int = 4
    temperature: float = 0.07



@dataclass
class DisagreementConfig:
    weight: float = 0.0
    mode: str = "attention_map"
    apply_to: str = "student"
    start_layer: int | None = None
    end_layer: int | None = None
    normalize: bool = True

@dataclass
class DiversityConfig:
    weight: float = 0.0


@dataclass
class OptimConfig:
    lr: LrInput = 1e-4
    clipnorm: float | None = None
    clipvalue: float | None = None
    weight_decay: float = 1e-4


@dataclass
class Stage1TrainerConfig:
    ssl_mode: SSLMode = "dino"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    byol: BYOLConfig = field(default_factory=BYOLConfig)
    moco: MoCoConfig = field(default_factory=MoCoConfig)
    simmim: SimMIMConfig = field(default_factory=SimMIMConfig)
    selfpatch: SelfPatchConfig = field(default_factory=SelfPatchConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "ssl_mode": self.ssl_mode,
            "encoder_name": self.encoder.name,
            "input_shape": self.encoder.input_shape,
            "embed_dim": self.encoder.embed_dim,
            "depth": self.encoder.depth,
            "num_heads": self.encoder.num_heads,
            "patch_size": self.encoder.patch_size,
            "use_pe": self.encoder.use_pe,
            "n_local": self.dino.n_local,
            "lambda_simmim": self.simmim.weight,
            "lambda_selfpatch": self.selfpatch.weight,
            "lambda_diversity": self.diversity.weight,
            "lr": self.optim.lr,
            "weight_decay": self.optim.weight_decay,
        }


def core_wandb_config(cfg: Stage1TrainerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "stage": 1,
        "ssl_mode": cfg.ssl_mode,
        "encoder_name": cfg.encoder.name,
        "input_shape": cfg.encoder.input_shape,
        "embed_dim": cfg.encoder.embed_dim,
        "depth": cfg.encoder.depth,
        "num_heads": cfg.encoder.num_heads,
        "patch_size": cfg.encoder.patch_size,
        "use_pe": cfg.encoder.use_pe,
        "n_local": cfg.dino.n_local,
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
    }

    if cfg.ssl_mode in ("moco", "dino", "dino_simmim"):
        out["temperature"] = cfg.moco.temperature if cfg.ssl_mode == "moco" else cfg.dino.student_temp
    if cfg.ssl_mode in ("dino", "dino_simmim"):
        out["teacher_temp"] = cfg.dino.teacher_temp
        out["center_momentum"] = cfg.dino.center_momentum
        out["ema_momentum"] = cfg.dino.ema_momentum
        out["projection_dim"] = cfg.dino.projection_dim
    if cfg.ssl_mode == "byol":
        out["projection_dim"] = cfg.byol.projection_dim
        out["predictor_dim"] = cfg.byol.predictor_dim
        out["ema_momentum"] = cfg.byol.ema_momentum
    if cfg.ssl_mode == "moco":
        out["projection_dim"] = cfg.moco.projection_dim
        out["ema_momentum"] = cfg.moco.ema_momentum

    if cfg.simmim.weight > 0.0:
        out["lambda_simmim"] = cfg.simmim.weight
        out["simmim_norm_target"] = cfg.simmim.norm_target
    if cfg.selfpatch.weight > 0.0:
        out["lambda_selfpatch"] = cfg.selfpatch.weight
        out["selfpatch_top_k"] = cfg.selfpatch.top_k
    if cfg.diversity.weight > 0.0:
        out["lambda_diversity"] = cfg.diversity.weight
    if cfg.encoder.attn_drop:
        out["attn_drop_rate"] = cfg.encoder.attn_drop_rate
        out["attn_drop_top_k"] = cfg.encoder.attn_drop_top_k
    return out
