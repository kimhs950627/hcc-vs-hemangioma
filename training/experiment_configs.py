"""Experiment preset factory for Stage-1 SSL training.

Usage (in notebook or script):
    from training.experiment_configs import build_experiment_cfg
    cfg = build_experiment_cfg(base_cfg, experiment_name)
"""
from __future__ import annotations

import copy
from training.stage1_config import Stage1TrainerConfig


# ---------------------------------------------------------------------------
# common diversity defaults (shared across all presets that enable diversity)
# ---------------------------------------------------------------------------
_DIV_DEFAULTS = dict(
    layer_mode="top_half",
    start_layer=None,
    end_layer=None,
    exclude_cls_col=True,
    entropy_weight=1.0,
    entropy_min=2.5,
)


def build_experiment_cfg(
    base_cfg: Stage1TrainerConfig,
    experiment_name: str,
) -> Stage1TrainerConfig:
    """Return a deep-copy of base_cfg overridden for the requested experiment.

    Supported experiment_name values:
        "dino"                    - pure DINO, no aux losses
        "dino_simmim"             - DINO + SimMIM (no diversity)
        "dino_simmim_diversity"   - DINO + SimMIM + cls_entropy diversity
        "dino_diversity"          - DINO + cls_entropy diversity
        "dino_diversity_ortho"    - DINO + ortho_entropy diversity  (NEW)
    """
    cfg = copy.deepcopy(base_cfg)

    if experiment_name == "dino":
        cfg.ssl_mode = "dino"
        cfg.simmim.weight = 0.0
        cfg.selfpatch.weight = 0.0
        cfg.diversity.weight = 0.0
        cfg.diversity.mode = "cls_entropy"
        _apply_div_defaults(cfg)

    elif experiment_name == "dino_simmim":
        cfg.ssl_mode = "dino_simmim"
        cfg.simmim.weight = 0.3
        cfg.selfpatch.weight = 0.0
        cfg.diversity.weight = 0.0
        cfg.diversity.mode = "cls_entropy"
        _apply_div_defaults(cfg)

    elif experiment_name == "dino_simmim_diversity":
        cfg.ssl_mode = "dino_simmim"
        cfg.simmim.weight = 0.3
        cfg.selfpatch.weight = 0.0
        cfg.diversity.weight = 1e-2
        cfg.diversity.mode = "cls_entropy"
        _apply_div_defaults(cfg)

    elif experiment_name == "dino_diversity":
        cfg.ssl_mode = "dino"
        cfg.simmim.weight = 0.0
        cfg.selfpatch.weight = 0.0
        cfg.diversity.weight = 1e-2
        cfg.diversity.mode = "cls_entropy"
        _apply_div_defaults(cfg)

    elif experiment_name == "dino_diversity_ortho":
        cfg.ssl_mode = "dino"
        cfg.simmim.weight = 0.0
        cfg.selfpatch.weight = 0.0
        cfg.diversity.weight = 1e-2
        cfg.diversity.mode = "ortho_entropy"   # <- NEW mode
        cfg.diversity.ortho_alpha = 0.5        # alpha * L_ortho + (1-alpha) * L_ent
        _apply_div_defaults(cfg)

    else:
        raise ValueError(
            f"Unknown experiment_name: {experiment_name!r}. "
            "Choose from: dino | dino_simmim | dino_simmim_diversity "
            "| dino_diversity | dino_diversity_ortho"
        )

    return cfg


def _apply_div_defaults(cfg: Stage1TrainerConfig) -> None:
    """Write shared diversity defaults onto cfg.diversity in-place."""
    for k, v in _DIV_DEFAULTS.items():
        setattr(cfg.diversity, k, v)
