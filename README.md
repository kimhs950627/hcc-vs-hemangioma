# hcc-vs-hemangioma

Lightweight ultrasound representation learning and downstream classification framework for HCC vs hemangioma, with a stage-1 self-supervised encoder and a stage-2 prototype-oriented classifier.

## Ultimate goal

The current project is not only to obtain a good-looking mean attention map. The actual goal is to build a **lightweight encoder** that can be reused across multiple downstream tasks, including dense tasks that depend on `encoded_patches`, while also enabling stage-2 **HCC prototype induction** and final HCC-vs-hemangioma classification.

Concretely, the project aims to achieve all of the following:

- Stable stage-1 self-supervised learning without late collapse.
- Headwise attention specialization, so different heads attend to different hepatic regions rather than converging to the same hyperechoic shortcut.
- High-quality `encoded_patches` that remain useful for dense tasks and local prototype reasoning.
- Strong stage-2 classification performance together with clinically interpretable HCC prototypes.

## Current stage-1 problem setting

Recent experiments showed two recurring failure modes.

1. **Shortcut monoculture**: plain DINO can avoid full representation collapse, but multiple attention heads often converge to the same hyperechoic region.
2. **Unstable diversification**: when auxiliary diversity pressure is applied too naively, the model may temporarily diversify and then exhibit late-stage attention collapse, with diversity loss rapidly dropping toward zero.

Because of this, the stage-1 objective has been reframed around two requirements:

- preserve DINO-style global invariance,
- enforce stable headwise specialization without low-entropy collapse.

## Stage-1 experiment plan

The current experiment plan is intentionally simplified to three runs.

| Experiment | `ssl_mode` | `simmim.weight` | `selfpatch.weight` | `diversity.weight` | Diversity type | Purpose |
|---|---:|---:|---:|---:|---|---|
| `dino` | `dino` | 0.0 | 0.0 | 0.0 | none | Global SSL baseline |
| `dino_simmim` | `dino_simmim` | 0.3 | 0.0 | 0.0 | none | Improve patch-token quality and dense usability |
| `dino_simmim_diversity` | `dino_simmim` | 0.3 | 0.0 | 1e-3 | top-half multi-layer CLS-row diversity + entropy floor | Add headwise specialization while controlling collapse |

`SelfPatch` is intentionally kept in the codebase as a legacy/ablation option, but it is not part of the current primary plan.

## Diversity design

The old diversity path used a last-layer full attention-map disagreement loss. The current design replaces it with:

- **CLS-row diversity**: headwise diversity is measured on CLS-to-patch attention rather than the full token-token attention matrix.
- **Multi-layer aggregation**: diversity is computed from multiple transformer layers rather than only the last layer.
- **Entropy floor**: a minimum entropy penalty is added so that attention heads do not collapse into pathological low-entropy states.

The default plan uses the **top half** of transformer layers for diversity aggregation.

## Data flow

```text
Stage 1: representation learning

raw ultrasound image
        |
        +------------------------------+
        |                              |
        v                              v
   DINO global/local views       masked image + patch mask
        |                              |
        |                              v
        |                         SimMIM branch
        |                              |
        +-------------> online encoder <-------------+
                               |                      |
                               |                      |
                               v                      v
                         CLS embedding         encoded_patches
                               |                      |
                               |                      +--> patch-quality objective
                               |
                               +--> projector --> DINO loss
                               |
                               +--> attention weights (multi-layer)
                                         |
                                         +--> CLS-row diversity
                                         +--> entropy floor

Stage 2: downstream learning

stage-1 encoder
      |
      +--> classifier head ----------------------> HCC vs hemangioma prediction
      |
      +--> prototype module / prototype mining --> HCC prototypes
      |
      +--> dense/local token usage --------------> encoded_patch-based downstream tasks
```

## Code status

The current codebase already supports the following pieces:

- all-layer attention extraction from the encoder,
- multi-layer CLS-row diversity primitives in `training/losses.py`,
- stage-1 DINO and DINO+SimMIM trainers wired to the new diversity path,
- config-level control of diversity settings,
- notebook-level experiment selection for the three-run plan.

## Main files

- `models/encoder.py`: ViT/CNN encoder definitions and attention return path.
- `training/losses.py`: DINO, SimMIM, and diversity-related losses.
- `training/stage1_dino.py`: stage-1 DINO trainer.
- `training/stage1_dino_simmim.py`: stage-1 DINO + SimMIM trainer.
- `training/stage1_config.py`: structured experiment config.
- `training/stage1_ssl.py`: trainer builder and config-to-model wiring.
- `Stage1_SSK_run.ipynb`: practical stage-1 experiment notebook.

## Recommended first run order

1. Run `dino` to establish the baseline.
2. Run `dino_simmim` to evaluate whether `encoded_patches` improve without diversity pressure.
3. Run `dino_simmim_diversity` to test whether headwise specialization improves without triggering late collapse.

## Notes

- `selfpatch.py` is preserved for legacy comparison and future ablation.
- The current README reflects the latest plan centered on lightweight encoder quality, dense usability, and stage-2 HCC prototype induction.
