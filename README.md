# hcc-vs-hemangioma

Lightweight ultrasound representation learning and downstream classification framework for HCC vs hemangioma, with a stage-1 self-supervised encoder and a stage-2 supervised fine-tuning benchmark.

## Ultimate goal

The project is not only to obtain a visually appealing attention map. The real target is to build a **lightweight encoder** that can be reused across downstream classification and dense-token tasks, while preserving interpretable attention behavior and enabling stage-2 HCC-vs-hemangioma decision support.

Concretely, the project aims to achieve all of the following:

- Stable stage-1 self-supervised learning without late collapse.
- Headwise attention specialization instead of shortcut monoculture.
- High-quality `encoded_patches` that remain useful for dense downstream tasks.
- Strong and reproducible stage-2 supervised performance under both pretrained and random-init settings.
- Clear experiment control from Kaggle notebooks, including model path, experiment mode, LR policy, WandB logging, confusion matrix, and attention-map upload.

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

Stage 2: supervised downstream learning

selected stage-1 encoder checkpoint
                |
                +------------------------------------------+
                |                                          |
                v                                          v
      finetune mode                                 scratch mode
(load MODEL_PATH weights)                    (random initialize encoder)
                |                                          |
                v                                          v
      supervised model                            supervised model
                |                                          |
                |                                          |
                +------------------+-----------------------+
                                   |
                                   v
                         train / val / test evaluation
                                   |
                                   +--> CE / SupCon / total loss logging
                                   +--> accuracy logging
                                   +--> test confusion matrix
                                   +--> test attention-map table
```

## Stage-2 experiment modes

Stage 2 is now controlled directly from `Stage2_SSK_run.ipynb`.

### Researcher-controlled inputs

- `MODEL_PATH`: path to the selected stage-1 encoder checkpoint.
- `EXPERIMENT_MODE`: one of `scratch`, `finetune`, or `both`.
- `LR_MODE`: one of `constant` or `cosine`.
- `LR`: base learning rate value.
- `WARMUP_EPOCHS`: warmup length when cosine schedule is used.

### Mode behavior

| Mode | Encoder initialization | Loss | Purpose |
|---|---|---|---|
| `scratch` | random initialization | CE only | Supervised baseline without stage-1 transfer |
| `finetune` | load `MODEL_PATH` weights | mean(CE, SupCon) | Evaluate benefit of stage-1 representation transfer |
| `both` | run `finetune` then `scratch` | each mode uses its own loss | Side-by-side comparison |

### WandB outputs

For each executed mode, the notebook logs:

- train metrics,
- validation metrics,
- test confusion matrix,
- test attention-map table based on the stage-1 attention callback path.

The final test artifact keys are intentionally separated by mode:

- `vit_파인튜닝_confusion_matrix`
- `vit_파인튜닝_att_map`
- `vit_from_scratch_confusion_matrix`
- `vit_from_scratch_att_map`

## Code status

The current codebase already supports the following pieces:

- all-layer attention extraction from the encoder,
- multi-layer CLS-row diversity primitives in `training/losses.py`,
- stage-1 DINO and DINO+SimMIM trainers wired to the diversity path,
- configurable stage-2 supervised trainer with CE-only vs mean(CE, SupCon) behavior,
- notebook-level experiment selection for stage 1 and stage 2,
- WandB logging for metrics, confusion matrix, and attention-map tables.

## Main files

- `models/encoder.py`: ViT/CNN encoder definitions and attention return path.
- `training/losses.py`: DINO, SimMIM, diversity, and supervised contrastive losses.
- `training/stage1_dino.py`: stage-1 DINO trainer.
- `training/stage1_dino_simmim.py`: stage-1 DINO + SimMIM trainer.
- `training/stage1_config.py`: structured experiment config.
- `training/stage1_ssl.py`: trainer builder and config-to-model wiring.
- `training/stage2_supcon.py`: stage-2 supervised trainer.
- `visualization/wandb_viz.py`: attention-map and WandB visualization helpers.
- `Stage1_SSK_run.ipynb`: practical stage-1 experiment notebook.
- `Stage2_SSK_run.ipynb`: practical stage-2 supervised experiment notebook.

## Recommended run order

1. Run stage 1 baseline and select a checkpoint.
2. Use `Stage2_SSK_run.ipynb` with `EXPERIMENT_MODE="finetune"` for transfer evaluation.
3. Run `EXPERIMENT_MODE="scratch"` for supervised baseline.
4. Use `EXPERIMENT_MODE="both"` only when a full paired comparison is needed in one notebook session.

## Notes

- `selfpatch.py` is preserved for legacy comparison and future ablation.
- Stage-2 attention logging intentionally reuses the proven stage-1 attention callback path for better stability.
- The current README reflects the latest plan centered on lightweight encoder quality, dense usability, and controlled stage-2 benchmarking.
