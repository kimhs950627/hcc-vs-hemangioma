# HCC vs. Hemangioma Classification

B-mode 간 초음파 이미지에서 **간세포암종(HCC)** 과 **혈관종(Hemangioma)** 을 분류하는 딥러닝 프로젝트.

> **Dataset**: [SMC-LUD (Samsung Medical Center – Liver Ultrasound Dataset)](https://doi.org/10.6084/m9.figshare.31112716)  
> *Nature Scientific Data, 2026-03-10 · 5,385 B-mode images · 1,021 patients*

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Dataset Structure](#dataset-structure)
3. [Installation](#installation)
4. [Dataloader](#dataloader)
5. [Stage 1 — DINO + SimMIM SSL Trainer](#stage-1--dino--simmim-ssl-trainer)
6. [Stage 1 — Other SSL Modes](#stage-1--other-ssl-modes)
7. [Stage 2 Supervised + SupCon](#stage-2-supervised--supcon-training)
8. [**Supervised Benchmark Baseline**](#supervised-benchmark-baseline)
9. [Prototype Bank Construction](#prototype-bank-construction)
10. [Inference and Scoring](#inference-and-scoring)
11. [W&B Visualization](#wb-visualization)
12. [Smoke-Test](#smoke-test)

---

## Repository Structure

```
hcc-vs-hemangioma/
├── dataloader.py
├── models/
│   └── encoder.py           # VisionTransformer, Swin-like, ConvNeXt, EfficientNet
├── training/
│   ├── stage1_ssl.py         # SSL router (moco / byol / dino)
│   ├── stage1_dino.py        # DINO + SimMIM combined trainer
│   ├── stage1_moco.py
│   ├── stage1_byol.py
│   ├── stage2_supcon.py
│   └── benchmark_supervised.py   # ← Scratch supervised baseline
├── callbacks/
│   └── epoch_visualization.py    # ← Per-epoch metrics + Grad-CAM + attention
├── utils/
│   ├── metrics.py                # Binary classification metrics
│   └── gradcam.py
├── inference/
│   └── build_prototype_bank.py
├── visualization/
│   └── wandb_viz.py
└── README.md
```

---

## Dataset Structure

```
clean_ver_for_train/          ← data_root
├── train_clean/
│   ├── HCC/
│   └── Hemangioma/
├── val_clean/
│   ├── HCC/
│   └── Hemangioma/
└── test_clean/
    ├── HCC/
    └── Hemangioma/
```

| 클래스 | 레이블 |
|--------|--------|
| Hemangioma (혈관종) | `0` |
| HCC (간세포암종)    | `1` |

---

## Installation

```bash
pip install keras tensorflow numpy matplotlib
```

- **Keras 3** (backend-agnostic), Python 3.11+, single consumer GPU (≤16 GB VRAM)

---

## Dataloader

### Directory to Provide

```python
DATA_ROOT = "/path/to/clean_ver_for_train"
```

### 1. Supervised Dataloader (`build_dataset`)

```python
from dataloader import build_dataset

ds_train, ds_val, ds_test = build_dataset(
    data_root        = "/path/to/clean_ver_for_train",
    img_size         = (224, 224),
    batch_size       = 32,
    use_augmentation = True,
    seed             = 42,
)
```

### 2. Multi-View SSL Dataset

```python
from dataloader import MultiViewDataset

# Global-only (BYOL / MoCo)
mv_ds = MultiViewDataset(
    data_root   = "/path/to/clean_ver_for_train",
    split       = "train",
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 0,        # 0 = 2 global views only
)

# Global + Local (DINO)
mv_ds = MultiViewDataset(
    data_root        = "/path/to/clean_ver_for_train",
    split            = "train",
    img_size         = (224, 224),
    batch_size       = 16,
    local_views      = 6,   # global 2 + local 6
    local_crop_scale = (0.05, 0.40),
)
```

### Augmentation Pipelines

| 함수 | 용도 | 강도 |
|------|------|------|
| `build_base_augmentation(img_size)` | Supervised train | 중간 |
| `build_strong_augmentation(img_size)` | SSL global view | 강 |
| `build_local_crop_augmentation(...)` | SSL local crop | 강+crop |

---

## Stage 1 — DINO + SimMIM SSL Trainer

`build_stage1_dino_simmim_trainer()` 는 **DINO self-distillation + SimMIM masked reconstruction** 을 동시에 학습하는 Stage 1 trainer를 반환한다.  
Teacher network는 EMA 기반 weight averaging이고, student는 두 global view + N local crop을 모두 본다.

### Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `encoder_name` | str | `"vit"` | `"vit"` / `"swin"` / `"convnext"` / `"efficientnet"` |
| `input_shape` | tuple | `(224,224,3)` | 이미지 입력 크기 |
| `projection_dim` | int | `256` | DINO projection head output dim |
| `n_local` | int | `4` | local crop 수 (MultiViewDataset의 `local_views`와 일치) |
| `mask_ratio` | float | `0.75` | SimMIM masked patch 비율 |
| `lambda_mim` | float | `0.1` | SimMIM reconstruction loss weight |
| `temperature` | float | `0.1` | student softmax temperature |
| `teacher_temp` | float | `0.04` | teacher softmax temperature |
| `teacher_temp_warmup_start` | float | `0.04` | teacher temp warmup 시작값 |
| `teacher_temp_target` | float | `0.07` | teacher temp warmup 목표값 |
| `warmup_epochs` | int | `10` | teacher temp warm-up epochs |
| `center_momentum` | float | `0.9` | teacher centering EMA momentum |
| `ema_momentum` | float | `0.996` | teacher weight EMA momentum |
| `alpha_final` | float | `0.996` | EMA decay 최종값 (cosine schedule) |
| `lr` | float | `1e-4` | Adam learning rate |

### Quickstart

```python
from dataloader import MultiViewDataset
from training.stage1_dino import build_stage1_dino_simmim_trainer
from training.ssl_callbacks import AlphaWarmupCallback, TeacherTempWarmupCallback

# 1. Multi-view dataset: global 2 + local n_local
mv_ds = MultiViewDataset(
    data_root   = "./clean_ver_for_train",
    split       = "train",
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 4,            # must match n_local below
).as_dataset()

# 2. Build trainer
ssl_model = build_stage1_dino_simmim_trainer(
    encoder_name                = "vit",
    input_shape                 = (224, 224, 3),
    projection_dim              = 256,
    n_local                     = 4,
    mask_ratio                  = 0.75,
    lambda_mim                  = 0.1,
    temperature                 = 0.1,
    teacher_temp                = 0.04,
    teacher_temp_warmup_start   = 0.04,
    teacher_temp_target         = 0.07,
    warmup_epochs               = 10,
    center_momentum             = 0.9,
    ema_momentum                = 0.996,
    alpha_final                 = 0.996,
    lr                          = 1e-4,
)

# 3. Callbacks
callbacks = [
    AlphaWarmupCallback(
        model         = ssl_model,
        alpha_start   = 0.996,
        alpha_final   = 0.9999,
        total_epochs  = 100,
    ),
    TeacherTempWarmupCallback(
        model         = ssl_model,
        temp_start    = 0.04,
        temp_final    = 0.07,
        warmup_epochs = 10,
    ),
]

# 4. Train
ssl_model.fit(mv_ds, epochs=100, callbacks=callbacks)

# 5. Export teacher encoder for Stage 2
teacher_enc = ssl_model.get_stage2_encoder(use_teacher=True)
teacher_enc.save_weights("output/vit_stage1_teacher.weights.h5")
```

### Outputs during SSL training

```
loss          — total = DINO loss + lambda_mim * MIM loss
loss_dino     — cross-entropy distillation loss
loss_mim      — pixel/patch reconstruction MSE
```

---

## Stage 1 — Other SSL Modes

```python
from training.stage1_ssl import build_stage1_trainer

# MoCo
moco_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, temperature=0.1, ema_momentum=0.996,
    lr=1e-4, ssl_mode="moco",
)

# BYOL
byol_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, predictor_dim=256, ema_momentum=0.996,
    lr=1e-4, ssl_mode="byol",
)

# DINO (via router — uses stage1_dino.py internally)
dino_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, temperature=0.1,
    teacher_temp=0.04, teacher_temp_warmup_start=0.04,
    teacher_temp_target=0.07, warmup_epochs=10,
    center_momentum=0.9, ema_momentum=0.996,
    lr=1e-4, ssl_mode="dino",
)
```

---

## Stage 2 Supervised + SupCon Training

```python
from dataloader import build_dataset
from training.stage2_supcon import build_stage2_trainer

ds_train, ds_val, ds_test = build_dataset(
    data_root="./clean_ver_for_train",
    img_size=(224,224), batch_size=16, use_augmentation=True,
)

stage2 = build_stage2_trainer(
    encoder_name              = "vit",
    input_shape               = (224, 224, 3),
    num_classes               = 2,
    supcon_weight             = 0.3,
    projection_dim            = 128,
    classifier_hidden_dim     = 256,
    dropout_rate              = 0.2,
    lr                        = 1e-4,
    teacher_encoder_weights   = "output/vit_stage1_teacher.weights.h5",
)

stage2.fit(ds_train, validation_data=ds_val, epochs=50)
stage2.model.save_weights("output/vit_stage2.weights.h5")
```

---

## Supervised Benchmark Baseline

**No pretraining. No regularizer. No dropout. Scratch supervised binary classification.**  
Goal: establish a clean lower-bound to compare against SSL-pretrained models.

### Python API

```python
from training.benchmark_supervised import run_supervised_benchmark

result = run_supervised_benchmark(
    data_root          = "./clean_ver_for_train",
    encoder_name       = "vit",       # "vit" | "swin" | "convnext" | "efficientnet"
    input_shape        = (224, 224, 3),
    batch_size         = 32,
    epochs             = 100,
    lr                 = 1e-4,
    # ── benchmark mode (default: all False) ──
    use_augmentation   = False,       # True for augmented ablation
    use_pretrained     = False,       # CNN only — ImageNet weights
    use_dropout        = False,
    use_weight_decay   = False,
    # ── output ──
    output_dir         = "outputs/benchmark_vit",
    seed               = 42,
    viz_every_n_epochs = 5,           # Grad-CAM + attention every 5 epochs
    max_viz_samples    = 8,
)

print(result)  # dict: accuracy, roc_auc, sensitivity, specificity, ppv, npv, ...
```

### CLI

```bash
# Minimal (pure scratch, no augmentation)
python training/benchmark_supervised.py \
    --data_root ./clean_ver_for_train \
    --encoder   vit \
    --epochs    100 \
    --batch_size 32 \
    --output_dir outputs/benchmark_vit

# With base augmentation (ablation)
python training/benchmark_supervised.py \
    --data_root ./clean_ver_for_train \
    --encoder   convnext \
    --epochs    100 \
    --use_augmentation \
    --output_dir outputs/benchmark_convnext_aug

# Full regularizer ablation (NOT the benchmark; for comparison)
python training/benchmark_supervised.py \
    --data_root ./clean_ver_for_train \
    --encoder   vit \
    --use_pretrained \
    --use_dropout \
    --use_weight_decay \
    --output_dir outputs/benchmark_vit_full
```

### Output Artifacts

```
outputs/benchmark_vit/
├── checkpoints/
│   └── best.weights.h5          ← best val_accuracy checkpoint
├── metrics/
│   ├── metrics_epoch_001.json   ← per-epoch metrics JSON
│   ├── metrics_epoch_002.json
│   ├── ...
│   └── metrics_all.csv          ← all epochs in one CSV
├── roc/
│   ├── roc_epoch_001.png        ← ROC curve (per epoch)
│   └── ...
├── gradcam/
│   ├── epoch_001_sample_00.png  ← Grad-CAM overlay (CNN only)
│   └── ...
├── attention/
│   ├── epoch_001_sample_00.png  ← ViT/Swin attention overlay
│   └── ...
└── test_report.json             ← final test-set metrics
```

### Metrics computed

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / N |
| ROC-AUC | Trapezoidal rule |
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| PPV | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| F1 | 2 × PPV × Sensitivity / (PPV + Sensitivity) |
| Youden threshold | argmax(sensitivity + specificity − 1) |

### Encoder compatibility

| Encoder | Grad-CAM | Attention map | Scratch supported |
|---|---|---|---|
| `vit` | — | ✅ CLS-to-patch | ✅ |
| `swin` | — | ✅ GAP attention | ✅ |
| `convnext` | ✅ last Conv2D | — | ✅ |
| `efficientnet` | ✅ last Conv2D | — | ✅ |

> CNN Grad-CAM requires the last Conv2D layer to be automatically detected.  
> If detection fails, Grad-CAM is silently skipped and a warning is printed.

### `build_supervised_benchmark_model` API

```python
from training.benchmark_supervised import build_supervised_benchmark_model

model = build_supervised_benchmark_model(
    encoder_name     = "vit",
    input_shape      = (224, 224, 3),
    use_pretrained   = False,   # CNN only
    use_dropout      = False,
    dropout_rate     = 0.3,
    use_weight_decay = False,
    weight_decay     = 1e-4,
)

# model.call() returns dict with keys:
#   'logit'       : (B, 1)  raw logit
#   'probability' : (B, 1)  sigmoid(logit)
#   'embedding'   : (B, D)  encoder embedding
#   'feature_map' : (B, H, W, C)  CNN feature map (or None for ViT)
#   'last_encoder_layer_attentional_weights' : ViT attention (or None for CNN)
```

### `EpochMetricsAndVisualizationCallback` standalone usage

```python
from callbacks.epoch_visualization import EpochMetricsAndVisualizationCallback

# Grab a fixed sample batch for visualization
sample_imgs, sample_lbls = next(iter(ds_val))

callback = EpochMetricsAndVisualizationCallback(
    val_dataset          = ds_val,
    output_dir           = "outputs/my_run",
    last_conv_layer_name = "block5c_project_conv",  # EfficientNet example
    core_model           = my_core_model,
    sample_images        = sample_imgs.numpy()[:8],
    sample_labels        = sample_lbls.numpy()[:8],
    viz_every_n_epochs   = 5,
    max_viz_samples      = 8,
    threshold            = 0.5,
)

fit_model.fit(
    ds_train,
    validation_data = ds_val,
    epochs          = 100,
    callbacks       = [callback],
)
```

---

## Prototype Bank Construction

```bash
python inference/build_prototype_bank.py \
  --data_root ./clean_ver_for_train \
  --encoder   vit \
  --weights   output/stage2_classifier.weights.h5 \
  --n_prototypes 8 \
  --output_dir   output/prototype_bank
```

```python
from inference.build_prototype_bank import build_dual_prototype_bank

result = build_dual_prototype_bank(
    data_root     = "./clean_ver_for_train",
    encoder       = "vit",
    weights       = "output/stage2_classifier.weights.h5",
    n_prototypes  = 8,
    output_dir    = "output/prototype_bank",
)
hema_bank = result["hemangioma_prototypes"]
hcc_bank  = result["hcc_prototypes"]
```

---

## Inference and Scoring

```python
import numpy as np
from models.encoder import build_classifier
from visualization.interpret import infer_with_prototypes

model = build_classifier("vit", input_shape=(224, 224, 3))
model.load_weights("output/stage2_classifier.weights.h5")

result = infer_with_prototypes(
    classifier_model       = model,
    image                  = image_np,   # (224, 224, 3) float32
    hemangioma_prototypes  = hema_bank,
    hcc_prototypes         = hcc_bank,
)
print(result["p_hcc"], result["malignancy_score"])
```

---

## W&B Visualization

```python
import wandb
from visualization.wandb_viz import init_wandb, get_wandb_callbacks, WandbStage2Visualizer, WandbVisualizationConfig

init_wandb(project="hcc-vs-hemangioma", run_name="stage2-vit-supcon")

vis_cfg = WandbVisualizationConfig(
    test_dir="./clean_ver_for_train/test_clean",
    num_images=8, image_size=(224, 224),
    log_every_n_epochs=1, stage="stage2",
)

clf_model.fit(
    ds_train, validation_data=ds_val, epochs=50,
    callbacks=get_wandb_callbacks() + [WandbStage2Visualizer(vis_cfg)],
)
```

---

## Smoke-Test

```bash
python dataloader.py /path/to/clean_ver_for_train
```

정상 실행 시 Train/Val/Test split 확인 + shape/dtype/range 검증 출력.

---

## Notes on Outputs

- **Transformer encoders** return: `cls_token`, `encoded_patches`, `last_encoder_layer_attentional_weights`
- **CNN encoders** return: `gap_vector`, `feature_map`
- `overlay_image`: Grad-CAM (CNN) or attention overlay (Transformer)
- `prototype_overlay_image`: patch-level prototype similarity heatmap

---

### Updated augmentation strategy

The training augmentation pipeline now includes stronger ultrasound-oriented photometric perturbation with:
- `RandomContrast`, `RandomBrightness`, custom `RandomGamma`

This reduces shortcut learning toward only bright echogenic regions and improves robustness to hypoechoic lesion interiors and gain/TGC variability.
