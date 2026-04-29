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
6. [Stage 1 — Unified Router (`ssl_mode`)](#stage-1--unified-router-ssl_mode)
7. [Stage 1 — Other SSL Modes](#stage-1--other-ssl-modes)
8. [Stage 2 Supervised + SupCon](#stage-2-supervised--supcon-training)
9. [**Supervised Benchmark Baseline**](#supervised-benchmark-baseline)
10. [Prototype Bank Construction](#prototype-bank-construction)
11. [Inference and Scoring](#inference-and-scoring)
12. [W&B Visualization](#wb-visualization)
13. [Smoke-Test](#smoke-test)

---

## Repository Structure

```
hcc-vs-hemangioma/
├── dataloader.py
├── models/
│   └── encoder.py           # VisionTransformer, Swin-like, ConvNeXt, EfficientNet
├── training/
│   ├── stage1_ssl.py         # Unified SSL router (moco/byol/dino/dino_simmim)
│   ├── stage1_dino.py        # DINOPretrainModel + DINOSimMIMModel
│   ├── stage1_moco.py
│   ├── stage1_byol.py
│   ├── stage2_supcon.py
│   └── benchmark_supervised.py
├── callbacks/
│   └── epoch_visualization.py
├── utils/
│   ├── metrics.py
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

`DINOSimMIMModel` 은 **DINO self-distillation** 과 **SimMIM masked image reconstruction** 을 단일 `train_step` 안에서 **동시에(jointly)** 최적화한다.

### 동작 원리

```
          ┌────────────────────────────────────────────────────┐
          │                  train_step                        │
          │                                                    │
  view[0] original_clean ──► Teacher Encoder ──► DINO center  │
  view[1] aug_global2    ──┐                                   │
  view[4+] local_i      ──┴► Student Encoder ──► DINO loss    │
                                                               │
  view[2] masked_clean  ──► Student Encoder ──► Pixel Head    │
  view[3] patch_mask    ──────────────────────► SimMIM loss    │
          │                                                    │
          │   loss = alpha * L_dino + (1-alpha) * λ * L_simmim│
          └────────────────────────────────────────────────────┘
```

- **DINO branch**: `original_clean`을 teacher global1로, `aug_global2`를 teacher/student global2로 사용한다. local crops(`views[4:]`)는 student 전용이며, **global view와 동일한 `online_encoder` 호출 경로**를 공유한다.
- **SimMIM branch**: `masked_clean`을 student encoder에 통과시켜 patch token을 추출하고, `pixel_pred_head`로 원본 pixel patch(`original_clean`)를 복원한다. `patch_mask`가 지정한 위치만 L1 loss에 반영된다.
- **Alpha warm-up**: 학습 초기에는 `alpha=1.0` (DINO 전용), `warmup_steps` 동안 **step(batch) 기준**으로 `alpha_final`까지 선형 감소 → 이후 DINO + SimMIM 균형 최적화. `TeacherTempWarmupCallback`과 동일하게 `on_train_batch_end` 기반.

### Input View 순서 (MaskedMultiViewDataset)

| 인덱스 | 텐서 | 역할 |
|--------|------|------|
| `views[0]` | `original_clean` | Teacher global1 / SimMIM reconstruction target |
| `views[1]` | `aug_global2` | Teacher global2 / Student global2 (DINO) |
| `views[2]` | `masked_clean` | SimMIM student input (patch들이 masking된 버전) |
| `views[3]` | `patch_mask` | `[B, N_patches]`, 1=masked 0=visible |
| `views[4:]` | `local_i` × n | DINO student local crops |

> **주의**: `patch_mask.shape[1]`은 encoder가 생성하는 patch token 수와 반드시 일치해야 한다.  
> `patch_size=16, input_shape=(224,224,3)` → `N = (224/16)^2 = 196` 토큰.  
> 불일치 시 `_simmim_forward()` 내 `tf.debugging.assert_equal`이 런타임 에러를 발생시킨다.

### Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `encoder_name` | str | — | `"vit"` / `"swin"` / `"convnext"` / `"efficientnet"` |
| `input_shape` | tuple | `(224,224,3)` | 이미지 입력 크기 |
| `projection_dim` | int | `256` | DINO projection head output dim |
| `patch_size` | int | `16` | SimMIM patch 크기 (encoder와 일치해야 함) |
| `n_local` | int | `4` | local crop 수 (`MaskedMultiViewDataset.local_views`와 일치) |
| `lambda_mim` | float | `1.0` | SimMIM reconstruction loss weight |
| `alpha_final` | float | `0.7` | alpha warm-up 완료 후 DINO 가중치 |
| `alpha_warmup_epochs` | int | `20` | model 내부 목표 warm-up epoch 수 (router 전달용) |
| `temperature` | float | `0.1` | student softmax temperature |
| `teacher_temp` | float | `0.04` | teacher softmax temperature (초기값) |
| `center_momentum` | float | `0.9` | teacher center EMA momentum |
| `ema_momentum` | float | `0.996` | teacher weight EMA momentum |
| `lr` | float | `1e-4` | AdamW learning rate |
| `clipnorm` | float\|None | `1.0` | gradient clip norm |
| `weight_decay` | float | `1e-4` | AdamW weight decay |

---

## Stage 1 — Unified Router (`ssl_mode`)

`build_stage1_trainer()` 는 `ssl_mode` 인자 하나로 모든 SSL 방식을 라우팅하는 **통합 진입점**이다.

```python
from training.stage1_ssl import build_stage1_trainer
```

### `ssl_mode='dino_simmim'` — DINO + SimMIM 동시 학습

> **`ssl_mode='dino_simmim'`로 지정하면 DINO와 SimMIM이 단일 `train_step` 안에서 동시에(jointly) 최적화된다.**

```python
from dataloader import MaskedMultiViewDataset          # masked view 제공
from training.stage1_ssl import build_stage1_trainer
from training.stage1_dino import AlphaWarmupCallback
from training.ssl_callbacks import TeacherTempWarmupCallback

# 1. Masked multi-view dataset
#    views: [original_clean, aug_global2, masked_clean, patch_mask, local_0, ..., local_(n-1)]
mv_ds = MaskedMultiViewDataset(
    data_root   = "./clean_ver_for_train",
    split       = "train",
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 4,        # must equal n_local below
    mask_ratio  = 0.75,
).as_dataset()

# 2. Build trainer via unified router
ssl_model = build_stage1_trainer(
    encoder_name         = "vit",
    input_shape          = (224, 224, 3),
    projection_dim       = 256,
    patch_size           = 16,          # SimMIM patch size
    n_local              = 4,           # DINO local crops
    lambda_mim           = 1.0,         # SimMIM loss weight
    alpha_final          = 0.7,         # DINO weight after warm-up
    alpha_warmup_epochs  = 20,          # model target (router 전달용)
    temperature          = 0.1,
    teacher_temp         = 0.04,
    center_momentum      = 0.9,
    ema_momentum         = 0.996,
    lr                   = 1e-4,
    clipnorm             = 1.0,
    weight_decay         = 1e-4,
    ssl_mode             = 'dino_simmim',   # ← 핵심 인자
)

# 3. Callbacks  ── 두 callback 모두 step(batch) 기준으로 동작
steps_per_epoch          = 116   # len(mv_ds)  (dataset 크기 / batch_size)
alpha_warmup_epochs      = 20
teacher_temp_warmup_epochs = 10

callbacks = [
    AlphaWarmupCallback(
        warmup_steps = alpha_warmup_epochs * steps_per_epoch,   # 2320 steps
        verbose      = True,
    ),
    TeacherTempWarmupCallback(
        start_value  = 0.04,
        end_value    = 0.07,
        warmup_steps = teacher_temp_warmup_epochs * steps_per_epoch,  # 1160 steps
        verbose      = True,
    ),
]

# 4. Train
ssl_model.fit(mv_ds, epochs=100, callbacks=callbacks)

# 5. Export teacher encoder for Stage 2
teacher_enc = ssl_model.get_stage2_encoder(use_teacher=True)
teacher_enc.save_weights("output/vit_stage1_dino_simmim.weights.h5")
```

### Loss 모니터링

```
loss          — total: alpha*L_dino + (1-alpha)*lambda_mim*L_simmim
l_dino        — DINO cross-entropy distillation loss
l_simmim      — SimMIM masked-patch L1 reconstruction loss
alpha         — current alpha blend ratio (step-wise: 1.0 → alpha_final)
teacher_temp  — current teacher softmax temperature
```

### SSL mode 비교

| `ssl_mode` | 필요 Dataset | Loss | 추천 encoder |
|---|---|---|---|
| `'moco'` | `MultiViewDataset` (2 global) | InfoNCE | vit / convnext |
| `'byol'` | `MultiViewDataset` (2 global) | MSE bootstrapping | vit / convnext |
| `'dino'` | `MultiViewDataset` (2 global + N local) | DINO cross-entropy | vit |
| `'dino_simmim'` | `MaskedMultiViewDataset` (2 global + masked + N local) | DINO + SimMIM L1 | **vit 전용** |

> CNN(`convnext`, `efficientnet`)은 `last_hidden_state` key를 반환하지 않으므로
> `dino_simmim`의 SimMIM branch에서 `feature_map` reshape 경로를 탄다.
> ViT 사용을 강력히 권장한다.

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

# DINO only
dino_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, temperature=0.1,
    teacher_temp=0.04, center_momentum=0.9, ema_momentum=0.996,
    n_local=4, lr=1e-4, ssl_mode="dino",
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
    teacher_encoder_weights   = "output/vit_stage1_dino_simmim.weights.h5",
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
    encoder_name       = "vit",
    input_shape        = (224, 224, 3),
    batch_size         = 32,
    epochs             = 100,
    lr                 = 1e-4,
    use_augmentation   = False,
    use_pretrained     = False,
    use_dropout        = False,
    use_weight_decay   = False,
    output_dir         = "outputs/benchmark_vit",
    seed               = 42,
    viz_every_n_epochs = 5,
    max_viz_samples    = 8,
)

print(result)
```

### CLI

```bash
python training/benchmark_supervised.py \
    --data_root ./clean_ver_for_train \
    --encoder   vit \
    --epochs    100 \
    --batch_size 32 \
    --output_dir outputs/benchmark_vit
```

### Output Artifacts

```
outputs/benchmark_vit/
├── checkpoints/best.weights.h5
├── metrics/metrics_all.csv
├── roc/roc_epoch_*.png
├── reports/
│   ├── classification_report.json
│   └── classification_report.txt
├── attention/
└── test_report.json
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

---

## Inference and Scoring

```python
import numpy as np
from models.encoder import build_classifier

model = build_classifier("vit", input_shape=(224, 224, 3))
model.load_weights("output/stage2_classifier.weights.h5")
```

---

## W&B Visualization

```python
import wandb
from visualization.wandb_viz import init_wandb, get_wandb_callbacks

init_wandb(project="hcc-vs-hemangioma", run_name="stage1-dino-simmim")
```

---

## Smoke-Test

```bash
python dataloader.py /path/to/clean_ver_for_train
```

정상 실행 시 Train/Val/Test split 확인 + shape/dtype/range 검증 출력.

---

## Notes on Outputs

- **Transformer encoders** return: `embedding`, `last_hidden_state`, `last_encoder_layer_attentional_weights`
- **CNN encoders** return: `embedding`, `feature_map`
- `dino_simmim` SimMIM branch: `last_hidden_state[:, 1:, :]` (CLS 제거 후 patch tokens) 또는 `feature_map` reshape

---

### Augmentation strategy

The training augmentation pipeline includes ultrasound-oriented photometric perturbation:

- `RandomContrast`, `RandomBrightness`, custom `RandomGamma`, `GaussianNoise`

This reduces shortcut learning toward only bright echogenic regions and improves robustness to hypoechoic lesion interiors and gain/TGC variability.
