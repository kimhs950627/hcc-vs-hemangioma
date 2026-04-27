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
   - [Directory to Provide](#directory-to-provide)
   - [Label Convention](#label-convention)
   - [1. Supervised Dataloader (`build_dataset`)](#1-supervised-dataloader-build_dataset)
   - [2. Multi-View — Global Only (`local_views=0`)](#2-multi-view--global-only-local_views0)
   - [3. Multi-View — Global + Local Crops](#3-multi-view--global--local-crops)
   - [Augmentation Pipelines](#augmentation-pipelines)
5. [Smoke-Test](#smoke-test)

---

## Repository Structure

```
hcc-vs-hemangioma/
├── dataloader.py       # SMC-LUD dataloader (this document)
└── README.md
```

---

## Dataset Structure

SMC-LUD `clean_ver_for_train` 디렉토리 기준. Train / Val / Test split이 **이미 완료**된 상태로 배포되며, 각 클래스 폴더 안에 이미지가 **flat하게** 존재한다 (환자별 sub-directory 없음).

```
clean_ver_for_train/          ← data_root 로 지정할 디렉토리
├── train_clean/
│   ├── HCC/
│   │   ├── img_0001.png
│   │   ├── img_0002.png
│   │   └── ...
│   └── Hemangioma/
│       ├── img_0001.png
│       └── ...
├── val_clean/
│   ├── HCC/
│   └── Hemangioma/
└── test_clean/
    ├── HCC/
    └── Hemangioma/
```

> `original_data/` 및 `split_for_train/` 폴더는 dataloader에서 사용하지 않는다.

---

## Installation

```bash
pip install keras tensorflow numpy
```

- **Keras 3** (backend-agnostic) 사용. TensorFlow는 `tf.data` I/O pipeline에만 의존.
- Python 3.11+, single consumer GPU (≤16 GB VRAM) 기준.

---

## Dataloader

### Directory to Provide

모든 함수에서 `data_root` 인자에 **`clean_ver_for_train/` 경로**를 넘긴다.

```python
DATA_ROOT = "/path/to/clean_ver_for_train"
```

내부적으로 다음 세 경로를 자동으로 탐색한다.

| split 인자 | 실제 탐색 경로 |
|-----------|--------------|
| `"train"` | `data_root/train_clean/{HCC,Hemangioma}/` |
| `"val"`   | `data_root/val_clean/{HCC,Hemangioma}/`   |
| `"test"`  | `data_root/test_clean/{HCC,Hemangioma}/`  |

---

### Label Convention

| 클래스 | 레이블 |
|--------|--------|
| Hemangioma (혈관종) | `0` |
| HCC (간세포암종)    | `1` |

---

### 1. Supervised Dataloader (`build_dataset`)

학습·검증·테스트용 `tf.data.Dataset` 세 개를 한번에 반환한다.  
Train split에만 data augmentation이 적용되고, Val/Test는 rescale만 수행한다.

```python
from dataloader import build_dataset

ds_train, ds_val, ds_test = build_dataset(
    data_root   = "/path/to/clean_ver_for_train",
    img_size    = (224, 224),   # 모든 이미지를 이 크기로 resize
    batch_size  = 32,
    use_augmentation = True,    # Train에만 base augmentation 적용
    seed        = 42,
)

# 사용 예시
for imgs, labels in ds_train:
    # imgs  : tf.float32  shape [32, 224, 224, 3],  range [0, 1]
    # labels: tf.int32    shape [32],  {0=Hemangioma, 1=HCC}
    print(imgs.shape, labels.numpy())
    break

# 검증 루프
for imgs, labels in ds_val:
    predictions = model(imgs, training=False)

# 최종 평가
for imgs, labels in ds_test:
    ...
```

---

### 2. Multi-View — Global Only (`local_views=0`)

`local_views=0` 이면 **global view 2개만** 반환한다.  
두 뷰 모두 `img_size` 그대로 (원본 해상도 유지), strong augmentation 적용.

```python
from dataloader import MultiViewDataset

multiview_ds = MultiViewDataset(
    data_root   = "/path/to/clean_ver_for_train",
    split       = "train",       # "train" / "val" / "test"
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 0,             # ← 0: global view 2개만 반환
    shuffle     = True,
    seed        = 42,
)

for g1, g2 in multiview_ds:
    # g1, g2 : tf.float32  shape [16, 224, 224, 3]  (동일 해상도)
    # teacher=g1, student=g2 or vice versa
    loss = ssl_loss(teacher(g1), student(g2))
    break
```

---

### 3. Multi-View — Global + Local Crops

`local_views=N` (N ≥ 1) 이면 **global 2개 + local N개** 를 list로 반환한다.  
Local view는 RandomCrop → resize → `img_size // 2` 해상도.

```python
from dataloader import MultiViewDataset

multiview_ds = MultiViewDataset(
    data_root        = "/path/to/clean_ver_for_train",
    split            = "train",
    img_size         = (224, 224),
    batch_size       = 16,
    local_views      = 6,                    # ← global 2 + local 6 = 총 8개 뷰
    local_crop_scale = (0.05, 0.40),         # 이미지 면적의 5~40% crop
    local_output_size= (96, 96),             # 원하면 직접 지정 (기본: img_size // 2)
    shuffle          = True,
    seed             = 42,
)

for views in multiview_ds:
    # views: Python list, len = 2 + local_views = 8
    g1, g2         = views[0], views[1]   # float32 [16, 224, 224, 3]
    local_crops    = views[2:]             # list of 6 × float32 [16, 96, 96, 3]

    # 예: self-supervised multi-view 손실
    global_feats   = [teacher(g1), teacher(g2)]
    student_feats  = [student(v) for v in views]
    break
```

---

### Augmentation Pipelines

세 가지 Keras Sequential augmentation layer가 내장돼 있다.  
모두 **Keras 3 `layers.*` API만** 사용하며 외부 라이브러리(cv2, scipy 등)에 의존하지 않는다.

| 함수 | 용도 | 강도 |
|------|------|------|
| `build_base_augmentation(img_size)` | Supervised train | 중간 |
| `build_strong_augmentation(img_size)` | global view | 강 |
| `build_local_crop_augmentation(parent_size, crop_scale, output_size)` | local crop | 강 + crop |

**적용된 augmentation 목록 (B-mode US 도메인 근거)**

| Transform | 파라미터 | 근거 |
|-----------|----------|------|
| `RandomFlip` (H+V) | — | 초음파 방향성 제한 없음 |
| `RandomRotation` | ±15° | 탐촉자 각도 변이 시뮬레이션 |
| `RandomZoom` | ±15~20% | 병변 크기 variability |
| `RandomTranslation` | 5~8% | 병변 위치 variability |
| `RandomBrightness` | 0.15~0.25 | US gain / depth attenuation 변동 |
| `RandomContrast` | 0.20~0.35 | TGC 변동 |
| `GaussianNoise` | std 0.025~0.04 | Speckle noise 시뮬레이션 |
| `Rescaling` | ÷255 | uint8 → float32 [0,1] |

> Blur는 Keras 3 stable API에 없으므로 GaussianNoise로 대체함.

---

## Smoke-Test

데이터셋 경로를 인자로 넘겨 전체 파이프라인을 검증한다.

```bash
python dataloader.py /path/to/clean_ver_for_train
```

정상 실행 시 아래와 같은 출력이 나온다.

```
[build_dataset] Collecting samples ...
  [train] Hemangioma= 2134  HCC= 2172  total= 4306
  [val  ] Hemangioma=  267  HCC=  272  total=  539
  [test ] Hemangioma=  268  HCC=  272  total=  540

Smoke-test 1: build_dataset (supervised)
  imgs  : (4, 224, 224, 3)  dtype=float32
  labels: [0 1 1 0]  (0=Hemangioma, 1=HCC)
  pixel range: [0.000, 1.000]

Smoke-test 2: MultiViewDataset (local_views=0)
  global_view_1 : (4, 224, 224, 3)  dtype=float32
  global_view_2 : (4, 224, 224, 3)  dtype=float32
  PASS: global-only mode

Smoke-test 3: MultiViewDataset (local_views=6)
  Total views returned : 8  (expected 8)
    [global view 0] shape=(4, 224, 224, 3)  dtype=float32
    [global view 1] shape=(4, 224, 224, 3)  dtype=float32
    [local  view 2] shape=(4, 112, 112, 3)  dtype=float32
    ...
  PASS: multi-crop mode
```


## Stage 1 SSL Training

Stage 1 supports selectable SSL modes in `training/stage1_ssl.py`: `moco`, `byol`, and `dino`.
For Stage 2 initialization, use the **teacher encoder weights by default** because the EMA teacher is a temporal ensemble and typically more stable.

```python
from dataloader import MultiViewDataset
from training.stage1_ssl import build_stage1_trainer

ssl_ds = MultiViewDataset(
    data_root="./clean_ver_for_train",
    split="train",
    img_size=(224, 224),
    batch_size=16,
    local_views=6,
).as_dataset()

ssl_model = build_stage1_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    temperature=0.1,
    ema_momentum=0.996,
    lr=1e-4,
    ssl_mode="moco",  # "moco" | "byol" | "dino"
    predictor_dim=256,  # used by BYOL
    teacher_temp=0.04,  # used by DINO
)

ssl_model.fit(ssl_ds, epochs=10)

# recommended for Stage 2
teacher_encoder = ssl_model.get_stage2_encoder(use_teacher=True)
teacher_encoder.save_weights("output/vit_stage1_teacher_encoder.weights.h5")
```

## Stage 2 Supervised + SupCon Training

Stage 2 can initialize the encoder directly from the **Stage 1 teacher weights**.

```python
from dataloader import build_dataset
from training.stage2_supcon import build_stage2_trainer, stage2_encoder_recommendation

print(stage2_encoder_recommendation())

ds_train, ds_val, ds_test = build_dataset(
    data_root="./clean_ver_for_train",
    img_size=(224, 224),
    batch_size=16,
    use_augmentation=True,
    shuffle_train=True,
    shuffle_val=True,
    shuffle_test=True,
)

stage2_model = build_stage2_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    num_classes=2,
    supcon_weight=0.3,
    lr=1e-4,
    teacher_encoder_weights="output/vit_stage1_teacher_encoder.weights.h5",
)

stage2_model.fit(ds_train, validation_data=ds_val, epochs=20)
```

## Prototype Bank Construction

The prototype bank is now built from the **train split only** and supports a **dual bank**:
- `hemangioma_prototypes.npy`
- `hcc_prototypes.npy`

```bash
python inference/build_prototype_bank.py   --data_root ./clean_ver_for_train   --encoder vit   --weights output/stage2_classifier.weights.h5   --n_prototypes 8   --output_dir output/prototype_bank
```

## Inference and Scoring

Dual-bank inference returns both **hemangioma score** and **HCC score**, as well as global/patch-level scores and heatmaps.

```python
import numpy as np
from models.encoder import build_classifier
from visualization.interpret import infer_with_prototypes

model = build_classifier("vit", input_shape=(224, 224, 3))
model.load_weights("output/stage2_classifier.weights.h5")

hema_bank = np.load("output/prototype_bank/hemangioma_prototypes.npy")
hcc_bank = np.load("output/prototype_bank/hcc_prototypes.npy")

result = infer_with_prototypes(
    classifier_model=model,
    image=image_np,
    hemangioma_prototypes=hema_bank,
    hcc_prototypes=hcc_bank,
)

print("P(Hemangioma):", result["p_hemangioma"])
print("P(HCC):", result["p_hcc"])
print("Hemangioma score:", result["hemangioma_score"])
print("HCC score:", result["hcc_score"])
print("Global HCC score:", result["hcc_global_score"])
print("Patch HCC score:", result["hcc_patch_score"])
print("Malignancy score:", result["malignancy_score"])

overlay = result["overlay_image"]
proto_overlay = result["prototype_overlay_image"]
```

## Notes on Outputs

- Transformer encoders return: `cls_token`, `encoded_patches`, `last_encoder_layer_attentional_weights`
- CNN encoders return: `gap_vector`, `feature_map`
- `overlay_image`: Grad-CAM for CNNs, attention overlay for Transformer-family encoders
- `prototype_overlay_image`: patch-level prototype similarity heatmap


## Prototype Bank Construction

The prototype bank is built from the **train split only** and supports a **dual bank**:
- `hemangioma_prototypes.npy`
- `hcc_prototypes.npy`

### CLI usage

```bash
python inference/build_prototype_bank.py \
  --data_root ./clean_ver_for_train \
  --encoder vit \
  --weights output/stage2_classifier.weights.h5 \
  --n_prototypes 8 \
  --output_dir output/prototype_bank
```

### Python / notebook usage

You can also import the script in a `.ipynb` notebook and build/save the dual prototype bank directly.

```python
from inference.build_prototype_bank import build_dual_prototype_bank

result = build_dual_prototype_bank(
    data_root="./clean_ver_for_train",
    encoder="vit",
    weights="output/stage2_classifier.weights.h5",
    n_prototypes=8,
    output_dir="output/prototype_bank",
    img_size=224,
    batch_size=16,
)

hema_bank = result["hemangioma_prototypes"]
hcc_bank = result["hcc_prototypes"]

print(result["hemangioma_path"])
print(result["hcc_path"])
print(hema_bank.shape, hcc_bank.shape)
```

This returns in-memory numpy arrays and also saves both prototype files to disk.



## Full Pipeline Example

The example below shows the intended end-to-end workflow:
1. Stage 1 SSL pretraining with EMA teacher-student learning
2. Export teacher encoder weights
3. Stage 2 supervised + SupCon training initialized from the teacher
4. Build dual prototype banks from the **train split**
5. Run prototype-based inference and obtain malignancy-related scores and heatmaps

```python
import numpy as np
from dataloader import MultiViewDataset, build_dataset
from training.stage1_ssl import build_stage1_trainer
from training.stage2_supcon import build_stage2_trainer
from inference.build_prototype_bank import build_dual_prototype_bank
from visualization.interpret import infer_with_prototypes
from models.encoder import build_classifier

# -----------------------------
# 1) Stage 1 SSL pretraining
# -----------------------------
ssl_ds = MultiViewDataset(
    data_root="./clean_ver_for_train",
    split="train",
    img_size=(224, 224),
    batch_size=16,
    local_views=6,
).as_dataset()

ssl_model = build_stage1_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    temperature=0.1,
    ema_momentum=0.996,
    lr=1e-4,
)

ssl_model.fit(ssl_ds, epochs=10)

# Export the EMA teacher encoder for Stage 2 initialization
teacher_encoder = ssl_model.get_stage2_encoder(use_teacher=True)
teacher_encoder.save_weights("output/vit_stage1_teacher_encoder.weights.h5")

# -----------------------------
# 2) Stage 2 supervised training
# -----------------------------
ds_train, ds_val, ds_test = build_dataset(
    data_root="./clean_ver_for_train",
    img_size=(224, 224),
    batch_size=16,
    use_augmentation=True,
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False,
)

stage2_model = build_stage2_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    num_classes=2,
    supcon_weight=0.3,
    lr=1e-4,
    teacher_encoder_weights="output/vit_stage1_teacher_encoder.weights.h5",
)

stage2_model.fit(ds_train, validation_data=ds_val, epochs=20)
stage2_model.model.save_weights("output/vit_stage2_classifier.weights.h5")

# -----------------------------
# 3) Build dual prototype banks
# -----------------------------
proto_result = build_dual_prototype_bank(
    data_root="./clean_ver_for_train",
    encoder="vit",
    weights="output/vit_stage2_classifier.weights.h5",
    n_prototypes=8,
    output_dir="output/prototype_bank",
    img_size=224,
    batch_size=16,
)

hema_bank = proto_result["hemangioma_prototypes"]
hcc_bank = proto_result["hcc_prototypes"]

print(proto_result["hemangioma_path"])
print(proto_result["hcc_path"])

# -----------------------------
# 4) Prototype-based inference
# -----------------------------
classifier_model = build_classifier("vit", input_shape=(224, 224, 3))
classifier_model.load_weights("output/vit_stage2_classifier.weights.h5")

# Replace this with a real preprocessed image array of shape [H, W, 3]
image_np = np.zeros((224, 224, 3), dtype=np.float32)

result = infer_with_prototypes(
    classifier_model=classifier_model,
    image=image_np,
    hemangioma_prototypes=hema_bank,
    hcc_prototypes=hcc_bank,
)

print("P(Hemangioma):", result["p_hemangioma"])
print("P(HCC):", result["p_hcc"])
print("Hemangioma score:", result["hemangioma_score"])
print("HCC score:", result["hcc_score"])
print("HCC global score:", result["hcc_global_score"])
print("HCC patch score:", result["hcc_patch_score"])
print("Malignancy score:", result["malignancy_score"])

# Heatmaps
overlay_image = result["overlay_image"]  # Grad-CAM / attention-based explanation
hcc_proto_overlay = result["prototype_overlay_image"]  # HCC prototype-only heatmap
margin_proto_overlay = result["prototype_margin_overlay_image"]  # (HCC - Hemangioma) dual-bank margin heatmap
```

### Expected artifacts

- `output/vit_stage1_teacher_encoder.weights.h5`
- `output/vit_stage2_classifier.weights.h5`
- `output/prototype_bank/hemangioma_prototypes.npy`
- `output/prototype_bank/hcc_prototypes.npy`

### Recommended order

- Use the **EMA teacher encoder** from Stage 1 to initialize Stage 2.
- Build prototype banks **after Stage 2**, not before.
- Build prototype banks from the **train split only**.
- Use the HCC bank (and the dual-bank margin) for malignancy-oriented prototype heatmaps.


- `output/vit_stage1_teacher_encoder.weights.h5`
- `output/vit_stage2_classifier.weights.h5`
- `output/prototype_bank/hemangioma_prototypes.npy`
- `output/prototype_bank/hcc_prototypes.npy`

### Recommended order

- Use the **EMA teacher encoder** from Stage 1 to initialize Stage 2.
- Build prototype banks **after Stage 2**, not before.
- Build prototype banks from the **train split only**.
- Use the HCC bank for malignancy-oriented prototype heatmaps.


### Stage 1 SSL modes

```python
from training.stage1_ssl import build_stage1_trainer

# MoCo-like EMA contrastive SSL
moco_model = build_stage1_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    temperature=0.1,
    ema_momentum=0.996,
    lr=1e-4,
    ssl_mode="moco",
)

# BYOL
byol_model = build_stage1_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    predictor_dim=256,
    ema_momentum=0.996,
    lr=1e-4,
    ssl_mode="byol",
)

# DINO
# local_views>0 in MultiViewDataset is recommended for DINO.
dino_model = build_stage1_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    temperature=0.1,
    teacher_temp=0.04,
    ema_momentum=0.996,
    lr=1e-4,
    ssl_mode="dino",
)
```

Implementation files:
- `training/stage1_moco.py`
- `training/stage1_byol.py`
- `training/stage1_dino.py`
- `training/stage1_ssl.py` (router)
