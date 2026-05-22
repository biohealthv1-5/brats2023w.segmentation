# 🧠 BraTS-GLI 뇌 MRI 종양 분류 + Explainable AI 프로젝트

> **과제명**: 2026-1 바이오헬스세미나02 — BraTS2023-GLI 기반 뇌 MRI 종양 슬라이스 이진 분류 + 해석성(XAI) 분석
> **데이터**: BraTS2023-GLI Challenge Training Data (1,251명 환자, T2-FLAIR 중심)
> **환경**: Python 3.13 / PyTorch 2.11 + CUDA 12.8 / NVIDIA RTX 4070 Laptop GPU (8GB)
> **진행 기간**: 2026-05-09 ~ 2026-05-21 (Day 1 ~ Day 6, 진행 중)

---

## 1. 프로젝트 개요

3D 뇌 MRI(NIfTI) 볼륨을 2D axial 슬라이스로 변환한 뒤, 각 슬라이스에 **종양이 존재하는지 자동 판별**하는 CNN 모델을 만들고, 그 모델이 정말로 종양을 보고 판단하는지 **Grad-CAM / Segmentation 출력**으로 검증한다.

당초 강의 일정은 3일차(전처리 → 학습 → XAI)로 끝나지만, 본 프로젝트는 3일차에서 발견한 **Shortcut Learning 문제**를 해결하기 위해 자율적으로 6일차까지 확장하고 있다.

### 1.1 6단계 실험 흐름

```
Day 1            Day 2            Day 3              Day 4              Day 5              Day 6
┌──────┐        ┌──────┐         ┌──────────┐       ┌──────────┐       ┌──────────────┐  ┌────────────────────┐
│전처리 │  ───▶ │Whole │  ───▶  │ Grad-CAM │ ───▶ │  Patch   │ ───▶ │ Multi-Task   │▶│ Multi-Modal MTL   │
│+EDA  │        │Slice │         │  XAI     │       │  Based   │       │ (Cls + Seg)  │  │ (T1ce + FLAIR)    │
└──────┘        └──────┘         └──────────┘       └──────────┘       └──────────────┘  └────────────────────┘
 1,251명         Acc 94.33%      IoU 0.145          F1 87.08%          F1 93.91%          F1 95~97% (목표)
 166K            F1  94.10%      FN 92% 소종양       IoU 0.128         IoU 0.706 ★       IoU 0.78+   (목표)
 슬라이스         AUC 0.9832     ❗ Shortcut 발견    (성능 ↓)          ✅ 5배 개선         🚧 진행 중
```

### 1.2 최종 결과 한 줄 요약

> **"Whole-Slice는 수치를 얻었지만 진실을 잃었고, Patch는 진실을 추구했지만 수치를 잃었으며, Multi-Task는 둘 다 잡았다."**

---

## 2. 데이터셋

| 항목 | 값 |
|------|------|
| 데이터셋 | BraTS-GLI 2023 (ASNR-MICCAI Challenge) |
| 환자 수 | 1,251명 |
| 모달리티 | T2-FLAIR (단일) — 부종 경계 가장 잘 보임 |
| 슬라이스 크기 | 224 × 224 (PNG, axial 단면) |
| 총 슬라이스 | 166,626개 |
| Positive (종양 존재) | 81,374개 (48.8%) |
| Negative (정상) | 85,252개 (51.2%) |
| 분할 (환자 단위) | Train 875 / Val 188 / Test 188 |
| Test 슬라이스 | 25,115개 |

> 환자 단위 분할로 data leakage 방지. 클래스 비율이 48.8% vs 51.2%로 거의 균형이라 별도의 oversampling 불필요.

---

## 3. 핵심 결과 (Test set N=25,115)

### 3.1 3-Way 비교

| 지표 | Whole-Slice (Day 2) | Patch-Based (Day 4) | **Multi-Task (Day 5)** |
|------|:-------------------:|:-------------------:|:----------------------:|
| Accuracy | 94.33% | 87.07% | 94.26% |
| Precision | 95.49% | 84.93% | **97.33%** ★ |
| Recall | **92.75%** | 89.34% | 90.73% |
| Specificity | 95.83% | 84.90% | **97.63%** ★ |
| F1-Score | **94.10%** | 87.08% | 93.91% |
| AUC-ROC | **0.9832** | 0.9104 | 0.9824 |
| **FP (오탐)** | 536 | 1,942 | **305** ★ (-43%) |
| FN (놓침) | 888 | 1,306 | 1,136 |
| **TP 평균 IoU (해석성)** | 0.145 | 0.128 | **0.7061** ★★ (4.87배) |
| 파라미터 | 11.2M | ~95K | ~14M |
| 학습 시간 | ~120분 | 128.8분 | 298.8분 |

### 3.2 핵심 발견

1. **Day 3 (Grad-CAM)**: Whole-Slice 모델의 TP/FP 히트맵이 동일하게 뇌 중심에 고정 → 모델은 종양이 아니라 **뇌 형태**를 학습하는 **Shortcut Learning** 진단.
2. **Day 4 (Patch)**: 광역 단서를 차단하는 passive 접근으로는 부족하다는 것을 확인 — F1도 IoU도 모두 하락.
3. **Day 5 (Multi-Task)**: 분류와 세분화를 동시에 학습시키는 active한 학습 신호 설계 → 분류 성능 유지하면서 IoU 5배 향상 + FP 43% 감소.
4. **Day 6 (Multi-Modal MTL)**: T1ce + FLAIR로 의학적 상보성 확보 → 마지막 약점인 소종양 FN 공략 중.

---

## 4. 디렉토리 구조

```
biohealth_lv.1/
├── README.md                          ← (본 문서)
├── 2026-1 바이오헬스세미나02.pdf       강의 자료
├── KakaoTalk_*.png                    팀 발표용 인포그래픽 (1일차/2일차/3일차)
├── brats2023_dataset_guide.md         데이터셋 가이드
├── project_spec_ideaB.md              프로젝트 상세 명세
├── day1_report.md                     1일차 결과 (전처리/EDA/분할)
├── day2_day3_report.md                2~3일차 결과 (Whole-Slice 학습 + Grad-CAM)
├── patch_pipeline_plan.md             4일차 패치 전환 계획서
├── organize_data.py                   원본 zip → BraTS-GLI/training 정리 스크립트
│
├── code/
│   ├── whole/                         Day 1-3: Whole-Slice 파이프라인
│   │   ├── step1_preprocess.py        3D NIfTI → 2D PNG 추출
│   │   ├── step2_eda.py               EDA 시각화
│   │   ├── step3_dataset.py           환자 단위 분할 + Dataset
│   │   ├── step4_model.py             ResNet-18 정의
│   │   ├── step5_train.py             2-Phase 학습
│   │   ├── step6_evaluate.py          테스트 평가
│   │   └── step7_gradcam.py           Grad-CAM 해석성 분석
│   ├── patch/                         Day 4: Patch-Based 파이프라인
│   │   ├── step8_patch_extract.py     패치 좌표 추출
│   │   ├── step9_patch_dataset.py     패치 Dataset
│   │   ├── step10_patch_model.py      PatchCNN (~95K params)
│   │   ├── step11_patch_train.py      패치 학습
│   │   ├── step12_patch_evaluate.py   패치 + 슬라이스 평가
│   │   └── step13_patch_gradcam.py    패치 Grad-CAM
│   ├── step14_comparison.py           Whole vs Patch 비교 자동 생성
│   ├── multitask/                     Day 5: Multi-Task (Cls + Seg)
│   │   ├── step15_prep_segmask.py
│   │   ├── step16_multitask_dataset.py
│   │   ├── step17_multitask_model.py  ResNet-18 enc + U-Net dec
│   │   ├── step18_multitask_train.py  α·BCE(cls) + β·(Dice+BCE)(seg)
│   │   └── step19_multitask_evaluate.py
│   ├── mmmt/                          Day 6: Multi-Modal Multi-Task
│   └── sota/                          확장 실험
│
├── data/
│   └── BraTS-GLI/training/            원본 NIfTI 1,251명
│
├── processed/                         (csv는 루트, 잘린 이미지는 하위 폴더)
│   ├── labels.csv                     슬라이스별 종양 유무
│   ├── splits.csv                     환자 단위 train/val/test
│   ├── patch_labels.csv               3,218,425개 패치 좌표
│   ├── slices/                        166,626개 FLAIR PNG (224×224)
│   ├── seg_masks/                     세분화 GT 마스크 (WT)
│   ├── mmmt/
│   │   └── t1ce_slices/               Day 6 T1ce 슬라이스
│   └── sota/
│       ├── seg_masks_wt/
│       ├── seg_masks_tc/
│       └── seg_masks_et/
│
└── outputs/                           (결과물은 figures/logs/checkpoints로 분리)
    ├── checkpoints/
    │   ├── whole/      best_model.pth, last_model.pth (~134MB)
    │   ├── patch/      patch_best_model.pth (~0.4MB)
    │   ├── multitask/  mt_best_model.pth (~170MB)
    │   ├── mmmt/
    │   └── sota/
    ├── figures/
    │   ├── whole/      EDA 5종 + 학습/평가 시각화 5종 + gradcam/
    │   ├── patch/      11_patch_training_curves.png + patch_eval/ + patch_gradcam/
    │   ├── multitask/  MT 학습/CM/ROC/Seg 샘플
    │   ├── mmmt/
    │   └── sota/
    └── logs/
        ├── step14_comparison.json     Whole vs Patch 자동 비교 결과
        ├── whole/      training_history.json, evaluation_results.json, ...
        ├── patch/      step8_stats.json, step11_train_log.json, step12_patch_eval.json
        ├── multitask/  mt_training_history.json, mt_evaluation_results.json, ...
        ├── mmmt/
        └── sota/
```

---

## 5. 모델별 아키텍처 한눈에 보기

### Whole-Slice (Day 2)
```
ResNet-18 (ImageNet Pretrained, 11.18M params)
└── AdaptiveAvgPool → Dropout(0.5) → Linear(512 → 1)   BCEWithLogitsLoss
```

### Patch-Based (Day 4)
```
PatchCNN (~95K params, from scratch)
Conv(3→32)→BN→ReLU→MaxPool   [64→32]
Conv(32→64)→BN→ReLU→MaxPool  [32→16]
Conv(64→128)→BN→ReLU→GAP     [16→1]
Dropout(0.5)→Linear(128→1)
패치 64×64 (stride=32) → 슬라이스 집계 (count ≥2 방식)
```

### Multi-Task (Day 5)
```
입력 슬라이스 (224×224)
        │
   Shared Encoder (ResNet-18, ImageNet Pretrained)
        │
   ┌───┴───┐
   │       │
 분류 Head    Segmentation Decoder (경량 U-Net, skip ×4)
 GAP→FC(512→1)
   │       │
P(tumor)  Tumor Mask (224×224)

Loss = α·BCE(cls) + β·(Dice + BCE)(seg)   α=1.0, β=0.5
```

---

## 6. 실행 방법

### 6.1 환경 설정

```bash
# Python 3.13.12 / PyTorch 2.11.0 + CUDA 12.8 권장
pip install torch torchvision nibabel pandas numpy scikit-learn matplotlib opencv-python pillow tqdm
```

### 6.2 데이터 준비

1. BraTS-GLI 2023 Training zip을 `data/BraTS-GLI/training/`에 압축 해제
2. `python organize_data.py` (필요 시)

### 6.3 단계별 실행

```bash
# Day 1: 전처리
python code/whole/step1_preprocess.py
python code/whole/step2_eda.py
python code/whole/step3_dataset.py

# Day 2: Whole-Slice 학습 + 평가
python code/whole/step4_model.py
python code/whole/step5_train.py
python code/whole/step6_evaluate.py

# Day 3: Grad-CAM 해석성 분석
python code/whole/step7_gradcam.py

# Day 4: Patch-Based 파이프라인
python code/patch/step8_patch_extract.py
python code/patch/step9_patch_dataset.py
python code/patch/step10_patch_model.py
python code/patch/step11_patch_train.py
python code/patch/step12_patch_evaluate.py
python code/patch/step13_patch_gradcam.py
python code/step14_comparison.py

# Day 5: Multi-Task Learning
python code/multitask/step15_prep_segmask.py
python code/multitask/step16_multitask_dataset.py
python code/multitask/step17_multitask_model.py
python code/multitask/step18_multitask_train.py
python code/multitask/step19_multitask_evaluate.py

# Day 6: Multi-Modal Multi-Task
python code/mmmt/step20_mmmt_preprocess.py
python code/mmmt/step24_mmmt_train.py
python code/mmmt/step25_mmmt_evaluate.py
```

---

## 7. 학술적 인사이트 (요약)

1. **단일 지표(Accuracy/AUC)만으로는 의료 AI 모델의 신뢰성을 보장할 수 없다.**
   94.33%라는 정확도가 shortcut learning을 가린 직접적 사례를 정량적으로 입증.
2. **Shortcut Learning은 명시적 보조 학습 신호(Auxiliary Task)로 극복 가능하다.**
   Multi-Task가 IoU를 0.145 → 0.706(4.87배) 끌어올린 직접 증거.
3. **광역 단서 차단(passive)보다 올바른 학습 강제(active)가 효과적이다.**
   Patch가 실패한 자리에서 Multi-Task가 성공한 사실이 이를 뒷받침.

---

## 8. 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 |
| Python | 3.13.12 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM) |
| 주요 라이브러리 | torchvision, nibabel, scikit-learn, matplotlib, opencv-python, pillow, pandas, tqdm |
