# 📋 2일차 · 3일차 실험 보고서

## BraTS-GLI 뇌종양 2D 슬라이스 이진 분류

> **프로젝트**: BraTS2023-GLI 뇌 MRI 종양 유무 분류  
> **모델**: ResNet-18 (ImageNet Pretrained)  
> **데이터**: T2-FLAIR 2D axial 슬라이스 (224×224)

---

# 📅 2일차: 모델 학습 및 평가

## 1. 실행 코드

| 파일 | 역할 |
|------|------|
| `step4_model.py` | ResNet-18 모델 정의 (FC: 512→1) |
| `step5_train.py` | 2-Phase 학습 루프 |
| `step6_evaluate.py` | 테스트셋 평가 + 시각화 |

## 2. 모델 아키텍처

```
ResNet-18 (Pretrained ImageNet)
├── conv1 → bn1 → relu → maxpool
├── layer1 (BasicBlock ×2, 64ch)
├── layer2 (BasicBlock ×2, 128ch)
├── layer3 (BasicBlock ×2, 256ch)
├── layer4 (BasicBlock ×2, 512ch)  ← Grad-CAM 타겟
├── AdaptiveAvgPool2d (1×1)
├── Dropout(0.5)
└── Linear(512 → 1)  ← BCEWithLogitsLoss
```

- 전체 파라미터: **11,177,025개**
- 손실 함수: `BCEWithLogitsLoss` (pos_weight 적용)

## 3. 학습 설정

| 항목 | Phase 1 (FC만) | Phase 2 (전체 Fine-tune) |
|------|:--------------:|:------------------------:|
| Epochs | 3 | 12 (Early Stop patience=5) |
| Learning Rate | 1e-3 | 1e-4 |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Batch Size | 32 | 32 |
| Weight Decay | 1e-4 | 1e-4 |
| Backbone | **동결** | **학습 가능** |

**Data Augmentation (학습 시)**:
- RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)
- RandomRotation(15°), RandomAffine(translate=0.1)
- ImageNet Normalize

## 4. 학습 결과

### 학습 곡선

| Epoch | Phase | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|:-----:|:-----:|:----------:|:---------:|:--------:|:-------:|:---:|
| 1 | P1 | 0.4520 | 79.46% | 0.4113 | 81.28% | 7.5e-4 |
| 2 | P1 | 0.4421 | 80.00% | 0.4064 | 81.46% | 2.5e-4 |
| 3 | P1 | 0.4360 | 80.39% | 0.4082 | 81.85% | 0 |
| 4 | P2 | 0.2067 | 92.06% | 0.2076 | 92.59% | 9.8e-5 |
| 5 | P2 | 0.1681 | 93.75% | 0.1889 | 93.33% | 9.3e-5 |
| 6 | P2 | 0.1543 | 94.28% | 0.1820 | 93.66% | 8.5e-5 |
| 7 | P2 | 0.1442 | 94.68% | 0.1746 | 93.84% | 7.5e-5 |
| 8 | P2 | 0.1327 | 95.16% | 0.1818 | 93.90% | 6.3e-5 |
| 9 | P2 | 0.1244 | 95.48% | **0.1713** | **94.15%** | 5.0e-5 |
| 10 | P2 | 0.1137 | 95.90% | 0.1928 | 93.59% | 3.7e-5 |
| 11 | P2 | 0.1042 | 96.27% | 0.1888 | 93.85% | 2.5e-5 |
| 12 | P2 | 0.0953 | 96.63% | 0.1873 | 94.16% | 1.5e-5 |
| 13 | P2 | 0.0868 | 96.96% | 0.2023 | 93.94% | 6.7e-6 |
| 14 | P2 | 0.0809 | 97.18% | 0.2209 | 93.71% | 1.7e-6 |

> Best model은 **Epoch 9** (Val Loss: 0.1713, Val Acc: 94.15%)에서 저장됨.
> Epoch 10 이후 Val Loss 증가 → **과적합 경향** 확인.

## 5. 테스트셋 평가 결과

### 핵심 성능 지표

| 지표 | 값 |
|------|:---:|
| **Accuracy** | **94.33%** |
| **Precision** | 95.49% |
| **Recall (Sensitivity)** | 92.75% |
| **Specificity** | 95.83% |
| **F1-Score** | 94.10% |
| **AUC-ROC** | **0.9832** |
| **AP (Average Precision)** | 0.9862 |
| **최적 Threshold (Youden's J)** | 0.6512 |

### Confusion Matrix

|  | Pred Negative | Pred Positive |
|:---:|:---:|:---:|
| **Actual Neg** | TN = **12,330** | FP = **536** |
| **Actual Pos** | FN = **888** | TP = **11,361** |

총 테스트 샘플: **25,115개**

### Classification Report

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|:---------:|:------:|:--------:|:-------:|
| Negative (종양 없음) | 0.933 | 0.958 | 0.945 | 12,866 |
| Positive (종양 존재) | 0.955 | 0.928 | 0.941 | 12,249 |
| **Macro Avg** | **0.944** | **0.943** | **0.943** | 25,115 |

## 6. 2일차 시각화 산출물

| 파일 | 내용 |
|------|------|
| `06_training_curves.png` | Loss / Accuracy / LR 학습 곡선 |
| `07_confusion_matrix.png` | 절대값 + 정규화 Confusion Matrix |
| `08_roc_curve.png` | ROC Curve (AUC=0.9832) |
| `09_pr_curve.png` | PR Curve (AP=0.9862) |
| `10_probability_distribution.png` | 예측 확률 분포 |

## 7. 2일차 소결

- **수치적 성능은 우수** (Acc 94%, AUC 0.98)
- 그러나 FN 888개(종양 놓침) + FP 536개(오탐) 존재
- 의료 AI에서 FN은 치명적 → Grad-CAM으로 원인 분석 필요

---

# 📅 3일차: Grad-CAM 해석성 분석

## 1. 실행 코드

| 파일 | 역할 |
|------|------|
| `step7_gradcam.py` | Grad-CAM 히트맵 생성 + TP/FN/FP 분석 |

### 분석 프로세스

```
테스트셋 전체 25,115개 → 예측 확률 수집
    ├── TP/FN/FP/TN 분류
    ├── [질문1] TP 8개 시각화 + 200개 IoU 통계
    ├── [질문2] FN 8개 시각화 + 300개 종양 크기 분석
    └── [질문3] FP 8개 시각화 + 확률 분포 비교
```

## 2. Grad-CAM 수치 결과 요약

| 지표 | 값 | 해석 |
|------|:---:|------|
| **TP 평균 IoU** | **0.145** | 히트맵과 종양 마스크 거의 안 겹침 |
| **FN 작은종양 비율** | **92.3%** | 놓친 종양의 92%가 매우 작음 (픽셀 <1%) |
| **FN 평균 예측 확률** | 0.191 | "없다"고 확신하며 놓침 |
| **FP 평균 예측 확률** | 0.741 | "있다"고 확신하며 오탐 |

## 3. [질문1] TP: Seg 마스크와 히트맵이 일치하는가?

### 결론: ❌ 일치하지 않음 (평균 IoU = 0.145)

**관찰 결과:**
- Grad-CAM 히트맵이 **항상 뇌 중심부에 큰 원형 blob**으로 나타남
- 종양이 좌측/우측에 위치해도 히트맵은 **가운데에 고정**
- 고확률 TP (prob=1.000)에서도 IoU는 0.23~0.25 수준
- 저확률 TP (prob≈0.50)에서는 IoU가 0.007~0.052로 극히 낮음

**IoU 분포:**
- 대부분 0.0~0.1 구간에 밀집
- IoU > 0.3인 경우는 극소수 (약 6%)

> 모델은 종양의 정확한 위치를 학습한 것이 아니라, "뇌의 전체적인 형태/밝기 분포"를 기반으로 판단하고 있음.

## 4. [질문2] FN 888개: 왜 놓쳤는가?

### 결론: 종양이 너무 작아서 feature map에서 소실

**핵심 발견:**
- FN의 **92.3%가 종양 픽셀 비율 1% 미만** (224×224 중 ~500픽셀 이하)
- 평균 종양 픽셀 비율: **0.36%** (약 180픽셀)
- 종양 크기 분포가 극도로 왼쪽 편향 (대부분 0~0.005 구간)

**히트맵 패턴:**
- 경계선 FN (prob≈0.497): 뇌 중심에 약한 반응, 종양 무시
- 극저확률 FN (prob≈0.002~0.003): 히트맵 완전 무반응 (IoU=0.000)
- 확률 분포: 0.0~0.05 구간에 가장 많이 밀집

> 작은 종양은 ResNet의 Global Average Pooling을 거치면서 feature가 희석되어 검출 불가능.
> 이는 2D whole-slice 분류의 **구조적 한계**임.

## 5. [질문3] FP 536개: 무엇을 착각했는가?

### 결론: 정상 뇌 구조를 종양으로 오인

**핵심 발견:**
- Seg 마스크가 완전히 비어있는데 히트맵이 강한 반응
- 고확신 FP (prob=1.000): 뇌 중심부에 강한 빨간 blob
- 히트맵 패턴이 TP와 동일 → 종양/비종양 구분 없이 같은 영역 주시
- FP 평균 확률 0.741 → 확신 있게 오탐

**확률 분포:**
- 0.8~1.0 고확률 구간에도 다수 분포
- threshold 조정으로 해결 불가능한 FP가 상당수 존재

> 모델이 "종양 특이적 feature"가 아닌 "뇌 존재 여부"를 학습했다는 결정적 증거.

## 6. 3일차 시각화 산출물

| 파일 | 내용 |
|------|------|
| `gradcam/gradcam_tp.png` | TP 8개: MRI / Seg / CAM / 오버레이 |
| `gradcam/gradcam_fn.png` | FN 8개: 경계선 + 극저확률 샘플 |
| `gradcam/gradcam_fp.png` | FP 8개: 고확신 + 경계선 오탐 |
| `gradcam/gradcam_iou_distribution.png` | TP 200개 IoU 분포 |
| `gradcam/gradcam_fn_tumor_size.png` | FN 종양 크기 분포 |
| `gradcam/gradcam_fn_fp_prob.png` | FN/FP 확률 비교 |
| `gradcam/gradcam_analysis.json` | 수치 결과 JSON |

## 7. 3일차 소결: 2D Whole-Slice 접근법의 한계

### 근본 원인

```
2D Whole-Slice 분류의 한계
├── 224×224 전체 이미지에서 종양은 극소 영역
├── GAP(Global Average Pooling)이 작은 종양 feature 희석
├── 모델이 "종양 위치"가 아닌 "뇌 형태" 학습
└── TP/FP의 히트맵 패턴이 동일 → 판별력 없음
```

| 항목 | 진단 |
|------|------|
| 히트맵 위치 | 종양이 아닌 **뇌 중심부에 고정** |
| 히트맵 패턴 | TP/FP 모두 동일한 blob |
| FN 원인 | 종양이 너무 작아 feature map에서 소실 |
| FP 원인 | 정상 뇌 구조를 종양으로 오인 |
| **해결 방향** | **Patch-based 접근법으로 전환 필요** |

---

# 📁 전체 파일 구조

```
biohealth_lv.1/
├── code/
│   ├── step1_preprocess.py      # 1일차: 3D → 2D 추출
│   ├── step2_eda.py             # 1일차: EDA
│   ├── step3_dataset.py         # 1일차: 데이터 분할
│   ├── step4_model.py           # 2일차: 모델 정의
│   ├── step5_train.py           # 2일차: 학습
│   ├── step6_evaluate.py        # 2일차: 평가
│   └── step7_gradcam.py         # 3일차: Grad-CAM
├── processed/
│   ├── slices/                   # 166,626개 PNG
│   ├── labels.csv
│   └── splits.csv
└── outputs/
    ├── checkpoints/              # best_model.pth, last_model.pth
    ├── figures/                  # EDA + 평가 + Grad-CAM 시각화
    └── logs/                     # JSON 로그들
```

## 환경 정보

| 항목 | 내용 |
|------|------|
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 4070 Laptop (8GB) |
| CUDA | 13.1 |
