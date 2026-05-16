# 🧠 뇌 MRI 종양 슬라이스 이진 분류 프로젝트 상세 명세서

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | BraTS-GLI 뇌 MRI 종양 존재 여부 2D 슬라이스 이진 분류 |
| **목표** | 3D MRI를 2D 슬라이스로 변환 후, 각 슬라이스에 종양이 존재하는지 자동 판별하는 CNN 모델 구축 |
| **데이터셋** | BraTS-GLI 2023 Training Data (~1,251명) |
| **분류 유형** | 이진 분류 (Tumor Present=1 / Tumor Absent=0) |
| **프레임워크** | PyTorch |
| **실행 환경** | 로컬 GPU |

---

## 1. 데이터 파이프라인 설계

### 1.1 원본 데이터 구조
```
BraTS-GLI-XXXXX-XXX/
├── *-t1c.nii.gz   # T1 Contrast Enhanced (조영증강)  ← 주 입력 모달리티
├── *-t1n.nii.gz   # T1 Native
├── *-t2f.nii.gz   # T2 FLAIR
├── *-t2w.nii.gz   # T2 Weighted
└── *-seg.nii.gz   # Segmentation Label (종양 마스크)
```

### 1.2 라벨 생성 방법
```python
# seg.nii.gz의 각 axial 슬라이스에 대해:
label = 1 if np.any(seg_slice > 0) else 0
# 종양 픽셀이 1개라도 있으면 → Positive (종양 존재)
# 종양 픽셀이 0개이면 → Negative (종양 없음)
```

### 1.3 입력 구성 방식

**단일 모달리티 (T2-FLAIR)** 사용:
- T2-FLAIR가 종양 경계를 가장 잘 보여주는 모달리티
- 입력: `(H, W, 1)` → 3채널 복제 `(H, W, 3)` → ResNet 호환

> [!TIP]
> 멀티모달리티(T1c + T2f + T2w → 3채널)도 가능하지만, 단일 모달리티가 학습 안정성이 높고 사진의 프로젝트 요구사항에 더 부합합니다. 필요시 확장 가능합니다.

### 1.4 예상 데이터 규모

| 구분 | 수량 |
|------|------|
| 환자 수 | ~1,251명 |
| 환자당 슬라이스 수 | ~155개 (240×240×155 볼륨 기준) |
| 전체 슬라이스 수 | ~194,000개 |
| 종양 포함 슬라이스 (예상) | ~30~40% (~58,000~77,000개) |
| 종양 미포함 슬라이스 (예상) | ~60~70% (~117,000~136,000개) |

---

## 2. 모델 아키텍처

### 2.1 기본 모델: **ResNet-18 (Pretrained on ImageNet)**

| 항목 | 내용 |
|------|------|
| **베이스 모델** | `torchvision.models.resnet18(pretrained=True)` |
| **입력 크기** | `224 × 224 × 3` |
| **수정 사항** | 마지막 FC layer를 `nn.Linear(512, 1)`로 교체 |
| **출력** | Sigmoid → 확률값 (0~1) |
| **선택 이유** | 의료 영상에서 검증된 성능, 학습 속도 빠름, Grad-CAM 적용 용이 |

### 2.2 모델 구조

```
ResNet-18 (Pretrained)
├── conv1 (7×7, 64)
├── bn1 → relu → maxpool
├── layer1 (BasicBlock × 2, 64)
├── layer2 (BasicBlock × 2, 128)
├── layer3 (BasicBlock × 2, 256)
├── layer4 (BasicBlock × 2, 512)  ← Grad-CAM 타겟 레이어
├── AdaptiveAvgPool2d (1×1)
├── Flatten
└── Linear (512 → 1)  ← 이진 분류 헤드 (수정)
```

### 2.3 Transfer Learning 전략
1. **Phase 1 (3 epochs)**: Feature extractor 동결, FC layer만 학습
2. **Phase 2 (7~12 epochs)**: 전체 unfreeze 후 낮은 LR로 fine-tuning

---

## 3. 학습 설정

### 3.1 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Optimizer | AdamW | weight_decay=1e-4 |
| Learning Rate | 1e-4 (Phase 1), 1e-5 (Phase 2) | |
| LR Scheduler | CosineAnnealingLR | T_max=epochs |
| Batch Size | 32 | GPU 메모리에 따라 조정 |
| Epochs | 15 | Early Stopping patience=5 |
| Loss Function | BCEWithLogitsLoss | pos_weight로 불균형 보정 |
| Image Size | 224 × 224 | ResNet 표준 입력 크기 |

### 3.2 Data Augmentation (학습 시)

| 증강 기법 | 파라미터 |
|----------|---------|
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.5 |
| RandomRotation | ±15° |
| RandomAffine | translate=(0.1, 0.1) |
| Normalize | ImageNet mean/std |

### 3.3 데이터 분할

| 분할 | 비율 | 수준 |
|------|------|------|
| Training | 70% | **환자 수준** |
| Validation | 15% | **환자 수준** |
| Test | 15% | **환자 수준** |

> [!IMPORTANT]
> **환자 수준 분할**: 같은 환자의 슬라이스가 train/val/test에 분산되면 데이터 누출(data leakage)이 발생합니다. 반드시 환자 단위로 분할해야 합니다.

---

## 4. 성능 평가 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| **Accuracy** | > 90% | 전체 정확도 |
| **Precision** | > 85% | 종양 예측의 정밀도 |
| **Recall** | > 90% | 종양 탐지율 (의료에서 중요) |
| **F1-Score** | > 87% | Precision과 Recall의 조화평균 |
| **AUC-ROC** | > 0.90 | ROC 곡선 아래 면적 |
| **Confusion Matrix** | - | TP/FP/TN/FN 시각화 |

---

## 5. 3주 일정 상세 계획

### 1주차 (1일차): 데이터 이해 및 전처리

| 단계 | 작업 내용 | 산출물 |
|------|----------|--------|
| Step 1 | NIfTI 데이터 로드 및 구조 확인 | 데이터 로드 코드 |
| Step 2 | 3D → 2D 슬라이스 추출 & 라벨 생성 | 슬라이스 PNG + CSV 라벨 |
| Step 3 | 데이터 EDA (분포 분석, 시각화) | EDA 차트 |
| Step 4 | Train/Val/Test 분할 (환자 수준) | 분할 CSV |
| Step 5 | PyTorch Dataset/DataLoader 구현 | 데이터 파이프라인 코드 |

### 2주차 (2일차): 모델 구축 및 학습

| 단계 | 작업 내용 | 산출물 |
|------|----------|--------|
| Step 1 | ResNet-18 모델 정의 | 모델 코드 |
| Step 2 | 학습 루프 구현 | 학습 코드 |
| Step 3 | Phase 1 학습 (FC만) | 학습 로그 |
| Step 4 | Phase 2 학습 (전체 fine-tune) | 최적 모델 체크포인트 |
| Step 5 | 성능 평가 & 시각화 | Confusion Matrix, ROC Curve |

### 3주차 (3일차): Explainable AI 및 결과 분석

| 단계 | 작업 내용 | 산출물 |
|------|----------|--------|
| Step 1 | Grad-CAM 구현 | Grad-CAM 코드 |
| Step 2 | 정답/오답 사례 Grad-CAM 시각화 | 히트맵 이미지 |
| Step 3 | seg 마스크와 Grad-CAM 비교 분석 | 비교 시각화 |
| Step 4 | 결과 해석 및 인사이트 도출 | 분석 보고서 |
| Step 5 | 최종 보고서 및 발표 자료 준비 | 최종 보고서 |

---

## 6. 프로젝트 디렉토리 구조

```
biohealth_lv.1/
├── data/
│   └── BraTS-GLI/training/...         # 원본 데이터
├── code/
│   ├── step1_preprocess.py            # 1일차: 전처리
│   ├── step2_eda.py                   # 1일차: EDA
│   ├── step3_dataset.py               # 1일차: Dataset/DataLoader
│   ├── step4_model.py                 # 2일차: 모델 정의
│   ├── step5_train.py                 # 2일차: 학습
│   ├── step6_evaluate.py              # 2일차: 평가
│   ├── step7_gradcam.py               # 3일차: Grad-CAM
│   └── step8_report.py                # 3일차: 최종 보고서
├── processed/
│   ├── slices/                         # 추출된 2D 슬라이스 (PNG)
│   ├── labels.csv                      # 슬라이스별 라벨
│   └── splits.csv                      # Train/Val/Test 분할
├── outputs/
│   ├── checkpoints/                    # 모델 체크포인트
│   ├── figures/                        # 시각화 결과
│   └── logs/                           # 학습 로그
└── requirements.txt
```
