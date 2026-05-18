# 🔬 Patch-Based 파이프라인 전환 계획서

## BraTS-GLI 뇌종양 분류 — 2D Whole-Slice → Patch-Based 전환

> **배경**: Grad-CAM 분석 결과 현행 2D whole-slice 분류 모델이 종양 위치를 정확히 학습하지 못하고  
> 뇌의 전체적인 형태만 학습하고 있음이 확인됨. 이를 해결하기 위해 patch-based 접근법으로 전환.

---

## 1. 현행 파이프라인의 문제점 요약

| 문제 | 원인 | 영향 |
|------|------|------|
| 낮은 IoU (0.145) | GAP이 종양 feature 희석 | 종양 위치 학습 실패 |
| FN 92% 소종양 | 작은 종양이 224×224에서 무시됨 | Recall 저하 (92.75%) |
| FP 고확신 오탐 | 정상 뇌 구조를 종양으로 착각 | Precision 저하 |
| TP/FP 동일 히트맵 | 종양 특이적 feature 부재 | 판별력 없음 |

---

## 2. Patch-Based 접근법 개요

### 핵심 아이디어

```
현행: 전체 슬라이스 (224×224) → "종양 있음/없음" 이진 분류
개선: 슬라이스를 패치로 분할 → 각 패치별 "종양 있음/없음" 분류
```

### 기대 효과

| 항목 | Whole-Slice (현행) | Patch-Based (개선) |
|------|:------------------:|:------------------:|
| 입력 크기 | 224×224 전체 | 64×64 또는 96×96 패치 |
| 종양 비율 | 전체 대비 <1% | 패치 대비 10~50% |
| Feature 해상도 | 종양 feature 희석 | 종양 feature 보존 |
| 히트맵 정밀도 | 뇌 중심 blob | 패치 단위 정밀 분석 |
| 소종양 감지 | ❌ 놓침 (92%) | ✅ 패치에서 큰 비율 |

---

## 3. 전체 파이프라인 설계

```
Phase A: 데이터 준비
    step8_patch_extract.py    ← 패치 추출 + 라벨링
    step9_patch_dataset.py    ← 패치 Dataset/DataLoader

Phase B: 모델 학습
    step10_patch_model.py     ← 패치 분류 모델 정의
    step11_patch_train.py     ← 학습 루프

Phase C: 평가 및 비교
    step12_patch_evaluate.py  ← 패치 단위 + 슬라이스 단위 평가
    step13_patch_gradcam.py   ← 패치 Grad-CAM 분석
    step14_comparison.py      ← Whole-Slice vs Patch 비교 보고서
```

---

## 4. 각 단계 상세 설계

### Step 8: 패치 추출 (`step8_patch_extract.py`)

**입력**: 원본 NIfTI (t2f + seg) 또는 기존 224×224 PNG 슬라이스  
**출력**: 패치 PNG + 패치 라벨 CSV

```
패치 추출 전략:
├── 패치 크기: 64×64 (stride=32, 50% overlap)
├── 224×224 슬라이스 → 최대 36개 패치 (6×6 grid)
├── 라벨링: seg 마스크에서 종양 픽셀 비율 계산
│   ├── 종양 비율 ≥ 5% → Positive (1)
│   └── 종양 비율 < 5% → Negative (0)
├── 빈 패치 필터링: 뇌 영역 비율 < 10% → 제외
└── 저장: {patient_id}_z{slice}_p{patch_idx}.png
```

**예상 데이터 규모**:
- 166,626 슬라이스 × ~20개 유효 패치 ≈ **~3,000,000개 패치**
- 클래스 비율: Negative 다수 → 샘플링 전략 필요

**클래스 균형 전략**:
- Negative 패치 랜덤 서브샘플링 (1:3 비율)
- 또는 학습 시 WeightedRandomSampler 사용

### Step 9: 패치 Dataset (`step9_patch_dataset.py`)

```python
class BrainTumorPatchDataset(Dataset):
    """패치 단위 Dataset"""
    # 기존 BrainTumorSliceDataset과 동일한 구조
    # 패치 크기에 맞는 transform 적용
    # 환자 단위 split 유지 (data leakage 방지)
```

**Transform 설계**:
- 학습: RandomFlip + RandomRotation + ColorJitter + Normalize
- 검증: Normalize만
- 패치는 이미 작으므로 Resize 불필요

### Step 10: 패치 모델 (`step10_patch_model.py`)

**방안 A: 경량 CNN (추천)**
```
Conv2d(3, 32, 3) → BN → ReLU → MaxPool
Conv2d(32, 64, 3) → BN → ReLU → MaxPool
Conv2d(64, 128, 3) → BN → ReLU → AdaptiveAvgPool
Dropout(0.5) → Linear(128, 1)
```
- 파라미터: ~200K (현행 11M 대비 1/50)
- 64×64 입력에 적합한 크기

**방안 B: ResNet-18 축소판**
```
ResNet-18 (Pretrained)
├── 입력: 64×64 → 내부적으로 처리
└── FC: 512 → 1
```
- Pretrained feature 활용 가능
- 단, 64×64에는 과대 모델일 수 있음

**권장**: 방안 A (경량 CNN)부터 시작, 성능 부족 시 방안 B로 전환

### Step 11: 패치 학습 (`step11_patch_train.py`)

| 항목 | 설정 |
|------|------|
| Optimizer | AdamW |
| LR | 1e-3 → CosineAnnealing |
| Batch Size | 128 (패치가 작아 큰 배치 가능) |
| Epochs | 20 |
| Early Stopping | patience=5 |
| Loss | BCEWithLogitsLoss + pos_weight |
| 클래스 균형 | WeightedRandomSampler |

### Step 12: 패치 평가 (`step12_patch_evaluate.py`)

**2단계 평가 체계:**

```
Level 1: 패치 단위 성능
├── Patch Accuracy, Precision, Recall, F1
├── Patch AUC-ROC
└── Confusion Matrix

Level 2: 슬라이스 단위 집계
├── 한 슬라이스의 패치 예측들을 집계
│   ├── 방법1: max pooling (패치 중 하나라도 양성 → 슬라이스 양성)
│   ├── 방법2: mean pooling (패치 평균 확률 > threshold)
│   └── 방법3: count (양성 패치 수 ≥ N개)
├── 슬라이스 단위 Accuracy, F1, AUC-ROC
└── Whole-Slice 모델과 직접 비교
```

### Step 13: 패치 Grad-CAM (`step13_patch_gradcam.py`)

- 패치 모델의 Grad-CAM 히트맵 생성
- 패치를 원래 슬라이스 위치에 재조립하여 **슬라이스 단위 히트맵 복원**
- seg 마스크와의 IoU 비교 (0.145 → 개선 목표)
- 패치 단위에서 모델이 종양 영역을 정확히 보는지 검증

### Step 14: 비교 보고서 (`step14_comparison.py`)

| 비교 항목 | Whole-Slice | Patch-Based |
|-----------|:-----------:|:-----------:|
| Accuracy | 94.33% | ? |
| F1-Score | 94.10% | ? |
| AUC-ROC | 0.9832 | ? |
| FN (놓친 종양) | 888개 | ? |
| FP (오탐) | 536개 | ? |
| Grad-CAM IoU | 0.145 | ? |
| 소종양 Recall | ~8% | ? |

---

## 5. 구현 일정 (예상)

| 단계 | 작업 | 예상 소요 |
|:----:|------|:---------:|
| 1 | Step 8: 패치 추출 | 30분 (코드) + 1~2시간 (실행) |
| 2 | Step 9: Dataset 구현 | 20분 |
| 3 | Step 10: 모델 정의 | 20분 |
| 4 | Step 11: 학습 | 30분 (코드) + 1~3시간 (학습) |
| 5 | Step 12: 평가 | 30분 |
| 6 | Step 13-14: Grad-CAM + 비교 | 1시간 |

---

## 6. 주요 고려사항

### 데이터 관련
- **Data Leakage 방지**: 패치 분할 시 기존 환자 단위 split(train/val/test) 유지 필수
- **클래스 불균형**: 패치 단위에서는 Negative가 압도적 다수 → 샘플링 전략 중요
- **저장 공간**: ~300만 패치 PNG ≈ 수 GB → 디스크 여유 확인 필요

### 모델 관련
- **패치 크기 선택**: 64×64 (작은 종양 포착) vs 96×96 (더 넓은 맥락)
- **Stride 선택**: 32 (overlap 50%) → 경계 종양 놓침 방지
- **종양 비율 threshold**: 5% (너무 낮으면 노이즈 패치 양성화)

### 평가 관련
- **슬라이스 집계 방법**: max pooling이 FN 최소화에 유리
- **소종양 Recall**: 핵심 개선 지표로 추적
- **Grad-CAM IoU**: 0.145 → 0.4+ 목표

---

## 7. 대안적 접근법 (참고)

| 방법 | 설명 | 장단점 |
|------|------|--------|
| **Sliding Window** | 고정 크기 패치를 슬라이딩 | 구현 간단, 중복 계산 多 |
| **ROI Proposal** | 후보 영역 추출 후 분류 | 2-stage, 복잡하지만 정밀 |
| **U-Net Segmentation** | 직접 픽셀 분류 | 가장 정밀, 학습 난이도 높음 |
| **Attention Mechanism** | 전체 슬라이스 + Attention | 위치 학습 개선, 구현 복잡 |

> 현재 프로젝트 수준에서는 **Sliding Window Patch** 방식이 가장 적합.
> 구현이 직관적이고, 기존 코드 구조를 최대한 재활용 가능.

---

## 8. 성공 기준

| 지표 | 현행 | 목표 |
|------|:----:|:----:|
| 슬라이스 단위 Accuracy | 94.33% | ≥ 95% |
| 소종양 Recall | ~8% | ≥ 50% |
| Grad-CAM IoU | 0.145 | ≥ 0.40 |
| FN 감소 | 888개 | ≤ 400개 |
| FP 감소 | 536개 | ≤ 300개 |
