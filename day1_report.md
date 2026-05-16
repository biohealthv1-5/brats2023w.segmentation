# ✅ 1일차 작업 완료 보고서

## 작업 요약

| 단계 | 상태 | 산출물 |
|------|:----:|--------|
| Step 1: 전처리 | ✅ 완료 | 166,626개 PNG 슬라이스 + labels.csv |
| Step 2: EDA | ✅ 완료 | 5종 시각화 차트 |
| Step 3: 데이터 분할 | ✅ 완료 | splits.csv + PyTorch DataLoader |

---

## 데이터 현황

| 항목 | 수치 |
|------|------|
| 처리된 환자 수 | **1,251명** |
| 총 슬라이스 수 | **166,626개** |
| Positive (종양 존재) | 81,374개 (48.8%) |
| Negative (종양 없음) | 85,252개 (51.2%) |
| 환자당 평균 슬라이스 | 133.2개 (중앙값: 134) |
| 환자당 평균 종양 슬라이스 | 65.0개 (중앙값: 66) |

> [!TIP]
> 클래스 비율이 **48.8% vs 51.2%**로 거의 완벽하게 균형잡혀 있어 별도의 클래스 불균형 처리가 불필요합니다.

---

## 데이터 분할 결과 (환자 수준)

| Split | 환자 수 | 슬라이스 수 | Positive | Negative |
|:-----:|:-------:|:----------:|:--------:|:--------:|
| **Train** | 875 | 116,629 | 48.8% | 51.2% |
| **Val** | 188 | 24,882 | 49.0% | 51.0% |
| **Test** | 188 | 25,115 | 48.8% | 51.2% |

---

## EDA 시각화 산출물

5종의 시각화 차트가 `outputs/figures/`에 저장되었습니다:

1. **01_class_distribution.png** - 전체 클래스 분포 (바 + 파이 차트)
2. **02_patient_analysis.png** - 환자당 슬라이스/종양 슬라이스/비율 분포
3. **03_slice_position_distribution.png** - Z축 위치별 종양 비율 (중앙부에 집중)
4. **04_sample_slices.png** - Positive/Negative 샘플 대비 시각화
5. **05_brain_fraction.png** - 뇌 영역 비율 분포 (종양 있는 슬라이스가 더 큰 뇌 영역)

---

## 생성된 파일 구조

```
biohealth_lv.1/
├── code/
│   ├── step1_preprocess.py     # 3D → 2D 슬라이스 추출
│   ├── step2_eda.py            # 탐색적 데이터 분석
│   └── step3_dataset.py        # 데이터 분할 + PyTorch Dataset
├── processed/
│   ├── slices/                  # 166,626개 PNG (224×224)
│   ├── labels.csv               # 슬라이스별 라벨
│   └── splits.csv               # Train/Val/Test 분할 정보
└── outputs/
    └── figures/                 # 5종 EDA 시각화
```

---

## 환경 정보

| 항목 | 내용 |
|------|------|
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 4070 Laptop (8GB) |
| CUDA | 13.1 |

---

## 다음 단계 (2일차)

2일차에서는 다음 작업을 수행합니다:
1. **ResNet-18 모델 정의** (Pretrained, FC layer 수정)
2. **학습 루프 구현** (Phase 1: FC만 → Phase 2: 전체 fine-tune)
3. **학습 실행** (15 epochs, Early Stopping)
4. **성능 평가** (Confusion Matrix, ROC Curve, AUC)
