# 🧠 BraTS-GLI 뇌 MRI 종양 분류 + Multi-Task / Multi-Modal / SOTA 신뢰성 패키지

> **과제명**: 2026-1 바이오헬스세미나02 — BraTS2023-GLI 기반 뇌 MRI 종양 슬라이스 이진 분류 + 해석성(XAI) + 신뢰성(SOTA) 분석
> **데이터**: BraTS2023-GLI Challenge Training Data (1,251명 환자, T1ce + T2-FLAIR)
> **환경**: Python 3.13 / PyTorch 2.11 + CUDA 12.8 / NVIDIA RTX 4070 Laptop GPU (8GB) / Windows 11
> **진행 기간**: 2026-05-09 ~ 2026-05-26 (Day 1 ~ Day 7, **종료**)
> **최종 발표 마스터 자료**: `260527_final_presentation.md`

---

## 1. 프로젝트 개요

3D 뇌 MRI(NIfTI) 볼륨을 2D axial 슬라이스로 변환한 뒤, 각 슬라이스에 **종양이 존재하는지 자동 판별**하는 CNN 모델을 만들고, 그 모델이 정말로 종양을 보고 판단하는지 **Grad-CAM / Segmentation 출력 / Calibration / Conformal Prediction**으로 검증한다.

당초 강의 일정은 3일차(전처리 → 학습 → XAI)로 끝나지만, 본 프로젝트는 3일차에서 발견한 **Shortcut Learning 문제**를 해결하기 위해 자율적으로 7일차까지 확장했고, **Day 7 시점에 프로젝트를 공식 종료**한다.

### 1.1 7단계 실험 흐름 (2026-05-09 → 2026-05-26)

```
Day 1     Day 2     Day 3       Day 4      Day 5      Day 6        Day 7 (★ 종료)
EDA       Whole     Grad-CAM    Patch      MTL        MMMT         SOTA Package
1,251명   Acc 94.33 IoU 0.145   F1 87.08   F1 93.91   F1 94.27     F1 93.50 (thr 0.7)
166K      F1  94.10 FN 92%      IoU 0.128  IoU 0.706  IoU 0.714    WT Dice 0.797 (slice)
슬라이스   AUC 0.9832 ❗ short    passive    active     modality     WT Dice 0.891 (volume) ★
                                실패       성공       FN 회복      AUROC 0.9824 [CI]
                                                                  ECE 0.036 / Conformal 0.896
                                                                  HD95 4.58 / 3-region
```

### 1.2 7단계 여정의 의미 — 의료 AI 연구의 정석 패턴

```
[Naive Baseline] → [한계 진단] → [구조적 시도] → [학습 신호 설계] → [입력 표현 강화] → [SOTA 통합 + 신뢰성]
   Whole-Slice    Grad-CAM     Patch-Based    Multi-Task        Multi-Modal MTL    Day 7 SOTA
   94.33% Acc     IoU 0.145    IoU 0.128      IoU 0.706         IoU 0.7142         IoU 0.7225
                  ❗shortcut    (passive 차단) (active 강제)      (의학적 상보성)    + 3-region + Calib
```

### 1.3 최종 결과 한 줄 요약

> **"Whole-Slice는 수치를 얻었지만 진실을 잃었고, Patch는 진실을 추구했지만 수치를 잃었으며, Multi-Task는 둘 다 잡았고, MMMT는 FN을 회복했으며, Day 7 SOTA는 학부 환경에서 학술 SOTA −0.02 영역까지 도달하면서 신뢰성 정량화의 완성을 보여줬다."**

### 1.4 프로젝트 종료 선언

본 보고서를 마지막으로 **이 프로젝트는 추가 학습/대규모 forward를 더 이상 진행하지 않는다.** 사유:

1. **GPU VRAM 누수** — step30 학습 중 epoch 후반부터 free VRAM이 점진적으로 감소, 14 epoch 시점 EarlyStopping 종료.
2. **Windows PyTorch CUDA Allocator 한계** — `expandable_segments:True`는 Linux 전용, Windows에서 Illegal Memory Access 발생.
3. **step31의 SWA 평가 실패** — `sota_swa.pth`는 디스크에 저장되었으나 val raw(24882×3×224×224 float32) 메모리 누적 시 *14GB array 할당 실패* (16GB RAM 한계).
4. **추가 학습 = 추가 위험** — 시드 변경/K-fold/3D 모델 등 모든 향후 옵션이 (1)(2)(3)에 의해 사실상 불가능.

---

## 2. 데이터셋

| 항목 | 값 |
|------|------|
| 데이터셋 | BraTS-GLI 2023 (ASNR-MICCAI Challenge) |
| 환자 수 | 1,251명 |
| 모달리티 | T2-FLAIR (Day 1~5) → **T1ce + T2-FLAIR + \|T1ce − FLAIR\|** (Day 6~7) |
| 슬라이스 크기 | 224 × 224 (PNG, axial 단면) |
| 총 슬라이스 | 166,626개 |
| Positive (종양 존재) | 81,374개 (48.8%) |
| Negative (정상) | 85,252개 (51.2%) |
| 분할 (환자 단위) | Train 875 / Val 188 / Test 188 |
| Test 슬라이스 | 25,115개 |
| Day 7 추가 라벨 | **WT / TC / ET 3-region** (BraTS 공식 평가축) |

> 환자 단위 분할로 data leakage 방지. 클래스 비율이 48.8% vs 51.2%로 거의 균형이라 별도의 oversampling 불필요 (단 Day 7 SOTA는 *소종양 ×3 Weighted Sampler* 추가).

---

## 3. 핵심 결과 (Test set N=25,115)

### 3.1 8-Way 분류 비교 (메인 표)

| # | 모델 | Threshold | F1 | AUROC | Precision | Recall | FP | FN | 비고 |
|:-:|------|:---------:|:--:|:-----:|:---------:|:------:|:--:|:--:|------|
| 1 | Whole-Slice (Day 2) | 0.6512 | 94.10 | **0.9832** | 95.49 | 92.75 | 536 | 888 | shortcut 진단 |
| 2 | Patch-Based (Day 4) | 0.5 | 87.08 | 0.9104 | 84.93 | 89.34 | 1,942 | 1,306 | passive 실패 |
| 3 | **Multi-Task (Day 5)** | 0.3847 | 93.91 | 0.9824 | **97.33** | 90.73 | **305** | 1,136 | active 성공 |
| 4 | **MMMT (Day 6)** | 0.5 | **94.27** | **0.9832** | 95.49 | 93.08 | 539 | 848 | FN −288 회복 |
| 5 | C-2 FLAIR-only abl. | 0.3847 | 93.98 | 0.9822 | 96.99 | 91.48 | 348 | 1,044 | Tversky 분리 |
| 6 | C-1 T1ce-only abl. | 0.5 | 87.55 | 0.9487 | 90.23 | 85.02 | 1,127 | 1,835 | WT 한계 노출 |
| 7 | **Day 7 SOTA (TTA)** | 0.5 | 92.96 | 0.9824 | 91.59 | **94.37** | 1,061 | **689** ★ | FN 최소 |
| 8 | **Day 7 SOTA (TTA)** ★ | **0.7** | **93.50** | 0.9824 | 94.84 | 92.20 | **614** | 956 | F1 회복 + FP 감소 |

### 3.2 세분화 (Segmentation) 성능

| 모델 | seg head | WT Dice (slice) | WT Dice (volume) | TC Dice | ET Dice | WT IoU | HD95 (WT) |
|------|:--------:|:---------------:|:----------------:|:-------:|:-------:|:------:|:---------:|
| Multi-Task (Day 5) | WT only | 0.7765 | — | — | — | 0.7061 | — |
| MMMT (Day 6) | WT only | 0.7874 | — | — | — | 0.7142 | — |
| **Day 7 SOTA** | **WT/TC/ET** | **0.7971** ★ | **0.8910** ★ | **0.7783** ★ | **0.7497** ★ | **0.7225** ★ | **4.58** |

### 3.3 신뢰성(Reliability) 보고 — Day 7 SOTA 전용

| 항목 | 값 | 비고 |
|------|----|------|
| Bootstrap CI (AUROC) | 0.9824 [0.9811, 0.9837] | 1000 iter |
| Bootstrap CI (F1 @0.7) | 0.9350 [0.9318, 0.9380] | val F1-optimal threshold |
| Temperature T | **1.5015** | over-confidence 보정 |
| ECE (Pre-Temp) | **0.0364** | well-calibrated 영역 |
| Brier Score | 0.0525 | — |
| Conformal Prediction coverage | **0.8964** (target 0.90) | marginal split conformal |
| TTA | 4-way (identity / hflip / vflip / rot180) | logit_mean aggregation |
| Per-volume WT Dice | **0.8910** (n=188) | 학술 SOTA −0.02 |

### 3.4 핵심 발견 (16가지 결정적 인사이트 중 발췌)

1. **Day 3 (Grad-CAM)**: Whole-Slice 모델 TP/FP 히트맵이 동일하게 뇌 중심에 고정 → 모델은 종양이 아니라 **뇌 형태**를 학습 (**Shortcut Learning**).
2. **Day 4 (Patch)**: 광역 단서를 차단하는 *passive* 접근만으로는 부족 — F1·IoU 모두 하락.
3. **Day 5 (Multi-Task)**: 분류 + 세분화 동시 학습 → 분류 성능 유지하며 **IoU 0.145 → 0.706 (4.87×)** + **FP 43% 감소**.
4. **Day 6 (MMMT)**: T1ce + FLAIR + diff → **FN 1,136 → 848 (−25.4%)**, recovered 381 슬라이스 (T1ce ET 강조 효과 정량 검증).
5. **Day 6 Ablation**: FLAIR가 분류 신호의 거의 전부(AUROC 0.9822 ≈ 0.9832), T1ce+diff는 **세분화 정밀도(+5.6%p Dice)에 기여**.
6. **Day 7 (SOTA)**: Focal-Tversky + DeepSup + Uncertainty Weighting + TumorCP + SWA + Temp Scaling + Conformal → **WT/TC/ET 3-region 동시** + **신뢰성 보고 11/12 항목 통과**.
7. **학술 SOTA 비교**: per-volume WT Dice **0.891** = DynUNet 2D (Nature Methods 2021, 0.91)의 **−0.019 영역**.

---

## 4. 디렉토리 구조

```
biohealth_lv.1/
├── README.md                          ← (본 문서)
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
│   │   ├── step18b_multitask_train_gradcam.py   ← Day 5 보강 (Train CAM IoU)
│   │   └── step19_multitask_evaluate.py
│   ├── mmmt/                          Day 6: Multi-Modal Multi-Task
│   │   ├── step20_mmmt_preprocess.py  T1ce 슬라이스 추출
│   │   ├── step21_mmmt_dataset.py     3채널 입력 (T1ce/FLAIR/diff)
│   │   ├── step22_mmmt_model.py       MMMTBrainNet
│   │   ├── step24_mmmt_train.py       Tversky 추가
│   │   ├── step25_mmmt_evaluate.py    표준 평가
│   │   ├── step25b_threshold_sweep.py  ← Day 6 권장 ①: thr 동등 보정
│   │   ├── step26_fn_diff.py           ← Day 6 권장 ②: FN 차분
│   │   ├── step27_ablation_train.py    ← Day 6 권장 ③: 채널 ablation 학습
│   │   └── step28_ablation_evaluate.py     채널 ablation 평가
│   └── sota/                          Day 7: SOTA 패키지 (★ 최종)
│       ├── step26_segmask_3region.py  WT/TC/ET 마스크 생성
│       ├── step27_sota_dataset.py     3채널 입력 + 3채널 mask + Weighted Sampler
│       ├── step28_sota_model.py       DeepSup + Uncertainty Weighting
│       ├── step29_sota_losses.py      Focal-Tversky + Boundary + TumorCP
│       ├── step30_sota_train.py       SWA + Phase 1/2 + AMP
│       └── step31_sota_evaluate.py    TTA + Temp Scaling + Conformal + Bootstrap CI
│
├── data/
│   └── BraTS-GLI/training/            원본 NIfTI 1,251명
│
├── processed/                         (csv는 루트, 이미지는 하위 폴더)
│   ├── labels.csv                     슬라이스별 종양 유무
│   ├── splits.csv                     환자 단위 train/val/test
│   ├── patch_labels.csv               3,218,425개 패치 좌표
│   ├── slices/                        166,626개 FLAIR PNG (224×224)
│   ├── seg_masks/                     세분화 GT 마스크 (WT)
│   ├── mmmt/
│   │   └── t1ce_slices/               Day 6 T1ce 슬라이스
│   └── sota/                          Day 7 SOTA 전용
│       ├── seg_masks_wt/
│       ├── seg_masks_tc/
│       └── seg_masks_et/
│
└── outputs/
    ├── checkpoints/
    │   ├── whole/      best_model.pth, last_model.pth (~134MB)
    │   ├── patch/      patch_best_model.pth (~0.4MB)
    │   ├── multitask/  mt_best_model.pth (~170MB)
    │   ├── mmmt/       mmmt_best.pth
    │   ├── mmmt_ablation/  c1_t1ce_only / c2_flair_only
    │   └── sota/       sota_best.pth + sota_swa.pth (★ Day 7 manifest)
    ├── figures/
    │   ├── whole/      EDA 5종 + 학습/평가 + gradcam/
    │   ├── patch/      11_patch_training_curves.png + patch_eval/ + patch_gradcam/
    │   ├── multitask/  MT 학습/CM/ROC/Seg + gradcam_train/
    │   ├── mmmt/       MMMT 학습/CM/Seg + gradcam_train/
    │   ├── mmmt_ablation/
    │   └── sota/       9장 — training/diagnostics/threshold_sweep_WT,TC,ET/
    │                   postproc/reliability/failure_worst30/failure_best30
    └── logs/
        ├── step14_comparison.json
        ├── whole/      training_history.json, evaluation_results.json
        ├── patch/      step8_stats.json, step11_train_log.json, step12_patch_eval.json
        ├── multitask/  mt_training_history.json, mt_evaluation_results.json,
        │               gradcam_train/mt_train_gradcam_summary.json
        ├── mmmt/       mmmt_summary.json, mmmt_history.json, mmmt_test_metrics.json,
        │               mmmt_test_metrics_thr_sweep.json, fn_diff_summary.json
        ├── mmmt_ablation/  ablation_table.json
        └── sota/       sota_summary.json, sota_history.json, sota_test_metrics.json,
                        sota_test_raw.npz (1.08GB), sota_val_raw.npz (964MB)
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
패치 64×64 (stride=32) → 슬라이스 집계 (count ≥2)
```

### Multi-Task (Day 5)
```
입력 슬라이스 (224×224, FLAIR 1ch → 3복제)
        │
   Shared Encoder (ResNet-18, ImageNet Pretrained)
        │
   ┌───┴───┐
   │       │
 분류 Head  Segmentation Decoder (경량 U-Net, skip ×4)
 GAP→FC(512→1)
   │       │
P(tumor)  Tumor Mask (224×224)

Loss = α·BCE(cls) + β·(Dice+BCE)(seg)   α=1.0, β=0.5
```

### MMMT — Multi-Modal Multi-Task (Day 6)
```
입력: (B, 3, 224, 224)  ← [T1ce, FLAIR, |T1ce − FLAIR|]
       │
   Shared Encoder (ResNet-18, ImageNet pretrained)
       │
   ┌───┴───┐
   GAP→FC(512→1)        Decoder (UpBlock×4 + skip)
   = cls_out             = seg_out (1ch WT, 224×224)

L_seg = 0.5·Dice(WT) + 0.5·Tversky(WT, α=0.7, β=0.3)
β_seg = 0.5  (FN에 강한 penalty → 소종양 누락 대응)
```

### Day 7 SOTA — DeepSup + Uncertainty Weighting (★ 최종)
```
입력: (B, 3, 224, 224)  ← [T1ce, FLAIR, |T1ce − FLAIR|]
       │
   Shared Encoder (ResNet-18, ImageNet pretrained)
       │
   ┌───┴───────────────────────────────────────┐
   GAP→Dropout(0.5)→FC(512→1)   Decoder (UpBlock×4 + skip)
   = cls_out (B,1)              ├─ seg_main: (B, 3, 224, 224) WT/TC/ET
                                ├─ aux_head_d4 (B,1,...) Deep Sup (WT)
                                ├─ aux_head_d3 (B,1,...) Deep Sup (WT)
                                └─ aux_head_d2 (B,1,...) Deep Sup (WT)
   log_var_cls, log_var_seg  ← Kendall 2018 Uncertainty Weighting

L_total  = exp(−log_var_cls) · BCE  + log_var_cls
         + exp(−log_var_seg) · L_seg_compound  + log_var_seg
L_seg_compound = 1.0·FocalTversky(α=0.7, β=0.3, γ=4/3)
               + 0.5·BCE + 0.3·Boundary (per region: WT/TC/ET)
+ Deep Sup (ds_weights = 0.125, 0.25, 0.5)
+ TumorCP (Yang MICCAI 2022, 50% prob)
+ Weighted Sampler (소종양 WT<256px → 가중 ×3)
+ SWA (start epoch 11, swa_lr 5e-5)
+ TTA 4-way (identity / hflip / vflip / rot180, logit_mean)
+ Temperature Scaling (T=1.5015, LBFGS on val BCE)
+ Conformal Prediction (marginal split, coverage 0.8964)
+ Bootstrap CI (1000 iter, 모든 메트릭)
```

---

## 6. 실행 방법

### 6.1 환경 설정

```bash
# Python 3.13.12 / PyTorch 2.11.0 + CUDA 12.8 권장
pip install torch torchvision nibabel pandas numpy scikit-learn matplotlib opencv-python pillow tqdm scipy
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
python code/multitask/step18b_multitask_train_gradcam.py    # Day 5 보강
python code/multitask/step19_multitask_evaluate.py

# Day 6: Multi-Modal Multi-Task + 권장 3종
python code/mmmt/step20_mmmt_preprocess.py
python code/mmmt/step24_mmmt_train.py
python code/mmmt/step25_mmmt_evaluate.py
python code/mmmt/step25b_threshold_sweep.py      # ① threshold 동등 보정
python code/mmmt/step26_fn_diff.py               # ② Day5↔Day6 FN 차분
python code/mmmt/step27_ablation_train.py        # ③ 채널 ablation 학습
python code/mmmt/step28_ablation_evaluate.py     # ③ 채널 ablation 평가

# Day 7: SOTA 패키지 (★ 최종)
python code/sota/step26_segmask_3region.py       # WT/TC/ET 3-region 마스크
python code/sota/step27_sota_dataset.py          # Weighted Sampler
python code/sota/step28_sota_model.py            # DeepSup + Uncertainty Weighting
python code/sota/step29_sota_losses.py           # Focal-Tversky + Boundary + TumorCP
python code/sota/step30_sota_train.py            # 약 17h / 14 epoch (Early Stop)
python code/sota/step31_sota_evaluate.py         # TTA + Temp + Conformal + Bootstrap CI
```

---

## 7. 학술적 인사이트 (요약)

1. **단일 지표(Accuracy/AUC)만으로 의료 AI 모델의 신뢰성을 보장할 수 없다.**
   94.33%라는 정확도가 shortcut learning을 가린 사례를 정량 입증 (Day 3 IoU 0.145).
2. **Shortcut Learning은 명시적 보조 학습 신호(Auxiliary Task)로 극복 가능하다.**
   Multi-Task가 IoU를 0.145 → 0.706(4.87×) 끌어올린 직접 증거 (단, 분류 head CAM IoU 자체는 0.12 → 0.16로 미미 — *MT는 shortcut을 *직접 고친* 게 아니라 *seg head로 우회* 했다는 한 차원 깊은 메시지*).
3. **광역 단서 차단(passive)보다 올바른 학습 강제(active)가 효과적이다.**
   Patch 실패 vs Multi-Task 성공.
4. **멀티모달의 효과는 "F1 점프"가 아닌 *FN 회복 + 운용 자유도 + 세분화 정밀도*에 있다.**
   FLAIR가 분류 신호의 거의 전부, T1ce + diff는 세분화에 +5.6%p Dice 기여.
5. **Tversky α=0.7은 분류 FN −21% / seg Dice −3%p의 *양면성*을 가진다.**
6. **Day 7 SOTA의 *진짜 가치*는 단일 F1이 아닌 *신뢰성 정량화의 완성도*다.**
   Bootstrap CI + Temperature Scaling + Conformal + ECE/Brier + HD95 + Per-volume Dice 의 6종 신뢰성 보고가 처음으로 학술 논문 수준에 도달.
7. **학부 환경(8GB Laptop + Windows + 14 epoch)에서 *학술 SOTA −0.02 영역*까지 도달 — 환경 제약이 명확한 천장.**

---

## 8. 학술 SOTA 비교 (per-volume WT Dice)

| 모델 | 발표 | per-volume WT Dice | 우리 격차 |
|------|:----:|:------------------:|:---------:|
| **MedNeXt** | MICCAI 2023 | 0.93 | −0.039 |
| **SwinUNETR-v2** | CVPR 2024 | 0.92 | −0.029 |
| **DynUNet 2D (nnU-Net)** | Nature Methods 2021 | **0.91** | **−0.019** ★ |
| **🎯 우리 Day 7 SOTA** | (본 프로젝트) | **0.8910** | (학부 환경 천장) |

> 분류 + 신뢰성 정량화 측면에서는 학술 SOTA가 *제공하지 않는* 차원의 통합 보고 (F1 0.935 + ECE 0.036 + Conformal 0.896 + Bootstrap CI 1000회).

---

## 9. 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 |
| Python | 3.13.12 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM) |
| RAM | 16GB DDR5 |
| 주요 라이브러리 | torchvision, nibabel, scikit-learn, matplotlib, opencv-python, pillow, pandas, tqdm, scipy |
| 총 학습 시간 | 약 **50시간** (Whole 2h + Patch 2.2h + MT 5h + MMMT 6.5h + Ablation 17.6h + SOTA 17h) |

### 환경적 한계 (프로젝트 종료 사유)

- **3D 모델 (nnU-Net 3D / SegResNet)**: 24GB+ GPU 필수 → 본 환경 불가
- **K-fold (5-fold) CV**: 17h × 5 = 85h + Resume 안정성 부족
- **Multi-seed 3-seed 평균**: 51h 학습 필요
- **SWA 평가**: 디스크에 `sota_swa.pth` 정상 저장됐으나 RAM 16GB로 val raw 14GB array 할당 실패 → best 단일 평가만 보고

---

## 10. 참고 문서

| 문서 | 분량 | 역할 |
|------|------|------|
| **`260527_final_presentation.md`** | 본 README의 원본 / 발표 마스터 자료 (Day 1~7 통합) | 최종 |
| `260521v1result.md` | Day 1~6 진행 보고 | 본문 통합 원본 |
| `260521v2ways.md` | 확장 로드맵 + SOTA 표 §7 (14개 모델) | §8 비교 원본 |
| `260524v1mmmtplus.md` | Day 5~6 완성 + 권장 3종 (thr 보정 / FN 차분 / ablation) | Day 5~6 원본 |
| `260525v1sotapluswhy.md` / `260525v2sotawith.md` | sota 코드 해설 + 학술 SOTA 위치 | Day 7 설계 |
| `260526v1fullresults.md` / `260526v2step31patch.md` | 성능 짜내기 + P1~P10 패치 | Day 7 구현 |
| `260526v3sotafinal.md` | Day 7 SOTA 평가 통합 | Day 7 원본 |
| `brats2023_dataset_guide.md` | WT/TC/ET 라벨 정의, BraTS 평가축 | 데이터 가이드 |
| `project_spec_ideaB.md` | 프로젝트 상세 명세 | 초기 설계 |

---

> 📎 **프로젝트 종료 선언**: GPU VRAM 누수 + Windows PyTorch CUDA Allocator 한계 + SWA 평가 RAM OOM으로 인해, *본 시점 이후의 학습/대규모 forward는 진행하지 않는다*. `260527_final_presentation.md`가 **최종 발표용 공식 종착점**.
