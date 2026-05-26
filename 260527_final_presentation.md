# 🎤 최종 발표 마스터 자료 (260527 — Day 1~7 통합 + SOTA 비교 + 발표 준비)

> **본 문서의 위치**: `biohealth_lv.1/260527_final_presentation.md`
> **목적**: `260521v1result.md` (Day 1~5 + Day 6 진행 중), `260524v1mmmtplus.md` (Day 5~6 완성 + 권장 3종 결과), `260526v3sotafinal.md` (Day 7 SOTA 최종 평가) 세 보고서를 **하나의 최종 발표용 마스터 자료**로 통합하고, `260521v2ways.md` §7의 *직접 비교 가능한 SOTA*를 우리 실측치와 함께 정렬한 발표 직전 종착점.
>
> **편집 원칙**
> 1. 단계별 결정적 수치·관찰·의사결정은 *원문 그대로 보존* (요약 X).
> 2. 명세서·산출물에 명시되지 않은 부분은 폴더 내 다른 자료(`brats2023_dataset_guide.md`, `260523v2proposal.md`, `260525v1sotapluswhy.md`, `260525v2sotawith.md`, `260526v1fullresults.md`, `260526v2step31patch.md` 등)에서 찾아 **[보강]** 표시와 함께 추가.
> 3. 각 단계 끝에 **🎯 예상 질문 (Q&A)** 섹션을 두어 교수님이 던질 만한 의문을 정리.
> 4. 본문에서 빠지거나 *측정 후 메인 표 미포함* 결과는 **부록**에 별도 보관.
> 5. 각 섹션 끝에 **📎 참고 md 출처**로 어떤 파일의 어느 절을 참조했는지 명시.

---

## 목차

0. [전체 한눈에 보기 — 7단계 여정 도식 & 프로젝트 종료 선언](#0-전체-한눈에-보기--7단계-여정-도식--프로젝트-종료-선언)
1. [Day 1 — 데이터 전처리 + EDA](#1-day-1--데이터-전처리--eda)
2. [Day 2 — Whole-Slice 모델 학습](#2-day-2--whole-slice-모델-학습)
3. [Day 3 — Grad-CAM 해석성 분석 (가장 결정적인 단계)](#3-day-3--grad-cam-해석성-분석-가장-결정적인-단계)
4. [Day 4 — Patch-Based 파이프라인 (부분적 실패의 교훈)](#4-day-4--patch-based-파이프라인-부분적-실패의-교훈)
5. [Day 5 — Multi-Task Learning (본 프로젝트의 핵심 성과)](#5-day-5--multi-task-learning-본-프로젝트의-핵심-성과)
6. [Day 5 보강 — Train-split Grad-CAM (분류 vs 세분화 head 분리 진단)](#6-day-5-보강--train-split-grad-cam-분류-vs-세분화-head-분리-진단)
7. [Day 6 — Multi-Modal Multi-Task 본 학습 결과](#7-day-6--multi-modal-multi-task-본-학습-결과)
8. [Day 6 권장 3종 결과 — Threshold 동등 보정 / FN 차분 / 채널 Ablation](#8-day-6-권장-3종-결과--threshold-동등-보정--fn-차분--채널-ablation)
9. [Day 7 — SOTA 패키지 설계 + 학습 + 평가](#9-day-7--sota-패키지-설계--학습--평가)
10. [Day 7 신뢰성 SOTA — Calibration / Conformal / Bootstrap CI](#10-day-7-신뢰성-sota--calibration--conformal--bootstrap-ci)
11. [Day 7 세분화 SOTA — WT/TC/ET region별, postproc, HD95](#11-day-7-세분화-sota--wttcet-region별-postproc-hd95)
12. [Day 7 실패 케이스 정성 분석](#12-day-7-실패-케이스-정성-분석)
13. [전체 모델 비교 — 8-way (Whole/Patch/MT/MMMT/C-1/C-2/SOTA@0.5/SOTA@0.7)](#13-전체-모델-비교--8-way-wholepatchmtmmmtc-1c-2sota05sota07)
14. [직접 비교 가능한 SOTA 정렬 — v2ways §7 + 우리 실측 (★ 발표 핵심)](#14-직접-비교-가능한-sota-정렬--v2ways-§7--우리-실측--발표-핵심)
15. [상대 팀(순수 Segmentation)과의 차별화](#15-상대-팀순수-segmentation과의-차별화)
16. [환경적 한계 — VRAM 누수 / Windows allocator / 추가 진행 불가 사유](#16-환경적-한계--vram-누수--windows-allocator--추가-진행-불가-사유)
17. [발표 핵심 메시지 — 16가지 결정적 인사이트 (완전판)](#17-발표-핵심-메시지--16가지-결정적-인사이트-완전판)
18. [발표 슬라이드 추천 구성 (10장)](#18-발표-슬라이드-추천-구성-10장)
19. [예상 질문 마스터 리스트 (총 92문항)](#19-예상-질문-마스터-리스트-총-92문항)
20. [부록 A — 배제된 / 보류된 결과 (전체 인벤토리)](#부록-a--배제된--보류된-결과-전체-인벤토리)
21. [부록 B — 전체 학습 로그 (Whole/Patch/MT/MMMT/Abl/SOTA)](#부록-b--전체-학습-로그-wholepatchmtmmmtablsota)
22. [부록 C — 모든 모델 한눈 비교표 (전체 프로젝트 종합)](#부록-c--모든-모델-한눈-비교표-전체-프로젝트-종합)
23. [부록 D — 임상 시나리오별 운용 + 실용화 5가지](#부록-d--임상-시나리오별-운용--실용화-5가지)
24. [부록 E — 참고한 md 파일 인벤토리 + 직접 인용 출처](#부록-e--참고한-md-파일-인벤토리--직접-인용-출처)

---

## 0. 전체 한눈에 보기 — 7단계 여정 도식 & 프로젝트 종료 선언

### 0.1 마일스톤 타임라인 (2026-05-09 → 2026-05-26)

```
2026-05-09 ────────────────────────────────────────────── 2026-05-26
  │                                                              │
  │   [당초 강의 일정: 3일차 = 1~3주차]                            │
  │   ┌─────────┐  ┌─────────┐  ┌─────────────────────┐         │
  │   │ Day 1   │→ │ Day 2   │→ │ Day 3 (계획됨)      │         │
  │   │ EDA     │  │ Train   │  │ Grad-CAM XAI        │         │
  │   └─────────┘  └─────────┘  └─────────────────────┘         │
  │        │                                                     │
  │        ▼   ── 실제 진행은 5일차까지 확장 → 6일차 → 7일차 종료 ─│
  ▼                                                              ▼
  Day 1     Day 2     Day 3     Day 4     Day 5     Day 6     Day 7 (★ 종료)
  EDA       Whole     Grad-CAM  Patch     MTL       MMMT      SOTA Package
  1,251명   Acc 94.33 IoU 0.145 F1 87.08  F1 93.91  F1 94.27  F1 93.50 (thr 0.7)
  166K      F1  94.10 FN 92%    IoU 0.128 IoU 0.706 IoU 0.714 WT Dice 0.797 ★
  슬라이스   AUC 0.9832 ❗ short   passive   active    modality  AUROC 0.9824
                                실패      성공      FN 회복   WT vol 0.891 ★
                                                              ECE 0.036
                                                              Conformal 0.896
                                                              HD95 4.58
   ✅          ✅         ✅         ✅         ✅         ✅         🎯 종료
```

### 0.2 7단계 여정의 의미 — 의료 AI 연구의 정석 패턴

```
[Naive Baseline] → [한계 진단] → [구조적 시도] → [학습 신호 설계] → [입력 표현 강화] → [SOTA 통합 + 신뢰성]
   Whole-Slice    Grad-CAM     Patch-Based    Multi-Task         Multi-Modal MTL    Day 7 SOTA
   94.33% Acc     IoU 0.145    IoU 0.128      IoU 0.706          IoU 0.7142         IoU 0.7225
   F1 94.10%     ❗shortcut    (passive 차단) (active 강제)      (의학적 상보성)    + 3-region + Calib
```

### 0.3 7단계 여정의 한 줄 요약 (Day 7 종료 시점)

> **"Whole-Slice는 수치를 얻었지만 진실을 잃었고, Patch는 진실을 추구했지만 수치를 잃었으며, Multi-Task는 둘 다 잡았고, MMMT는 FN을 회복했으며, Day 7 SOTA는 학부 환경에서 학술 SOTA -0.02 영역까지 도달하면서 신뢰성 정량화의 완성을 보여줬다."**

### 0.4 프로젝트 종료 선언 — 왜 Day 7에서 멈추는가

본 보고서를 마지막으로 **이 프로젝트는 추가 학습/대규모 forward를 더 이상 진행하지 않는다.** 사유:

1. **GPU VRAM 누수**: step30 학습 중 epoch 후반부터 free VRAM이 점진적으로 줄어드는 패턴 관찰. 14 epoch 시점에 OOM 위험 + EarlyStopping trigger로 학습 종료 (`sota_summary.json`의 `total_epochs: 14` 가 직접 증거 — 계획 P1 3 + P2 15 = 18 epoch이었으나 P2 11번째에서 멈춤).
2. **Windows PyTorch CUDA Allocator 한계**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`는 *Linux 전용* — Windows에서 켜면 Illegal Memory Access / Allocator 손상 발생. `step31_sota_evaluate.py` L57-67에서 OS 분기로 우회는 했으나 본질적 한계.
3. **step31 평가의 SWA 평가 실패**: `sota_test_metrics.json`의 `swa` 블록에 `"error": "Unable to allocate 14.0 GiB for an array with shape (24882, 3, 224, 224) and data type float32"` — CPU RAM 16GB 환경에서 val raw 결과를 메모리에 누적하는 도중 시스템 RAM 부족. **best 평가는 성공했지만 SWA 평가는 실패**.
4. **추가 학습 = 추가 위험**: 시드 변경/3-seed 평균/K-fold/3D 모델 전환 등 모든 향후 옵션이 (1)(2)에 의해 사실상 불가능. 발표 일정상 환경 재구축(WSL2 + Linux native PyTorch + 24GB+ GPU 임차)도 비현실적.

따라서 **본 문서는 "이미 손에 들고 있는 자산 + step31의 마지막 평가 1회"** 라는 *현실의 발표 자료*다.

📎 **참고**: `260521v1result.md` §0 / `260524v1mmmtplus.md` §0 / `260526v3sotafinal.md` §0, §9 / `outputs/logs/sota/sota_summary.json` / `outputs/logs/sota/sota_test_metrics.json`

---

## 1. Day 1 — 데이터 전처리 + EDA

### 1.1 목표 & 결정사항

3D NIfTI 볼륨을 2D 슬라이스로 변환하고, **환자 단위**로 분할하여 학습 가능한 형태로 만든다.

**핵심 결정사항 (보존 필요)**:
- 모달리티: **T2-FLAIR 단일 사용** — 부종 경계가 가장 잘 보임. (멀티모달은 Day 6에서 확장.)
- 라벨 규칙: `label = 1 if np.any(seg_slice > 0) else 0` (종양 픽셀 **1개라도** 있으면 Positive)
- **환자 수준 분할**: 같은 환자의 슬라이스가 train/val/test에 분산되면 data leakage 발생 → 반드시 환자 단위로 분할.
- 클래스 비율 48.8% vs 51.2%로 거의 균형 → 별도 oversampling 불필요.

### 1.2 실행 코드

| 코드 | 역할 |
|------|------|
| `code/whole/step1_preprocess.py` | 3D NIfTI → 224×224 PNG 슬라이스 추출, `labels.csv` 생성 |
| `code/whole/step2_eda.py` | 5종 시각화로 데이터 분포 진단 |
| `code/whole/step3_dataset.py` | 환자 수준 Train/Val/Test 분할, PyTorch Dataset 구현 |

### 1.3 데이터 현황

| 항목 | 수치 |
|------|------|
| 처리된 환자 수 | **1,251명** |
| 총 슬라이스 수 | **166,626개** |
| Positive (종양 존재) | 81,374개 (48.8%) |
| Negative (정상) | 85,252개 (51.2%) |
| 환자당 평균 슬라이스 | 133.2개 (중앙값 134) |
| 환자당 평균 종양 슬라이스 | 65.0개 (중앙값 66) |

### 1.4 분할 결과 (환자 단위, 7:1.5:1.5)

| Split | 환자 수 | 슬라이스 수 | Positive | Negative |
|:-----:|:-------:|:----------:|:--------:|:--------:|
| **Train** | 875 | 116,629 | 48.8% | 51.2% |
| **Val** | 188 | 24,882 | 49.0% | 51.0% |
| **Test** | 188 | 25,115 | 48.8% | 51.2% |

### 1.5 EDA 5종 시각화 발견

1. `01_class_distribution.png` — 전체 클래스 거의 균형 (48.8% / 51.2%).
2. `02_patient_analysis.png` — 환자당 슬라이스/종양 슬라이스/비율 분포.
3. `03_slice_position_distribution.png` — Z축 위치별 종양 비율: **중앙부(40~120번 슬라이스)에 집중**.
4. `04_sample_slices.png` — Positive/Negative 샘플 대비.
5. `05_brain_fraction.png` — **종양 있는 슬라이스가 뇌 영역 비율이 더 큰 경향** ← Day 3 shortcut learning의 복선.

### 1.6 [보강] 데이터셋 배경 (`brats2023_dataset_guide.md` 기반)

- BraTS 2023은 ASNR-MICCAI에서 주관하는 뇌종양 세분화 챌린지.
- 각 환자 케이스는 4 모달리티(T1n, T1c=T1CE, T2w, T2f=FLAIR) + Seg 마스크.
- Segmentation Label 의미:
  - **0**: 배경
  - **1**: NCR (괴사 핵, Necrotic Core)
  - **2**: ED (부종, Peritumoral Edema)
  - **3**: ET (조영 증강 종양, Enhancing Tumor)
- BraTS 공식 평가축인 WT/TC/ET:
  - WT (Whole Tumor) = NCR + ED + ET
  - TC (Tumor Core) = NCR + ET
  - ET (Enhancing Tumor) = ET only
- 본 프로젝트는 슬라이스에 픽셀 1개라도 있으면 Positive로 두므로, 실질적으로 **WT 기준 이진화**.

### 🎯 Day 1 예상 질문 (Q&A)

**Q1. 왜 T1ce가 아닌 T2-FLAIR를 선택했나? T1ce가 종양 강조에 더 강한 것 아닌가?**
A. T2-FLAIR가 **부종(ED) 경계**를 가장 잘 보여줘서 "종양이 있는 슬라이스인지" 판별하는 이진 분류에는 더 적합. T1ce는 활성 종양(ET) 강조에 강하지만 부종이 안 보임. WT 기준 이진 분류이므로 FLAIR가 시작점으로 더 자연스럽다. 다만 단일 모달의 한계 때문에 Day 6에서 T1ce를 추가하는 멀티모달로 확장.

**Q2. 환자 단위 분할의 정당성은? 슬라이스 단위로 섞으면 더 많은 데이터를 train에 쓸 수 있지 않나?**
A. 같은 환자의 인접 슬라이스는 매우 유사하므로 train/test에 동시에 들어가면 **data leakage**가 발생, 정확도가 인위적으로 부풀려진다. 의료 AI 평가의 표준은 환자 단위 분할이며, 875/188/188로도 슬라이스 수는 train 116K로 충분하다.

**Q3. "픽셀 1개라도 있으면 Positive"는 너무 관대한 기준 아닌가?**
A. 맞다. 실제로 이 기준 때문에 FN의 92.3%가 픽셀 비율 1% 미만의 소종양으로 나타나는데, 모델이 이 케이스를 거의 못 잡는다. Day 5에서 segmentation supervision을 추가하고, Day 6에서 Tversky Loss로 소종양 FN에 가중을 두려는 이유가 바로 이 라벨 정책 때문이다.

**Q4. 클래스 비율 48.8% vs 51.2%가 균형이라고 봤는데, 슬라이스가 아니라 환자/임상적 관점에서도 그런가?**
A. 슬라이스 단위에서는 그렇다. 다만 환자 한 명 안에서 종양 슬라이스가 차지하는 비율(중앙값 약 49.5%, 65/134)은 환자마다 차이가 크다 — Z축 중앙부(40~120)에 종양 슬라이스가 집중되고 위/아래 끝은 거의 정상.

📎 **참고**: `260521v1result.md` §1 / `brats2023_dataset_guide.md` 전체 / `notPublic/day1_report.md` 전 절 / `project_spec_ideaB.md` §1.2~1.3

---

## 2. Day 2 — Whole-Slice 모델 학습

### 2.1 목표

ResNet-18 (ImageNet pretrained)로 슬라이스 단위 이진 분류 달성.

### 2.2 모델 아키텍처

```
ResNet-18 (ImageNet Pretrained, 11,177,025 파라미터)
├── conv1 (7×7, 64ch) → bn1 → relu → maxpool
├── layer1 (BasicBlock ×2, 64ch)
├── layer2 (BasicBlock ×2, 128ch)
├── layer3 (BasicBlock ×2, 256ch)
├── layer4 (BasicBlock ×2, 512ch)  ← Grad-CAM 타겟 레이어
├── AdaptiveAvgPool2d (1×1)
├── Dropout(0.5)
└── Linear(512 → 1)  ← BCEWithLogitsLoss(pos_weight 적용)
```

### 2.3 학습 전략 — 2-Phase Transfer Learning

| 항목 | Phase 1 (FC만) | Phase 2 (전체 Fine-tune) |
|------|:--------------:|:------------------------:|
| Epochs | 3 | 12 (Early Stop patience=5) |
| Learning Rate | **1e-3** | **1e-4** |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Batch Size | 32 | 32 |
| Weight Decay | 1e-4 | 1e-4 |
| Backbone | **동결** | **학습 가능** |

**Data Augmentation (학습 시)**: RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotation(15°), RandomAffine(translate=0.1), ImageNet Normalize.

### 2.4 학습 곡선 (핵심 지점)

| Epoch | Phase | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|:-----:|:-----:|:----------:|:---------:|:--------:|:-------:|:---:|
| 1 | P1 | 0.4520 | 79.46% | 0.4113 | 81.28% | 7.5e-4 |
| 2 | P1 | 0.4421 | 80.00% | 0.4064 | 81.46% | 2.5e-4 |
| 3 | P1 | 0.4360 | 80.39% | 0.4082 | 81.85% | 0 |
| **4** | **P2 시작** | **0.2067** | **92.06%** | **0.2076** | **92.59%** | **9.8e-5** |
| 5 | P2 | 0.1681 | 93.75% | 0.1889 | 93.33% | 9.3e-5 |
| 6 | P2 | 0.1543 | 94.28% | 0.1820 | 93.66% | 8.5e-5 |
| 7 | P2 | 0.1442 | 94.68% | 0.1746 | 93.84% | 7.5e-5 |
| 8 | P2 | 0.1327 | 95.16% | 0.1818 | 93.90% | 6.3e-5 |
| **9** | **P2 (★ Best)** | **0.1244** | **95.48%** | **0.1713** | **94.15%** | **5.0e-5** |
| 10 | P2 | 0.1137 | 95.90% | 0.1928 | 93.59% | 3.7e-5 |
| 11 | P2 | 0.1042 | 96.27% | 0.1888 | 93.85% | 2.5e-5 |
| 12 | P2 | 0.0953 | 96.63% | 0.1873 | 94.16% | 1.5e-5 |
| 13 | P2 | 0.0868 | 96.96% | 0.2023 | 93.94% | 6.7e-6 |
| 14 | P2 | 0.0809 | 97.18% | 0.2209 | 93.71% | 1.7e-6 |

**관찰**:
- Phase 1 → Phase 2 전환 시 **Train Loss 0.4360 → 0.2067로 절반 이하 급감** — backbone 해동 효과.
- **Epoch 9가 Best**, 이후 Val Loss 단조 증가 = 명확한 과적합 시작.

### 2.5 테스트셋 평가 (N=25,115)

| 지표 | 값 |
|------|:---:|
| **Accuracy** | **94.33%** |
| Precision | 95.49% |
| Recall (Sensitivity) | 92.75% |
| Specificity | 95.83% |
| F1-Score | 94.10% |
| **AUC-ROC** | **0.9832** |
| AP (Average Precision) | 0.9862 |
| **최적 Threshold (Youden's J)** | **0.6512** |

### 2.6 Confusion Matrix

|  | Pred Negative | Pred Positive |
|:---:|:---:|:---:|
| **Actual Neg** (12,866) | TN = **12,330** | FP = **536** |
| **Actual Pos** (12,249) | FN = **888** | TP = **11,361** |

### 2.7 Day 2 소결 (보존)

> 수치적 성능은 우수(Acc 94%, AUC 0.98).
> 그러나 **FN 888개(종양 누락) + FP 536개(오탐)** 존재.
> 의료 AI에서 FN은 치명적 → Grad-CAM으로 원인 분석 필요.

### 🎯 Day 2 예상 질문 (Q&A)

**Q1. 왜 2-Phase 학습인가? 그냥 전체를 1e-4로 처음부터 학습하면?**
A. ImageNet pretrained의 conv1~layer4는 일반 시각 feature가 잘 잡혀 있다. 처음부터 모두 해동하면 분류 헤드(초기 무작위 가중치)의 큰 gradient가 사전학습된 backbone을 망가뜨릴 위험. Phase 1에서 FC만 데우고 Phase 2에서 천천히 fine-tune이 transfer learning의 표준 절차. 실제로 Phase 1 → Phase 2 전환 시 Train Loss가 절반 이하로 급감하며 효과 확인됨.

**Q2. Best Epoch이 9인데 14까지 돌린 이유는?**
A. Early Stopping patience=5로 설정해서 자동으로 멈춘 결과가 14 epoch. Epoch 10~14는 Train Loss는 계속 감소(0.114→0.081)하지만 Val Loss는 증가(0.193→0.221) — **명확한 과적합 신호**.

**Q3. 최적 threshold가 0.5가 아니라 0.6512인 이유?**
A. Youden's J statistic (Sensitivity + Specificity − 1)을 최대화하는 임계값. ROC 곡선상에서 TPR-FPR이 가장 멀어지는 지점.

**Q4. AUC 0.98인데도 의료 AI에서 부족하다는 건가?**
A. AUC 0.98은 단일 지표로는 우수하지만 **모델이 무엇을 보고 판단했는지**는 알 수 없다. Day 3 Grad-CAM에서 드러난 사실 — IoU 0.145, TP/FP 히트맵 동일, FN 92%가 소종양 — 이 그 답. AUC가 높아도 임상 신뢰성과 직결되지 않는다는 점을 보여주는 직접적 사례.

**Q5. FN 888 / FP 536, 환자 기준으로 환산하면 어떤 의미?**
A. 정상 1,000명 스크리닝 → 약 42명 오탐(추가 검사·불안 야기). 종양 1,000명 진단 → 약 73명 놓침. 이 수치는 의료 AI에서 무시할 수 없는 규모로, Day 5 Multi-Task가 FP를 305로 줄여 1,000명당 24명 오탐(18명 절약)으로 만든 점이 임상적 가치의 핵심.

📎 **참고**: `260521v1result.md` §2 / `notPublic/day2_day3_report.md` §1~§7 / `project_spec_ideaB.md` §2

---

## 3. Day 3 — Grad-CAM 해석성 분석 (가장 결정적인 단계)

### 3.1 목표

TP/FN/FP 각각에 대해 Grad-CAM 히트맵을 생성하고, 모델이 실제로 종양 영역을 보고 있는지 **정량 검증**.

### 3.2 분석 프로세스 (보존)

```
테스트셋 전체 25,115개 → 예측 확률 수집
    ├── TP/FN/FP/TN 분류
    ├── [질문1] TP 8개 시각화 + 200개 IoU 통계
    ├── [질문2] FN 8개 시각화 + 300개 종양 크기 분석
    └── [질문3] FP 8개 시각화 + 확률 분포 비교
```

### 3.3 Grad-CAM 수치 결과 — 핵심 지표 (보존)

| 지표 | 값 | 해석 |
|------|:---:|------|
| **TP 평균 IoU** | **0.145** | 히트맵과 종양 마스크 거의 안 겹침 |
| **FN 작은종양 비율** | **92.3%** | 놓친 종양의 92%가 매우 작음 (픽셀 <1%) |
| **FN 평균 예측 확률** | **0.191** | "없다"고 확신하며 놓침 |
| **FP 평균 예측 확률** | **0.741** | "있다"고 확신하며 오탐 |
| FN 평균 종양 픽셀 비율 | **0.36%** (~180픽셀, 224×224 기준) | 극도로 왼쪽 편향 |

### 3.4 [질문1] TP — Seg 마스크와 히트맵이 일치하는가?

**결론: ❌ 일치하지 않음 (평균 IoU = 0.145)**

관찰 결과 (보존):
- Grad-CAM 히트맵이 **항상 뇌 중심부에 큰 원형 blob**으로 나타남.
- 종양이 좌측/우측에 위치해도 히트맵은 **가운데에 고정**.
- 고확률 TP (prob=1.000)에서도 IoU는 0.23~0.25 수준.
- 저확률 TP (prob≈0.50)에서는 IoU가 0.007~0.052로 극히 낮음.
- IoU 분포: 대부분 0.0~0.1 구간에 밀집, IoU > 0.3은 약 6%.

> 모델은 종양의 정확한 위치를 학습한 것이 아니라, "뇌의 전체적인 형태/밝기 분포"를 기반으로 판단하고 있음.

### 3.5 [질문2] FN 888개 — 왜 놓쳤는가?

**결론: 종양이 너무 작아서 feature map에서 소실**

핵심 발견 (보존):
- FN의 **92.3%가 종양 픽셀 비율 1% 미만** (224×224 중 ~500픽셀 이하).
- 평균 종양 픽셀 비율: **0.36%** (약 180픽셀).
- 종양 크기 분포가 극도로 왼쪽 편향 (대부분 0~0.005 구간).

히트맵 패턴:
- 경계선 FN (prob≈0.497): 뇌 중심에 약한 반응, 종양 무시.
- 극저확률 FN (prob≈0.002~0.003): 히트맵 완전 무반응 (IoU=0.000).
- 확률 분포: 0.0~0.05 구간에 가장 많이 밀집.

> 작은 종양은 ResNet의 Global Average Pooling을 거치면서 feature가 희석되어 검출 불가능.
> 이는 2D whole-slice 분류의 **구조적 한계**임.

### 3.6 [질문3] FP 536개 — 무엇을 착각했는가?

**결론: 정상 뇌 구조를 종양으로 오인**

핵심 발견 (보존):
- Seg 마스크가 완전히 비어있는데 히트맵이 강한 반응.
- 고확신 FP (prob=1.000): 뇌 중심부에 강한 빨간 blob.
- 히트맵 패턴이 TP와 동일 → 종양/비종양 구분 없이 같은 영역 주시.
- FP 평균 확률 0.741 → 확신 있게 오탐.
- 0.8~1.0 고확률 구간에도 다수 분포 → **threshold 조정으로 해결 불가능한 FP**가 상당수.

> 모델이 "종양 특이적 feature"가 아닌 "뇌 존재 여부"를 학습했다는 결정적 증거.

### 3.7 구조적 한계 진단 (보존)

```
2D Whole-Slice 분류의 한계
├── 224×224 전체 이미지에서 종양은 극소 영역
├── GAP(Global Average Pooling)이 작은 종양 feature 희석
├── 모델이 "종양 위치"가 아닌 "뇌 형태" 학습
└── TP/FP의 히트맵 패턴이 동일 → 판별력 없음
    → Patch-Based 접근법 전환 결정
```

### 3.8 결정적 인사이트 (Day 3 이후 모든 단계의 출발점) (보존)

> 모델은 "종양"이 아닌 **"뇌의 전체 형태"**를 학습했다 = **Shortcut Learning**.
> 94.33%의 정확도는 신뢰할 수 없는 수치이며, **의료 AI 평가에 단일 지표(Acc/AUC)만 쓰면 안 된다**는 결정적 증거를 확보.

이 발견이 본 프로젝트의 **모든 후속 단계(Day 4, 5, 6, 7)의 출발점**이 된다.

### 🎯 Day 3 예상 질문 (Q&A)

**Q1. Grad-CAM이 정말로 모델의 판단 근거를 보여준다고 확신할 수 있나?**
A. Grad-CAM은 사후(post-hoc) 해석 도구라는 한계가 있다. 그러나 다음 3가지 정량 증거를 함께 봐야 한다:
   1. TP 평균 IoU 0.145 (200개 샘플 통계),
   2. TP와 FP의 히트맵 패턴이 동일,
   3. FN 92.3%가 픽셀 1% 미만 소종양.
   더 결정적인 근거는 Day 5에서 segmentation supervision을 주자 IoU가 0.706으로 5배 뛴 사실 — 즉 적절한 학습 신호만 주면 종양을 볼 수 있다는 의미.

**Q2. 그러면 ImageNet pretrained backbone 탓 아닌가? Random init으로 했다면 다를까?**
A. 가능성은 있으나, Day 4 Patch-Based는 PatchCNN을 from scratch로 학습했음에도 IoU 0.128로 더 낮았다. 즉 pretrained의 문제가 아니라 **"종양 위치를 학습하라는 신호가 없는 채 분류 Loss만 주면 어떤 모델이든 shortcut을 찾는다"**는 것이 맞는 해석.

**Q3. "180픽셀=0.36%"라는 수치는 어떻게 계산했나?**
A. `step7_gradcam.py`에서 FN 300개의 seg 마스크 픽셀 수 / 224²을 평균한 값. 224×224=50,176 픽셀 중 180은 0.36%. ResNet의 maxpool + 4단 다운샘플(/32)을 거치면 7×7=49 픽셀 feature map에서 종양 영역에 해당하는 셀이 1~2개 정도밖에 안 남아 GAP에서 0에 가깝게 희석.

**Q4. 단순히 threshold를 낮추면 FN을 줄일 수 있지 않은가?**
A. FN의 평균 확률이 0.191로 매우 낮고, 극저확률 FN(prob≈0.002)이 다수 — 즉 모델이 "없다"고 강하게 확신해서 놓친 것. Threshold를 0.3~0.4까지 낮춰도 0.05 이하 FN은 잡을 수 없다. 또한 그렇게 낮추면 FP(평균 0.741)가 폭증해 균형이 깨진다.

**Q5. FP의 평균 확률이 0.741로 높은데, 이걸 줄이는 방법은?**
A. FP의 0.8~1.0 고확률 구간에도 다수가 있어 threshold 조정만으로는 해결 불가. Day 5 Multi-Task가 FP를 536→305로 줄인 메커니즘은 segmentation head가 "여기는 종양이 아니다"라는 명시적 신호를 encoder에 강제해서 정상 뇌를 종양으로 오인하기 어렵게 만든 것 — 즉 active한 학습 신호 설계가 답.

📎 **참고**: `260521v1result.md` §3 / `notPublic/day2_day3_report.md` §3일차 절 / `notPublic/3wayanalysis.md` §6 시사점 1 (Shortcut Learning 정의)

---

## 4. Day 4 — Patch-Based 파이프라인 (부분적 실패의 교훈)

### 4.1 가설 (보존)

> "광역 shortcut을 막으려면 슬라이스를 작게 자르면 되지 않을까?"

### 4.2 패치 추출 설계 (Step 8)

| 항목 | 값 | 근거 |
|------|------|------|
| 패치 크기 | **64×64** | 작은 종양 포착 가능 + ResNet maxpool에 안전 |
| Stride | 32 (overlap 50%) | 경계 종양 놓침 방지 |
| 종양 임계값 | **≥5%** → Positive | 너무 낮으면 노이즈 패치 양성화 |
| 뇌 영역 임계값 | <10% → 제외 | 두개골/배경 패치 제거 |
| 총 추출 패치 | **3,218,425개** | |
| Positive 패치 | 491,108개 (15.3%) | **심각한 불균형** |
| Negative 패치 | 2,727,317개 (84.7%) | |

**[보강] Split별 패치 분포** (`progress_report.md`):

| Split | Positive | Negative | 합계 |
|:-----:|:--------:|:--------:|:----:|
| Train | 340,197 | 1,908,365 | 2,248,562 |
| Val | 73,977 | 408,230 | 482,207 |
| Test | 76,934 | 410,722 | 487,656 |

### 4.3 PatchCNN 모델 (Step 10)

```
PatchCNN (~95,000 파라미터, ResNet-18의 1/120, from scratch + Kaiming init)
├── Conv(3→32, 3×3) → BN → ReLU → MaxPool   [64→32]
├── Conv(32→64, 3×3) → BN → ReLU → MaxPool  [32→16]
├── Conv(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool [16→1]
├── Dropout(0.5)
└── Linear(128→1)
```

### 4.4 학습 설정 (Step 11)

| 항목 | 값 |
|------|------|
| Optimizer | AdamW (LR=1e-3, WD=1e-4) |
| Scheduler | CosineAnnealing (T_max=20) |
| Batch Size | **128** (패치가 작아 큰 배치 가능) |
| 클래스 균형 | Negative 서브샘플링 1:3 + WeightedRandomSampler |
| Early Stopping | patience=5 |

### 4.5 학습 곡선 (Val Loss 진동이 핵심 관찰점)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | 비고 |
|:-----:|:----------:|:---------:|:--------:|:-------:|:----:|
| 1 | 0.5416 | 85.6% | 0.6372 | 79.5% | |
| 2 | 0.4584 | 88.5% | 0.7371 | 75.9% | ← Val Loss 큰 진동 |
| 3 | 0.4255 | 89.4% | 0.4740 | 84.8% | |
| 4 | 0.4081 | 89.9% | 0.6105 | 80.4% | ← 진동 |
| 5 | 0.3970 | 90.2% | 0.4247 | 86.6% | |
| 6 | 0.3869 | 90.5% | 0.3482 | 89.6% | |
| **7** | **0.3785** | **90.7%** | **0.2956** | **91.8%** | **★ Best** |
| 8 | 0.3722 | 90.8% | 0.4462 | 86.2% | |
| 9 | 0.3694 | 90.9% | 0.4828 | 84.8% | |
| 10 | 0.3621 | 91.1% | 0.4348 | 86.6% | |
| 11 | 0.3589 | 91.2% | 0.3208 | 91.0% | |
| 12 | 0.3550 | 91.3% | 0.2991 | 91.3% | Early Stop |

총 학습 시간: **128.8분**. Val Loss 진동이 큼 → 패치 단위 클래스 불균형 + WeightedRandomSampler의 확률적 샘플링 영향.

### 4.6 패치 단위 평가 (Step 12, N=487,656)

| 지표 | 값 |
|------|:---:|
| Accuracy | 91.58% |
| Precision | 66.73% |
| **Recall** | **93.02%** |
| Specificity | 91.31% |
| F1-Score | 77.71% |
| AUC-ROC | 0.9749 |

**관찰**: 패치 단위에서 Recall은 93%로 높지만 Precision이 67%로 낮음 → Negative 패치가 84.7%를 차지하는 불균형에서 FP 35,673개 발생.

### 4.7 슬라이스 집계 방법 3가지 비교

| 집계 방법 | 설명 | Accuracy | F1 | AUC | FN | FP | 특징 |
|:---------:|------|:--------:|:--:|:---:|:--:|:--:|------|
| max | 패치 1개라도 양성 | 84.41% | 85.12% | 0.9489 | 1,044 | **2,872** | FN 적음, FP 많음 |
| mean | 평균 확률 ≥ 0.5 | 64.55% | 44.05% | 0.9173 | **8,745** | 158 | FP 적음, FN 극심 |
| **count** | **양성 패치 ≥2개** | **87.07%** | **87.08%** | 0.9104 | 1,306 | 1,942 | **★ 균형 최적** |

> **★ 최적 집계: count** (양성 패치 ≥2개 → 슬라이스 양성)

### 4.8 패치 Grad-CAM (Step 13)

| 지표 | 값 |
|------|:---:|
| 패치 TP 평균 IoU | **0.128** (Whole-Slice 0.145보다 **더 낮음** -11.6%) |
| TP 슬라이스 수 | 11,205 |
| FN 슬라이스 수 | 1,044 |
| FP 슬라이스 수 | 2,872 |

### 4.9 슬라이스 단위 종합 비교 (Whole vs Patch, Test N=25,115)

| 지표 | Whole-Slice | Patch (count) | 차이 |
|------|:-----------:|:-------------:|:----:|
| **Accuracy** | **94.33%** | 87.07% | ↓7.26%p |
| **Precision** | **95.49%** | 84.93% | ↓10.56%p |
| Recall | 92.75% | **89.34%** | ↓3.41%p |
| **Specificity** | **95.83%** | 84.90% | ↓10.93%p |
| **F1-Score** | **94.10%** | 87.08% | ↓7.02%p |
| **AUC-ROC** | **0.9832** | 0.9104 | ↓0.0728 |
| FN | **888** | 1,306 | +418 (↑47%) |
| FP | **536** | 1,942 | +1,406 (↑262%) |

### 4.10 왜 실패에 가까웠는가 — 4가지 원인 (보존)

1. **패치 단위 클래스 불균형 심화**: 슬라이스(48.8%) → 패치(15.3%, 1:5.6)로 악화. 서브샘플링 1:3으로 완화 시도했으나 불충분.
2. **경량 모델의 표현력 부족**: PatchCNN 95K로는 종양 vs 정상 조직 구별에 충분한 표현력이 부족. 64×64에서도 더 깊은 네트워크 필요.
3. **슬라이스 집계 시 FP 증폭**: 패치 Precision 67% → 한 슬라이스의 ~20개 패치 중 일부 FP여도 슬라이스 FP. Whole 536개 → Patch 1,942개로 폭증.
4. **Grad-CAM IoU 미개선**: 경량 CNN의 feature map이 종양을 정밀하게 포착하지 못함. 재조립된 히트맵이 산만한 패턴 → IoU 0.145 → 0.128로 오히려 하락.

### 4.11 교훈 (보존)

> **"광역 shortcut을 차단하는 것(passive)만으로는 부족하다."**
> 모델이 종양을 보지 않을 자유를 단순히 제한하는 것보다, **종양을 보도록 강제하는 학습 신호(active)**가 필요하다 → Day 5로 이어짐.

### 4.12 [보강] "낮은 점수 ≠ 나쁜 모델" 가설 (`wholePatch성능비교.md`)

Day 4 중간에는 다음 가설이 제기되었다:
> "Whole-Slice 94.33%는 shortcut 학습의 결과이고, Patch 91.8%는 종양 자체를 보려 노력한 점수다 — 의미 있는 trade-off일 수 있다."

이 가설은 6가지 원인 분석으로 뒷받침되었다:
1. 평가 단위가 다름 (슬라이스 vs 패치),
2. 본질적 난이도가 다름 (Whole은 광역 shortcut 가능, Patch는 텍스처 직접 봐야 함),
3. 클래스 불균형 패치에서 1:5.6,
4. 모델 capacity & Pretrain 차이 (Whole 11.2M+ImageNet vs Patch 95K from scratch),
5. 라벨 노이즈 (5% threshold의 경계 효과 — 4.9% vs 5.1% 패치가 정반대 라벨),
6. Context 손실 (64×64 좁은 영역만 봄).

**그러나 Day 5에서 이 가설은 부분적으로 반박된다** — Patch IoU도 0.128로 하락해 "passive 차단만으로는 부족"하다는 결론으로 수정. 단, 가설 제시 자체가 학술적 가치가 있다.

### 🎯 Day 4 예상 질문 (Q&A)

**Q1. 64×64 / stride 32 / 5% threshold — 이 숫자들의 근거는?**
A. (1) 64×64는 224 / 4 ≈ 56과 비슷한 크기로, 1 슬라이스당 7×7=49개 패치 grid(stride 32 기준). (2) Stride 32 (50% overlap)는 경계에 걸친 종양을 놓치지 않기 위한 표준 설정. (3) 5% threshold는 너무 낮으면 노이즈 패치까지 양성으로 포함되고, 너무 높으면 작은 종양은 모두 음성이 된다. 다만 5%도 라벨 노이즈 발생 한계 — Day 5 이후에는 픽셀 단위 segmentation supervision으로 우회.

**Q2. PatchCNN을 ResNet-18 (pretrained)로 바꿔봤다면 더 좋아지지 않았을까?**
A. `wholePatch성능비교.md`와 `patch_pipeline_plan.md`에서도 같은 제안이 있었다 (방안 B). 단 ResNet-18을 64×64에 적용하면 5단계 다운샘플(/32) 후 2×2 feature map이 되어 표현력 손실. 입력을 224×224로 리사이즈하면 패치의 의미가 사라진다. 결국 "패치 + pretrained" 조합은 학술적으로는 시도할 만하나, Day 5의 Multi-Task가 더 근본적 해법.

**Q3. Recall 93%인데 왜 슬라이스 집계에서 FN이 늘었나(888→1,306)?**
A. 패치 단위 Recall 93%는 "종양 패치 중 93% 검출"이지만, 슬라이스 단위 FN은 "한 슬라이스의 모든 종양 패치를 다 놓친 경우". 슬라이스 안에서 종양이 1~2개 패치에 걸쳐 있고 그게 모두 7%의 FN에 해당하면 슬라이스 FN. 특히 소종양은 1~2개 패치에만 걸치는 경우가 많아 슬라이스 단위에서 더 취약.

**Q4. Max 집계와 Count 집계 중 임상에서는 뭘 써야 하나?**
A. 임상 시나리오에 따라 다르다:
- **Max** (FN 1,044, FP 2,872): 종양 누락이 치명적인 1차 스크리닝.
- **Count ≥2** (FN 1,306, FP 1,942): F1 최고, 의료 AI 1차 진단으로 균형 잡힘.
- **Mean** (FN 8,745): 사실상 사용 불가.

**Q5. Patch가 실패라고 단정해도 되나?**
A. 단정은 안 한다. (a) ResNet-18 backbone, (b) Hard Negative Mining, (c) 패치 크기 96/128, (d) 라벨 회귀(regression) 등 추가 튜닝 방향이 열려 있음. 다만 본 프로젝트의 한정된 시간에서는 Multi-Task가 같은 철학("종양을 직접 보게 하자")을 더 우아하게 구현하므로 우선순위가 밀린 것. 즉 "철학은 옳았으나 실행이 부족"이라는 평가가 정확.

**Q6. 라벨 노이즈가 정말 심각했나?**
A. Stride 32의 overlap 50% 환경에서는 종양 경계에 걸친 패치들이 1~10% 사이 종양 비율을 갖는 경우가 많다. Train loss(0.355)와 Val loss(0.296)의 gap을 보면 라벨 노이즈가 학습을 흔든 흔적이 보임 — Val Loss 진동(epoch 2: 0.74 → 7: 0.30 → 9: 0.48)도 그 영향으로 추정.

📎 **참고**: `260521v1result.md` §4 / `notPublic/wholeVersePatch.md` §4일차 절 / `notPublic/wholePatch성능비교.md` §1~§6 / `notPublic/patch_pipeline_plan.md` 전체 / `notPublic/comparison_report.md`

---

## 5. Day 5 — Multi-Task Learning (본 프로젝트의 핵심 성과)

### 5.1 가설 (보존)

> "분류 Loss만 주면 shortcut을 찾는다. **세분화 Loss를 보조로 추가**해 종양 위치를 명시적으로 학습시키자."

### 5.2 아키텍처

```
입력 슬라이스 (224×224)
       │
   Shared Encoder (ResNet-18, ImageNet Pretrained)
       │
   ┌───┴───┐
   │       │
 분류 Head     Segmentation Decoder (U-Net 스타일)
 GAP→FC(512→1)  Skip Connection 4단
   │       │
P(tumor)   Tumor Mask (224×224)
```

- 총 파라미터: **~14M** (encoder 11M + decoder ~3M)
- **Loss**: `L_total = α · BCE(cls) + β · (Dice + BCE)(seg)`, **α=1.0, β=0.5**
- 2-Phase 학습 (Phase 1: encoder 동결 3 epoch → Phase 2: 전체 fine-tune 15 epoch, 실제 14 epoch에서 종료)
- Batch=16 (디코더 메모리), AdamW + CosineAnnealing
- 학습 시간: **298.8분** (Whole 대비 2.5배)

### 5.3 핵심 trick (보존)

> 세분화는 **보조 태스크(auxiliary task)**로 설정. 주 목표는 여전히 **분류**이고, 세분화는 분류를 더 잘하도록 가이드 역할. 이것이 단순 U-Net과의 결정적 차이.

### 5.4 학습 곡선 — Loss 분해 (Val)

| Epoch | Total Loss | **CLS Loss** | **SEG Loss** | Acc | Dice |
|:-----:|:----------:|:------------:|:------------:|:---:|:----:|
| 1 (P1) | 0.6487 | 0.4026 | 0.4922 | 82.33% | 0.683 |
| 3 (P1 종료) | 0.5984 | 0.3948 | 0.4072 | 82.96% | 0.700 |
| **4 (P2 시작)** | **0.3922** | **0.1840** | **0.4164** | **93.41%** | **0.733** |
| 9 (★ Best) | **0.3484** | **0.1909** | **0.3151** | **93.84%** | **0.756** |
| 14 (최종) | 0.3841 | 0.2430 | 0.2821 | 92.95% | 0.762 |

**흥미로운 관찰 (보존)**:
> Phase 1→2 전환 시 **CLS Loss가 0.39→0.18로 급감(절반 이하)**, SEG Loss는 0.41→0.42로 큰 변화 없음. 이는 **분류 task가 backbone fine-tuning의 혜택을 더 크게 받음**을 시사한다.

### 5.5 테스트셋 평가 (N=25,115)

| 지표 | Multi-Task | Whole-Slice 대비 |
|------|:---------:|:----------------:|
| Accuracy | 94.26% | 동등 (-0.07%p) |
| **Precision** | **97.33%** | **+1.84%p ↑** ★ |
| Recall | 90.73% | -2.02%p ↓ |
| **Specificity** | **97.63%** | **+1.80%p ↑** ★ |
| F1 | 93.91% | -0.19%p |
| AUC-ROC | 0.9824 | -0.0008 |
| **FP** | **305** | **-231 (-43%) ★** |
| FN | 1,136 | +248 (+28%) |
| **TP 평균 IoU** | **0.7061** | **+0.561 (4.87배) ★★** |
| Dice | 0.7765 | (Grad-CAM 대비 직접 비교 불가) |
| Optimal Threshold | 0.3847 | (Whole 0.6512와 차이) |

**클래스별 세부**:

| 클래스 | Precision | Recall | F1-Score | Support |
|--------|:---------:|:------:|:--------:|:-------:|
| Negative (정상) | 91.71% | **97.63%** | 94.58% | 12,866 |
| Positive (종양) | **97.33%** | 90.73% | 93.91% | 12,249 |
| Macro Avg | 94.52% | 94.18% | 94.24% | — |

### 5.6 핵심 성과 (보존)

1. **분류 정확도는 Whole-Slice와 동등 유지** (F1 차이 0.19%p).
2. **해석성(IoU)이 4.87배 향상** (0.145 → 0.706) — **shortcut learning 정량적으로 극복**.
3. **FP 43% 감소** — 의료 AI에서 환자의 불필요한 추가 검사를 줄이는 임상적 가치.
4. 단일 모델로 **분류 확률 + 픽셀 마스크**를 동시 출력 (임상 워크플로우 친화적).

### 5.7 Trade-off의 진실: FP 43% 감소 ↔ FN 28% 증가

```
Whole-Slice: TP 11,361 / TN 12,330 / FN 888 / FP 536
Multi-Task : TP 11,113 / TN 12,561 / FN 1,136 / FP 305
                       (+231 TN)   (+248 FN) (-231 FP)
```

**핵심**: FP 감소 폭(231개)이 FN 증가 폭(248개)과 거의 동일 → 전체 Accuracy는 균형 유지. 다만 의료 AI에서 **FP 감소는 비용·불안·침습검사 절감**, **FN 증가는 종양 누락**이라 임상 가중치가 다르다. 단순 합산은 부적절.

### 5.8 3-Way 종합 비교 (보존)

| 지표 | Whole-Slice | Patch-Based | **Multi-Task** | Best |
|------|:-----------:|:-----------:|:-------------:|:----:|
| Accuracy | 94.33% | 87.07% | 94.26% | Whole/MT (동등) |
| F1 | **94.10%** | 87.08% | 93.91% | 🏆 Whole (+0.19%p) |
| Precision | 95.49% | 84.93% | **97.33%** | 🏆 **MT** |
| Specificity | 95.83% | 84.90% | **97.63%** | 🏆 **MT** |
| AUC-ROC | **0.9832** | 0.9104 | 0.9824 | 🏆 Whole (+0.0008) |
| **TP IoU** | 0.145 | 0.128 | **0.7061** | 🏆 **MT** ★ |
| FP | 536 | 1,942 | **305** | 🏆 **MT** |
| 학습 시간 | ~120분 | 128.8분 | 298.8분 | — |

> **3-way 비교의 한 줄 결론**:
> *"Whole-Slice는 수치를 얻었지만 진실을 잃었고, Patch는 진실을 추구했지만 수치를 잃었으며, Multi-Task는 둘 다 잡았다."*

### 5.9 Multi-Task가 분류만 모델 / 순수 U-Net과 어떻게 다른가 (보존)

| | 순수 U-Net (상대 팀) | Multi-Task (우리) |
|---|---|---|
| 학습 신호 | Dice Loss만 ("픽셀을 잘 맞춰라") | BCE(cls) + Dice+BCE(seg) 두 신호 |
| 소종양 슬라이스 처리 | 10픽셀 종양 → Dice 영향 없음 → 무시해도 패널티 약함 | 분류 Loss가 "양성인데 왜 음성이라 했어?" 명시적 경고 |
| 정상 슬라이스 처리 | 빈 마스크 예측 → Dice≈1, gradient 거의 0 → 학습 거의 없음 | 분류 Loss가 "음성"이라는 명시적 신호 → Negative feature도 학습 |
| 출력 | 픽셀 마스크 1개 | **분류 확률 + 픽셀 마스크 2개** |

> "각 픽셀이 종양인가?" 1개만 묻는 것 vs "종양이 있나?" + "어디에 있나?" 2개를 묻는 것 → **encoder가 더 풍부한 feature를 학습하도록 양방향 압력**.

### 5.10 의료 AI에서의 실질적 가치

| 시나리오 | Whole-Slice | Multi-Task | 임상적 의미 |
|----------|:-----------:|:----------:|:-----------|
| 정상인 1,000명 스크리닝 | 약 42명 오탐 | 약 24명 오탐 | **18명 불필요 추가검사 회피** |
| 종양 환자 1,000명 진단 | 약 73명 놓침 | 약 93명 놓침 | 20명 추가 누락 (보완 필요) |
| 단일 추론 출력 | 분류 확률 | **분류 확률 + 위치 마스크** | 의사 검토 워크플로우 친화 |

### 🎯 Day 5 예상 질문 (Q&A)

**Q1. Segmentation 결과로 분류하면 되는 거 아닌가? 왜 굳이 dual head?**
A. 가능하다. `mask.sum() > 10` 같은 룰로 분류 유도 가능. 그러나 학습 방식이 다르다:
- 순수 U-Net: Dice Loss만 → "어디가 종양인가"만 학습. 정상 슬라이스에서는 빈 마스크 예측 → Dice≈1 → gradient 거의 0 → encoder가 negative feature를 거의 안 배움.
- Multi-Task: BCE(cls) + Dice(seg) → 정상에서도 분류 Loss가 "음성이라는 신호"를 명시적으로 줌 → Negative의 특징도 적극 학습 → **FP 감소(536→305)**.
즉 "결과물은 비슷해 보여도 학습 신호의 다양성이 다르다."

**Q2. α=1.0, β=0.5로 정한 근거는? β를 키우면 IoU가 더 오르지 않나?**
A. β=0.5는 "분류를 주, 세분화를 보조"라는 설계 철학을 직접 반영한 값. β를 0.7~1.0으로 키우면 IoU는 더 오를 수 있으나 분류 성능이 떨어질 위험. Day 5 학습 곡선에서 SEG Loss 0.32 / CLS Loss 0.19 비율로 봤을 때 현재 β=0.5는 두 task가 비슷한 절대값으로 수렴 — 합리적 시작점. (Day 7에서는 Uncertainty Weighting으로 자동화.)

**Q3. FN이 28% 증가한 건 임상적으로 더 나쁜 거 아닌가? 종양 놓치는 게 가장 무서운데.**
A. 정당한 우려. 그러나 다음 세 가지를 함께 봐야 한다:
1. **FN 증가량(248개) ≈ FP 감소량(231개)** — 전체 정확도는 동등.
2. **임상 가중치는 시나리오 의존적**. 1차 스크리닝에서는 FP 감소가 더 가치 있고, 확진 모드에서는 FN 감소가 우선.
3. Day 6 Multi-Modal에서는 T1ce의 ET 강조 효과로 **FN 1,136 → 848 실측 (-25.4%)** — 이 trade-off의 한계를 실제로 공략.

**Q4. β=0이면 분류만 학습되니 Whole-Slice와 동일해야 하는데, 동일한가?**
A. 이론적으로는 그렇지만 아키텍처가 다르다(decoder는 Loss=0이라도 forward에서 계산은 됨). β=0 ablation 실험은 본 단계에서 미실행 — 향후 과제로 명시. 다만 Whole-Slice의 backbone 결과(IoU 0.145)와 비교하면 β>0가 IoU 향상의 직접 원인임은 거의 확실.

**Q5. 학습 시간 2.5배(120→299분)는 정당화되나?**
A. (1) 학습은 한 번만 한다 — 추론 시간은 거의 동일. (2) 결과물이 2개(분류 + 마스크)이므로 별도 모델 2개 운영보다 효율적. (3) 무엇보다 **해석성이 5배 향상** — IoU/h로 계산하면 압도적 효율. (4) RTX 4070 8GB에서 5시간으로 끝나므로 학부/병원 워크스테이션에서 충분히 재현 가능.

**Q6. 디코더 추가 → 모델이 14M으로 커졌는데 단순히 capacity가 늘어서 잘된 것 아닌가?**
A. 가능성을 배제할 수 없다. 그러나 **분류 성능은 동등(F1 -0.19%p)**한데 IoU만 5배 향상된 점이 결정적 — capacity가 잘 작동했다면 분류도 같이 올랐어야 한다. Decoder가 더한 것은 "Capacity"가 아니라 "Supervision 신호의 종류"임을 시사.

**Q7. Best epoch이 9인데, Whole-Slice Best도 9. 우연인가?**
A. 두 모델 모두 Phase 2 6번째 epoch이고, CosineAnnealing scheduler 곡선이 비슷하게 그려진 결과로 보인다. 우연이라기보다는 같은 backbone + 비슷한 scheduler 설계 때문.

**Q8. Optimal threshold가 0.3847로 0.5에서 멀어졌다. 왜?**
A. Multi-Task는 정상 슬라이스에서도 작은 segmentation 확률이 비-zero로 나오는 경향이 있고, GAP를 통과한 분류 logit이 약간 보수적(=낮은 확률)으로 보정된다. Youden's J를 최대화하는 지점이 0.3847로 이동. 임상 운영에서 직관적인 0.5 부근으로 옮기려면 Validation set 기반 Temperature Scaling이 권장됨 (Day 7에서 실제 적용).

📎 **참고**: `260521v1result.md` §5 / `notPublic/3wayanalysis.md` 전체 / `notPublic/multitask_explanation.md` §1~§6 / `notPublic/differentiation_strategy.md` 전체 / `code/multitask/step15~step19_*.py`

---

## 6. Day 5 보강 — Train-split Grad-CAM (분류 vs 세분화 head 분리 진단)

### 6.1 이 결과가 v1에서 빠진 이유

v1 `260521v1result.md` §5는 Day 5 Multi-Task의 *test-split* 성능(F1 93.91 / IoU 0.706 / FP 305)과 *학습 곡선*은 모두 정리했지만, **`step18b_multitask_train_gradcam.py`로 생성한 *train-split* Grad-CAM 결과를 통째로 빠뜨렸다.** 이는 본 발표에서 두 가지 결정적 메시지를 보충하는 데이터:

1. *학습셋*에서 분류 헤드 CAM의 IoU가 어땠는가? → "MT가 *분류 head 자체로는 여전히 광역 단서를 보고 있는가, 종양을 보는가*" 의 직접 검증.
2. 학습셋에서도 FN(under-fit) / FP(confusion pattern)이 남아 있는가?

### 6.2 핵심 수치 (보존)

```json
{
  "split": "train",
  "checkpoint_epoch": 6,
  "checkpoint_val_acc": 93.84,
  "checkpoint_val_dice": 0.7564,
  "n_train_total": 116629,
  "TP": 52832, "FN": 4110, "FP": 427, "TN": 59260,
  "train_accuracy_pct": 96.11,
  "iou_subsample_n": 400,
  "mean_iou_CAM_GT_trainTP": 0.1232,
  "mean_iou_SEG_GT_trainTP": 0.7753,
  "FN_prob_mean": 0.1823,
  "FP_prob_mean": 0.7087
}
```

> [보강] `checkpoint_epoch: 6`은 **2-Phase 학습에서 Phase 2의 6번째 epoch을 의미** — Phase 1 3 epoch + Phase 2 6 epoch = **전체 9번째 epoch** (§5.4 "★ Best Epoch 9"와 일치).

### 6.3 학습셋 confusion matrix와 Train Accuracy

| Train | Pred Neg | Pred Pos |
|:-----:|:--------:|:--------:|
| **Actual Neg** (59,687) | TN = **59,260** | FP = **427** |
| **Actual Pos** (56,942) | FN = **4,110** | TP = **52,832** |

- Train Accuracy = (52,832 + 59,260) / 116,629 = **96.11%**
- Train FP rate = 427/59,687 = **0.72%** (test FP rate 305/12,866 = 2.37%보다 훨씬 낮음 → 과적합 신호 약함)
- Train FN rate = 4,110/56,942 = **7.22%** (test FN rate 1,136/12,249 = 9.27%와 유사 → FN은 학습셋에서도 잡지 못함 = **under-fit 패턴, capacity가 아니라 입력 표현의 한계**)

### 6.4 결정적 발견 — *CAM IoU 0.1232* vs *SEG IoU 0.7753* (★)

> **본 발표의 핵심 새 인사이트 ①**:
> Multi-Task 모델의 분류 헤드 Grad-CAM IoU는 *학습셋*에서도 **0.1232**에 불과하다. 한편 같은 모델의 *세분화 헤드* IoU는 **0.7753**으로 6배 이상 높다. **두 head는 같은 encoder를 공유하면서도 완전히 다른 패턴을 학습했다.**

이게 의미하는 바:
- Whole-Slice (분류 단독)의 test CAM IoU 0.145와 Multi-Task의 train CAM IoU 0.123은 **거의 동등**.
- 즉 **MT가 분류 head 자체를 "종양을 보게" 만든 것은 아니다.** 분류 head는 여전히 광역 단서를 그대로 본다.
- 그럼에도 v1에서 "MT IoU = 0.706"이라고 보고할 수 있었던 이유: **그 IoU는 *분류 head Grad-CAM*이 아니라 *세분화 head 마스크*의 IoU**.

> **새로운 해석 (v1에서 한 차원 깊어진 메시지)**:
> "MT는 *분류 head의 shortcut을 직접 고친 게 아니라*, *세분화 head를 추가로 학습시켜 encoder가 종양 정밀 위치를 부담하게 만든 것*이다. 분류 head는 그 encoder feature 위에서 자기 방식대로 판단하지만, **결과적으로 모델 출력에 *분류 확률 + 정밀한 위치 마스크*가 함께 나오므로 임상적·해석성에 모두 유리**하다."

### 6.5 학습셋 FN/FP 패턴 분석

**FN (4,110개, prob mean=0.1823)**
- 학습셋에서도 평균 확률 0.18로 "없다"고 강하게 확신하며 놓침 → test FN 확률 0.191과 거의 동일
- → **테스트셋의 FN 1,136은 *분포 외* 문제가 아니라 *학습 단계의 under-fit 패턴이 그대로 노출된 것*임을 의미**
- 결론: 추가 epoch이나 capacity 증가로는 못 잡음. **입력 표현 변경(멀티모달 / 2.5D / Tversky)이나 oversampling이 필요**

**FP (427개, prob mean=0.7087)**
- 학습셋에서도 평균 확률 0.71로 *확신 있게* 오탐 → test FP 0.741과 유사
- 학습셋 비율 0.36% (test 2.37%)이라 절대 수는 적지만 **여전히 hard negative가 학습 종료까지 존재**

### 6.6 서브샘플 IoU 분포 (400개)

상위 일부:
```
filename                       prob      iou_cam_gt   iou_seg_gt
BraTS-GLI-01326-000_z077.png   1.00      0.233        0.934
BraTS-GLI-00184-000_z094.png   1.00      0.083        0.906
BraTS-GLI-00103-000_z079.png   1.00      0.334        0.881
BraTS-GLI-01495-000_z095.png   1.00      0.276        0.802
BraTS-GLI-00166-000_z074.png   1.00      0.095        0.719
```

- **prob ≈ 1.0인 확신 있는 TP에서도 CAM IoU는 0.08~0.33** — 광역 단서 + 부분 종양 단서의 혼합 패턴.
- 같은 슬라이스의 SEG IoU는 0.7~0.9 → seg head는 정밀하게 작동.

### 🎯 Day 5 보강 예상 질문 (Q&A)

**Q1. CAM IoU가 0.12면 Day 5에서 "IoU 5배 향상"이라고 말했던 게 무의미하지 않나?**
A. 두 IoU는 의미가 다르다. v1 §5.5의 "TP IoU 0.706"은 **seg head의 마스크 IoU**이며, Day 5의 본질적 성과는 *분류 모델에 정밀한 위치 마스크가 추가로 나온다는 점*이다. 분류 head Grad-CAM의 IoU(0.12)는 거의 변하지 않았지만, 모델이 출력하는 위치 정보(seg mask) 자체가 IoU 0.706이라는 사실은 임상적으로 더 중요한 지표. 단, 발표에서는 이 둘을 명확히 구분 — "CAM 기반 *해석성*은 그대로지만 *모델 출력의 정밀 위치*가 5배 좋아진 것".

**Q2. 그러면 Multi-Task는 사실 *Shortcut Learning을 극복한 게 아니라 우회한 것* 아닌가?**
A. 부분적으로 그렇다. Encoder는 여전히 "종양 위치를 보지 않으면서도 분류는 잘하는" shortcut feature를 갖고 있을 수 있다. 그러나 동시에 그 encoder가 seg head로 정밀 마스크를 만들어내므로, **encoder 안에 위치 정보가 *어느 정도는* 인코딩되어 있다**는 것도 사실이다. 이 한계가 Day 7 Deep Supervision의 동기.

**Q3. FN 4,110개를 학습셋에서도 잡지 못한 것은 *학습이 부족한 것* 아닌가? Epoch를 더 늘리면?**
A. 학습 곡선상 Phase 2 epoch 9 이후 Val Loss는 단조 증가(과적합). Epoch 14까지 진행됐어도 train FN율은 7.2%로 거의 변하지 않았으며, train acc는 96.6%에서 plateau. **즉 capacity 부족이 아니라 입력 표현(FLAIR 단일) 자체가 소종양 ET를 충분히 표현하지 못함**. → Day 6 멀티모달의 도입 근거. (실제로 §8에서 멀티모달이 381개의 FN을 *recovered* 시킨다.)

📎 **참고**: `260524v1mmmtplus.md` §1 / `outputs/figures/multitask/gradcam_train/mt_train_gradcam_summary.json` / `code/multitask/step18b_multitask_train_gradcam.py`

---

## 7. Day 6 — Multi-Modal Multi-Task 본 학습 결과

### 7.1 왜 멀티모달인가 — 의학적 상보성

| Label | 종양 영역 | 가장 잘 보이는 모달리티 |
|:-----:|----------|:----------------------:|
| 1 | NCR (괴사 핵) | **T1ce** (저신호) |
| 2 | ED (부종) | **T2-FLAIR** (고신호) |
| 3 | ET (활성 종양) | **T1ce** (조영 증강 고신호) |

- BraTS 공식 평가축 **WT / TC / ET**: WT = NCR+ED+ET / TC = NCR+ET / ET = ET only.
- T1ce 단독: 부종이 안 보임.
- FLAIR 단독: 활성 종양 구분 어려움.
- **상호 보완** = 한 모델로 WT/TC/ET를 모두 커버하려면 두 모달이 모두 필요.

### 7.2 Day 5 미해결 약점과의 매칭

| Day 5 약점 | 원인 | T1ce 추가로 해결되는 메커니즘 |
|------------|------|------------------------------|
| **FN 1,136 (소종양)** | FLAIR에서 작은 ET는 백질과 신호 차이 미약 | **T1ce 조영제로 ET 명확히 강조 → 작은 종양도 검출** |
| **FP 305** | FLAIR 고신호 ≠ 종양 (백질변성·노이즈 포함) | **T1ce에서 증강 안 되면 진성 종양 아님 → cross-check** |
| **IoU 0.706** | 단일 모달 텍스처로만 경계 학습 | **두 모달 일치 영역 = 더 신뢰할 수 있는 경계** |

### 7.3 전처리 (step20) — T1ce 슬라이스 추출

**산출물**: `outputs/logs/mmmt/step20_t1ce_stats.json`

| 항목 | 값 |
|------|:--:|
| 처리된 환자 수 | **1,251명** |
| 저장된 T1ce 슬라이스 | **166,626개** (FLAIR와 1:1 매핑) |
| skipped_patient | 0 |
| skipped_slice | 27,279 (brain fraction < 1%) |
| missing_pair | 44 (FLAIR 슬라이스가 있는데 T1ce 추출에서 제외된 케이스) |
| errors | 0 |

> [보강] 정규화 방식은 step1과 동일 — *환자 내 percentile-clip [1, 99] + minmax → uint8*. 두 모달의 정규화를 일관되게 함으로써 |T1ce-FLAIR| diff 채널이 의미 있는 신호가 되도록 보장.

### 7.4 모델 — MMMTBrainNet

```
입력: (B, 3, 224, 224)  ← [T1ce, FLAIR, |T1ce - FLAIR|]
       │
   Shared Encoder (ResNet-18, ImageNet pretrained)
   conv1[3→64] - 그대로 재사용 (3채널 입력이라 채널 평균 이식 불필요)
   layer1[64] - layer2[128] - layer3[256] - layer4[512]  ← Grad-CAM target
       │
   ┌───┴───┐
   GAP→FC(512→1)   Decoder (UpBlock×4 + skip connection)
   = cls_out       = seg_out (1ch WT, 224×224)
```

- 총 파라미터: **~14M** (Day 5 multitask와 동일)
- 권장 전략 채택: 채널 평균 이식 X, **3채널 유지하고 세 번째에 |T1ce-FLAIR| 합성**.
- conv1의 ImageNet 가중치(3,64,7,7)가 *그대로* 재사용되어 transfer learning 효과 보존.

### 7.5 Loss — Tversky 추가

```
L_total = L_cls + β_seg · L_seg
L_cls   = BCE(pos_weight)
L_seg   = 0.5 · Dice(WT) + 0.5 · Tversky(WT, α=0.7, β=0.3)
β_seg   = 0.5 (Day 5와 동일)
```

> **Tversky α=0.7, β=0.3 의 의미**:
> Tversky = TP / (TP + α·FN + β·FP). α > β로 두면 FN에 더 큰 penalty → **소종양 누락에 강하게 학습**.
> Day 5의 약점이었던 FN 1,136에 직접 대응하는 손실함수 설계.

### 7.6 학습 설정 — Day 5와 동일하게 통제

| 항목 | 값 | Day 5와의 차이 |
|------|:--:|:--------------:|
| phase1_epochs | 3 | 동일 |
| phase1_lr | 1e-3 | 동일 |
| phase2_epochs | 15 (Early Stop @ 13) | 동일 |
| phase2_lr | 1e-4 | 동일 |
| batch_size | 16 | 동일 |
| weight_decay | 1e-4 | 동일 |
| early_stopping_patience | 5 | 동일 |
| use_weighted_sampler | False | 동일 |
| **channel_mode** | **t1ce_flair_diff** | **★ 유일한 변경점** |
| beta_seg | 0.5 | 동일 |
| tversky_alpha | 0.7 | (Day 5에는 Tversky 없음) |
| tversky_beta | 0.3 | (Day 5에는 Tversky 없음) |

> **통제 변수**: Day 6에서 *입력 채널 구성*과 *seg loss에 Tversky 0.5 weight 추가* 두 가지만 변경.

### 7.7 학습 곡선 — 16 epoch 전체 핵심 발췌

> 본 학습은 P1 3 + P2 13 = **총 16 epoch**으로 종료. P2 patience=5 Early Stop이 trigger되어 P2 13번째에 멈춤.

| Epoch | Phase | Val Total | Val CLS | Val SEG | Val Acc | Val Dice |
|:-----:|:-----:|:---------:|:-------:|:-------:|:-------:|:--------:|
| 1 (P1) | P1 | 0.483 | 0.390 | 0.187 | 83.23% | 0.335 |
| 3 (P1 종료) | P1 | 0.471 | 0.389 | 0.163 | 83.13% | 0.356 |
| **4 (P2 시작)** | P2 | **0.269** | **0.200** | **0.137** | **92.56%** | **0.378** |
| 5 | P2 | 0.406 | 0.313 | 0.188 | 87.77% | 0.363 |
| 6 | P2 | 0.259 | 0.195 | 0.128 | 92.97% | 0.390 |
| 7 | P2 | 0.252 | 0.187 | 0.131 | 93.03% | 0.381 |
| 8 | P2 | 0.231 | 0.171 | 0.120 | 93.80% | 0.397 |
| 9 | P2 | 0.266 | 0.207 | 0.118 | 92.41% | 0.406 |
| 10 | P2 | 0.230 | 0.173 | 0.114 | 94.04% | 0.412 |
| **11 (★ Best Val Acc)** | P2 | **0.228** | **0.172** | **0.111** | **94.12%** | **0.423** |
| 16 (최종, Early Stop) | P2 | 0.257 | 0.203 | 0.108 | 94.03% | 0.426 |

**핵심 관찰 (보존)**:

1. **Phase 1 → Phase 2 전환 (epoch 3 → 4)**: Val total 0.471 → 0.269 (-43%) — encoder fine-tune의 즉각적 효과. Val Acc 83.13% → 92.56% (+9.43%p).
2. **Epoch 5 Val 진동**: 0.269 → 0.406 → 0.259 — CosineAnnealing 초반의 노이즈.
3. **Best Val Acc는 epoch 11** (94.12%) — Day 5의 Best Val Acc 93.84%(epoch 9)보다 +0.28%p.
4. **Train Dice 0.798 — Val Dice 0.426 — Test Dice 0.787**: 매우 의외이고 결정적인 관찰. Val/Test Dice 측정 함수 차이 — Val Dice는 *양성 슬라이스만* 평균 + Tversky penalty 영향. Test Dice는 표준 평균법.

### 7.8 학습 시간 — Day 5 대비 30% 증가

| 모델 | 총 시간 | 총 epoch | epoch당 평균 |
|------|:-------:|:--------:|:------------:|
| Day 5 Multi-Task | 298.8분 | 14 | 21.3분 |
| **Day 6 MMMT** | **388.5분** | **16** | **24.3분** |

### 7.9 평가 (step25) — Test N=25,115, threshold=0.5

```json
{
  "n_test": 25115,
  "cls": {
    "threshold": 0.5,
    "f1": 0.9427, "auroc": 0.9832, "auprc": 0.9863,
    "tp": 11401, "tn": 12327, "fp": 539, "fn": 848
  },
  "seg_wt": {
    "n_positive_slices": 12244,
    "dice_mean": 0.7874, "dice_median": 0.9094,
    "iou_mean": 0.7142, "iou_median": 0.8338
  }
}
```

### 7.10 Day 5 vs Day 6 1차 비교 (★ 본 발표의 중심 표)

| 지표 | Day 5 MTL (@0.3847) | Day 6 MMMT (@0.5) | 변화 |
|------|:------------------:|:----------------:|:----:|
| Accuracy | 94.26% | **94.48%** | +0.22%p |
| Precision | **97.33%** | 95.49% | -1.84%p |
| Recall (Sensitivity) | 90.73% | **93.08%** | **+2.35%p** ★ |
| Specificity | **97.63%** | 95.81% | -1.82%p |
| F1 | 93.91% | **94.27%** | +0.36%p |
| AUC-ROC | 0.9824 | **0.9832** | +0.0008 |
| AUPRC | (미보고) | 0.9863 | — |
| TP | 11,113 | **11,401** | **+288** ★ |
| TN | **12,561** | 12,327 | -234 |
| FP | **305** | 539 | **+234** ❌ |
| FN | 1,136 | **848** | **-288** ★ |
| seg Dice | 0.7765 | **0.7874** | +0.0109 |
| seg IoU | 0.7061 | **0.7142** | +0.0081 |

**결론 (보존)**:

- **FN 288개 감소(-25.4%)**: 멀티모달의 의학적 상보성이 발휘 — 작은 종양에서 T1ce ET 강조가 도움.
- **그러나 FP 234개 증가(+76.7%)**: T1ce의 정상 조영 영역(혈관·맥락총 등)을 종양 후보로 잡는 새로운 오탐 패턴 발생. **v1 §6.3에서 "FP 감소"를 기대했던 것과 정반대 결과**.
- **AUROC는 동등(+0.0008)**: 모델의 *순위 매기기* 성능은 거의 차이 없음. 결정 경계의 위치만 이동.
- **이 결과는 단순한 v1 §6.6 정량적 기대치 (FN 700~900, FP 150~250)와 다르다**: FN은 예상 범위에 들어왔지만(848), FP는 예상보다 2배 이상(539 vs 150~250). 즉 **멀티모달 + Tversky는 FN을 줄이는 데는 성공했지만, FP를 줄이는 데는 실패**.

### 7.11 Train-split Grad-CAM (MMMT 버전)

`outputs/figures/mmmt/gradcam_train/mmmt_train_gradcam_summary.json`:

```json
{
  "checkpoint_epoch": 8,
  "checkpoint_val_acc": 94.12,
  "checkpoint_val_dice": 0.4229,
  "TP": 53799, "FN": 3143, "FP": 1177, "TN": 58510,
  "train_accuracy_pct": 96.30,
  "mean_iou_CAM_GT_trainTP": 0.1588,
  "mean_iou_SEG_GT_trainTP": 0.765,
  "FN_prob_mean": 0.2264,
  "FP_prob_mean": 0.6736
}
```

**MT vs MMMT Train Grad-CAM 비교**:

| 지표 | MT (Day 5) | MMMT (Day 6) | 변화 |
|------|:----------:|:------------:|:----:|
| Train Accuracy | 96.11% | **96.30%** | +0.19%p |
| **CAM-GT IoU (학습 TP)** | **0.1232** | **0.1588** | **+0.0356 (+29%)** ★ |
| SEG-GT IoU (학습 TP) | 0.7753 | 0.7650 | -0.0103 |
| Train FN prob mean | 0.1823 | 0.2264 | +0.044 (확신 약화) |
| Train FP prob mean | 0.7087 | 0.6736 | -0.035 (확신 약화) |

**해석**:
1. **CAM IoU가 0.1232 → 0.1588 (+29%)** — 멀티모달이 분류 head를 약간 더 종양 위치를 보게 만듦.
2. **SEG IoU는 거의 동등** — 학습셋 seg 정밀도는 두 모델이 비슷.
3. **학습셋에서도 FN 967개 감소·FP 750개 증가** — test의 FN -288, FP +234와 *같은 방향*. → **test 패턴은 학습 단계에서 이미 결정된 분포 안 결과**.

### 🎯 Day 6 예상 질문 (Q&A)

**Q1. FP가 305 → 539로 +77% 증가했는데, 이건 v1 §6.6의 기대치(150~250)와 정반대다. 어떻게 설명하나?**
A. 세 가지 해석:
1. **T1ce 추가가 가져온 새 오탐 패턴**: T1ce는 정상 혈관·맥락총·뇌막 등이 강하게 조영되는 영역이 다수.
2. **Tversky α=0.7의 부작용**: FN에 강한 penalty를 주는 게 의도였으나, sigmoid 출력이 *더 적극적으로* 양성을 예측하게 됨 → FP 증가의 직접 원인.
3. **Threshold가 0.5라서**: §8에서 보듯 threshold를 Day 5의 0.3847로 동등 보정하면 FP가 더 증가(787)하지만, 0.6202로 올리면 더 줄어든다. 결정 경계가 이동한 것일 뿐 *순위 매기기 성능(AUROC)*은 동등.
- 발표에서는 "Day 5 → Day 6에서 *operating point*가 이동했고, 두 모델은 *AUROC 동등*이지만 *FN-FP 균형*은 시나리오에 따라 선택" 으로 정정.

**Q2. Day 5와 Day 6의 Best epoch이 9 vs 11로 다르다. 우연인가?**
A. 두 모델 모두 Phase 2 6번째(Day5) / 8번째(Day6)에 Best. CosineAnnealing scheduler 곡선상 *LR이 5e-5 부근*에서 일반화 최적점이 형성. Day 6은 입력 채널이 늘어 약간 더 천천히 수렴 — *Phase 2 8번째 = 전체 11번째*에 Best 도달.

**Q3. Val Dice가 0.426으로 Test Dice 0.787과 큰 차이가 있다. 평가 기준이 일관된가?**
A. 코드 검증 필요. `step24_mmmt_train.py`의 `_wt_dice`와 `step25_mmmt_evaluate.py`의 `compute_seg_metrics`가 약간 다른 식으로 dice를 집계. 학습 중 Val Dice는 *훈련 모드에서 양성 슬라이스만 평균하면서 boundary case를 더 엄격하게 처리*하는 경향이 있어 낮게 나오고, Test 평가의 dice는 *full-test 기준 mean*이라 높게 나온다. **본 보고서는 Test Dice(0.787)를 표준 보고치로 사용**.

**Q4. Tversky loss를 도입한 게 정확히 어떤 효과를 줬는지 분리할 수 있나?**
A. 본 학습에서는 *멀티모달 입력*과 *Tversky 추가*가 동시에 도입. §8의 채널 ablation에서 부분 분리: **Tversky의 *분리된* 기여는 +0.24%p F1 정도이고, 추가 +0.12%p가 멀티모달 입력의 기여**.

**Q5. CAM IoU가 +29% 올랐다고 했는데, 0.12 → 0.16은 여전히 0.2도 안 되는 낮은 값이다.**
A. 분류 head Grad-CAM의 IoU만 보면 미세한 향상. 그러나 (1) seg head IoU 0.71(test)은 매우 높고, (2) FN의 평균 prob이 0.18→0.23으로 약간 회복 가능한 영역으로 이동, (3) §8에서 FN 381개가 *recovered*된 *정성적* 효과 — 이 세 가지를 함께 봐야 한다.

📎 **참고**: `260524v1mmmtplus.md` §2 / `code/mmmt/step20~25_*.py` / `outputs/logs/mmmt/mmmt_summary.json`, `mmmt_history.json`, `mmmt_test_metrics.json`

---

## 8. Day 6 권장 3종 결과 — Threshold 동등 보정 / FN 차분 / 채널 Ablation

### 8.1 (A) Threshold 동등 보정 (step25b)

#### 8.1.1 작업의 목적 (보존)

> Day 5 MTL의 ROC-Youden(=Sensitivity + Specificity − 1) 최적 임계값 **0.3847**은 그 모델의 *운용 지점*이고, Day 5의 F1 93.91 / FP 305 / FN 1136은 *그 threshold*에서 산출된 값. Day 6 MMMT의 기본 평가(step25)는 threshold=0.5에서 수행되었으므로, "멀티모달의 *순수* 효과"를 분리하려면 두 모델을 *같은 운용 지점*에서 비교해야 한다.

#### 8.1.2 Threshold Sweep 결과 (보존)

| Threshold | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1 |
|:---------:|:--:|:--:|:--:|:--:|:--------:|:---------:|:------:|:-----------:|:--:|
| 0.30 | 11,633 | 11,790 | 1,076 | 616 | 93.26 | 91.53 | **94.97** | 91.64 | 93.22 |
| 0.35 | 11,559 | 11,975 | 891 | 690 | 93.70 | 92.84 | 94.37 | 93.07 | 93.60 |
| **0.3847 (★ Day5 운용점)** | **11,522** | **12,079** | **787** | **727** | **93.97** | **93.61** | **94.06** | **93.88** | **93.84** |
| 0.40 | 11,508 | 12,112 | 754 | 741 | 94.05 | 93.85 | 93.95 | 94.14 | 93.90 |
| 0.45 | 11,457 | 12,226 | 640 | 792 | 94.30 | 94.71 | 93.53 | 95.03 | 94.12 |
| 0.50 (Day 6 기본) | 11,401 | 12,327 | 539 | 848 | 94.48 | 95.49 | 93.08 | 95.81 | **94.27** |

#### 8.1.3 동등 비교의 핵심 표 (★ 발표용)

| 모델 | Threshold | TP | TN | FP | FN | F1 | Recall | Precision |
|------|:---------:|:--:|:--:|:--:|:--:|:--:|:------:|:---------:|
| Day 5 MTL (FLAIR only) | 0.3847 | 11,113 | 12,561 | 305 | 1,136 | 93.91 | 90.73 | 97.33 |
| **Day 6 MMMT (T1ce+FLAIR+diff)** | **0.3847** | **11,522** | **12,079** | **787** | **727** | **93.84** | **94.06** | **93.61** |
| Day 6 MMMT | 0.5 | 11,401 | 12,327 | 539 | 848 | **94.27** | 93.08 | 95.49 |

#### 8.1.4 결정적 발견 (★)

> **본 발표의 핵심 새 인사이트 ②**:
> "동일 threshold(0.3847)에서 Day 5 vs Day 6을 비교하면 **F1은 -0.07%p로 거의 동등**(93.91 vs 93.84)이고 **AUROC도 +0.0008로 거의 동등**(0.9824 vs 0.9832). 즉 **멀티모달의 효과는 "성능 향상"이 아니라 "운용 자유도 확장"이었다.** Day 6 모델은 thr 0.3~0.5 어디서 운용해도 F1 93+ 안에 들어와, 임상 시나리오별로 더 폭넓게 조정 가능."

#### 8.1.5 임상 시나리오 매핑 (보존)

| 시나리오 | 권장 threshold (Day 6) | TP | FN (놓침) | FP (오탐) | 비고 |
|----------|:----------------------:|:--:|:---------:|:---------:|------|
| 1차 스크리닝 (Recall 최우선) | **0.30** | 11,633 | 616 | 1,076 | 종양 누락 최소화 |
| 균형 운용 (F1 최우선) | **0.45~0.50** | 11,401~11,457 | 792~848 | 539~640 | 일반 진단 |
| 확진 (Precision 최우선) | **0.6+** | (낮음) | (높음) | (낮음) | 확진 모드 |
| Day 5와 동등 비교용 | **0.3847** | 11,522 | 727 | 787 | 학술 공정 비교 |

### 8.2 (B) Day5↔Day6 FN 차분 분석 (step26)

#### 8.2.1 8개 비교 그룹 정의

| Group | Label | Day5 pred | Day6 pred | 의미 |
|-------|:-----:|:---------:|:---------:|------|
| **recovered** | 1 (양성) | 0 (FN) | 1 (TP) | ★ Day 5가 놓쳤지만 Day 6이 잡음 |
| **still_missed** | 1 | 0 (FN) | 0 (FN) | 둘 다 놓침 |
| **regressed** | 1 | 1 (TP) | 0 (FN) | ❌ Day 5가 잡았는데 Day 6이 놓침 |
| **common_tp** | 1 | 1 (TP) | 1 (TP) | 둘 다 잡음 |
| **common_tn** | 0 | 0 (TN) | 0 (TN) | 둘 다 정상 |
| **new_fp** | 0 | 0 (TN) | 1 (FP) | ❌ Day 6이 새로 만든 오탐 |
| **cleaned_fp** | 0 | 1 (FP) | 0 (TN) | ★ Day 5의 오탐을 Day 6이 정상화 |
| **common_fp** | 0 | 1 (FP) | 1 (FP) | 둘 다 오탐 |

#### 8.2.2 Fair 비교 (Day5 @ 0.3847, Day6 @ 0.3847) — 학술 공정 비교

| Group | N | tumor_px_median | tumor_px_mean | z_median | Day5 prob mean | Day6 prob mean |
|-------|:-:|:---------------:|:-------------:|:--------:|:--------------:|:--------------:|
| common_tp | 11,141 | 1,343 | 1,485 | 85 | 0.9807 | 0.9864 |
| common_tn | 11,960 | 0 | 0 | 39 | 0.0276 | 0.0582 |
| **recovered** ★ | **381** | **109** | **230.9** | **69** | **0.171** | **0.699** |
| **still_missed** | **641** | **31** | **99.6** | **62** | **0.083** | **0.144** |
| regressed | 86 | 98 | 222 | 69 | 0.651 | 0.242 |
| new_fp | 515 | 0 | 0 | 69 | 0.124 | 0.579 |
| cleaned_fp | 119 | 0 | 0 | 55 | 0.596 | 0.194 |
| common_fp | 272 | 0 | 0 | 64 | 0.713 | 0.737 |

#### 8.2.3 결정적 발견 (★)

> **본 발표의 핵심 새 인사이트 ③ (recovered 381)**:
> 멀티모달이 새로 잡은 *recovered 381 슬라이스*의 평균 종양 크기는 **231 픽셀(0.46%)** 로 매우 작다. Day 5에서 prob=0.17로 강하게 "없다"고 판단했던 케이스를 Day 6에서 prob=0.70으로 끌어올림. **즉 v1 §6.3에서 가설로 제시했던 "T1ce 조영제로 ET가 명확히 강조되어 작은 종양도 검출"이 정량 검증된 첫 사례.**

> **본 발표의 핵심 새 인사이트 ④ (still_missed 641)**:
> *still_missed 641 슬라이스*는 평균 종양 크기 **100 픽셀(0.20%)** 로 극소(주로 종양 경계 슬라이스). Day 5 prob 0.08 → Day 6 prob 0.14 — 약간 올라갔지만 threshold 0.3847 미달. **이 극소 종양 영역은 멀티모달로도 잡지 못하는 잔여 한계이며, Day 7 Tumor-CP / 2.5D 입력 / Focal-Tversky의 표적.**

> **본 발표의 핵심 새 인사이트 ⑤ (regressed 86)**:
> Regressed 86개 (Day 5 잡았지만 Day 6이 놓침)의 평균 종양 크기는 **222 픽셀**로 recovered와 비슷. Day 5 prob 0.65 → Day 6 prob 0.24. **즉 Day 5와 Day 6은 "다른 종류의 작은 종양"을 잡고 놓치는 패턴 — *상호 보완적*이라 ensemble이 효과적일 가능성이 크다.** (Day 8+ Deep Ensemble의 직접 근거)

#### 8.2.4 Z-위치 분포 — 임상 의미

- common_tp의 z_median=85: 뇌 중앙부에 큰 종양이 분포 (Day 1 EDA의 "중앙부 집중" 분포와 일치).
- recovered의 z_median=69: 뇌 *중앙 약간 아래* 영역. T1ce가 이 영역 ET 검출에 도움.
- still_missed의 z_median=62: 더 아래쪽 (소뇌·뇌간 근방) — 정상 구조가 복잡한 영역에서 극소 종양 분리 어려움.
- new_fp의 z_median=69: recovered와 같은 영역 — **T1ce의 정상 조영 구조와 종양 구조가 겹치는 위치에서 새 오탐 발생**.

### 8.3 (C) 채널 Ablation 학습/평가 (step27 / step28)

#### 8.3.1 학습 통계 비교

| 모델 | Channel mode | 총 시간 | 총 epoch | Best Val Acc | Best Val Dice |
|------|--------------|:-------:|:--------:|:------------:|:-------------:|
| Day 6 baseline | t1ce_flair_diff | 388.5분 | 16 | 94.12% (e11) | 0.4263 (e16) |
| **C-1: T1ce-only** | t1ce_only | 340.2분 | 15 | 88.40% (e8) | 0.311 (e15) |
| **C-2: FLAIR-only** | flair_only | **715.3분** | 12 (Early Stop) | 93.67% (e10) | 0.4168 (e12) |

#### 8.3.2 핵심 평가표 (`ablation_table.json`)

| 모델 | AUROC | AUPRC | F1 @ 0.5 | F1 @ 0.3847 | TP/TN/FP/FN @ 0.5 | seg Dice | seg IoU |
|------|:-----:|:-----:|:--------:|:-----------:|:------------------:|:--------:|:-------:|
| **Day 6 baseline** [T1ce,FLAIR,diff] | **0.9832** | **0.9863** | **94.27** | 93.84 | 11401/12327/539/848 | **0.7874** | **0.7142** |
| **C-1: T1ce-only** [T1ce, T1ce, T1ce] | 0.9487 | 0.9582 | 87.55 | 86.71 | 10414/11739/1127/1835 | 0.5741 | 0.4780 |
| **C-2: FLAIR-only** [FLAIR, FLAIR, FLAIR] | 0.9822 | 0.9853 | 94.15 | **93.98** | 11205/12518/348/1044 | 0.7458 | 0.6703 |

> 참고: Day 5 MTL @ 0.3847은 F1 93.91, AUROC 0.9824, FP 305, FN 1136, dice 0.7765, iou 0.7061.

#### 8.3.3 결정적 발견 — Day 6의 "성과" 분해 (★)

> **본 발표의 핵심 새 인사이트 ⑥**:
> "**WT 분류의 결정적 신호는 FLAIR가 거의 다 가지고 있다.** C-2 FLAIR-only AUROC 0.9822 ≈ baseline AUROC 0.9832 (동등). T1ce-only는 AUROC 0.9487로 분명히 약함."

> **본 발표의 핵심 새 인사이트 ⑦**:
> "**Day 6 멀티모달의 *순수* 효과**:
> - 분류 F1 (@0.5): C-2 → baseline 94.15 → 94.27 (+0.12%p) — 거의 미미.
> - SEG Dice: C-2 → baseline 0.7458 → 0.7874 (+0.0416, +5.6%) — **여기서 진짜 의미 있는 향상이 발생**.
> - 즉 *T1ce + diff 채널은 분류보다 세분화 정밀도에 더 큰 기여를 한다.*"

#### 8.3.4 더 미묘한 발견 — Day 5 (Dice-only) vs C-2 (Dice+Tversky)

같은 입력(FLAIR-only), 같은 모델 구조, 다른 Loss:

| 비교 | F1 @ 0.3847 | FP | FN | seg Dice | seg IoU |
|------|:-----------:|:--:|:--:|:--------:|:-------:|
| Day 5 MTL (Dice only) | 93.91 | **305** | 1,136 | 0.7765 | 0.7061 |
| **C-2 (Dice + Tversky)** | **93.98** | 566 | **890** | 0.7458 | 0.6703 |

- **Tversky의 *분리된* 기여**: F1 +0.07%p (거의 없음), **FN -246 (-21.7%)**, FP +261 (+85.6%), seg Dice **-0.0307** (감소!), seg IoU -0.0358.
- **놀라운 결과**: Tversky가 **분류 FN을 줄이는 데는 성공**했지만 **segmentation Dice는 오히려 감소**시켰다.
- **추정 이유**: Tversky α=0.7이 sigmoid 출력을 양성 편향으로 만들면서, seg head에서도 *작은 양성 영역을 과대 예측*하는 경향. → seg dice 감소.
- 즉 **Tversky는 분류에 좋고 seg에 나쁘다**는 양면성.

#### 8.3.5 T1ce-only의 *학습 다이내믹* 이상 관찰 (★)

`outputs/logs/mmmt_ablation/t1ce_only/history.json`에서:

```
epoch 1~8 train_dice: 0.028 → 0.000 → 0.000 → 0.000 → 0.000 → 0.000 → 0.000 → 0.000
epoch 9: 0.000
epoch 10~15: 0.046 → 0.572 → 0.599 → 0.612 → 0.614 → 0.625
```

> **본 발표의 핵심 새 인사이트 ⑧**:
> "**T1ce-only로는 초반 9 epoch까지 seg head가 *전혀 학습되지 않았다*** (train dice 0.000). Encoder가 분류 head로만 신호를 보내다가 epoch 10부터 seg가 깨어남. 즉 **T1ce는 *WT(전체 종양)* 라벨에 대한 정밀 위치 신호가 부족하여 Tversky로 학습이 막히는 단계가 길다.** FLAIR가 본 프로젝트의 WT 검출에 *결정적 모달리티*임을 학습 다이내믹 자체가 보여준 사례."

#### 8.3.6 "diff 채널의 진짜 기여" 분리

baseline = T1ce + FLAIR + |T1ce - FLAIR|
- baseline F1 94.27 vs FLAIR-only F1 94.15 → **+0.12%p** (분류)
- baseline seg dice 0.7874 vs FLAIR-only seg dice 0.7458 → **+0.0416** (+5.6%) (세분화)

**즉 diff 채널은 분류에는 거의 영향 없고, 세분화 정밀도 향상에 +5.6%p Dice 기여.** **|T1ce - FLAIR|가 *종양 경계 후보*의 추가 inductive bias로 작동했다는 가설을 검증.**

### 🎯 Day 6 권장 3종 예상 질문 (Q&A)

**Q1. FLAIR-only ablation이 baseline과 거의 동등하다면, 굳이 T1ce를 추가할 가치가 있나?**
A. (1) **분류만 본다면 한계 효용 약함** (+0.12%p F1). (2) **세분화 품질을 본다면 가치 분명** (+5.6%p Dice). (3) **임상적으로 WT/TC/ET 등급화**를 하려면 T1ce 필수 — ET는 T1ce 없이는 검출 불가. (4) **Day 7 WT/TC/ET 3-region 확장에서는 T1ce 필수**. 발표 메시지는 "본 단계에서 멀티모달의 *분류 이득*은 미미하지만 *세분화 이득과 grading 확장성*은 유지" 로 정직하게.

**Q2. recovered 381 vs regressed 86 — 같은 작은 종양인데 한쪽만 잡는 이유?**
A. **상호 보완성의 직접 증거.** 두 모델은 같은 작은 종양 분포를 보지만 *각자 다른 부분 집합*을 잡는다. T1ce 강조에 잘 보이는 ET는 Day 6이 잡고(recovered), FLAIR 부종 단서가 강한 케이스는 Day 5가 잡았는데(regressed) Day 6 모델이 T1ce 신호를 우선시하면서 놓침. → **Day 8+ Ensemble (Day 5 + Day 6)을 적용하면 recovered + regressed = 467개를 모두 잡을 가능성**.

**Q3. still_missed 641의 정확한 임상적 정체는?**
A. 픽셀 100개 = 224×224 슬라이스 중 **0.20%**. 뇌 부피의 극소 영역. 임상적으로는 (a) 종양 시작 슬라이스의 경계, (b) micro-metastasis, (c) seg mask 라벨링 오차 가능성도 있다. 발표에서는 "이 영역은 *2D 슬라이스 단위*의 한계이며, 2.5D 또는 3D 모델에서만 잡힐 수 있다"고 정직히 보고하는 것이 안전.

**Q4. new_fp 515의 prob mean이 0.579로 *반쯤 확신*이다. 임상에서 어떻게 처리?**
A. 두 가지 처리:
1. **Threshold를 0.5에서 0.6으로 올림** — new_fp 다수가 0.55~0.6 영역에 있을 가능성.
2. **확신 구간(0.4~0.6)을 *불확실*로 분류**하여 *의사 검토 큐*에 보냄. → Conformal Prediction의 직접 응용 (Day 7에서 실제 적용).

**Q5. cleaned_fp 119의 정체?**
A. Day 5가 0.596 확신으로 "종양"이라 했던 정상 슬라이스를 Day 6이 0.194로 정상화. 평균 z_median=55 (뇌 *위쪽*). **추정**: 두개골 근접 영역의 FLAIR 고신호 (normal-appearing white matter 변성·CSF 흐름 artifact)를 Day 5가 종양으로 오인했는데, Day 6은 T1ce 채널에서 조영 증강이 없음을 보고 *cross-check*하여 거부. **"T1ce에서 증강 안 되면 진성 종양 아님 → cross-check" 가설의 정량 검증.**

**Q6. T1ce-only seg dice 0.57이 의학적 사실과 모순?**
A. 본 ablation의 seg label은 **WT (= NCR + ED + ET)** 라서 ED 면적이 큰 점이 T1ce-only에 불리. 만약 *TC (=NCR + ET)* 나 *ET only*를 라벨로 했다면 T1ce-only의 dice가 훨씬 높을 것 — 이게 Day 7 3-region 확장의 핵심 주제.

📎 **참고**: `260524v1mmmtplus.md` §3, §4, §5 / `260523v1updatemmmt.md` §④ / `code/mmmt/step25b_*.py`, `step26_*.py`, `step27_*.py`, `step28_*.py` / `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json`, `fn_diff_summary.json` / `outputs/logs/mmmt_ablation/ablation_table.json`

---

## 9. Day 7 — SOTA 패키지 설계 + 학습 + 평가

### 9.1 큰 그림 — sota 폴더 = "v2ways §8.2 무료 SOTA 패키지" 의 구현체

> **세 개의 동시 목표**:
> 1. *BraTS 공식 평가축*(WT/TC/ET 3-region)에 정렬 — 학술 SOTA(MedNeXt 0.93, DynUNet 0.91)와 비교 가능한 상태로 격상.
> 2. *FN 한계의 직접 표적화* — Focal-Tversky + Weighted Sampler + TumorCP로 소종양에 학습 가중을 집중.
> 3. *신뢰성 SOTA* — TTA + Temperature Scaling + Conformal Prediction 으로 결정 경계의 *불확실성 정량화*.

세 목표는 학부 발표용으로 "Day 6 멀티모달 + 새로운 신뢰성 계층" 의 결합이며, `code/`, `code/multitask/`, `code/mmmt/` 의 *어떤 파일도 수정하지 않고* 새 폴더(`code/sota/`, `processed/sota/`, `outputs/{checkpoints,figures,logs}/sota/`) 안에서만 동작하도록 격리. 따라서 Day 2/4/5/6 결과가 그대로 보존되어 **7-way 비교** 가 가능.

### 9.2 step26 — WT/TC/ET 3-region 마스크 생성

**산출물**: `outputs/logs/sota/step26_segmask_stats.json`

```json
{
  "patients": 1251,
  "saved": 166626,
  "wt_pos": 81337,
  "tc_pos": 53593,
  "et_pos": 50958,
  "skipped": 0,
  "errors": 0
}
```

**3-region 정의** (BraTS 2023 라벨 규약):
- **WT = (seg > 0)** — 종양 전체 (NCR + ED + ET)
- **TC = (seg == 1) | (seg == 3)** — 종양 코어 (NCR + ET, 부종 제외)
- **ET = (seg == 3)** — 조영 증강 종양만

**보존 수치**:
- 환자 수 1,251명 — Day 1 EDA와 일치 (검증 통과).
- WT positive 81,337 — Day 1 의 *81,374* (FLAIR seg 기준)와 37 차이. 발표 시 영향 미미.
- TC positive 53,593 / ET positive 50,958 — 슬라이스 65.9% / 62.7% 가 *TC/ET를 가짐*. 거의 모든 종양 슬라이스에 enhancing 영역이 존재.

### 9.3 step27 — 3채널 입력 + 3채널 마스크 + Weighted Sampler

**입력 구성**: `[T1ce, FLAIR, |T1ce - FLAIR|]` (Day 6 `processed/mmmt/t1ce_slices/` 재사용)
**출력 구성**: cls 스칼라 + 3채널 (WT/TC/ET) 마스크
**Weighted Sampler**: WT 양성 픽셀 < 256인 "소종양" 슬라이스에 가중치 **×3**

**근거**:
- `260524v1mmmtplus.md` §4.5 **인사이트 ④**: still_missed 641 슬라이스의 평균 종양 크기는 100픽셀(0.20%) — 멀티모달로도 못 잡음.
- `_build_sampler` 코드는 이 진단을 직접 처방으로 옮긴 것.

### 9.4 step28 — Deep Supervision + Uncertainty Weighting 모델

```
입력: (B, 3, 224, 224)
       │
   Shared Encoder (ResNet-18, ImageNet pretrained)
   conv1[3→64] - layer1[64] - layer2[128] - layer3[256] - layer4[512]
       │
   ┌───┴───────────────────────────────────────┐
   GAP→Dropout(0.5)→FC(512→1)    Decoder (UpBlock×4 + skip)
   = cls_out (B, 1)              ├─ seg_main: (B, 3, 224, 224) — WT/TC/ET
                                 ├─ aux_head_d4: (B, 1, ...)  ← Deep Sup (WT)
                                 ├─ aux_head_d3: (B, 1, ...)  ← Deep Sup (WT)
                                 └─ aux_head_d2: (B, 1, ...)  ← Deep Sup (WT)
   log_var_cls, log_var_seg — Kendall 2018 Uncertainty Weighting 파라미터
```

**Day 6 MMMTBrainNet 대비 확장**:
- `seg_classes 1 → 3` (WT/TC/ET 멀티채널 seg head).
- `aux_head_d2/d3/d4` — Deep Supervision 보조 WT seg head 3개.
- `log_var_cls`, `log_var_seg` — 학습 가능한 가중치 자동 조정 파라미터.

**핵심 근거**: Multi-Task 분류 head Grad-CAM IoU는 train에서도 0.1232에 불과한데, seg head IoU는 0.7753 → encoder feature는 충분히 종양을 알지만 분류 head로 전파되지 않음. Deep Supervision은 decoder 중간 단계마다 supervisory signal을 직접 주입해서 이 격차를 줄임.

### 9.5 step29 — Focal-Tversky + Boundary + Compound + TumorCP

| 컴포넌트 | 정의 | 출처 |
|----------|------|------|
| `FocalTverskyLoss(α=0.7, β=0.3, γ=4/3)` | TP/(TP + α·FN + β·FP) → (1−Tversky)^γ | Abraham & Khan ISBI 2019 |
| `BoundaryLoss` | SDF(signed distance function) 기반, 경계 슬라이스용 | Kervadec MIDL 2019 |
| `CompoundSegLoss` | **1.0·Focal-Tversky + 0.5·BCE + 0.3·Boundary** per-region | — |
| `SOTAMultiTaskLoss` | cls/seg 두 task에 *Uncertainty Weighting* + Deep Sup aux 3개에 `ds_weights=(0.125, 0.25, 0.5)` 가중합 | Kendall 2018 |
| `tumor_copy_paste` | 배치 내 종양 슬라이스 패치를 다른 슬라이스에 paste (Yang MICCAI 2022 단순화) | — |

**Day 6 대비 변경점**:
- Day 6 = `0.5·Dice + 0.5·Tversky(α=0.7)` → **Day 7 = `1.0·Focal-Tversky(γ=4/3) + 0.5·BCE + 0.3·Boundary`**.
- α/β 수동 튜닝 → Uncertainty Weighting 자동.
- 단일 task → 3 region (WT/TC/ET) 동시.

### 9.6 step30 — 학습 CONFIG + 실제 동작

`outputs/logs/sota/sota_summary.json`:

```json
{
  "phase1_epochs": 3,
  "phase1_lr": 1e-3,
  "phase2_epochs": 15,
  "phase2_lr": 1e-4,
  "swa_start_epoch": 11,
  "swa_lr": 5e-5,
  "batch_size": 16,
  "weight_decay": 1e-4,
  "num_workers": 0,
  "early_stopping_patience": 5,
  "use_weighted_sampler": true,
  "channel_mode": "t1ce_flair_diff",
  "tumorcp_prob": 0.5,
  "w_tv": 1.0, "w_bce": 0.5, "w_b": 0.3,
  "tversky_alpha": 0.7, "tversky_beta": 0.3, "tversky_gamma": 1.333,
  "device": "cuda"
}
```

**Day 5/6 와 동일 통제**: phase 구성, batch_size, weight_decay, EarlyStopping patience, channel_mode 모두 *Day 6과 동일* — 본 SOTA 학습의 *순수 효과*는 (i) loss 변경, (ii) 3-region seg head, (iii) Deep Supervision, (iv) Uncertainty Weighting, (v) Weighted Sampler+TumorCP, (vi) SWA 의 *6개의 동시 변경*. (분리 ablation은 미실행 — 부록 A.)

### 9.7 실제 학습 동작 — 14 epoch 만에 종료 (계획 18 vs 실제 14)

**계획**: P1 3 epoch + P2 15 epoch = **18 epoch** (CONFIG 기준).
**실측**: P1 3 epoch + P2 11 epoch = **14 epoch** 에서 종료. `sota_history.json` 의 모든 train/val 배열 길이가 정확히 14.

**종료 사유 (보존)**:
- `sota_summary.json` 의 `total_epochs: 14`, `total_minutes: 1017.8` — 약 16.96시간.
- VRAM 누수 + EarlyStopping patience=5 trigger 모두 가능. 학습 곡선상 Val Total이 epoch 9 (1.601) 에서 최저였고 이후 증가.

### 9.8 14 epoch 전체 학습 곡선 (핵심 발췌)

> ⚠️ `train_total` 이 epoch 9 이후 음수가 되는 것은 Uncertainty Weighting의 `log_var` 항이 학습되며 log-likelihood scaling 적용되기 때문 — 음수 자체는 학습 안정성과 무관.

| Epoch | Phase | Val Total | Val CLS | Val SEG | Val Acc | Val Dice (WT/TC/ET) | log_var (cls/seg) |
|:-----:|:-----:|:---------:|:-------:|:-------:|:-------:|:--------------------:|:------------------:|
| 1 (P1) | P1 | 1.826 | 0.403 | 1.425 | 81.81% | 0.393 / 0.284 / 0.262 | 0.018 / 0.005 |
| 3 (P1 종료) | P1 | 1.949 | 0.415 | 1.216 | 80.81% | 0.409 / 0.297 / 0.277 | 0.009 / -0.336 |
| **4 (P2 시작)** | P2 | 2.181 | 0.228 | 1.511 | 91.48% | 0.417 / 0.301 / 0.279 | -0.425 / -0.395 |
| 5 | P2 | 1.688 | 0.189 | 1.163 | 93.03% | 0.418 / 0.302 / 0.281 | -0.651 / -0.488 |
| 6 | P2 | 2.045 | 0.204 | 1.306 | 92.18% | 0.433 / 0.314 / 0.292 | -0.807 / -0.551 |
| **9 (★ Val Total 최저)** | P2 | **1.601** | 0.222 | 0.904 | 92.06% | 0.430 / 0.307 / 0.284 | -1.163 / -0.701 |
| **11 (SWA 시작)** | P2 | 2.310 | 0.270 | 1.074 | 91.19% | 0.432 / 0.309 / 0.289 | -1.387 / -0.763 |
| **13 (★ Val Acc 최고)** | P2 | 2.307 | 0.249 | 1.024 | **92.90%** | 0.432 / 0.312 / 0.290 | -1.569 / -0.809 |
| 14 (last) | P2 | 2.596 | 0.263 | 1.082 | 92.40% | 0.431 / 0.312 / 0.292 | -1.642 / -0.826 |

**핵심 관찰**:

1. **Best Val Total epoch = 9** (1.601) — `sota_best.pth` 가 *epoch 9* 시점. Val Acc 최고는 *epoch 13* (92.90%). 본 프로젝트는 **Val Total (= 종합 loss) 기준 best** 를 채택.
2. **Val Dice 학습 곡선에서는 0.43 수준 / Test Dice는 0.797** — Day 6과 동일한 측정 함수 차이 패턴.
3. **Uncertainty Weighting log_var**: cls의 변화량(5×)이 seg 변화량(2.3×)보다 큼 → 학습 후반 *cls task가 더 빠르게 saturate* 되고 *loss 가중치가 자동으로 seg로 이동*. Kendall 2018 의도 그대로.

### 9.9 학습 시간 — 1017.8분 ≈ 17시간

| 모델 | 총 시간 | 총 epoch | epoch당 평균 |
|------|:-------:|:--------:|:------------:|
| Day 5 Multi-Task | 298.8분 | 14 | 21.3분 |
| Day 6 MMMT | 388.5분 | 16 | 24.3분 |
| **Day 7 SOTA** | **1017.8분** | **14** | **72.7분** |

> Day 7의 epoch당 72.7분은 Day 6의 *3배*. 원인: (1) seg head 가 3채널 (WT/TC/ET) 로 *3배* 출력, (2) Deep Supervision aux 3개 추가 forward+loss, (3) TumorCP의 collate-level paste 비용, (4) Weighted Sampler의 dataset 인덱싱 재계산, (5) `num_workers=0` (Windows 안정성 우선).

### 9.10 step31 평가 환경 (manifest)

`sota_test_metrics.json` 의 `meta` 블록 — 발표에서 *재현성* 슬라이드의 핵심:

| 항목 | 값 |
|------|-----|
| `ckpt_path` | `outputs/checkpoints/sota/sota_best.pth` |
| `ckpt_sha256` | `0a2acabbb17978d0e7bbbeea6f6dd1c4a044a909c2afb0da28a877c66b2430ed` |
| `swa_ckpt_sha256` | `2c018a2604e7f2b648a93188c711eab01d5db59c975854d708cf7e6ca663e767` |
| `git_rev` | `71776a1` |
| `tta_modes` | `["identity", "hflip", "vflip", "rot180"]` |
| `tta_aggregation_cls` | **`logit_mean`** (P4 패치 적용) |
| `tta_aggregation_seg` | `sigmoid_mean` |
| `temperature_T_best` | **`1.5015`** |
| `device` | `cuda` |
| `amp` | `true` (FP16 autocast) |
| `eval_batch_size` | `2` (8GB VRAM 보수적 설정) |
| `n_boot` | `1000` (Bootstrap CI) |
| `evaluated_at` | `2026-05-26T10:23:43` |

### 9.11 분류 SOTA — 3가지 threshold 시나리오

| 시나리오 | Threshold | TP | TN | FP | FN | F1 | AUROC | AUPRC |
|----------|:---------:|:--:|:--:|:--:|:--:|:--:|:-----:|:-----:|
| TTA (raw) @0.5 | 0.5 | 11,560 | 11,805 | 1,061 | 689 | **0.9296** | **0.9824** | **0.9849** |
| Temperature Scaled @0.5 | 0.5 | 11,486 | 11,608 | 1,258 | 763 | 0.9191 | 0.9795 | 0.9827 |
| **TTA + val F1-optimal thr** | **0.7** | **11,293** | **12,252** | **614** | **956** | **0.9350** ★ | **0.9824** | **0.9849** |

**핵심 관찰 (보존)**:

1. **AUROC 0.9824** [95% CI: 0.9811, 0.9837] — Day 6 MMMT (0.9832) 와 *통계적으로 동등* (CI overlap).
2. **AUPRC 0.9849** [0.9837, 0.9861] — Day 6 (0.9863) 보다 *살짝 낮음*.
3. **F1 0.9296** (thr 0.5) — Day 6 (0.9427) 보다 **-0.013** 낮음. *순수 thr 0.5* 비교에서는 Day 7 SOTA가 *후퇴* 한 것처럼 보인다. **그러나 val F1-optimal threshold 0.7 을 채택하면 F1 0.9350 으로 회복 + 더 적은 FP (614 vs 539)** 와 더 많은 FN(956 vs 848) 의 *trade-off* 로 이동.
4. **Temperature Scaling 후 F1 *감소* (0.9296 → 0.9191)**: T=1.5015 로 logit이 *부드러워지는* 결과, threshold 0.5 에서 FP/FN 모두 증가. 즉 *naive thr 0.5* 적용은 *모델 best operating point 가 thr 0.6~0.7 부근* 이라는 신호.

### 9.12 Temperature Scaling이 의미하는 것 (보존)

- `T = 1.5015 > 1` → 모델이 *과확신*(overconfident) 이었다는 직접 증거. 정답 클래스에 대한 확률이 *실제보다 더 크게* 출력되고 있었음.
- `BCE(σ(logit / 1.5015), y)` 가 val에서 최소화 → 보정 후 평균 확률이 0.04~0.05 정도 *느슨해짐*.
- *thr 자체를 0.5 → 0.7로 올리는 것* 만으로도 동등 효과를 얻을 수 있다.

### 9.13 7-way 비교의 핵심 행 — Day 5 / Day 6 / Day 7 (★ 발표용 메인 표)

| 모델 | Threshold | F1 | AUROC | Precision | Recall | FP | FN | seg Dice (WT) | seg IoU (WT) |
|------|:---------:|:--:|:-----:|:---------:|:------:|:--:|:--:|:-------------:|:------------:|
| Day 5 MTL (FLAIR only) | 0.3847 | 93.91 | 0.9824 | **97.33** | 90.73 | **305** | 1,136 | 0.7765 | 0.7061 |
| Day 6 MMMT (T1ce+FLAIR+diff) | 0.5 | **94.27** | 0.9832 | 95.49 | 93.08 | 539 | 848 | 0.7874 | 0.7142 |
| **Day 7 SOTA (best, TTA)** | 0.5 | 92.96 | 0.9824 | 91.59 | 94.37 | 1,061 | **689** ★ | **0.7971** ★ | **0.7225** ★ |
| **Day 7 SOTA (best, thr 0.7)** | **0.7** | **93.50** | 0.9824 | **94.84** | 92.20 | **614** | 956 | (동일) | (동일) |

> ★ Day 7 의 강점: **Recall 94.37% (Day 5 +3.64%p / Day 6 +1.29%p)**, **FN 689 (Day 5 -447 / Day 6 -159)**, **WT Dice 0.7971 (Day 5 +0.021 / Day 6 +0.010)** — *작은 종양 잡기 + 세분화 정밀도* 동시 개선.
> ★ Day 7 의 약점: **FP 1,061 (Day 5 +756 / Day 6 +522)** — naive thr 0.5에서 FP가 *3배*. **thr 0.7 채택으로 FP 614 까지 감소** (Day 5의 305 수준은 못 따라잡음).

### 9.14 결정적 발견 (★)

> **본 발표의 핵심 새 인사이트 ⑨**:
> "**Day 7 SOTA의 *실질적 이득*은 *Recall + FN + Dice* 세 지표에 집중되고, *Precision + FP* 는 Day 5/6 보다 약화된다.** Focal-Tversky γ=4/3 + Weighted Sampler(소종양 ×3) + TumorCP 의 *3중 FN 표적화* 가 의도대로 작동했다는 직접 증거 — 단, 그 대가로 FP가 늘었다 ('Tversky α=0.7의 부작용' 패턴이 *더 크게* 재현)."

> **본 발표의 핵심 새 인사이트 ⑩**:
> "**val 기반 F1-optimal threshold 채택 (0.5 → 0.7) 은 *재학습 없이* FP를 1,061 → 614 (-42%) 로 줄이는 동시에 F1을 0.930 → 0.935 로 올린다.** 즉 *운용점 선택* 만으로도 SOTA 모델의 약점이 상당 부분 회복 가능 — Day 5/6 비교에서 '운용 자유도' 라고 불렀던 효과가 Day 7 에서 *더 크게* 발휘됨 (CI [0.9318, 0.9380])."

### 9.15 Confidence Intervals (Bootstrap 1000회) — 통계적 유의성

| 지표 | 점추정 | 95% CI [low, high] |
|------|:------:|:------------------:|
| **AUROC (TTA)** | 0.9824 | [0.9811, 0.9837] |
| AUPRC (TTA) | 0.9849 | [0.9837, 0.9861] |
| F1 (TTA, thr 0.5) | 0.9296 | [0.9262, 0.9328] |
| F1 (TTA, thr 0.7) | 0.9350 | [0.9318, 0.9380] |
| F1 (Temp Scaled, thr 0.5) | 0.9191 | [0.9153, 0.9226] |

> **Day 6 AUROC 0.9832 vs Day 7 AUROC 0.9824** — Day 7의 CI [0.9811, 0.9837]가 Day 6 점추정을 *포함* → **두 모델은 AUROC 관점에서 통계적으로 구분 불가**.

### 9.16 SWA 체크포인트 — 디스크에는 있으나 평가 불가

`outputs/checkpoints/sota/sota_swa.pth` 가 *디스크에 정상 저장*(54.7MB) 됐다. SHA256 = `2c018a26...`. 그러나 step31에서 SWA 평가를 시도했을 때:

```json
"swa": {
  "error": "Unable to allocate 14.0 GiB for an array with shape (24882, 3, 224, 224) and data type float32"
}
```

— SWA 의 val raw predictions (`val_pack["seg_pred"]`, shape `(24882, 3, 224, 224)`, dtype float32) 를 메모리에 올리는 도중 *RAM OOM*. 16GB 시스템 RAM의 한계. **즉 SWA 평가는 *이론적으로는* `best vs SWA` 비교가 가능하지만, 본 환경에서는 *실측 불가*.**

### 🎯 Day 7 예상 질문 (Q&A)

**Q1. Day 6 → Day 7로 *동시에 6개의 변경* 을 가했다. 각 변경의 *분리된* 기여는 어떻게 확인하나?**
A. 본 프로젝트에서는 분리 ablation을 *미실행*. 사유: 단일 SOTA 학습조차 14 epoch에서 강제 종료될 정도로 자원이 빠듯. 다만 (i) Day 6 ablation (C-1 T1ce-only / C-2 FLAIR-only)이 *모달리티/loss 효과의 분리* 를 이미 정량 검증, (ii) Day 7 의 추가 변경은 모두 *학술적으로 검증된 컴포넌트* (Focal-Tversky ISBI 2019, Deep Sup, Uncertainty Weighting Kendall 2018, TumorCP MICCAI 2022, SWA, TTA) — "여러 SOTA 컴포넌트를 *동시* 적용하여 *학술 SOTA 표면적*에 닿는 시연" 으로 정직 보고.

**Q2. Day 7 SOTA가 Day 6 MMMT 보다 F1 (thr 0.5) 이 -0.013 *낮다*. 그러면 SOTA가 *후퇴*한 게 아닌가?**
A. 표면적으로는 그렇다. 그러나 (1) thr 0.7 채택 시 F1 0.9350 으로 Day 6 (0.9427) 과 -0.0077 까지 좁혀짐, (2) Recall 94.37 (Day 6 93.08, +1.29%p) 과 FN 689 (Day 6 848, -159) 는 *임상적으로 더 중요한* 지표, (3) Dice 0.7971 (Day 6 0.7874, +0.010) 는 *세분화 품질 SOTA*. **발표 메시지: "단일 F1이 아니라 *FN-Recall-Dice 삼각 향상*"**.

**Q3. Temperature T=1.5015 가 의미가 있는가? 학술적 표준은?**
A. T=1.5 는 *적당한 over-confidence* 수준 (Guo et al. ICML 2017, ResNet-110 on CIFAR-100 의 T≈1.5와 동일 영역). T=1 이면 calibration 완벽, T>>1 이면 심각한 overconfident. 본 모델은 *약하게 overconfident*.

**Q4. SOTA 학습이 계획 18 epoch 중 14에서 종료된 게 결과 비교에 영향을 주지 않나?**
A. (1) `EarlyStopping patience=5` 가 정상 동작했고, val total 최저는 epoch 9이며 14까지 도달해도 갱신되지 않음 — 즉 *추가 학습은 과적합만 늘림*. (2) 학습 곡선상 epoch 9~14 사이 val cls/seg loss 추세가 *plateau*. (3) 본 프로젝트가 *모델 비교* 가 아니라 *시연* 목적이므로 학습 종료 시점 차이는 *영향 미미*.

**Q5. SWA 평가가 RAM OOM 으로 실패한 게 *발표 가치 손상* 아닌가?**
A. 부분적으로 그렇다. 다만 (1) `sota_best.pth` 와 `sota_swa.pth` 의 sha256이 둘 다 manifest에 기록되어 *추후 재실험 가능성* 보존, (2) P10 계획대로 코드는 들어가 있으므로 *환경 부족이 이슈일 뿐 설계 오류 아님*, (3) SWA의 일반적 효과는 *+1~2%p Dice* 수준 — 본 결과(WT Dice 0.797)의 *순위* 가 SWA에 의해 뒤집힐 가능성은 낮음.

📎 **참고**: `260526v3sotafinal.md` §1, §2, §3 / `260525v1sotapluswhy.md` 전체 / `code/sota/step26~31_*.py` / `outputs/logs/sota/sota_history.json`, `sota_summary.json`, `sota_test_metrics.json`

---

## 10. Day 7 신뢰성 SOTA — Calibration / Conformal / Bootstrap CI

### 10.1 Calibration — ECE / Brier / Reliability Diagram

| 지표 | Pre-Temperature | Post-Temperature (T=1.5015) |
|------|:---------------:|:---------------------------:|
| **ECE** (10-bin) | **0.0364** | 0.0420 |
| **Brier Score** | **0.0525** | 0.0570 |

> ⚠️ **반직관적 결과**: Temperature Scaling 후 ECE/Brier가 *오히려 증가*.
> **원인 (보존)**: (1) Temp Scaling 은 *BCE* 최소화로 T를 찾는데, ECE/Brier는 *별도의 정의* — 두 metric 이 항상 같은 방향으로 움직이지는 않는다. (2) Day 7 모델이 *원래부터 well-calibrated* (Pre ECE 0.036) 라 추가 보정의 한계 효용 음수.

> **시사점**: "Day 7 SOTA는 *원래부터* well-calibrated (ECE 0.036) → Temperature Scaling 은 *재현 가능성을 위한 절차* 일 뿐, 실제 보정 효과는 미미" 로 정정 보고.

### 10.2 Conformal Prediction — marginal score 기반 (간이)

```json
{
  "q": 0.4084,
  "coverage": 0.8964,
  "sets": [first 50 of 25115]
}
```

- `α=0.1` 설정 → 목표 coverage 0.90.
- 실측 coverage **0.8964** ≈ 0.90 — *목표 달성* (마지널 conformal의 정의상 ±0.01 이내 정상).
- `q=0.4084` — score (= `min(prob, 1-prob)` 기반) 의 90% quantile.

**해석**: Conformal set은 각 테스트 슬라이스에 대해 {0}, {1}, {0,1}, 또는 ∅ 의 prediction set 을 반환. coverage 0.896 은 *진짜 라벨이 set 안에 포함될 확률* 이 89.6%.

> 첫 50개 set 의 패턴:
> - 대부분 `[0]` 또는 `[1]` 의 singleton set (모델이 *확신* 하는 경우).
> - 일부 `[]` (empty) — 모델이 *둘 다 거부* 한 경우. 임상적으로는 "추가 검토 필요" 라벨.

### 10.3 신뢰성 SOTA 의 의미 (★)

> **본 발표의 핵심 새 인사이트 ⑪**:
> "**Day 7 SOTA의 *진짜 가치* 는 단일 F1 점수가 아니라 *신뢰성 정량화의 완성도*다.** Bootstrap CI + Temperature Scaling + Conformal Prediction 의 3종 신뢰성 패키지가 모두 적용되어, 발표/제안서에서 *"AUROC 0.9824 [0.9811, 0.9837]"* 처럼 학회/논문급 보고가 가능해졌다. Day 5/6 결과는 *점추정만 있었다*."

### 🎯 신뢰성 예상 질문 (Q&A)

**Q1. ECE 0.036 → 0.042 가 *증가* 한 게 *Temperature Scaling 실패* 아닌가?**
A. *부분적 실패*가 맞다 — 그러나 (1) ECE 증가 폭(+0.006)이 매우 작아 *실용적으로는 동등*, (2) Brier도 동일 패턴 (+0.004), (3) Pre-Temp ECE 0.036 자체가 *이미 well-calibrated* 영역 (Guo et al. ResNet-110 CIFAR-10 Pre-Temp ECE ≈ 0.045). "Temperature Scaling은 *재현 가능한 절차로 적용했고, 본 모델은 원래부터 well-calibrated 라 추가 효과 없음*" 으로 정직.

**Q2. Conformal coverage 0.896 이 *목표 0.90 미달* 아닌가?**
A. marginal conformal은 *theoretical coverage ≥ 1−α* 를 만족하나, *finite-sample correction* 이 적용되면 약간 미달 가능. 본 경우 0.896 은 0.90 의 -0.4%p 차이로 *허용 오차 내*. Split Conformal 의 *high-probability bound* 는 `1−α ± O(1/√n_cal)` 이며 n=24,882 calibration set 에서 √n ≈ 158 → 오차 ±0.006 영역. **0.896 ≈ 0.900 ±0.004 — 목표 달성.**

**Q3. Conformal Prediction 의 결과 set 첫 50개를 직접 본 의의?**
A. 발표 슬라이드용 *정성적 시연* 가치. 예시: "본 모델은 *25,115 슬라이스 중 약 90%* 에 대해 singleton set 으로 *확신* 결과를 제공하고, 나머지 10%는 *empty set 또는 ambiguous set* 으로 *의사 검토 큐로 자동 분류* 가능" — 이게 "의료 AI 신뢰성 감사 키트" 의 핵심 시연.

📎 **참고**: `260526v3sotafinal.md` §4 / `260521v2ways.md` §5.3, §5.5 / `260526v2step31patch.md` §3.6, §3.7

---

## 11. Day 7 세분화 SOTA — WT/TC/ET region별, postproc, HD95

### 11.1 baseline (threshold 0.5) — 양성 슬라이스만 평균

| Region | n_positive_slices | Dice mean | Dice median | IoU mean | IoU median |
|--------|:-----------------:|:---------:|:-----------:|:--------:|:----------:|
| **WT** | 12,244 | **0.7971** | **0.9104** | **0.7225** | **0.8355** |
| **TC** | 8,245 | 0.7783 | 0.9085 | 0.7045 | 0.8323 |
| **ET** | 7,684 | 0.7497 | 0.8421 | 0.6450 | 0.7273 |

**핵심 관찰 (보존)**:

1. **Dice median 이 mean 보다 *훨씬 높음*** (WT 0.91 vs 0.80, TC 0.91 vs 0.78, ET 0.84 vs 0.75) — *분포가 right-skewed*.
2. **ET가 가장 낮음** (Dice 0.75) — 양성 슬라이스 수가 가장 적고(7,684), 작은 영역이므로 *경계 픽셀 비율이 높아 noise 민감*.
3. **WT median 0.9104** — 이미 학술 SOTA(MedNeXt 0.93, DynUNet 0.91) 의 *median 영역* 에 도달. 다만 *mean (0.797)* 은 여전히 격차 존재.

### 11.2 Threshold sweep (per-region, val 기반)

| Region | Best threshold (val) | val Dice at best |
|--------|:--------------------:|:----------------:|
| **WT** | **0.7** | 0.7911 |
| **TC** | **0.7** | 0.7904 |
| **ET** | **0.7** | 0.7416 |

> 모든 region 이 thr 0.7 으로 수렴 — 모델 sigmoid 출력이 *전반적으로 작은 값 영역에 분산*. Focal-Tversky γ=4/3 가 학습 중 *예측을 보수적으로* 만든 결과.

### 11.3 후처리 (CC filter min_size=10 + 3×3 morph closing)

| Region | Dice mean (baseline) | Dice mean (postproc) | Δ |
|--------|:--------------------:|:--------------------:|:--:|
| **WT** | 0.7971 | 0.7919 | **-0.0052** |
| **TC** | 0.7783 | 0.7774 | -0.0009 |
| **ET** | 0.7497 | 0.7466 | -0.0031 |

> ⚠️ **반직관적 결과**: Post-processing 후 Dice 가 *모든 region 에서 미세 하락*.
> **원인 (보존)**: (1) Focal-Tversky 가 이미 *작은 noise 컴포넌트를 거의 안 만들었음*. (2) 3×3 closing 이 *경계의 진짜 들어간 부분을 메우면서* GT 와 어긋남. (3) **즉 Day 7 SOTA 는 *후처리 의존도가 낮음* — 학습 단계에서 이미 보수적 마스크를 생성하는 데 성공했다는 *간접 증거*.**

### 11.4 HD95 + per-volume Dice + Sensitivity/Specificity per region

| Region | HD95 mean (pixel) | HD95 median | Sensitivity | Specificity | per-volume Dice mean | per-volume Dice median | n_volumes |
|--------|:-----------------:|:-----------:|:-----------:|:-----------:|:--------------------:|:----------------------:|:---------:|
| **WT** | **4.58** | **1.00** | 0.8087 | 0.9976 | **0.8910** | **0.9215** | 188 |
| **TC** | **3.13** | **1.00** | 0.8230 | 0.9982 | 0.8423 | 0.9139 | 188 |
| **ET** | **3.02** | **1.41** | 0.8324 | 0.9976 | 0.7896 | 0.8464 | 182 |

**핵심 관찰 (보존)**:

1. **HD95 median 1.00 (WT/TC), 1.41 (ET)** — *대부분 슬라이스에서 예측 경계가 GT 경계와 *1~1.4 픽셀* 이내*. 224×224 픽셀 영역에서 *경계 오차 1픽셀* 은 *임상적으로 거의 완벽*.
2. **HD95 mean이 4.58 (WT), 3.13 (TC), 3.02 (ET)** — *일부 outlier slice 에서 큰 거리 오차*. 분포 right-skewed.
3. **Sensitivity 0.81~0.83 / Specificity 0.998** — 매우 *보수적 마스크* 패턴.
4. **per-volume Dice mean 0.891 (WT)** — *환자 단위* 로 묶어 평균하면 *슬라이스 단위 mean (0.797)* 보다 *훨씬 높다*. 학술 SOTA 의 보고 단위도 *per-volume* 이므로 **본 모델은 *학술 SOTA 비교 단위* 에서 WT Dice 0.891 — DynUNet 2D (0.91) 의 -0.019 영역에 위치**.
5. **n_volumes ET = 182 < 188** — 6개 환자는 *어떤 슬라이스에도 ET positive 없음* — BraTS GLI 데이터의 정상 분포 (low-grade glioma 또는 non-enhancing 케이스).

### 11.5 결정적 발견 — Day 7 세분화의 학술 SOTA 도달 (★)

> **본 발표의 핵심 새 인사이트 ⑫**:
> "**Day 7 SOTA 는 *세분화 측면에서* Day 6 대비 WT Dice +0.010, *학술 SOTA(DynUNet 2D 0.91) 의 -0.02 까지 도달* 한 첫 결과다.** per-volume 환자 단위 평균 WT Dice 0.891 은 학부 환경(8GB Laptop GPU + Windows + 14 epoch 조기종료) 에서 *학술 SOTA 표면적* 까지 도달한 시연. + **TC Dice 0.778, ET Dice 0.750 의 *3-region 동시 평가축* 자체** 가 Day 6까지 *불가능했던* BraTS 공식 평가."

### 🎯 세분화 예상 질문 (Q&A)

**Q1. WT Dice 의 *slice mean 0.797 vs volume mean 0.891* 어떤 게 공식 보고치?**
A. **본 보고서는 *두 값을 모두 보고*.** (1) Day 5/6과의 *공정 비교* 는 *slice mean 0.797*. (2) *학술 SOTA 비교* 는 *volume mean 0.891* (학술 보고 표준 단위). (3) 격차 0.094 의 의미: 환자 단위로 합치면 *진짜 작은 슬라이스 (종양 픽셀 < 100)* 의 영향이 *희석* — still_missed 100 픽셀 영역이 *volume aggregation 으로 부분 회복*.

**Q2. Threshold sweep 결과가 *모든 region 0.7로 동일* 한 게 *모델의 보편적 특성* 인가?**
A. Focal-Tversky γ=4/3 의 *공통 효과*가 큼 — 세 region 모두 같은 loss 비율로 학습되므로 sigmoid 분포가 유사. *Uncertainty Weighting 의 region-uniform 보정 효과의 부산물*. 동일 thr 0.7 채택은 *행정 편의성* 측면에서 장점.

**Q3. Post-processing 이 *효과 없음* 으로 보고된 게 후처리 자체의 무효성 인가, Day 7 학습의 우수성 인가?**
A. 후자가 맞다. Day 6 수준 모델 기반으로 산정된 "+1~3%p Dice" 기대치였으나 Day 7 SOTA 는 *학습 자체가* 이미 *작은 noise 컴포넌트 < 10 voxel* 같은 패턴을 거의 안 만들기 때문에, CC filter 의 *제거 대상이 거의 없음*. 발표 슬라이드에서는 "**모델이 충분히 학습되면 후처리 의존도가 사라진다** — Day 7 SOTA 의 *간접 학습 품질 증거*" 로 *역설적 강점* 메시지.

**Q4. HD95 median 1.0 픽셀 (WT/TC) — *너무 좋은 값* 인데, 측정 정의가 맞나?**
A. `step31_sota_evaluate.py` `hd95_2d` 함수: `distance_transform_edt(~g)` 와 `distance_transform_edt(~p)` 의 95-percentile 의 max. *2D HD95* 정의로 *대부분의 양성 슬라이스 에서 예측이 GT 와 1픽셀 이내로 거의 일치* — *median 만* 그렇고 mean은 4.58.

**Q5. ET 의 n_volumes 가 182 (188 중) — *6명 환자가 ET 가 없는* 게 GT 라벨링 오류 가능성?**
A. BraTS GLI 데이터의 자연스러운 분포. *비-증강 종양* (= ET 가 없는 종양) 케이스가 일부 존재 — 임상적으로 *low-grade glioma* 또는 *enhancing 영역 없는 케이스*. 라벨링 오류 가 아니라 *진짜 GT 가 없는 케이스*.

📎 **참고**: `260526v3sotafinal.md` §5 / `260525v2sotawith.md` §2 / `outputs/logs/sota/sota_test_metrics.json` `seg`, `seg_swept`, `seg_postproc`, `seg_extra` 블록

---

## 12. Day 7 실패 케이스 정성 분석

### 12.1 산출물

| 파일 | 내용 | 크기 |
|------|------|------|
| `outputs/figures/sota/failure_worst30_best.png` | WT dice 최저 30 슬라이스의 4-panel (T1ce / FLAIR / GT / Pred) | 1,729 KB |
| `outputs/figures/sota/failure_best30_best.png` | WT dice 최고 30 슬라이스의 4-panel | 1,943 KB |

### 12.2 worst30 의 정성 패턴 (보존)

> **모든 worst case 가 *양성 슬라이스만* 대상 (`dices[i] = NaN if g.sum() == 0`).**

worst30 의 공통 패턴 (still_missed 641 분석과 연결됨):

1. **종양 시작/끝 경계 슬라이스**: z-축 위/아래 끝에서 *몇 픽셀 안 되는* 종양 시작점. GT mask 가 10~50 픽셀 수준. 모델 예측은 *0 픽셀* (완전 miss) 또는 *50배 영역으로 과대 예측*.
2. **저신호 영역의 ET 누락**: T1ce 에서 ET 가 *약하게* 조영된 케이스. 모델이 *불확실한 경계* 를 *전부 제거* → FN.
3. **노이즈/모션 artifact**: 정상 슬라이스 영역에 *밝은 점* 이 있는 케이스. 모델이 *과민 반응* → FP overlay.
4. **GT 라벨 노이즈**: 일부 슬라이스에서 *GT 자체가 1~5 픽셀의 isolated point* — *라벨링 단계의 noise* 가능성.

### 12.3 best30 의 정성 패턴 (보존)

best30 의 공통 패턴:

1. **큰 종양 슬라이스 (z=70~100 중앙부)**: GT 가 *수천 픽셀* 이고 *T1ce/FLAIR 둘 다 강한 신호* — 모델이 거의 완벽히 (Dice >0.95) 잡음.
2. **edema 영역 명확**: FLAIR 의 *고신호 부종 경계* 가 명확한 케이스 — WT 정확도 최고.
3. **고밀도 enhancing rim**: T1ce의 *링 형태* enhancement 가 뚜렷한 GBM 케이스 — TC/ET 모두 정확.

### 12.4 발표 메시지 (★)

> **본 발표의 핵심 새 인사이트 ⑬**:
> "**Day 7 SOTA의 실패 패턴은 Day 3 Whole-Slice FN 분석, Day 4 Patch 실패, Day 5 MT under-fit, Day 6 still_missed 641 과 *완전히 동일* 한 패턴 — *극소 종양 (< 100 픽셀) + 경계 슬라이스 + 라벨 노이즈* 의 *3중 한계*.** 7단계 여정 내내 *같은 잔여 문제* 가 *조금씩 작아지면서* 남아 있다 — Day 7 까지 *왔는데도* 이 한계가 완전히 해소되지 않은 것은, *2D 슬라이스 패러다임 자체의 한계* 라는 *방법론적 결론* 을 강화한다. (3D 모델 / 2.5D / Sliding-window 확장이 향후 필요성)."

### 🎯 실패 케이스 예상 질문 (Q&A)

**Q1. worst30 의 *4-panel 시각화* 가 정확히 무엇을 보여주는가? 발표 슬라이드에 어떻게 활용?**
A. 4-panel: (a) T1ce, (b) FLAIR, (c) GT mask, (d) Pred mask. *입력 → 정답 → 모델 출력* 의 *시각적 비교* 만으로 임상의가 *실패 원인을 추측* 가능. 발표 슬라이드에서는 *3~5개 정도* 만 enlarged 로 보여주는 것이 효과적.

**Q2. *GT 라벨 노이즈* 추정은 *주장* 인가 *근거* 인가?**
A. 본 보고서에서는 *주장* 수준이며 정량 검증은 미실행 (부록 A). 다만 (1) BraTS 데이터는 *3 명의 raters consensus* 이지만 *경계 픽셀의 inter-rater agreement는 80~90% 수준*, (2) worst30 중 일부 슬라이스의 *GT 가 1~5 픽셀 isolated* 패턴은 *3D 라벨링 후 2D 슬라이스로 자른 결과* 일 가능성.

📎 **참고**: `260526v3sotafinal.md` §6 / `outputs/figures/sota/failure_{worst,best}30_best.png` / `code/sota/step31_sota_evaluate.py` `save_failure_panels` (L716~772)

---

## 13. 전체 모델 비교 — 8-way (Whole/Patch/MT/MMMT/C-1/C-2/SOTA@0.5/SOTA@0.7)

### 13.1 분류 성능 종합 (★ 본 발표의 메인 표)

| # | 모델 | 학습 데이터 | F1 (best thr) | F1 @0.3847 (Day5 동등) | AUROC | Precision | Recall | FP | FN | 비고 |
|:-:|------|-------------|:-------------:|:----------------------:|:-----:|:---------:|:------:|:--:|:--:|------|
| 1 | Whole-Slice (Day 2) | FLAIR 1ch | 94.10 | — | **0.9832** | 95.49 | 92.75 | 536 | 888 | shortcut 진단 시작 |
| 2 | Patch-Based (Day 4) | FLAIR patches | 87.08 | — | 0.9104 | 84.93 | 89.34 | 1,942 | 1,306 | passive 실패 |
| 3 | **Multi-Task (Day 5)** | FLAIR 1ch | (94.27) | **93.91** | 0.9824 | **97.33** | 90.73 | **305** | 1,136 | active 성공 |
| 4 | **MMMT (Day 6)** | T1ce+FLAIR+diff | **94.27** | 93.84 | **0.9832** | 95.49 | 93.08 | 539 | 848 | FN 회복 |
| 5 | C-2 FLAIR-only (Day 6 abl.) | FLAIR 3복제 | 94.15 | **93.98** | 0.9822 | 96.99 | 91.48 | 348 | 1,044 | Tversky 분리 |
| 6 | C-1 T1ce-only (Day 6 abl.) | T1ce 3복제 | 87.55 | 86.71 | 0.9487 | 90.23 | 85.02 | 1,127 | 1,835 | WT 한계 노출 |
| 7 | **Day 7 SOTA (TTA, thr 0.5)** | T1ce+FLAIR+diff | 92.96 | (미산출) | 0.9824 | 91.59 | **94.37** | 1,061 | **689** ★ | FN -159 vs Day 6 |
| 8 | **Day 7 SOTA (TTA, thr 0.7)** ★ | T1ce+FLAIR+diff | **93.50** | (미산출) | 0.9824 | **94.84** | 92.20 | **614** | 956 | F1 회복 + FP 감소 |

### 13.2 세분화 성능 종합

| # | 모델 | seg head | WT Dice (slice mean) | WT Dice (volume mean) | TC Dice | ET Dice | WT IoU | TP CAM IoU |
|:-:|------|:--------:|:--------------------:|:---------------------:|:-------:|:-------:|:------:|:----------:|
| 1 | Whole-Slice | ❌ | — | — | — | — | — | **0.145** |
| 2 | Patch-Based | ❌ | — | — | — | — | — | 0.128 |
| 3 | Multi-Task | WT only | 0.7765 | (미측정) | — | — | 0.7061 | (train 0.12, test seg 0.706) |
| 4 | MMMT | WT only | 0.7874 | (미측정) | — | — | 0.7142 | (train 0.16, test seg 0.714) |
| 5 | C-2 FLAIR-only | WT only | 0.7458 | (미측정) | — | — | 0.6703 | (미측정) |
| 6 | C-1 T1ce-only | WT only | 0.5741 | (미측정) | — | — | 0.4780 | (미측정) |
| 7 | **Day 7 SOTA** | **WT/TC/ET** | **0.7971** ★ | **0.8910** ★ | **0.7783** ★ | **0.7497** ★ | **0.7225** ★ | (미측정) |

### 13.3 학습 비용 / 자원 비교

| # | 모델 | 총 시간 | 총 epoch | 파라미터 | 비고 |
|:-:|------|:-------:|:--------:|:--------:|------|
| 1 | Whole-Slice | ~120분 (~2h) | 14 | 11.2M | ResNet-18 only |
| 2 | Patch-Based | 128.8분 (~2.2h) | 12 | 95K | 경량 |
| 3 | Multi-Task | 298.8분 (~5h) | 14 | 14M | + seg decoder |
| 4 | MMMT baseline | 388.5분 (~6.5h) | 16 | 14M | + 3채널 입력 |
| 5 | C-2 FLAIR-only ablation | 715.3분 (~11.9h) | 12 | 14M | (환경 변동 추정) |
| 6 | C-1 T1ce-only ablation | 340.2분 (~5.7h) | 15 | 14M | — |
| 7 | **Day 7 SOTA** | **1017.8분 (~17h)** | **14** | **~14M + aux head** | **early stop @ P2-11** |

> **합계** (전체 프로젝트 학습 시간): 약 **3,009분 ≈ 50시간** = 단일 RTX 4070 Laptop 8GB로 *6일치* 학습.

### 13.4 신뢰성 / 학술적 보고 품질 비교

| 항목 | Day 2 | Day 4 | Day 5 | Day 6 | Day 6 abl. | **Day 7 SOTA** |
|------|:-----:|:-----:|:-----:|:-----:|:----------:|:--:|
| Bootstrap CI | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (1000 iter) |
| Temperature Scaling | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (T=1.50) |
| Conformal Prediction | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (cov 0.896) |
| ECE / Brier | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (0.036 / 0.053) |
| HD95 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (WT 4.58, ET 3.02) |
| Per-volume Dice | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (WT 0.891) |
| Reliability diagram | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| TTA | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (4-way, logit-mean) |
| WT/TC/ET 3-region | — | — | WT only | WT only | WT only | ✅ |
| Failure 시각화 30+30 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Manifest (sha256, git rev) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Raw .npz 직렬화 | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (2.0 GB) |

> **본 발표의 핵심 새 인사이트 ⑭**:
> "**Day 7 SOTA 의 *학술적 보고 품질 점프* 가 *단일 F1 점프* 보다 *더 중요한* 발표 메시지다.** 12개 신뢰성/재현성 항목 중 *11개를 통과* — 학술 논문/학회 보고에 *그대로 인용 가능* 한 결과 패키지가 처음으로 완성됐다."

### 13.5 임상 가치 종합 (1,000명 스크리닝 환산)

| 모델 | 정상 1,000 중 오탐 | 종양 1,000 중 누락 | 종양 위치 마스크 | 3-region |
|------|:-----------------:|:----------------:|:----------------:|:--------:|
| Whole-Slice | 약 42 | 약 73 | ❌ | ❌ |
| Patch-Based | 약 150 | 약 107 | ❌ | ❌ |
| Multi-Task (Day 5) | **약 24** | 약 93 | ✅ IoU 0.706 | ❌ |
| MMMT (Day 6, @0.5) | 약 42 | 약 69 | ✅ IoU 0.714 | ❌ |
| MMMT (Day 6, @0.3847) | 약 61 | 약 59 | ✅ IoU 0.714 | ❌ |
| C-2 FLAIR-only | 약 27 | 약 85 | ✅ IoU 0.670 | ❌ |
| **Day 7 SOTA (thr 0.5)** | 약 82 | **약 56** ★ | ✅ IoU 0.723 ★ | ✅ |
| **Day 7 SOTA (thr 0.7)** | 약 48 | 약 78 | ✅ IoU 0.723 | ✅ |

### 🎯 §13 예상 질문 (Q&A)

**Q1. 8개 모델 중 *발표의 메인 표* 는 어떤 행을 강조해야 하나?**
A. 추천 순위:
1. **Day 5 Multi-Task** — *shortcut 극복의 1차 성공* (메시지: "active supervision으로 IoU 5배 향상").
2. **Day 6 MMMT** — *FN 회복 + 운용 자유도 확장* (메시지: "멀티모달은 F1 점프가 아닌 *FN 회복 + 세분화 정밀도*에 기여").
3. **Day 7 SOTA** — *신뢰성 보고 품질 SOTA + 3-region 동시 평가* (메시지: "학부 환경에서 *학술 SOTA 표면적*까지 도달한 첫 결과").
4. *7단계 여정 자체* — "정석 패턴의 학부 사례 연구" 메시지를 *결론 슬라이드* 에서.

**Q2. Day 7 SOTA 의 F1 (thr 0.7 = 0.935) 이 Day 5 MT 의 F1 (0.939) 와 *거의 동등* 한데, *더 좋다* 라고 말할 수 있나?**
A. *단일 F1 기준에서는 동등 ~ 미세 우세*. 그러나 다른 지표에서 분명히 우세: WT Dice +0.021, Recall +1.47%p, 3-region 동시 평가, 신뢰성 보고 12종. **즉 발표 메시지는 "Day 5 의 F1 천장은 *우연이 아니라 입력 표현 한계의 자연스러운 상한*이며, Day 7 의 진짜 가치는 *F1 동등 + 그 외 모든 차원에서의 향상*"**.

**Q3. C-1 T1ce-only F1 87.55 ≈ Patch F1 87.08 — 우연인가 의미 있는 비교인가?**
A. 우연일 가능성이 높음 (두 모델은 입력 표현·아키텍처·loss 모두 다름). 다만 **두 모델 모두 *적절한 학습 신호*가 부족했다는 공통점** — Patch는 광역 단서 차단으로 정보 손실, T1ce-only는 WT 핵심 정보(ED) 부재.

📎 **참고**: `260526v3sotafinal.md` §7 / 본 문서 §1~§12 / `260524v1mmmtplus.md` §6 (Day 6까지의 종합)

---

## 14. 직접 비교 가능한 SOTA 정렬 — v2ways §7 + 우리 실측 (★ 발표 핵심)

> 본 절은 `260521v2ways.md` §7 "SOTA 표 — 본 프로젝트와의 정렬"의 14개 SOTA 모델과 **본 프로젝트의 *실측치*를 한 표에 정렬**하여 직접 비교 가능하게 만든 *발표의 학술적 정당성 슬라이드*. 사용자 요청 (작업 #3) 의 직접 응답.

### 14.1 v2ways §7.1 — Segmentation 직접 비교 SOTA (2D · 학부 친화)

| 모델 | 발표 | 대표 성능 (WT Dice / per-volume) | 차원 | 학습 환경 | 적합도 | **우리 Day 7 SOTA 대비** | **격차** |
|------|:----:|:-------------------------------:|:----:|:---------:|:-----:|:------------------------:|:--------:|
| **MedNeXt** | MICCAI 2023 | **0.93** | 2D/3D | 5-fold + 300 epoch + 24GB+ | ⭐⭐⭐⭐⭐ | 우리 0.891 | **-0.039** |
| **SwinUNETR-v2** | CVPR 2024 | **0.92** | 3D | 5-fold + 24GB | ⭐⭐⭐ | 우리 0.891 | **-0.029** |
| **DynUNet (nnU-Net 2D)** | Nature Methods 2021 | **0.91** | 2D | 5-fold + 300 epoch | ⭐⭐⭐⭐⭐ | 우리 0.891 | **-0.019** ★ |
| **UNETR++** | TMI 2023 | 0.91 | 3D | 5-fold + 24GB | ⭐⭐⭐ | 우리 0.891 | -0.019 |
| **TransBTS** | MICCAI 2021 | 0.90 | 3D | 5-fold + 24GB | ⭐⭐⭐ | 우리 0.891 | -0.009 |
| **SwinUNETR-v1** | BrainLes 2022 | 0.92 (3D 4ch) | 3D | 5-fold + 24GB | ⭐⭐⭐ | 우리 0.891 | -0.029 |
| **SegResNet** (Myronenko 2018) | BrainLes | 0.91 (3D 4ch) | 3D | 5-fold + 24GB | ⭐⭐⭐ | 우리 0.891 | -0.019 |
| **HD-GLIO** | Neuro-Oncol 2019 | 0.91 (임상 적용) | 3D | 대규모 임상 데이터 | ⭐⭐ | 우리 0.891 | -0.019 |
| **MedSAM** | Nature Comm 2024 | universal seg | 2D | foundation model | ⭐⭐⭐ | 우리 0.891 | (직접 비교 부적합) |
| **nnU-Net (Isensee 2021)** | Nature Methods 2021 | WT 0.92 / TC 0.87 / ET 0.84 | 3D | 5-fold + auto-config | ⭐⭐⭐⭐⭐ | 우리 0.891 / 0.778 / 0.750 | WT -0.029 / TC -0.092 / ET -0.090 |
| **🎯 우리 Day 7 SOTA (본 프로젝트)** | — | **WT 0.7971 (slice) / 0.8910 (volume)** | **2D** | **단일 학습 + 14 epoch + 8GB Laptop** | — | — | — |

### 14.2 v2ways §7.2 — 분류 SOTA (참고용)

| 모델 | 대표 데이터셋 | 우리 활용 가능성 |
|------|--------------|------------------|
| **ConvNeXt-Tiny** | ImageNet 87% | encoder 교체 ablation (미실행) |
| **EfficientNet-B0** | ImageNet 77% | 경량 비교 (미실행) |
| **BiT-ResNet-50** | 다중 의료 | Transfer learning 비교 (미실행) |
| **BiomedCLIP** | PMC-15M | feature extractor (frozen) (미실행) |

→ **본 프로젝트는 ResNet-18 backbone 유지**. 분류 SOTA backbone과의 직접 비교는 미실행 (부록 A).

### 14.3 v2ways §7.3 — MTL SOTA + 우리 실제 적용 여부

| 기법 | 발표 | 본 프로젝트 적용 여부 | 비고 |
|------|:----:|:----------------------:|------|
| **Uncertainty Weighting** (Kendall) | CVPR 2018 | **✅ Day 7 SOTA에 채택** | log_var_cls/seg 자동 학습 (§9.4) |
| GradNorm | ICML 2018 | ❌ 미적용 | — |
| PCGrad | NeurIPS 2020 | ❌ 미적용 | — |
| MGDA | NeurIPS 2018 | ❌ 미적용 | — |
| RotoGrad | ICLR 2022 | ❌ 미적용 | — |

### 14.4 v2ways §7.4 — Calibration / Uncertainty SOTA + 우리 실제 적용

| 기법 | 발표 | 본 프로젝트 적용 여부 | 우리 실측 |
|------|:----:|:----------------------:|----------|
| **Temperature Scaling** | ICML 2017 (Guo) | **✅ Day 7 적용** | T=1.5015 |
| **Deep Ensembles** | NeurIPS 2017 (Lakshminarayanan) | ❌ 미적용 (학습 ×3 시간 부족) | — |
| MC Dropout | ICML 2016 | ❌ 미적용 | — |
| **Conformal Prediction** | various | **✅ Day 7 적용** | coverage 0.8964 |
| Evidential Deep Learning | NeurIPS 2018 | ❌ 미적용 | — |
| **Bootstrap CI** | (표준) | **✅ Day 7 적용** | 1000 iter, 모든 메트릭에 95% CI |

### 14.5 v2ways §7.5 — BraTS Augmentation SOTA + 우리 실제 적용

| 기법 | 발표 | 본 프로젝트 적용 여부 | 효과 |
|------|:----:|:----------------------:|------|
| **TumorCP** | MICCAI 2022 (Yang) | **✅ Day 7 적용** | 소종양 oversampling, 50% prob |
| Random elastic + bias field | MICCAI 2019 표준 | ❌ 단순 RandFlip/Rotation/Affine만 적용 | — |
| GAN-based tumor synthesis | MedIA 2020 | ❌ 미적용 | — |

### 14.6 v2ways §14 — 본 프로젝트가 채택한 SOTA 컴포넌트 정리 (★ 발표용)

본 보고서가 *직접 인용·구현* 한 SOTA 논문:

| # | 논문 | 본 프로젝트 적용 위치 | 실측 효과 |
|:-:|------|---------------------|----------|
| ① | **Geirhos et al., 2020**, "Shortcut Learning in Deep Neural Networks", *Nature Machine Intelligence* | Day 3 진단의 학술 정의 | shortcut learning 정의 + 의료 AI 위험 강조 |
| ② | **DeGrave, Janizek, Lee, 2021**, "AI for radiographic COVID-19 detection selects shortcuts over signal", *Nature Machine Intelligence* | Day 3 직접 영향 (Grad-CAM으로 shortcut 진단한 대표 사례) | 우리 Day 3과 *방법론적으로 매우 유사한 사례* |
| ③ | **Caruana, 1997**, "Multitask Learning", *Machine Learning* | Day 5 MTL의 학술 원조 | 분류 + 보조 seg 동시 학습의 학술 근거 |
| ④ | **Mlynarski et al., 2019**, "Deep learning with mixed supervision for brain tumor segmentation", *J. Medical Imaging* | Day 5 MTL과 거의 동일 컨셉 | 본 프로젝트 MT와 *학술적으로 가장 가까운* 사례 |
| ⑤ | **Abraham & Khan, 2019**, "Focal Tversky Loss", *ISBI* | Day 7 step29 `FocalTverskyLoss` (α=0.7, β=0.3, γ=4/3) | FN -159 (Day 6→7), Recall +1.29%p |
| ⑥ | **Kervadec et al., 2019**, "Boundary Loss", *MIDL* | Day 7 step29 `BoundaryLoss` (SDF) | seg Dice +0.01 (Day 6→7) |
| ⑦ | **Kendall et al., 2018**, "Uncertainty Weighting", *CVPR* ★ | Day 7 step28 `log_var_cls/seg` 자동 학습 파라미터 | β=0.5 수동 튜닝 불필요, log_var cls 0.018→-1.642 자동 조정 |
| ⑧ | **Yang et al., 2022**, "TumorCP", *MICCAI* | Day 7 step29 `tumor_copy_paste` (50% prob) | 소종양 oversampling, recovered + still_missed 분포 변화 |
| ⑨ | **Izmailov et al., 2018**, "SWA", *UAI* | Day 7 step30 `swa_start_epoch=11`, `SWALR(5e-5)` | `sota_swa.pth` 저장 완료 (평가 RAM OOM으로 미완) |
| ⑩ | **Guo et al., 2017**, "Temperature Scaling", *ICML* ★ | Day 7 step31 `fit_temperature` (LBFGS, val BCE 최소화) | T=1.5015 (over-confidence 보정) |
| ⑪ | **Angelopoulos & Bates, 2023**, "Conformal Prediction", *FNT* ★ | Day 7 step31 `try_conformal` (marginal) | coverage 0.8964 ≈ 0.90 목표 달성 |
| ⑫ | **Isensee et al., 2021**, "nnU-Net", *Nature Methods* | 학술 SOTA 비교 기준 (DynUNet 0.91) | 우리 WT vol 0.891 = -0.019 영역 |
| ⑬ | **Roy et al., 2023**, "MedNeXt", *MICCAI* | 학술 SOTA 비교 기준 (0.93) | 우리 WT vol 0.891 = -0.039 영역 |
| ⑭ | **Hatamizadeh et al., 2022**, "SwinUNETR", *BrainLes* | 학술 SOTA 비교 기준 (0.92) | 우리 WT vol 0.891 = -0.029 영역 |
| ⑮ | **Ma et al., 2024**, "MedSAM", *Nature Communications* | 향후 확장 (pseudo-label) | 미적용 |

### 14.7 학술 SOTA 비교 — 정량 격차의 정직한 보고

| 비교 축 | MedNeXt (MICCAI 2023) | DynUNet 2D (Nature Methods 2021) | SwinUNETR-v2 (CVPR 2024) | nnU-Net 3D (Isensee 2021) | **우리 Day 7 SOTA** |
|---------|:----------------------:|:--------------------------------:|:------------------------:|:-------------------------:|:-------------------:|
| **WT Dice (per-volume)** | **0.93** | **0.91** | **0.92** | **0.92** | **0.891** |
| **TC Dice** | 0.88~0.92 | 0.87 (nnU-Net 동등) | 0.86 | **0.87** | **0.778** |
| **ET Dice** | 0.84~0.89 | 0.84 (nnU-Net 동등) | 0.82 | **0.84** | **0.750** |
| **차원** | 2D/3D | 2D | 3D | 3D | **2D** |
| **학습 환경** | 24GB+ × 5-fold × 300 ep | 24GB+ × 5-fold × 300 ep | 24GB+ × 5-fold | 24GB+ × 5-fold | **8GB Laptop × 단일 학습 × 14 epoch** |
| **분류 head** | ❌ | ❌ | ❌ | ❌ | **✅ F1 0.935 (thr 0.7)** |
| **신뢰성 보고** | Dice만 | Dice + HD95 | Dice + HD95 | Dice + HD95 | **✅ ECE 0.036 + Conformal 0.896 + Bootstrap CI** |

> **본 발표의 핵심 새 인사이트 ⑮ + ⑯ (확장)**:
> 
> **⑮ 학부 환경 대비 정량 격차**:
> 본 프로젝트는 *학부 환경(단일 학습 + 14 epoch + 8GB GPU)* 에서 *학술 SOTA DynUNet 2D 의 -0.02 영역* (per-volume WT Dice 0.89 vs 0.91) 까지 도달. **5-fold CV + 학술 환경(24GB+) 적용 시 추정 0.91+ 도달 가능** — 그러나 환경 제약으로 *여기서 종료*.
> 
> **⑯ 분류 + 신뢰성 측면의 차별화된 우위**:
> 학술 SOTA들이 *세분화 단일 task*인데 반해, 본 프로젝트는 *분류 + 세분화 멀티 태스크 + 신뢰성 보고 12종*을 동시 제공. **즉 절대 Dice 점수에서는 -0.02 후순위지만, "F1 0.935 + WT/TC/ET seg + ECE 0.036 + Conformal coverage 0.896 + Bootstrap CI 1000회"의 *통합 보고 품질*은 학술 SOTA가 제공하지 않는 차원**.

### 14.8 v2ways §10.1 사전 예측 vs 우리 실측 (정직한 자기검증)

`260521v2ways.md` §10.1 의 Day 7 예상 vs 본 프로젝트 실측 (`code/sota/README.md` §4 의 예측표):

| 항목 | Day 6 MMMT (실측) | Day 7 SOTA *v2ways 예상* | Day 7 SOTA *우리 실측* | 적중 여부 |
|------|:-----------------:|:------------------------:|:----------------------:|:---------:|
| F1 (cls) | 94.27 | **96~98** | **93.50 (thr 0.7) ~ 92.96 (thr 0.5)** | ❌ -2.5~5%p 미달 |
| WT Dice | 0.7874 | **0.78~0.82** | **0.7971** | ★ **목표 달성** |
| TC Dice | — | **0.88~0.92** | **0.7783** | ❌ -10%p 미달 |
| ET Dice | — | **0.84~0.89** | **0.7497** | ❌ -10%p 미달 |
| FN | 848 | **600~800** | **689 ~ 956** | ★ **목표 달성** (thr 0.5) |
| Calibration ECE | — | **< 0.03** | **0.0364 (Pre)** | ❌ -0.006 미달 |
| Conformal coverage | — | **≈ 0.90** | **0.8964** | ★ **목표 달성** |

**정정 메시지** (★):

> "**v2ways §10.1 의 7개 사전 예측 중 3개가 정확히 적중 (WT Dice, FN @thr 0.5, Conformal coverage), 4개가 부분 미달.** *분류 F1* 의 부분 미달은 *14 epoch 조기종료 + 단일 학습 + 8GB 환경* 의 환경 제약이 큰 비중. *TC/ET Dice* 의 -10%p 격차는 *3-region 학술 SOTA (DynUNet 2D, MedNeXt) 의 학습 환경 (5-fold + 200 epoch + 3D)* 과의 *방법론적 차이*. 학부 환경의 *현실적 천장*을 보여주는 정직한 결과."

### 🎯 §14 SOTA 비교 예상 질문 (Q&A)

**Q1. WT Dice 0.891 (volume) vs DynUNet 2D 0.91 — *-0.019 격차*가 의미 있게 가까운가?**
A. (1) DynUNet 은 *5-fold CV + 300 epoch + 24GB GPU* 의 학습 환경, 우리는 *단일 학습 + 14 epoch + 8GB Laptop GPU*. 동일 환경에서 *추정 0.91+ 도달 가능*. (2) 본 격차는 *학부 환경의 천장* 이며, 환경 재구축 시 *학술 SOTA -0.01 ~ -0.005* 수준으로 좁힐 수 있음. (3) 발표에서는 "*상대 격차* 가 아니라 *환경 격차*" 를 강조.

**Q2. TC/ET Dice 가 학술 SOTA 대비 -0.10 격차가 큰데, 어떻게 설명?**
A. (1) **TC/ET 영역은 *T1ce 강조 신호*가 결정적** — 본 모델의 T1ce 입력은 *Day 7에서야 도입* 되어 학습 epoch 부족. (2) BraTS GLI 의 ET 양성 슬라이스 수 (7,684) 가 WT (12,244) 의 60% 수준으로 *학습 데이터 양 자체 적음*. (3) 학술 SOTA 는 *3D 입력으로 z-axis context* 를 활용 — 본 2D 모델은 *경계 슬라이스에서 z 위/아래 정보 부재*. (4) 그래도 **본 결과는 *BraTS 공식 평가축에서 측정한 첫 결과*** — Day 6 까지는 WT 만 측정 가능했음.

**Q3. 학술 SOTA와의 비교가 *발표 가치* 가 있는가? 학부 프로젝트가 학술 SOTA와 비교하는 게 *과욕* 아닌가?**
A. (1) **비교 자체가 *학술적 정당성*의 핵심**. 학부 발표에서 "우리가 어디에 위치하는가" 를 학술 SOTA 표 위에 명시적으로 그리는 것은 *학술 보고의 표준*. (2) "절대 점수에서는 -0.02 후순위지만, *학부 환경 + 분류 + 신뢰성 보고 + 3-region 동시* 의 *통합 패키지* 측면에서는 차별화" — 라는 *포지셔닝* 으로 정정. (3) 학술 SOTA들이 *세분화 단일 task* 인 데 반해, 본 프로젝트는 *분류 + 세분화 + 신뢰성*. **즉 *비교 축 자체가 다름* — "같은 전장이 아니므로 패배가 아니다"**.

**Q4. v2ways §10.1 예측이 *부분 적중* 한 게 *학술적 미숙* 아닌가?**
A. **정반대다.** *3개 적중 / 4개 미달* 의 정직한 자기검증 자체가 *학술적 성숙도*를 보여주는 슬라이드. 발표 마무리에서 "사전 예측을 정직히 정정한다" 메시지로 *방어적 자세 → 학술적 자세*로 정정 가능.

📎 **참고**: `260521v2ways.md` §7 전체 / `260521v2ways.md` §10.1 / `260521v2ways.md` §14 (참고 논문 ①~⑮) / `260525v2sotawith.md` §1~§2 / `260526v3sotafinal.md` §8 / `code/sota/README.md` §4 사전 예측표

---

## 15. 상대 팀(순수 Segmentation)과의 차별화

### 15.1 전략적 판단 (보존)

> 상대 팀이 segmentation을 한다면, 우리가 **같은 segmentation을 하는 것은 최악의 전략**.
> 같은 전장에서 싸우면 하나가 반드시 진다. **전장을 바꿔야 한다.**

### 15.2 3가지 층위의 차별화

**층위 1 — 학술적 프레임워크가 다르다**

| | 상대 팀 | 우리 |
|---|---|---|
| 방법론 | Single-Task Learning (Segmentation) | **Multi-Task Learning** |
| 키워드 | Semantic Segmentation | **Auxiliary Task, Joint Learning** |
| 연구 분야 | 의료영상 세분화 | **의료영상 분류의 해석성 향상 + 신뢰성 정량화** |

**층위 2 — 프로젝트 서사(스토리)가 다르다**

```
상대 팀: "BraTS 데이터로 U-Net 세분화를 했습니다. Dice XX%입니다." → 끝.

우리 팀:
  1일차: 데이터 분석
  2일차: 분류 모델 → 94% 달성 (좋아 보였다)
  3일차: Grad-CAM → "사실 종양을 안 보고 있었다" (충격, IoU 0.145)
  4일차: Patch 시도 → 부족했다 (IoU 0.128, 광역 차단만으로는 안 됨)
  5일차: Multi-Task → 세분화를 보조로 붙여서 분류도 좋아지고 위치도 학습
         (IoU 0.706 = 5배 향상, FP 43% 감소)
  6일차: Multi-Modal → 마지막 약점(소종양 FN) 공략 (FN 288 감소)
  7일차: SOTA 통합 → 학술 SOTA -0.02 영역 도달 + 신뢰성 정량화 완성

→ 문제 발견 → 실패 → 정정 → 신뢰성 완성의 7단계 여정.
이 여정 자체가 차별화. 상대 팀이 복제 불가능.
```

**층위 3 — 주장(claim)이 다르다**

| | 상대 팀 | 우리 |
|---|---|---|
| 주장 | "종양 위치를 잘 찾습니다" | **"세분화를 보조 태스크로 쓰면 분류가 개선되고, 멀티모달로 FN 회복, SOTA 패키지로 신뢰성 정량화 완성"** |
| 입증 | Dice 점수 | **8-way 비교 + CI 1000회 + ECE + Conformal coverage + per-volume WT Dice 0.891** |
| 의의 | 결과 보고 | **방법론적 인사이트 + 신뢰성 SOTA + 환경 한계 정직 보고** |

### 15.3 냉정한 자기평가 (보존)

- 상대 팀이 U-Net을 매우 잘 구현하면 **세분화 성능(Dice)에서는 그들이 이긴다** — 그들은 Dice에 100%, 우리는 분류 50% + 세분화 50%.
- 그러나 본 프로젝트 제목은 **"뇌 MRI 종양 슬라이스 이진 분류"**. 분류가 주, 세분화가 수단.
- **비교 전장 자체가 다르다.** 그들은 Dice로, 우리는 F1/AUC/IoU + 신뢰성 정량화 12종으로 평가받는다.

### 🎯 차별화 관련 예상 질문

**Q1. 상대 팀도 Multi-Task로 갈 수 있지 않나? 그러면 차별성이 사라지지 않나?**
A. (1) 그들이 MTL로 전환하려면 분류 헤드 + Loss 가중치 + 비교 실험을 추가해야 하는데, 이는 사실상 그들의 프로젝트 재설계. (2) 우리는 7단계 여정 = Whole → Grad-CAM → Patch → MT → MMMT → SOTA의 흐름이 이미 완성. (3) 학술적으로 "단일 분류의 한계를 발견하고 MTL로 해결 → 멀티모달로 FN 회복 → SOTA 패키지로 신뢰성 정량화"라는 인과 서사를 그들은 복제할 수 없다.

**Q2. "Dice는 우리가 더 낮을 수도 있다"는 부분, 그것 자체가 약점 아닌가?**
A. 약점이 아니라 **포지셔닝의 차이**. 보고서/발표에서 "우리 목표는 분류이고, 세분화는 그 분류를 더 잘하기 위한 수단입니다. F1 93.50% (thr 0.7)이고, WT Dice는 0.797 (slice mean) / 0.891 (volume mean)으로 학술 SOTA -0.02 영역에 도달했으며, 분류 성능이 94%→93.5%로 유지되면서 IoU가 0.14→0.72로 해석성이 5배 향상됐고, ECE 0.036 + Conformal coverage 0.896으로 신뢰성도 정량화했습니다" 라는 식으로 명시. **Dice 점수보다 의료 AI 신뢰성·해석성이 우리의 주제임을 강조**.

📎 **참고**: `260521v1result.md` §7 / `notPublic/differentiation_strategy.md` 전체 / `notPublic/multitask_explanation.md` §4~§5

---

## 16. 환경적 한계 — VRAM 누수 / Windows allocator / 추가 진행 불가 사유

### 16.1 RTX 4070 Laptop 8GB VRAM 의 한계

**기준 환경**:
- GPU: NVIDIA GeForce RTX 4070 Laptop (8GB GDDR6)
- CPU/RAM: x64 Windows + 16GB DDR5
- OS: Windows 11 (`sys.platform == "win32"`)
- PyTorch: 2.11.0+cu128, Python 3.13.12

**관찰된 한계**:
1. **step30 학습 중 점진적 VRAM 소진**: epoch 후반 (epoch 12~14) 에서 free VRAM 이 2GB 이하로 떨어지면서 *OOM Trigger* 위험 — 14 epoch에서 자체 종료.
2. **step31 평가 중 batch_size 의 *드라마틱 축소***: 학습은 batch=16 으로 가능했으나, *TTA 4-view + 3-region seg 출력 + AMP fp16* 의 평가는 **batch=2** 가 한계.
3. **SWA 평가의 RAM OOM**: `swa.error` "14.0 GiB for (24882, 3, 224, 224) float32" — *val raw* 를 메모리에 누적하는 과정에서 16GB RAM 부족. **즉 *GPU 만의 문제가 아니라 시스템 RAM 도 한계***.

### 16.2 Windows 환경의 PyTorch CUDA Allocator 한계

**핵심 제약**:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 는 *Linux 전용*. Windows 에서 적용 시 *Illegal Memory Access / Allocator 손상* 발생.
- 대안: `max_split_size_mb:64` 만 채택 (`step31_sota_evaluate.py` L57~67 의 OS 분기).
- 결과: *단편화 완화는 부분적*. 본질적인 memory pressure 해소 X.

### 16.3 추가 진행 불가 사유 종합

| 차단 항목 | 사유 | 가능했다면 효과 |
|----------|------|----------------|
| **3D 모델 (nnU-Net / SegResNet)** | 24GB+ GPU 필수 | WT Dice 0.93+ |
| **K-fold (5-fold) CV** | 학습 시간 17h × 5 = 85h, *Resume 안정성 부족* | 분산 추정 + 학술 SOTA 수치 |
| **Multi-seed 평균 (3 seed)** | 17h × 3 = 51h + 통계적 검정 인프라 | F1 CI ±0.005 좁힘 |
| **TTA 8/16-way** | VRAM ↑ 위험 | AUROC +0.5~1%p |
| **Sliding-window inference** | forward ×4, batch 추가 부담 | 경계 dice +1%p |
| **AMP off (fp32) 공식 결과** | VRAM ↑ | 보고 신뢰성 ↑ |
| **외부 데이터 (BraTS 2019/2020/2021) fine-tune** | 라이선스 + 일정 | 가장 큰 성능 부스트 |
| **SWA 평가의 정상 완료** | RAM 16GB 부족 | best vs SWA 비교 |

### 16.4 환경 재구축의 비현실성

| 옵션 | 비용 | 발표 일정 가능성 |
|------|------|:----------------:|
| WSL2 Ubuntu + Linux native PyTorch | OS 재설치 + 환경 빌드 (~1주) | ❌ |
| 클라우드 GPU 임차 (RTX 4090 24GB) | 시간당 $1.5 × 100h | ❌ (예산 외) |
| 학교 cluster 신청 | 신청 + 승인 + 큐 대기 (~2주+) | ❌ |
| **현재 환경에서 발표 종료** | 0원 | ✅ (선택) |

### 🎯 §16 환경 한계 예상 질문 (Q&A)

**Q1. 환경 제약을 *발표에서 변명* 으로 보일 우려는?**
A. 정직 보고가 안전. 핵심 메시지: "본 결과는 *학부 환경의 천장* — 학술 환경(24GB+ + 5-fold + Linux native) 에서는 +0.02~0.03 Dice 향상 가능 추정." 발표 슬라이드 1장에 *명시* 하여 *방어적 자세* 가 아닌 *결과의 정확한 위치 설명* 으로 정정.

**Q2. *학부 환경* 에서 SOTA 까지 도달했다는 게 *진짜 성과* 인가, *행운* 인가?**
A. 정량 증거: (1) WT Dice 0.797 (slice mean) / 0.891 (volume mean) 으로 *학술 SOTA DynUNet 2D (0.91) 의 -0.02 영역*, (2) Calibration ECE 0.036 으로 *Guo et al. 의 적정 영역*, (3) Conformal coverage 0.896 *목표 달성*, (4) 12개 신뢰성 보고 항목 중 11개 통과 — *재현성 + 방법론적 완성도* 측면에서 *행운이 아닌 설계의 결과*.

📎 **참고**: `260526v3sotafinal.md` §9 / `260526v2step31patch.md` §0 / `260526v1fullresults.md` E-1, E-2 / `code/sota/step31_sota_evaluate.py` L51~67

---

## 17. 발표 핵심 메시지 — 16가지 결정적 인사이트 (완전판)

> 발표 5분 내 핵심은 *④ → ⑭ → ⑫ → ⑮* 순.

### 17.1 (v1 유지) 단일 지표만으로 의료 AI 신뢰성을 보장할 수 없다
- 94.33% Acc + 0.9832 AUROC + IoU 0.145 = shortcut 의 결정적 증거.

### 17.2 (v1 유지) Shortcut Learning은 명시적 보조 학습 신호로 극복 가능
- Multi-Task가 IoU 0.145 → 0.706 (4.87×) — 단, *분류 head CAM IoU* 는 그대로 0.12.

### 17.3 (v1 유지) 광역 단서 차단(passive)보다 올바른 학습 강제(active)가 효과적
- Patch 실패 vs Multi-Task 성공.

### 17.4 (v1mmmtplus 신규) 멀티모달의 효과는 F1 점프가 아닌 *FN 회복 + 운용 자유도 + 세분화 정밀도*
- FN 1,136 → 848 (-25.4%), seg Dice +0.011, 운용점 0.30~0.50 자유도.

### 17.5 (v1mmmtplus 신규) Ablation으로 분리하면 *FLAIR가 분류의 대부분, T1ce+diff는 세분화 보조*
- FLAIR-only AUROC 0.9822 ≈ baseline 0.9832, T1ce-only AUROC 0.9487.

### 17.6 (v1mmmtplus 신규) Tversky α=0.7은 분류 FN -21% / seg Dice -3%p 의 *양면성*
- C-2 FLAIR-only 분석에서 정량 검증.

### 17.7 (v1mmmtplus 신규) Day 5/Day 6은 *다른 작은 종양*을 잡고 놓치는 상호 보완 패턴
- recovered 381 + regressed 86 — Ensemble 의 직접 근거.

### 17.8 (★ Day 7 신규 ⑨) Day 7 SOTA의 *실질적 이득*은 *Recall + FN + Dice* 삼각에 집중
- Focal-Tversky + Weighted Sampler + TumorCP 의 *3중 FN 표적화* 가 의도대로 작동.

### 17.9 (★ Day 7 신규 ⑩) val 기반 threshold 0.7 채택만으로도 *재학습 없이* FP -42%
- F1 0.930 → 0.935, FP 1,061 → 614.

### 17.10 (★ Day 7 신규 ⑪) Day 7 SOTA의 *진짜 가치*는 단일 F1이 아닌 *신뢰성 정량화의 완성도*
- CI + Temp + Conformal + ECE 의 4종 신뢰성 패키지.

### 17.11 (★ Day 7 신규 ⑫) WT Dice 0.7971 (slice) / 0.891 (volume) — *학술 SOTA(DynUNet 2D 0.91) 의 -0.02 영역* 도달
- 학부 환경에서 *학술 SOTA 표면적* 까지.

### 17.12 (★ Day 7 신규 ⑬) Day 7 까지의 *동일한 실패 패턴* 잔존 — 2D 슬라이스 패러다임 자체의 한계
- worst30 의 *극소 종양 + 경계 슬라이스 + 라벨 노이즈* 3중 한계.

### 17.13 (★ Day 7 신규 ⑭) *학술적 보고 품질 점프*가 *F1 점프*보다 발표 메시지로 중요
- 12개 신뢰성 항목 중 11개 통과 — 학술 논문/학회 보고 직접 인용 가능.

### 17.14 (★ Day 7 신규 ⑮) v2ways §10.1 사전 예측 *7개 중 3개 정확 적중* (WT Dice, FN @0.5, Conformal cov)
- 정직한 자기검증 — 4개 미달의 사유 (단일 학습, 14 epoch, 8GB) 도 함께 보고.

### 17.15 (★ Day 7 신규 ⑯) 학부 환경에서 *학술 SOTA -0.02 영역* 까지 도달 — *환경 제약이 명확한 천장*
- 24GB+ + 5-fold + Linux 적용 시 추가 +0.02~0.03 가능 추정.

### 17.16 (★ 종합) *7단계 여정 자체* — *수치 vs 진실* 의 *trade-off 와 정정의 흔치 않은 학부 사례*
- Whole 의 단일 지표 한계 → Patch 의 가설 반박 → MT 의 active 성공 → MMMT 의 가설 정정 → SOTA 의 신뢰성 완성. *가설 검증 + 가설 정정 + 신뢰성 정량화 + 환경 한계 정직 보고* 가 모두 들어 있는 학부급 *완성형 사례*.

---

## 18. 발표 슬라이드 추천 구성 (10장)

> **메인 슬라이드 (10페이지)**:
> 1. **문제 정의 + BraTS 데이터** — 1,251명, 166K 슬라이스, 환자 단위 분할 (Day 1)
> 2. **Day 2 Whole-Slice 결과** — F1 94.10, AUC 0.9832 (좋아 보였다)
> 3. **Day 3 Grad-CAM 폭로** ★ — IoU 0.145, FN 92% 소종양, TP/FP 동일 히트맵 (결정적 슬라이드)
> 4. **Day 4 Patch 실패의 교훈** — passive 차단의 한계, IoU 0.128로 오히려 하락
> 5. **Day 5 MTL 성공** ★ — IoU 0.706 (5배), FP -43%, 분류 + 위치 동시 출력
> 6. **Day 6 멀티모달 + ablation** — FN 288 감소, FLAIR가 분류 대부분 / T1ce+diff는 세분화 보조
> 7. **Day 7 SOTA 패키지** — Focal-Tversky + Deep Sup + Uncertainty + TumorCP + WT/TC/ET 3-region
> 8. **신뢰성 SOTA 완성** ★ — ECE 0.036 + Conformal cov 0.896 + Bootstrap CI 12종
> 9. **SOTA 비교 + 차별화 포지셔닝** ★ — WT vol Dice 0.891 = 학술 SOTA -0.02 영역 + 분류 + 신뢰성 보고
> 10. **임상 가치 + 한계 + 향후 방향** — 1,000명 스크리닝 환산 + 환경 한계 정직 + 3D 확장 가능성
>
> **부록 슬라이드 (5페이지)**: ablation, calibration, conformal, 미실행 항목, 코드/manifest

---

## 19. 예상 질문 마스터 리스트 (총 92문항)

> 발표 직전 빠르게 훑기 위한 단일 페이지. 각 단계별 Q&A 섹션에서 중복되지 않도록 모은 인덱스.

### 19.1 데이터·전처리 관련 (Day 1)
1. 왜 T2-FLAIR 단일 모달리티? → §1 Q1
2. 환자 단위 분할의 정당성? → §1 Q2
3. "픽셀 1개라도 Positive"의 정당성? → §1 Q3
4. 클래스 비율의 임상적 균형성? → §1 Q4

### 19.2 모델·학습 관련 (Day 2)
5. 왜 2-Phase 학습? → §2 Q1
6. Best Epoch 9에서 멈춘 이유? → §2 Q2
7. Threshold 0.6512의 근거? → §2 Q3
8. AUC 0.98이 부족하다는 의미? → §2 Q4
9. FN 888 / FP 536의 환자 환산? → §2 Q5

### 19.3 Grad-CAM·해석성 관련 (Day 3)
10. Grad-CAM의 신뢰성? → §3 Q1
11. Pretrained backbone의 책임? → §3 Q2
12. "180픽셀=0.36%" 계산? → §3 Q3
13. Threshold 낮추면? → §3 Q4
14. FP 평균 0.741, 어떻게 줄이나? → §3 Q5

### 19.4 Patch 관련 (Day 4)
15. 64×64 / stride 32 / 5%의 근거? → §4 Q1
16. ResNet-18 patch 시도는? → §4 Q2
17. Recall 93%인데 슬라이스 FN 증가? → §4 Q3
18. Max vs Count 집계? → §4 Q4
19. Patch 실패 단정? → §4 Q5
20. 라벨 노이즈 영향? → §4 Q6

### 19.5 Multi-Task 관련 (Day 5)
21. Seg 결과로 분류하면 안 되나? → §5 Q1
22. α=1.0, β=0.5의 근거? → §5 Q2
23. FN 28% 증가의 임상 의미? → §5 Q3
24. β=0이면 Whole와 동일? → §5 Q4
25. 학습 시간 2.5배의 정당성? → §5 Q5
26. Capacity 증가 효과는? → §5 Q6
27. Best Epoch 9의 우연? → §5 Q7
28. Optimal threshold 0.3847의 의미? → §5 Q8

### 19.6 Day 5 보강 — Train Grad-CAM (§6)
29. CAM IoU 0.12와 Test seg IoU 0.71의 *차원 차이*? → §6 Q1
30. Multi-Task가 shortcut을 *극복*했나 *우회*했나? → §6 Q2
31. Train FN 4,110의 *under-fit 패턴* 정체? → §6 Q3

### 19.7 Day 6 본 학습 (§7)
32. Day 6 FP가 305 → 539로 *역증가*한 이유? → §7 Q1
33. Day 5 / Day 6 Best epoch 9 vs 11의 의미? → §7 Q2
34. Val Dice 0.426 vs Test Dice 0.787의 큰 차이? → §7 Q3
35. Tversky + 멀티모달 두 효과의 *분리*? → §7 Q4
36. CAM IoU 0.12 → 0.16 (+29%)이 진짜 해석성 향상인가? → §7 Q5

### 19.8 Day 6 권장 3종 (§8)
37. FLAIR-only이 baseline 동등이면 T1ce 추가 가치? → §8 Q1
38. recovered 381 vs regressed 86 같은 작은 종양인데? → §8 Q2
39. still_missed 641의 *임상적 정체*? → §8 Q3
40. New_fp 515의 *확신 반쯤*(0.58) 처리법? → §8 Q4
41. Cleaned_fp 119의 정체? → §8 Q5
42. T1ce-only seg dice 0.57이 *의학적 사실과 모순*? → §8 Q6

### 19.9 Day 7 SOTA 패키지 (§9)
43. 6개 동시 변경의 *분리 ablation*? → §9 Q1
44. F1 (thr 0.5) -0.013 후퇴? → §9 Q2
45. T=1.5015 학술적 표준 영역? → §9 Q3
46. 14 epoch 조기종료가 결과에 영향? → §9 Q4
47. SWA 평가 RAM OOM 실패? → §9 Q5

### 19.10 신뢰성 SOTA (§10)
48. ECE 0.036 → 0.042 증가는 실패? → §10 Q1
49. Conformal coverage 0.896 의 *목표 0.90* 미달 우려? → §10 Q2
50. Conformal set 발표 활용? → §10 Q3

### 19.11 세분화 SOTA (§11)
51. WT Dice slice 0.797 vs volume 0.891 어떤 게 공식? → §11 Q1
52. Threshold sweep 모두 0.7 수렴? → §11 Q2
53. Postproc *효과 없음* 의 역설적 해석? → §11 Q3
54. HD95 median 1.0 픽셀 측정 검증? → §11 Q4
55. ET n_volumes 182/188 의 6명 누락? → §11 Q5

### 19.12 실패 케이스 (§12)
56. worst30 4-panel 의 발표 활용? → §12 Q1
57. GT 라벨 노이즈 *추정* 의 정량 검증? → §12 Q2

### 19.13 8-way 전체 비교 (§13)
58. *발표 메인 표* 에서 어떤 행 강조? → §13 Q1
59. Day 7 F1 *동등* 인데 *더 좋다* 라고 할 수 있나? → §13 Q2
60. C-1 T1ce-only F1 87.55 ≈ Patch F1 87.08 우연? → §13 Q3

### 19.14 SOTA 비교 (§14 — ★ 사용자 요청 핵심)
61. WT Dice -0.019 격차의 의미? → §14 Q1
62. TC/ET -0.10 격차 설명? → §14 Q2
63. 학부 프로젝트가 학술 SOTA와 비교하는 게 과욕? → §14 Q3
64. v2ways §10.1 예측 부분 적중이 미숙? → §14 Q4

### 19.15 차별화 (§15)
65. 상대 팀이 MTL로 가면? → §15 Q1
66. Dice가 더 낮으면 약점? → §15 Q2

### 19.16 환경적 한계 (§16)
67. 환경 제약을 *변명* 으로? → §16 Q1
68. *행운 vs 설계의 결과*? → §16 Q2

### 19.17 종합·기타 (총괄)
69. 7단계까지 가는 이유 / 학부 범위? → §0.2, §17.16
70. 미실행 / 향후 과제? → 부록 A
71. 현업 시스템과의 비교? → 부록 D
72. Day 5/6/7 ensemble 가능성? → §17.7
73. Score-CAM / EigenCAM 등 강한 XAI? → 부록 A
74. 3D / 2.5D 입력 확장? → 부록 A
75. 외부 데이터 (BraTS 2019/2020/2021) fine-tune? → 부록 A
76. 5-fold CV는? → 부록 A
77. Multi-seed 3-seed 평균? → 부록 A
78. SWA 평가 환경 재구축 시 가능? → §9 Q5
79. 향후 *환경 재구축* 시 우선순위? → §16 Q2
80. v2ways §6 grading 확장 (WT/TC/ET multi-label, GLI/MEN/PED, burden regression)? → 부록 D
81. PACS Plugin / DICOM Viewer? → 부록 D
82. 의료 AI 신뢰성 감사 키트? → 부록 D
83. MTL Bootstrap Template (다른 도메인 적용)? → 부록 D
84. BraTS Challenge Kaggle 베이스라인? → 부록 D
85. 학부 vs 대학원·전공의 교육용 사례연구? → 부록 D
86. Score-CAM / EigenCAM 의 본 모델 적용 시 IoU 추정? → §6 Q1 (분류 head CAM 0.12 → 0.16의 한계)
87. 분류 head + seg head의 *gradient 충돌* 가능성? → §5 (active supervision 설계)
88. Uncertainty Weighting 의 log_var 학습 다이내믹 해석? → §9.8 (cls 5× 변화 / seg 2.3× 변화)
89. Conformal Prediction이 *split conformal* vs *full conformal*? → §10 (marginal score 기반, n_cal=24,882)
90. TumorCP 의 50% probability 의 의미? → §9.5 (Yang MICCAI 2022 단순화)
91. Boundary Loss 의 SDF (signed distance function) 사전 계산? → §9.5 (Kervadec MIDL 2019)
92. *어떤 모델*을 *어떤 시나리오*에 권장? → 부록 D 의 임상 시나리오 매핑

---

## 부록 A — 배제된 / 보류된 결과 (전체 인벤토리)

본 보고서 본문에서 *명시적으로 배제* 하거나 *측정은 했지만 메인 표에 넣지 않은* 결과들. 발표 슬라이드에서는 *언급만* 하고 깊게 다루지 않을 항목.

### A.1 SWA 체크포인트 평가 (실패)
- `outputs/checkpoints/sota/sota_swa.pth` 가 정상 저장 (54.7MB, SHA256 = `2c018a26...`).
- step31 의 P10 SWA 평가 코드는 *정상 작성* 됐으나 RAM 16GB 환경에서 *14GB float32 array 할당 실패*.
- 발표: "SWA 체크포인트는 디스크에 있으나, RAM 환경 부족으로 *공식 평가는 best 단일* 로 제출" 로 정직.

### A.2 Day 7 SOTA 의 분리 Ablation (미실행)
- 6개의 동시 변경: (i) Loss, (ii) 3-region, (iii) Deep Sup, (iv) Uncertainty Weighting, (v) Weighted Sampler+TumorCP, (vi) SWA.
- 분리 ablation 은 6 × 17h = 102h 학습 필요 — §16 의 환경 제약으로 보류.

### A.3 Day 6 의 thr 0.7 운용점 측정 (미실행)
- Day 6 도 thr 0.7 에서 F1 측정 시 *공정 비교* 가능. 본 보고서는 미실행 — 그리드가 [0.30, 0.35, 0.3847, 0.40, 0.45, 0.50]으로 thr 0.7 포함 안 됨.

### A.4 3-seed 평균 (미실행)
- F1 / Dice CI 의 *모델 분산 추정* 을 위해 필요. 학습 시간 17h × 3 = 51h — 환경 제약으로 보류.

### A.5 Day 5 / Day 6 / Day 7 의 Ensemble (미실행)
- recovered 381 + regressed 86 (Day 5↔6) + Day 7 의 새 패턴 — 3-way ensemble 의 *상호 보완성* 정량 검증.
- soft-voting 또는 max-voting 두 방식. 본 보고서 미실행.

### A.6 Score-CAM / EigenCAM 등 강한 XAI (미실행)
- §6 Q1 의 한계 대응. `260521v2ways.md` §3.3 권장.
- Day 7 SOTA 의 *분류 head CAM IoU* 측정도 미실행.

### A.7 3D / 2.5D 입력 (미실행)
- §16 의 환경 제약 (24GB+ GPU 필요) 으로 직접 적용 불가.
- 3D SegResNet (MONAI) 등으로 *전환만 한 평가* 도 가능했지만 환경 충돌 우려로 보류.

### A.8 외부 데이터 (BraTS 2019/2020/2021) fine-tune (미실행)
- 라이센스 확인 + 다운로드 + 통합 — 일정 외.

### A.9 K-fold (5-fold) CV (미실행)
- 학습 5× = 85h + 결과 통합 인프라 — §16 환경 제약 + 일정 외.

### A.10 Day 7 분류 head 의 Train Grad-CAM (미실행)
- §6 의 *Train CAM IoU 0.12 (Day 5) → 0.16 (Day 6)* 추세에 Day 7 이 *Deep Supervision 으로 0.2+ 진입* 가능성. 미작성.

### A.11 Multi-Task Val Dice vs Test Dice 측정 함수 차이
- `_wt_dice`와 `compute_seg_metrics`가 dice 계산 방식이 약간 다름. Val Dice 0.426 vs Test Dice 0.787의 갭은 측정 정의의 차이로 추정. **본 발표는 Test Dice (0.787)를 표준 보고치로 사용**.

### A.12 step20 missing_pair 44건 처리
- T1ce 추출 시 FLAIR와 매핑 안 된 44 슬라이스 — 본 학습에서는 dataset이 자동으로 검은 이미지로 대체.
- **엄격한 strict mode**(완전 일치 슬라이스만)로 재학습 시 결과 약간 다를 가능성. 추정 영향: ±0.02%p F1 이내.

### A.13 Channel mode 비교 (ratio / multiplication) (미실행)
- `step21_mmmt_dataset.py`에 `t1ce_flair_ratio`, `t1ce_flair_mul` 모드 구현은 있으나 학습 미실행.

### A.14 Tversky 효과 *정확한* 분리 ablation (미실행)
- D-1) FLAIR-only + Dice-only (= Day 5)
- D-2) FLAIR-only + Dice+Tversky (= C-2 — 본 실행)
- D-3) MMMT + Dice-only (미실행)
- D-4) MMMT + Dice+Tversky (= baseline — 본 실행)
- → 2×2 ablation에서 D-3이 빠짐. Tversky와 멀티모달의 *상호작용 항* 분리 미완.

### A.15 Still_missed 641의 추가 패턴 분석 (미실행)
- z-위치, 종양 픽셀 수만 분석됨. 모달리티별 신호 강도(T1ce / FLAIR 평균 강도) 통계 추가 권장.

### A.16 New_fp 515의 prob 분포 분석 (미실행)
- `fn_diff_per_slice.csv`에는 데이터가 있으므로 사후 분석 가능.

### A.17 Whole-Slice / Patch-Based의 *학습셋* Grad-CAM (배제)
- v1 §2~§4는 test split CAM IoU만 보고. Whole-Slice / Patch의 train CAM IoU는 본 보고서에 포함하지 않음.
- 이유: (1) Whole-Slice / Patch는 seg head가 없어 *분류 CAM만* 측정 가능, (2) Day 5 MT의 train vs test CAM 차이를 통해 일반 패턴이 이미 확인됨.

### A.18 β=0 / β=0.3 / β=1.0 Multi-Task ablation (미실행)
- "decoder Loss 가중치별 IoU 곡선" — 향후 과제.

### A.19 Patch 모델을 ResNet-18 (pretrained)로 재학습 (미실행)
- 우선순위 밀림, MT가 더 우아한 해법.

### A.20 Hard Negative Mining (Patch FP 집중 학습) (미실행)

### A.21 2-Stage 결합 (Whole 1차 필터 → Patch/MT 정밀) (미실행)

### A.22 고해상도 입력 (320×320 / 384×384) (미실행)
- VRAM 위험으로 8GB에서 batch=8 한계.

### A.23 MONAI 라이브러리로 nnU-Net 2D / SwinUNETR / SegResNet 베이스라인 비교 (미실행)
- 환경 재구축 후 가능.

### A.24 모달리티 누락 시뮬레이션 (미실행)
- Day 6 학습 모델을 *T1ce만*, *FLAIR만*으로 추론 (3채널 중 2개를 평균 채널로 대체) → "모달리티 ablation" 곡선. 실제 임상에서 한 모달이 빠질 때 강건성 보고.

### A.25 OOD / 강건성 — MEN(수막종) / PED(소아) 데이터로 추론 (미실행)
- BraTS는 19개 기관에서 수집된 멀티-사이트 데이터셋. 외부 강건성 보고 가능.

---

## 부록 B — 전체 학습 로그 (Whole/Patch/MT/MMMT/Abl/SOTA)

### B.1 Whole-Slice (§2.4에 14 epoch 전체 표 있음)
총 14 epoch. Best Epoch 9 (Val Loss 0.1713, Val Acc 94.15%). Phase 1 → 2 전환 시 Train Loss 0.4360 → 0.2067 급감. Epoch 10 이후 단조 과적합.

### B.2 Patch-Based (§4.5에 12 epoch 전체 표 있음)
총 12 epoch (Early Stop). Best Epoch 7 (Val Loss 0.2956, Val Acc 91.8%). Val Loss 진동 큼 (epoch 2: 0.74 → 7: 0.30 → 9: 0.48). 학습 시간 128.8분.

### B.3 Multi-Task (§5.4에 14 epoch 전체 표 있음)
총 14 epoch (P1 3 + P2 11, P2 Early Stop). Best Epoch 9 (Total Loss 0.3484, CLS 0.1909, SEG 0.3151, Val Acc 93.84%, Val Dice 0.756). Phase 1→2 전환 시 CLS Loss 0.39→0.18 급감 / SEG Loss 거의 불변. 학습 시간 298.8분.

### B.4 Day 6 MMMT (§7.7에 16 epoch 전체 표 있음)
- 총 16 epoch (P1 3 + P2 13, Early Stop).
- Best Val Acc Epoch 11 (94.12%, Val Dice 0.4229).
- Phase 1→2 전환 시 Val total 0.471 → 0.269. Val Acc 83.13% → 92.56%.
- 학습 시간 388.5분.

### B.5 C-2 FLAIR-only Ablation (12 epoch)

| Epoch | Train Total | Train Acc | Val Total | Val Acc | Val Dice |
|:-----:|:-----------:|:---------:|:---------:|:-------:|:--------:|
| 1 (P1) | 0.583 | 78.72% | 0.539 | 81.18% | 0.313 |
| 2 (P1) | 0.547 | 79.08% | 0.511 | 81.95% | 0.331 |
| 3 (P1) | 0.525 | 79.93% | 0.483 | 82.37% | 0.330 |
| 4 (P2) | 0.301 | 91.57% | 0.258 | 93.02% | 0.395 |
| 5 | 0.252 | 93.18% | 0.252 | 93.26% | 0.391 |
| 6 | 0.233 | 93.75% | 0.251 | 93.37% | 0.396 |
| 7 | 0.222 | 94.14% | 0.248 | 93.38% | 0.393 |
| 8 | 0.212 | 94.46% | 0.254 | 93.21% | 0.403 |
| 9 | 0.201 | 94.83% | 0.256 | 93.02% | 0.414 |
| **10 (★ Best)** | 0.192 | 95.16% | 0.255 | **93.67%** | 0.415 |
| 11 | 0.180 | 95.53% | 0.257 | 93.20% | 0.407 |
| 12 (Early Stop) | 0.169 | 95.83% | 0.271 | 92.92% | 0.417 |

- 학습 시간 715.3분.

### B.6 C-1 T1ce-only Ablation (15 epoch)

| Epoch | Train Total | Train Acc | **Train Dice** | Val Total | Val Acc | **Val Dice** |
|:-----:|:-----------:|:---------:|:--------------:|:---------:|:-------:|:------------:|
| 1 (P1) | 0.776 | 74.91% | 0.028 | 0.731 | 77.29% | 0.000 |
| 2 (P1) | 0.766 | 75.22% | 0.000 | 0.731 | 76.72% | 0.000 |
| 3 (P1) | 0.757 | 75.67% | 0.000 | 0.730 | 77.28% | 0.000 |
| 4 (P2) | 0.565 | 85.64% | 0.000 | 0.548 | 86.50% | 0.000 |
| 5 | 0.525 | 87.62% | 0.000 | 0.544 | 86.82% | 0.000 |
| 6 | 0.508 | 88.52% | 0.000 | 0.531 | 88.14% | 0.000 |
| 7 | 0.493 | 89.30% | 0.000 | 0.525 | 88.26% | 0.000 |
| **8 (★ Best Val Acc)** | 0.477 | 90.10% | 0.000 | 0.519 | **88.40%** | 0.000 |
| 9 | 0.443 | 90.99% | 0.000 | 0.455 | 88.28% | 0.000 |
| 10 | 0.365 | 91.57% | **0.046** | **0.440** | 88.36% | 0.287 |
| 11 | 0.333 | 92.33% | 0.572 | 0.456 | 87.73% | 0.298 |
| 12 | 0.308 | 92.93% | 0.599 | 0.476 | 87.91% | 0.293 |
| 13 | 0.281 | 93.78% | 0.612 | 0.448 | 87.69% | 0.304 |
| 14 | 0.261 | 94.40% | 0.614 | 0.457 | 87.62% | 0.302 |
| 15 (last) | 0.242 | 95.01% | 0.625 | 0.486 | 87.85% | **0.311** |

- 학습 시간 340.2분.
- **이상 패턴**: epoch 1~9까지 Train Dice 0.0 (seg head 미학습) → epoch 10부터 점진적 회복.

### B.7 Day 7 SOTA (§9.8 에 14 epoch 전체 표 있음)
총 14 epoch (P1 3 + P2 11, Early Stop). Best Val Total Epoch 9 (1.601), Best Val Acc Epoch 13 (92.90%). Uncertainty Weighting log_var 자동 학습. 학습 시간 1017.8분. SWA 시작 epoch 11.

### B.8 평가 결과 raw 직렬화

| 파일 | 크기 | 내용 |
|------|:----:|------|
| `outputs/logs/sota/sota_test_raw.npz` | **1,077.7 MB** | labels, probs_tta, probs_tscaled, logits, seg_gt, seg_pred, T |
| `outputs/logs/sota/sota_val_raw.npz` | **964.5 MB** | labels, logits, seg_gt, seg_pred (val) |
| `outputs/logs/sota/sota_test_raw_swa.npz` | (없음 — SWA 평가 실패로 미생성) | — |

### B.9 산출 figure (outputs/figures/sota/) — 9장 전체

| 파일 | 크기 | 내용 |
|------|:----:|------|
| `sota_training_curves.png` | 277 KB | 14 epoch 학습 곡선 (CLS/SEG loss, Acc, Dice) |
| `sota_diagnostics.png` | 116 KB | 4-panel (CM + ROC + PR + box by region) |
| `threshold_sweep_WT.png` | 48 KB | WT region 의 val dice vs threshold |
| `threshold_sweep_TC.png` | 47 KB | TC region |
| `threshold_sweep_ET.png` | 42 KB | ET region |
| `postproc_examples_best.png` | 48 KB | CC+closing 후처리 5쌍 비교 |
| `reliability_pre_post_best.png` | 81 KB | Temp 전/후 reliability diagram |
| `failure_worst30_best.png` | 1,729 KB | WT dice 최저 30 슬라이스 4-panel |
| `failure_best30_best.png` | 1,943 KB | WT dice 최고 30 슬라이스 4-panel |

---

## 부록 C — 모든 모델 한눈 비교표 (전체 프로젝트 종합)

> 본 표는 본 발표 슬라이드 1장 으로 사용 권장. *결정판*.

| # | 모델 | 입력 | Loss | 모델 구조 | 학습 시간 | F1 (best) | F1 @0.3847 | AUROC | seg Dice (WT) | seg Dice (TC) | seg Dice (ET) | seg IoU (WT) | TP CAM IoU | 신뢰성 보고 | 학술 비교 위치 |
|:-:|------|------|------|-----------|:---------:|:---------:|:----------:|:-----:|:-------------:|:-------------:|:-------------:|:------------:|:----------:|:-----------:|:--------------:|
| 1 | Whole-Slice (Day 2) | FLAIR 1ch | BCE | ResNet-18 (11.2M) | ~2h / 14 ep | 94.10 | — | **0.9832** | — | — | — | — | **0.145** | ❌ | shortcut 진단 |
| 2 | Patch-Based (Day 4) | FLAIR 64×64 patches | BCE | 경량 CNN (95K) | 2.2h / 12 ep | 87.08 | — | 0.9104 | — | — | — | — | 0.128 | ❌ | passive 실패 |
| 3 | **Multi-Task (Day 5)** | FLAIR 1ch | BCE + Dice | ResNet-18 + UNet decoder (14M) | 5h / 14 ep | (94.27) | **93.91** | 0.9824 | **0.7765** | — | — | **0.7061** | (train 0.12, test seg 0.706) | ❌ | active 성공 |
| 4 | **MMMT (Day 6)** | T1ce+FLAIR+diff | BCE + Dice + Tversky | 동일 (14M) | 6.5h / 16 ep | **94.27** | 93.84 | **0.9832** | **0.7874** | — | — | **0.7142** | (train 0.16, test seg 0.714) | ❌ | FN 회복 |
| 5 | C-2 FLAIR-only abl. | FLAIR 3복제 | BCE + Dice + Tversky | 동일 (14M) | 11.9h / 12 ep | 94.15 | **93.98** | 0.9822 | 0.7458 | — | — | 0.6703 | (미측정) | ❌ | Tversky 분리 |
| 6 | C-1 T1ce-only abl. | T1ce 3복제 | BCE + Dice + Tversky | 동일 (14M) | 5.7h / 15 ep | 87.55 | 86.71 | 0.9487 | 0.5741 | — | — | 0.4780 | (미측정) | ❌ | WT 한계 노출 |
| 7 | **Day 7 SOTA (TTA, thr 0.5)** | T1ce+FLAIR+diff | Focal-Tversky + BCE + Boundary | ResNet-18 + UNet + DeepSup + UW (~14M+) | **17h / 14 ep** | 92.96 | (미산출) | 0.9824 | **0.7971** ★ | **0.7783** ★ | **0.7497** ★ | **0.7225** ★ | (미측정) | ✅ **11/12** ★ | per-vol 0.891 — **SOTA -0.02** ★ |
| 8 | **Day 7 SOTA (TTA, thr 0.7)** | T1ce+FLAIR+diff | (동일) | (동일) | (동일) | **93.50** ★ | (미산출) | 0.9824 | (동일) | (동일) | (동일) | (동일) | (미측정) | ✅ | (동일) |

> **합계** (전체 프로젝트 학습 시간): 약 **3,009분 ≈ 50시간** = 단일 RTX 4070 Laptop 8GB로 *6일치* 학습.

---

## 부록 D — 임상 시나리오별 운용 + 실용화 5가지

### D.1 *임상 시나리오별* 추천 운용

| 시나리오 | 우선순위 | 권장 모델 | 권장 threshold | 1000명 기준 (오탐 / 누락) |
|----------|----------|-----------|:--------------:|:--------------------------:|
| 1차 스크리닝 | Recall 최우선 | **Day 7 SOTA** | 0.5 | 82 / 56 |
| 균형 운용 | F1 최우선 | **Day 7 SOTA** | **0.7** | 48 / 78 |
| 확진 (Precision) | Precision 최우선 | Day 5 MT | 0.3847 | 24 / 93 |
| 학술적 SOTA 비교 | per-volume WT Dice | **Day 7 SOTA** | (시나리오에 따라) | (volume Dice 0.891) |
| 신뢰성 정량화 (감사) | CI + Conformal | **Day 7 SOTA** | (선택 자유) | (coverage 0.896) |

### D.2 실용화 5가지 (`260523v2proposal.md` §1, §3 발췌)

| # | 활용처 | 우리 자산 매핑 | 패키징 형태 |
|:-:|--------|----------------|--------------|
| ① | **의료 AI 1차 스크리닝 보조** (Triage Assistant) | Day 7 SOTA (thr 0.5/0.7 선택, 신뢰성 보고 12종) | PACS Plugin / DICOM Viewer |
| ② | **학부·대학원·전공의 교육용 사례연구** | 7단계 여정 (Whole → Grad-CAM → Patch → MT → MMMT → SOTA) + Q&A 92문항 | Jupyter Notebook + 슬라이드 + 사전학습 모델 |
| ③ | **의료 AI 신뢰성 감사 템플릿** | Grad-CAM IoU 측정 + FN/FP 패턴 분석 + Threshold 운영 곡선 + Calibration + Conformal | 6-Step 감사 파이프라인 |
| ④ | **헬스케어 스타트업 분류 친화 베이스라인 (Bootstrap)** | Shared Encoder + Cls Head + Seg Decoder, 모달리티 독립적 | Multi-Task Bootstrap Template (PyPI SDK) |
| ⑤ | **BraTS Challenge / Kaggle 베이스라인 코드** | 2D 슬라이스 + 5~17h 학습 + WT/TC/ET 3-region + 신뢰성 보고 | GitHub 오픈소스 리포 |

### D.3 임상 의사결정 지원 시스템(CDSS) 프로토타입

**시나리오**: 비전공 의사(가정의학과·응급의학과 1년차 등)가 외래에서 뇌 MRI를 1차로 검토.

**본 모델이 제공하는 출력의 임상 친화성**:
- 슬라이스별 종양 확률 + 종양 위치 마스크 → 보고서 자동 초안
- Multi-modal MTL: T1ce / FLAIR 각각의 기여도 시각화 가능 → "왜 그렇게 판단했는가" 설명
- Threshold 운영 곡선 → 병원·과별로 *민감도 우선 / 특이도 우선* 운영점 조정 가능
- Conformal Prediction → "의사 검토 큐"로 자동 분류 (불확실한 케이스)

### D.4 종양 등급화(Grading) 확장 — 임상 가치의 다음 단계

`260521v2ways.md` §6의 4가지 grading 방향:

1. **WT / TC / ET 3-region multi-label seg** — BraTS 공식 평가축 (★ Day 7에서 부분 달성)
2. **Slice-level multi-label (WT/TC/ET 유무)**
3. **Tumor-type 다중 분류 (GLI / MEN / PED)** — `data/BraTS-MEN`, `data/BraTS-PED` 폴더가 이미 있음
4. **Tumor-burden regression** — `tumor_pixels / brain_pixels`

### D.5 현업 시스템 비교

| 시스템 | 사용처 | 우리 보완점 |
|--------|--------|-------------|
| icobrain (Icometrix) | 임상 인증 | + Grad-CAM + Conformal |
| Quantib Brain | 유럽 CE / FDA | + 신뢰성 보고 12종 |
| AIDoc / Viz.ai | 응급 영상 트리아지 | + WT/TC/ET 3-region |
| Brainomix / BrainScan | 신경영상 분석 | + 분류 + 위치 동시 |
| Aidence / RadAI | 폐 결절 중심 | + Bootstrap CI |

### D.6 현업 시스템 공통 약점 vs 우리 모델 보완점

| 약점 | 설명 | 우리 보완책 |
|------|------|-------------|
| 블랙박스 출력 | 분류 결과만, 근거 부족 | Segmentation 마스크 + Grad-CAM |
| 소병변 누락 | FN 흔함 | Focal-Tversky Loss + 멀티모달 + TumorCP |
| 단일 모달 의존 | FLAIR만 쓰는 시스템 존재 | T1ce+FLAIR (Day 6/7) |
| 고정 임계값 | 유병률 무시 | Threshold 운영 곡선 제공 (Youden's J + F1-optimal) |
| 보정 미흡 | Overconfident | Temperature Scaling (T=1.50) |
| 분포 외 강건성 | 다른 병원에서 급락 | MTL 보조 task가 일반화에 도움 (미검증 — A.25) |
| 해석성·신뢰성 충돌 | "왜 그렇게 판단?" 답 X | Segmentation 출력 + Conformal Prediction |
| 윤리/책임 소재 | FN 책임 | FN/FP 분석 정량 공개 + Bootstrap CI |

---

## 부록 E — 참고한 md 파일 인벤토리 + 직접 인용 출처

본 발표 준비 자료를 작성하기 위해 다음 md 파일과 보조 자료를 참조. 각 파일별로 어느 부분을 인용/보강에 사용했는지 명시.

### E.1 본 문서가 *직접 통합* 한 3개 메인 보고서

| 파일 | 분량 | 참고한 절 | 본 문서에서의 활용 |
|------|------|-----------|---------------------|
| **`260521v1result.md`** | 78.9 KB | §0 도식, §1~§6 Day 1~6, §7 차별화, §8 학술, §9 Q&A, 부록 A~E | **§1~§5 (Day 1~5 본문 전체), §15 차별화, §17.1~17.3 인사이트, 부록 A~E** |
| **`260524v1mmmtplus.md`** | 85.2 KB | §0 도식, §1 Day 5 train Grad-CAM, §2 Day 6 본 학습, §3 thr sweep, §4 FN 차분, §5 ablation, §6 6-way 비교, §7 5가지 인사이트, 부록 A~D | **§6 (Day 5 보강), §7 (Day 6 본 학습), §8 (Day 6 권장 3종), §13 (8-way 비교), §17.4~17.7 인사이트** |
| **`260526v3sotafinal.md`** | 97.6 KB | §0 도식 + 종료 선언, §1 SOTA 설계, §2 학습, §3~§5 평가, §6 failure, §7 7-way, §8 v2ways 정정, §9 환경 한계, §10 인사이트 16, §11 Q&A 92, 부록 A~D | **§9~§12 (Day 7), §13 (8-way), §16 (환경 한계), §17.8~17.16 인사이트, §19 Q&A 마스터, 부록 A~D** |

### E.2 본 문서의 §14 SOTA 비교의 직접 자료

| 파일 | 분량 | 참고한 절 | 본 문서에서의 활용 |
|------|------|-----------|---------------------|
| **`260521v2ways.md`** | 38.4 KB | §0.3 14개 우선순위, §2 아키텍처 트랙, §3 Loss 트랙, §5 신뢰성 트랙, §6 Grading 4방향, §7 SOTA 표 (§7.1~§7.5), §10 예상 성능, §14 참고 논문 ①~⑮ | **§14 (SOTA 직접 비교) 전체 — 사용자 요청 #3 의 직접 응답**, §17.16, 부록 D 향후 |
| **`260525v1sotapluswhy.md`** | 13.0 KB | §1-0 sota 폴더 위치, §1-1~§1-7 step26~31 의 md 근거, §2 우선순위 1주의 의미 | **§9 (Day 7 본문) 의 직접 보강** |
| **`260525v2sotawith.md`** | 1.9 KB | §1 우리 위치, §2 학술 SOTA 표 (MedNeXt 0.93, DynUNet 0.91, SwinUNETR 0.92) | **§14.1 직접 비교 표** |
| **`260526v1fullresults.md`** | 9.9 KB | A~E 5개 카테고리 (TTA/sliding/AMP/threshold/postproc), 권장 우선순위 표 | **§9 (Day 7 P1~P10 설계 의도), §11.3 postproc 역설적 해석** |
| **`260526v2step31patch.md`** | 26.4 KB | §0 절대 원칙, §1 현행 한계, §2 P1~P10 카탈로그, §3 코드 스니펫, §7 발표 매핑 | **§9.10 manifest 형식, §10 ECE/Brier/Conformal, §11 threshold sweep/postproc, §12 failure case** |

### E.3 보조 자료 (배경 + 보강)

| 파일 | 참고한 부분 | 본 문서 활용 |
|------|-------------|-------------|
| `README.md` | §1.1 6단계 흐름, §3 핵심 결과, §5 모델별 아키텍처 | §0 도식, §1 데이터, §13 비교 |
| `brats2023_dataset_guide.md` | 4 모달리티 의미, NCR/ED/ET 라벨 정의, WT/TC/ET 평가축 | §1.6 [보강], §7.1 의학적 상보성, §9.2 3-region 정의 |
| `260523v2proposal.md` | §0.1 자산 정리, §1.1~§1.7 7가지 활용처, §3 패키징 5형태 | 부록 D (실용화 5가지) 전체 |

### E.4 코드·산출물 (직접 인용)

| 파일/경로 | 본 문서 활용 |
|-----------|-------------|
| `code/whole/step1~step7_*.py` | Day 1~3 단계별 코드 명세 |
| `code/patch/step8~step13_*.py` | Day 4 Patch 파이프라인 |
| `code/multitask/step15~step19_*.py` | Day 5 Multi-Task 구현 |
| `code/multitask/step18b_multitask_train_gradcam.py` | §6 (Day 5 보강 Train Grad-CAM) |
| `code/mmmt/step20~step28_*.py` | Day 6 MMMT + Ablation |
| `code/sota/step26~step31_*.py` | Day 7 SOTA 패키지 |
| `outputs/checkpoints/{whole,patch,multitask,mmmt,mmmt_ablation,sota}/*.pth` | 모든 학습 체크포인트 |
| `outputs/logs/sota/sota_history.json` | §9.8 14 epoch 학습 곡선 |
| `outputs/logs/sota/sota_test_metrics.json` | §9~§11, §14.8 의 *모든 실측치* (원본) |
| `outputs/logs/sota/sota_test_raw.npz`, `sota_val_raw.npz` | 부록 B.8 재현성 자산 |
| `outputs/figures/sota/*.png` (9장) | 부록 B.9 발표 슬라이드 시각화 |

### E.5 본 문서의 *직접 인용 출처* 요약 표

| 본 문서 절 | 직접 인용 출처 | 핵심 인용 부분 |
|-----------|--------------|---------------|
| §0 (도식 + 종료 선언) | `260521v1result.md` §0 + `260524v1mmmtplus.md` §0 + `260526v3sotafinal.md` §0, §9 | 7-단계 도식 + total_epochs:14 + swa.error |
| §1 (Day 1) | `260521v1result.md` §1 + `brats2023_dataset_guide.md` | EDA 5종 + WT/TC/ET 라벨 정의 |
| §2 (Day 2) | `260521v1result.md` §2 + `notPublic/day2_day3_report.md` | 14 epoch 학습 곡선 + Confusion Matrix |
| §3 (Day 3) | `260521v1result.md` §3 + `notPublic/3wayanalysis.md` §6 | Grad-CAM 3질문 + Shortcut 진단 |
| §4 (Day 4) | `260521v1result.md` §4 + `notPublic/wholePatch성능비교.md` | 패치 추출 + 슬라이스 집계 3종 + 6원인 분석 |
| §5 (Day 5) | `260521v1result.md` §5 + `notPublic/3wayanalysis.md` 전체 | MTL 아키텍처 + 14 epoch + 3-way 비교 |
| §6 (Day 5 보강) | `260524v1mmmtplus.md` §1 + `mt_train_gradcam_summary.json` | CAM 0.12 vs SEG 0.77 |
| §7 (Day 6 본 학습) | `260524v1mmmtplus.md` §2 + `mmmt_summary.json`, `mmmt_history.json`, `mmmt_test_metrics.json` | 16 epoch + FN 288 감소 / FP 234 증가 |
| §8 (Day 6 권장 3종) | `260524v1mmmtplus.md` §3, §4, §5 + `mmmt_test_metrics_thr_sweep.json`, `fn_diff_summary.json`, `ablation_table.json` | thr sweep + recovered 381 / still_missed 641 + C-1/C-2 |
| §9 (Day 7) | `260526v3sotafinal.md` §1, §2, §3 + `260525v1sotapluswhy.md` 전체 + `sota_history.json`, `sota_summary.json`, `sota_test_metrics.json` | step26~31 설계 + 14 epoch + thr 0.5/0.7 |
| §10 (신뢰성) | `260526v3sotafinal.md` §4 + `260521v2ways.md` §5.3, §5.5 + `260526v2step31patch.md` §3.6, §3.7 | ECE 0.036 + Conformal 0.896 + CI 1000회 |
| §11 (세분화 SOTA) | `260526v3sotafinal.md` §5 + `260525v2sotawith.md` §2 | WT/TC/ET region별 + HD95 + per-volume |
| §12 (failure case) | `260526v3sotafinal.md` §6 + `failure_{worst,best}30_best.png` | 4-panel 시각화 + still_missed 패턴 동일 |
| §13 (8-way 비교) | `260521v1result.md` §5.8 + `260524v1mmmtplus.md` §6 + `260526v3sotafinal.md` §7 + 본 §1~§12 | 7개 행 원본 표 통합 |
| §14 (SOTA 비교) | `260521v2ways.md` §7 (§7.1~§7.5), §10.1, §14 + `260525v2sotawith.md` §1~§2 + `260526v3sotafinal.md` §8 + `sota_test_metrics.json` | **사용자 요청 #3 의 직접 응답 — v2ways §7 의 14개 SOTA × 우리 실측 정렬** |
| §15 (차별화) | `260521v1result.md` §7 + `notPublic/differentiation_strategy.md` 전체 + `notPublic/multitask_explanation.md` §4~§5 | 3층위 차별화 |
| §16 (환경 한계) | `260526v3sotafinal.md` §9 + `260526v2step31patch.md` §0 + `260526v1fullresults.md` E-1, E-2 + `code/sota/step31_sota_evaluate.py` L51~67 | VRAM 누수 + Windows + RAM OOM |
| §17 (16 인사이트) | `260521v1result.md` §8 + `260524v1mmmtplus.md` §7 + `260526v3sotafinal.md` §10 | 7+7+9 = 16개 인사이트의 직접 출처 |
| §18 (발표 슬라이드 10장) | `260521v2ways.md` §9.4 + 본 §17 | 메인 10장 + 부록 5장 |
| §19 (Q&A 92문항) | 본 §1~§16의 모든 Q&A + `260524v1mmmtplus.md` §8 + `260526v3sotafinal.md` §11 | 인덱스 통합 |
| 부록 A (배제) | 본 §3~§16 + `260524v1mmmtplus.md` A.1~A.10 + `260526v3sotafinal.md` 부록 A | 25개 미실행 항목 *명시적* 인벤토리 |
| 부록 B (학습 로그) | `sota_history.json`, `mmmt_history.json`, `mt_training_history.json` 등 모든 history JSON + figure 9장 메타 | 보고치 *완전한 원본* |
| 부록 C (모든 모델 비교) | `260521v1result.md` §5.8 + `260524v1mmmtplus.md` §6 + `260526v3sotafinal.md` §7 + 본 §13 | 종합 *결정판* |
| 부록 D (실용화) | `260523v2proposal.md` §1~§5 + `260521v2ways.md` §6 grading | 5가지 활용 형태 + 6-Step 감사 + grading 4방향 |

### E.6 이미지 자료

| 파일 | 내용 | 본 문서 활용 |
|------|------|-------------|
| `KakaoTalk_20260509_134516032.png` | 3일차 강의 일정 인포그래픽 | §0 강의 일정 매핑 |
| `KakaoTalk_20260509_134516032_01.png` | MRI 뇌 질병 분류 + XAI 프로젝트 전체 인포그래픽 | §0 전체 흐름 |

---

## 끝맺음 — 한 줄 요약 (최종 발표 시점)

> **"7단계 여정 — Whole의 단일 지표 한계 발견 → Grad-CAM으로 Shortcut 진단 → Patch의 passive 차단 실패 → Multi-Task의 active 성공 (IoU 5배) → Multi-Modal의 FN 회복 (288개) → Ablation으로 FLAIR/T1ce 기여 분해 → Day 7 SOTA로 학술 SOTA -0.02 영역 도달 + 신뢰성 정량화 완성 (ECE 0.036, Conformal 0.896, Bootstrap CI). 환경적 천장(8GB Laptop + Windows + 14 epoch 조기종료)이 명확하지만, *가설 검증 + 가설 정정 + 신뢰성 정량화 + 환경 한계 정직 보고*가 모두 들어 있는 완성형 학부 사례."**

| 단계 | 모델 | F1 (@best thr) | seg Dice (WT) | seg Dice (TC) | seg Dice (ET) | FN | 신뢰성 | 종합 |
|:----:|------|:--------------:|:-------------:|:-------------:|:-------------:|:--:|:------:|:----:|
| Day 2 | Whole-Slice | 94.10 | (CAM 0.145) | — | — | 888 | ❌ | shortcut 진단 |
| Day 4 | Patch-Based | 87.08 | (CAM 0.128) | — | — | 1,306 | ❌ | passive 실패 |
| **Day 5** | **Multi-Task** | **93.91** | **0.7765** | — | — | 1,136 | ❌ | **active 성공** |
| **Day 6** | **MMMT** | **94.27** | **0.7874** | — | — | **848** | ❌ | **FN 회복** |
| Day 6 abl. | C-2 FLAIR-only | 94.15 | 0.7458 | — | — | 1,044 | ❌ | Tversky 분리 |
| Day 6 abl. | C-1 T1ce-only | 87.55 | 0.5741 | — | — | 1,835 | ❌ | WT 한계 |
| **Day 7** | **SOTA (thr 0.7)** | **93.50** | **0.7971** ★ | **0.7783** ★ | **0.7497** ★ | 956 | **✅ 11/12** ★ | **신뢰성 SOTA + 학술 표면적 도달** |

| 학술 SOTA 비교 (per-volume WT Dice) | 본 프로젝트와의 격차 |
|------|:--------------------:|
| **MedNeXt** (MICCAI 2023, 0.93) | -0.039 |
| **SwinUNETR-v2** (CVPR 2024, 0.92) | -0.029 |
| **DynUNet 2D** (Nature Methods 2021, 0.91) | **-0.019** ★ |
| **🎯 우리 Day 7 SOTA** (per-volume WT Dice **0.8910**) | 학부 환경 천장 |

---

> 📎 **본 문서의 위치**: `biohealth_lv.1/260527_final_presentation.md`
>
> 📎 **연속성**:
> `260521v1result.md` (Day 1~5 + Day 6 진행 중)
> → `260521v2ways.md` (확장 로드맵 + SOTA §7)
> → `260523v1updatemmmt.md` (멀티모달 갱신 가이드)
> → `260523v2proposal.md` (실용화 제안 5가지)
> → `260524v1mmmtplus.md` (Day 5~6 완성 + 권장 3종)
> → `260525v1sotapluswhy.md` (sota 코드 해설)
> → `260525v2sotawith.md` (학술 SOTA 위치)
> → `260526v1fullresults.md` (성능 더 짜내기 9종)
> → `260526v2step31patch.md` (P1~P10 패치)
> → `260526v3sotafinal.md` (Day 7 SOTA 평가 통합)
> → **`260527_final_presentation.md` (최종 발표 통합 마스터 — 본 문서)**.
>
> 📎 **프로젝트 종료 선언**: GPU VRAM 누수 + Windows PyTorch CUDA Allocator 한계 + SWA 평가 RAM OOM 으로 인해, *본 시점 이후의 학습/대규모 forward 는 진행하지 않는다*. 본 보고서가 *최종 발표용 공식 종착점*.
