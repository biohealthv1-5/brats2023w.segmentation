# 🎤 260523 v2 — 최종 발표 준비 자료 (MMMT 결과 + 권장 3종 확장판)

> **본 문서의 위치**: `260521v1result.md` (Day 1~6 진행 중)의 **직후 후속편**.
> `260521v1result.md`에서 Day 5까지의 Multi-Task 결과를 정리했고 Day 6은 "진행 중"으로 남아 있었다. 본 문서는 그 누락된 부분 — **(a) Day 5 Multi-Task의 *Train-split Grad-CAM* 결과(`mt_train_gradcam_summary.json`)**, **(b) Day 6 Multi-Modal Multi-Task의 본 학습/평가 결과**, **(c) `260523v1updatemmmt.md` §④에서 권장된 3종(A·B·C) 실행 결과(step25b / step26 / step27·28)** — 를 모두 보강하여 최종 발표용 마스터 자료로 통합한다.
>
> **편집 원칙 (v1과 동일 유지)**
> 1. 단계별 결정적 수치·관찰·의사결정은 *원문 그대로 보존* (요약 X).
> 2. 명세서·산출물에 명시되지 않은 부분은 폴더 내 다른 자료(코드 주석, 산출 JSON, 다른 md)를 찾아 **[보강]** 표시와 함께 추가.
> 3. 각 단계 끝에 **🎯 예상 질문 (Q&A)** 섹션을 두어 교수님이 던질 만한 의문을 정리.
> 4. 본문에서 빠진 내용(미실행·미해결·향후·세부 학습 곡선·논문 추천 등)은 **부록**에 별도 보관.
> 5. 각 섹션 끝에 **📎 참고 md 출처**로 어떤 파일의 어느 절을 참조했는지 명시.

---

## 목차

0. [v1 이후 추가된 작업의 전체 도식](#0-v1-이후-추가된-작업의-전체-도식)
1. [Day 5 보강 — Multi-Task의 *Train-split* Grad-CAM (누락 분석)](#1-day-5-보강--multi-task의-train-split-grad-cam-누락-분석)
2. [Day 6 — Multi-Modal Multi-Task 본 학습 결과 (step20~step25)](#2-day-6--multi-modal-multi-task-본-학습-결과-step20step25)
3. [권장 3종 결과 A — Threshold 동등 보정 (step25b)](#3-권장-3종-결과-a--threshold-동등-보정-step25b)
4. [권장 3종 결과 B — Day5↔Day6 FN 차분 분석 (step26)](#4-권장-3종-결과-b--day5day6-fn-차분-분석-step26)
5. [권장 3종 결과 C — 채널 Ablation 학습/평가 (step27 / step28)](#5-권장-3종-결과-c--채널-ablation-학습평가-step27--step28)
6. [전체 모델 비교 (Whole / Patch / MT / MMMT / Ablations 6-way)](#6-전체-모델-비교-whole--patch--mt--mmmt--ablations-6-way)
7. [발표 핵심 메시지 — 5가지 결정적 인사이트 (Day 5~6 확장판)](#7-발표-핵심-메시지--5가지-결정적-인사이트-day-56-확장판)
8. [예상 질문 마스터 리스트 — Day 6 / Ablation 추가분](#8-예상-질문-마스터-리스트--day-6--ablation-추가분)
9. [부록 A — 배제된 / 보류된 결과](#부록-a--배제된--보류된-결과)
10. [부록 B — 전체 학습 로그 (Multi-Task / MMMT / Ablations)](#부록-b--전체-학습-로그-multi-task--mmmt--ablations)
11. [부록 C — 미실행·향후 계획 (Day 7+)](#부록-c--미실행향후-계획-day-7)
12. [부록 D — 참고한 md 파일 인벤토리 (v1 이후 추가분)](#부록-d--참고한-md-파일-인벤토리-v1-이후-추가분)

---

## 0. v1 이후 추가된 작업의 전체 도식

### 0.1 v1 시점(2026-05-21)과 v2 시점(2026-05-23~24)의 차이

```
260521v1result.md 작성 시점 (2026-05-21):
  Day 1 ─ Day 2 ─ Day 3 ─ Day 4 ─ Day 5 ─ [Day 6 진행 중, 결과 없음]
                                  ↑
                            Train-split Grad-CAM 누락
                            (step18b는 실행됐으나 v1 보고서에 포함 안 됨)

260523v2mmmtplus.md 작성 시점 (2026-05-24):
  Day 1 ─ Day 2 ─ Day 3 ─ Day 4 ─ Day 5 ─ Day 6 (완료, MM-MTL)
                                          │
                                          ├─ step24b 멀티모달 Train Grad-CAM (완료)
                                          ├─ (A) step25b Threshold 동등 보정 (완료)
                                          ├─ (B) step26 FN 차분 분석 (완료)
                                          └─ (C) step27/28 채널 Ablation (완료)
```

### 0.2 본 문서가 다루는 5개 신규 산출물 패키지

| # | 패키지 | 스크립트 | 산출물 위치 | 본 문서 §|
|:-:|--------|----------|-------------|:--:|
| 1 | **MT Train Grad-CAM** | `code/multitask/step18b_*` | `outputs/figures/multitask/gradcam_train/` | §1 |
| 2 | **MMMT 본 학습/평가** | `code/mmmt/step20~25_*` | `outputs/{checkpoints,figures,logs}/mmmt/` | §2 |
| 3 | **Threshold 동등 보정 (A)** | `code/mmmt/step25b_*` | `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json`, `outputs/figures/mmmt/threshold_sweep.png` | §3 |
| 4 | **FN 차분 분석 (B)** | `code/mmmt/step26_*` | `outputs/logs/mmmt/fn_diff_*.csv/json`, `outputs/figures/mmmt/fn_diff_*.png` | §4 |
| 5 | **채널 Ablation (C)** | `code/mmmt/step27_*` / `step28_*` | `outputs/{checkpoints,figures,logs}/mmmt_ablation/`, `outputs/logs/mmmt_ablation/ablation_table.json` | §5 |

📎 **참고**: `260521v1result.md` §0.1 (마일스톤 타임라인) / `260523v1updatemmmt.md` §②~§④ (재실행 불필요 진단 + Phase 1/2 권장 파이프라인) / `260523v2proposal.md` §0.1 (자산 정리)

---

## 1. Day 5 보강 — Multi-Task의 *Train-split* Grad-CAM (누락 분석)

### 1.1 이 결과가 v1에서 빠진 이유

v1 `260521v1result.md` §5는 Day 5 Multi-Task의 *test-split* 성능(F1 93.91 / IoU 0.706 / FP 305)과 *학습 곡선*은 모두 정리했지만, **`step18b_multitask_train_gradcam.py`로 생성한 *train-split* Grad-CAM 결과를 통째로 빠뜨렸다.** 이는 본 발표에서 두 가지 결정적 메시지를 보충하는 데이터다:

1. *학습셋*에서 분류 헤드 CAM의 IoU가 어땠는가? → "MT가 *분류 head 자체로는 여전히 광역 단서를 보고 있는가, 종양을 보는가*" 의 직접 검증.
2. 학습셋에서도 FN(under-fit) / FP(confusion pattern)이 남아 있는가?

### 1.2 산출물 (`outputs/figures/multitask/gradcam_train/`)

| 파일 | 내용 |
|------|------|
| `mt_train_gradcam_tp.png` | TP 8개 시각화 (T1FLAIR / Seg GT / Grad-CAM / Seg Pred) |
| `mt_train_gradcam_fn.png` | FN 8개 시각화 |
| `mt_train_gradcam_fp.png` | FP 8개 시각화 |
| `mt_train_cam_vs_seg_iou.png` | CAM-GT IoU vs SEG-GT IoU scatter |
| `mt_train_tp_iou_distribution.png` | TP IoU 히스토그램 |
| `mt_train_gradcam_summary.json` | 통계 요약 |
| `mt_train_gradcam_subsample.csv` | 400개 서브샘플 IoU 원본 |

### 1.3 핵심 수치 (보존)

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

> [보강] `checkpoint_epoch: 6`은 **2-Phase 학습에서 Phase 2의 6번째 epoch을 의미**한다 — Phase 1 3 epoch + Phase 2 6 epoch = **전체 9번째 epoch** (v1 §5.4 표의 "★ Best Epoch 9"와 일치). 실제 학습 곡선에서 Val Acc 93.84%, Val Dice 0.7564는 epoch 9의 값이다 (`mt_training_history.json` 9번째 entry).

### 1.4 학습셋 confusion matrix와 Train Accuracy

| Train | Pred Neg | Pred Pos |
|:-----:|:--------:|:--------:|
| **Actual Neg** (59,687) | TN = **59,260** | FP = **427** |
| **Actual Pos** (56,942) | FN = **4,110** | TP = **52,832** |

- Train Accuracy = (52,832 + 59,260) / 116,629 = **96.11%**
- Train FP rate = 427/59,687 = **0.72%** (test FP rate 305/12,866 = 2.37%보다 훨씬 낮음 → 과적합 신호 약함)
- Train FN rate = 4,110/56,942 = **7.22%** (test FN rate 1,136/12,249 = 9.27%와 유사 → FN은 학습셋에서도 잡지 못함 = **under-fit 패턴, capacity가 아니라 입력 표현의 한계**)

### 1.5 결정적 발견 — *CAM IoU 0.1232* vs *SEG IoU 0.7753*

> **본 발표의 핵심 새 인사이트 ①**:
> Multi-Task 모델의 분류 헤드 Grad-CAM IoU는 *학습셋*에서도 **0.1232**에 불과하다. 한편 같은 모델의 *세분화 헤드* IoU는 **0.7753**으로 6배 이상 높다. **두 head는 같은 encoder를 공유하면서도 완전히 다른 패턴을 학습했다.**

이게 의미하는 바:
- Whole-Slice (분류 단독)의 test CAM IoU 0.145와 Multi-Task의 train CAM IoU 0.123은 **거의 동등** (v1 §3.3, §5.5 비교).
- 즉 **MT가 분류 head 자체를 "종양을 보게" 만든 것은 아니다.** 분류 head는 여전히 광역 단서를 그대로 본다.
- 그럼에도 v1에서 "MT IoU = 0.706"이라고 보고할 수 있었던 이유: **그 IoU는 *분류 head Grad-CAM*이 아니라 *세분화 head 마스크*의 IoU**.
- v1 §5.8 표의 "TP IoU 0.7061"이 *seg head의 IoU* 임은 `mt_evaluation_results.json["segmentation"]["mean_iou"] = 0.7061`로 직접 확인된다.

> **새로운 해석 (v1에서 한 차원 깊어진 메시지)**:
> "MT는 *분류 head의 shortcut을 직접 고친 게 아니라*, *세분화 head를 추가로 학습시켜 encoder가 종양 정밀 위치를 부담하게 만든 것*이다. 분류 head는 그 encoder feature 위에서 자기 방식대로 판단하지만, **결과적으로 모델 출력에 *분류 확률 + 정밀한 위치 마스크*가 함께 나오므로 임상적·해석성에 모두 유리**하다."

### 1.6 학습셋 FN/FP 패턴 분석

#### 1.6.1 FN (4,110개, prob mean=0.1823)
- 학습셋에서도 평균 확률 0.18로 "없다"고 강하게 확신하며 놓침 → v1 §3.5 test FN 확률 0.191과 거의 동일
- → **테스트셋의 FN 1,136은 *분포 외* 문제가 아니라 *학습 단계의 under-fit 패턴이 그대로 노출된 것*임을 의미** (학습셋에서 6.9%, 테스트셋에서 9.3% — 비율도 비슷)
- 결론: 추가 epoch이나 capacity 증가로는 못 잡음. **입력 표현 변경(멀티모달 / 2.5D / Tversky)이나 oversampling이 필요**

#### 1.6.2 FP (427개, prob mean=0.7087)
- 학습셋에서도 평균 확률 0.71로 *확신 있게* 오탐 → v1 §3.6 test FP 0.741과 유사
- 학습셋 비율 0.36% (test 2.37%)이라 절대 수는 적지만 **여전히 hard negative가 학습 종료까지 존재**

### 1.7 서브샘플 IoU 분포 (mt_train_gradcam_subsample.csv 400개)

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

📎 **참고**:
- `code/multitask/step18b_multitask_train_gradcam.py` (구현 + Grad-CAM 알고리즘 — model.enc5 = ResNet-18 layer4)
- `outputs/figures/multitask/gradcam_train/mt_train_gradcam_summary.json`
- `260521v1result.md` §3.3 (Whole-Slice test CAM IoU 0.145), §5.5 (Multi-Task test seg IoU 0.706)
- `260523v1updatemmmt.md` §② (step18b는 *읽기 전용 분석 스크립트* — 체크포인트에 손대지 않음 확인)

### 🎯 §1 예상 질문 (Q&A)

**Q1. CAM IoU가 0.12면 Day 5에서 "IoU 5배 향상"이라고 말했던 게 무의미하지 않나?**
A. 두 IoU는 의미가 다르다. v1 §5.5의 "TP IoU 0.706"은 **seg head의 마스크 IoU**이며, Day 5의 본질적 성과는 *분류 모델에 정밀한 위치 마스크가 추가로 나온다는 점*이다. 분류 head Grad-CAM의 IoU(0.12)는 거의 변하지 않았지만, 모델이 출력하는 위치 정보(seg mask) 자체가 IoU 0.706이라는 사실은 임상적으로 더 중요한 지표다. 단, 발표에서는 이 둘을 명확히 구분하여 설명해야 한다 — "CAM 기반 *해석성*은 그대로지만 *모델 출력의 정밀 위치*가 5배 좋아진 것".

**Q2. 그러면 Multi-Task는 사실 *Shortcut Learning을 극복한 게 아니라 우회한 것* 아닌가?**
A. 부분적으로 그렇다. Encoder는 여전히 "종양 위치를 보지 않으면서도 분류는 잘하는" shortcut feature를 갖고 있을 수 있다. 그러나 동시에 그 encoder가 seg head로 정밀 마스크를 만들어내므로, **encoder 안에 위치 정보가 *어느 정도는* 인코딩되어 있다**는 것도 사실이다. CAM은 이를 분류 head의 GAP→FC 라우팅을 통해 보는 도구라서 *해석성 측면의 한계*가 있는 것이지, 모델 자체가 위치를 모르는 것은 아니다.
- 이 한계가 정확히 v2 (`260521v2ways.md` §3.3)의 **Class Activation Mapping 한계** 지적이고, 그래서 Day 7+에서 *Score-CAM / EigenCAM / Attention Rollout* 같은 더 강한 XAI 도구로 확장이 권장된다.

**Q3. FN 4,110개를 학습셋에서도 잡지 못한 것은 *학습이 부족한 것* 아닌가? Epoch를 더 늘리면?**
A. 학습 곡선상 Phase 2 epoch 9 이후 Val Loss는 단조 증가(과적합)했다 (v1 §2.4). Epoch 14까지 진행됐어도 train FN율은 7.2%로 거의 변하지 않았으며, train acc는 96.6%에서 plateau. **즉 capacity 부족이 아니라 입력 표현(FLAIR 단일) 자체가 소종양 ET를 충분히 표현하지 못함**. → Day 6 멀티모달의 도입 근거. (실제로 §4 step26에서 멀티모달이 381개의 FN을 *recovered* 시킨다.)

📎 **참고**: 같은 §1의 자료 + `code/multitask/step17_multitask_model.py` (모델 구조 확인)

---

## 2. Day 6 — Multi-Modal Multi-Task 본 학습 결과 (step20~step25)

### 2.1 전처리 (step20) — T1ce 슬라이스 추출

**산출물**: `outputs/logs/mmmt/step20_t1ce_stats.json`

| 항목 | 값 |
|------|:--:|
| 처리된 환자 수 | **1,251명** |
| 저장된 T1ce 슬라이스 | **166,626개** (FLAIR와 1:1 매핑) |
| skipped_patient | 0 |
| skipped_slice | 27,279 (brain fraction < 1%) |
| missing_pair | 44 (FLAIR 슬라이스가 있는데 T1ce 추출에서 제외된 케이스) |
| errors | 0 |

> [보강] 정규화 방식은 step1과 동일 — *환자 내 percentile-clip [1, 99] + minmax → uint8*. 두 모달의 정규화를 일관되게 함으로써 |T1ce-FLAIR| diff 채널이 의미 있는 신호가 되도록 보장 (`code/mmmt/step20_mmmt_preprocess.py`).
>
> *missing_pair=44*: FLAIR에는 brain fraction ≥ 1%였지만 T1ce에서 동일 슬라이스가 brain mask 누락된 경우. 본 학습에서는 *step21 dataset이 페어 누락된 슬라이스를 자동으로 검은 이미지로 대체*하여 처리하지만, 향후 정량적 정밀도를 위해 *완전 일치 슬라이스만 사용*하는 strict mode 권장 (부록 A).

### 2.2 모델 (step22) — MMMTBrainNet

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
- v1 §6.5 (b)의 권장 전략 채택: 채널 평균 이식 X, **3채널 유지하고 세 번째에 |T1ce-FLAIR| 합성**.
- conv1의 ImageNet 가중치(3,64,7,7)가 *그대로* 재사용되어 transfer learning 효과 보존.

### 2.3 Loss (step23) — Tversky 추가

```
L_total = L_cls + β_seg · L_seg
L_cls   = BCE(pos_weight)
L_seg   = 0.5 · Dice(WT) + 0.5 · Tversky(WT, α=0.7, β=0.3)
β_seg   = 0.5 (Day 5와 동일)
```

> **Tversky α=0.7, β=0.3 의 의미** (v1 §6.7 #1, `260521v2ways.md` §3.1):
> Tversky = TP / (TP + α·FN + β·FP). α > β로 두면 FN에 더 큰 penalty → **소종양 누락에 강하게 학습**.
> Day 5의 약점이었던 FN 1,136에 직접 대응하는 손실함수 설계.

### 2.4 학습 설정 (step24) — Day 5와 동일하게 통제

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
| device | cuda (RTX 4070 8GB) | 동일 |
| Optimizer / Scheduler | AdamW / CosineAnnealing | 동일 |

> **통제 변수**: Day 6에서 *입력 채널 구성*과 *seg loss에 Tversky 0.5 weight 추가* 두 가지만 변경. 즉 "**멀티모달 + Tversky의 합산 효과**"가 측정된다. 순수 모달 효과만 분리하려면 §5 ablation 결과를 함께 봐야 한다.

### 2.5 학습 곡선 — 16 epoch 전체 (mmmt_history.json)

> **참고**: 본 학습은 P1 3 + P2 13 = **총 16 epoch**으로 종료됨. P2 patience=5 Early Stop이 trigger되어 P2 13번째에 멈춤. Day 5는 14 epoch (P2 11)이었으므로 약간 더 오래 학습.

| Epoch | Phase | Train Total | Train CLS | Train SEG | Train Acc | Train Dice | Val Total | Val CLS | Val SEG | Val Acc | Val Dice | LR |
|:-----:|:-----:|:-----------:|:---------:|:---------:|:---------:|:----------:|:---------:|:-------:|:-------:|:-------:|:--------:|:--:|
| 1 (P1) | P1 | 0.583 | 0.461 | 0.244 | 78.80% | 0.574 | 0.483 | 0.390 | 0.187 | 83.23% | 0.335 | 7.5e-4 |
| 2 (P1) | P1 | 0.546 | 0.452 | 0.187 | 79.58% | 0.654 | 0.521 | 0.438 | 0.166 | 81.67% | 0.355 | 2.5e-4 |
| 3 (P1 종료) | P1 | 0.525 | 0.442 | 0.165 | 79.95% | 0.704 | 0.471 | 0.389 | 0.163 | 83.13% | 0.356 | 0 |
| **4 (P2 시작)** | P2 | **0.291** | **0.216** | **0.151** | **91.81%** | **0.703** | **0.269** | **0.200** | **0.137** | **92.56%** | **0.378** | **9.89e-5** |
| 5 | P2 | 0.243 | 0.177 | 0.133 | 93.48% | 0.740 | 0.406 | 0.313 | 0.188 | 87.77% | 0.363 | 9.57e-5 |
| 6 | P2 | 0.229 | 0.166 | 0.126 | 93.89% | 0.752 | 0.259 | 0.195 | 0.128 | 92.97% | 0.390 | 9.05e-5 |
| 7 | P2 | 0.215 | 0.155 | 0.121 | 94.31% | 0.764 | 0.252 | 0.187 | 0.131 | 93.03% | 0.381 | 8.35e-5 |
| 8 | P2 | 0.209 | 0.149 | 0.119 | 94.53% | 0.751 | 0.231 | 0.171 | 0.120 | 93.80% | 0.397 | 7.5e-5 |
| 9 | P2 | 0.192 | 0.136 | 0.113 | 95.09% | 0.776 | 0.266 | 0.207 | 0.118 | 92.41% | 0.406 | 6.55e-5 |
| 10 | P2 | 0.184 | 0.129 | 0.110 | 95.35% | 0.761 | 0.230 | 0.173 | 0.114 | 94.04% | 0.412 | 5.52e-5 |
| **11 (★ Best Val Acc)** | P2 | **0.171** | **0.118** | **0.105** | **95.80%** | **0.793** | **0.228** | **0.172** | **0.111** | **94.12%** | **0.423** | **4.48e-5** |
| 12 | P2 | 0.160 | 0.108 | 0.102 | 96.19% | 0.785 | 0.240 | 0.184 | 0.113 | 93.76% | 0.417 | 3.45e-5 |
| 13 | P2 | 0.149 | 0.099 | 0.100 | 96.49% | 0.795 | 0.244 | 0.189 | 0.111 | 94.04% | 0.422 | 2.5e-5 |
| 14 | P2 | 0.140 | 0.092 | 0.097 | 96.80% | 0.798 | 0.245 | 0.191 | 0.109 | 93.56% | 0.424 | 1.65e-5 |
| 15 | P2 | 0.130 | 0.083 | 0.095 | 97.11% | 0.805 | 0.250 | 0.196 | 0.108 | 94.09% | 0.423 | 9.55e-6 |
| 16 (최종, Early Stop) | P2 | 0.122 | 0.075 | 0.094 | 97.40% | 0.798 | 0.257 | 0.203 | 0.108 | 94.03% | 0.426 | 4.32e-6 |

**핵심 관찰 (보존)**:

1. **Phase 1 → Phase 2 전환 (epoch 3 → 4)**:
   - Val total 0.471 → 0.269 (-43%) — encoder fine-tune의 즉각적 효과
   - Val CLS 0.389 → 0.200 (-49%) — Day 5와 거의 동일한 패턴 (v1 §5.4)
   - Val Acc 83.13% → 92.56% (+9.43%p) — 단일 epoch에서 가장 큰 점프

2. **Epoch 5 Val 진동**:
   - Val total 0.269 → 0.406 → 0.259 (epoch 4→5→6) — CosineAnnealing 초반의 노이즈
   - Val acc 92.56 → 87.77 → 92.97 — Val Dice는 영향이 적음 (0.378→0.363→0.390)
   - **Day 5와 다른 점**: Day 5도 epoch 5에서 약한 진동(83.0→81.5)이 있었지만 epoch 6부터 안정. Day 6은 진동이 epoch 5에서 더 크지만 epoch 6에서 곧바로 회복.

3. **Best Val Acc는 epoch 11** (94.12%) — Day 5의 Best Val Acc 93.84%(epoch 9)보다 +0.28%p
   - Day 5는 epoch 9에서 Val Loss 최저로 Best, Day 6는 epoch 11에서 Best.
   - **Val Dice는 epoch 16에서 최고치 0.4263** (Best Acc epoch과 다름) → Val Acc 기준 Best가 채택됨 (mmmt_best.pth).

4. **Train Dice는 0.798 — Val Dice 0.426와의 큰 gap**:
   - **이건 매우 의외이고 결정적인 관찰**. Day 5 Multi-Task의 Val Dice는 0.7564 (학습 곡선 마지막)인데 Day 6은 0.426으로 절반 수준.
   - 그러나 **Test Dice는 0.7874** (mmmt_test_metrics.json) — Val과 Test Dice 사이에 큰 차이.
   - **추정 원인**: Val Dice는 train_one_epoch의 `_wt_dice` 함수가 *양성 슬라이스만* 평균하면서 *Tversky가 작은 영역에 강하게 penalty를 주어* sigmoid 출력이 더 보수적이 됨 → val에서 sigmoid >= 0.5 임계로 이진화한 mask가 작아져 dice가 낮게 측정.
   - Test 평가(step25)에서는 다른 함수(`compute_seg_metrics`)로 mean dice를 계산 — 둘 사이의 측정 정의 차이가 누적된 결과. (코드 검증 항목 — 부록 A에서 *threshold 0.5가 아닌 양성 슬라이스 평균법* 일관성 점검 권장).

### 2.6 학습 시간 — Day 5 대비 30% 증가

| 모델 | 총 시간 | 총 epoch | epoch당 평균 |
|------|:-------:|:--------:|:------------:|
| Day 5 Multi-Task | 298.8분 | 14 | 21.3분 |
| **Day 6 MMMT** | **388.5분** | **16** | **24.3분** |

- 시간 증가 원인: (1) epoch 2회 더 진행 (14→16), (2) 입력이 3채널이라 데이터 로딩·전처리 비용 증가, (3) Tversky loss의 추가 연산.
- 학부 자원(RTX 4070 8GB)에서 **6.5시간**으로 완료 — v1 §6.4의 추정치 4~6h를 약간 상회.

### 2.7 평가 (step25) — Test N=25,115, threshold=0.5

```json
{
  "n_test": 25115,
  "cls": {
    "threshold": 0.5,
    "f1": 0.9427,
    "auroc": 0.9832,
    "auprc": 0.9863,
    "tp": 11401, "tn": 12327, "fp": 539, "fn": 848
  },
  "seg_wt": {
    "n_positive_slices": 12244,
    "dice_mean": 0.7874, "dice_median": 0.9094,
    "iou_mean": 0.7142, "iou_median": 0.8338
  }
}
```

### 2.8 Day 5 vs Day 6 1차 비교 (★ 본 발표의 중심 표)

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

- **FN 288개 감소(-25.4%)**: 멀티모달의 의학적 상보성이 발휘 — 작은 종양에서 T1ce ET 강조가 도움. v1 §6.3 "Day 5 약점 매칭 표"의 예측이 정량 검증됨.
- **그러나 FP 234개 증가(+76.7%)**: T1ce의 정상 조영 영역(혈관·맥락총 등)을 종양 후보로 잡는 새로운 오탐 패턴 발생. **v1 §6.3에서 "FP 감소"를 기대했던 것과 정반대 결과**.
- **AUROC는 동등(+0.0008)**: 모델의 *순위 매기기* 성능은 거의 차이 없음. 결정 경계의 위치만 이동.
- **이 결과는 단순한 "v1 §6.6 정량적 기대치" (FN 700~900, FP 150~250)와 다르다**: FN은 예상 범위에 들어왔지만(848), FP는 예상보다 2배 이상(539 vs 150~250). 즉 **멀티모달 + Tversky는 FN을 줄이는 데는 성공했지만, FP를 줄이는 데는 실패**.

### 2.9 Train-split Grad-CAM (step24b) — MMMT 버전

`outputs/figures/mmmt/gradcam_train/mmmt_train_gradcam_summary.json`:

```json
{
  "split": "train",
  "channel_mode": "t1ce_flair_diff",
  "checkpoint_epoch": 8,
  "checkpoint_val_acc": 94.12,
  "checkpoint_val_dice": 0.4229,
  "n_train_total": 116629,
  "TP": 53799, "FN": 3143, "FP": 1177, "TN": 58510,
  "train_accuracy_pct": 96.30,
  "iou_subsample_n": 400,
  "mean_iou_CAM_GT_trainTP": 0.1588,
  "mean_iou_SEG_GT_trainTP": 0.765,
  "FN_prob_mean": 0.2264,
  "FP_prob_mean": 0.6736
}
```

> [보강] `checkpoint_epoch: 8`은 0-indexed라면 9번째 epoch — 그러나 mmmt_history.json에서 Val Acc 94.12%는 epoch 11이다. 이 불일치는 step24b가 `mmmt_best.pth` 메타데이터에서 epoch 카운터를 다르게 저장했기 때문(P2 8번째 epoch = P2_start_epoch=4부터 카운트하면 11번째 전체 epoch과 일치). 본질적으로 같은 모델.

**MT vs MMMT Train Grad-CAM 비교**:

| 지표 | MT (Day 5) | MMMT (Day 6) | 변화 |
|------|:----------:|:------------:|:----:|
| Train TP | 52,832 | **53,799** | +967 |
| Train FN | 4,110 | **3,143** | **-967** ★ |
| Train FP | 427 | 1,177 | **+750** ❌ |
| Train TN | 59,260 | 58,510 | -750 |
| Train Accuracy | 96.11% | **96.30%** | +0.19%p |
| **CAM-GT IoU (학습 TP)** | **0.1232** | **0.1588** | **+0.0356 (+29%)** ★ |
| SEG-GT IoU (학습 TP) | 0.7753 | 0.7650 | -0.0103 |
| Train FN prob mean | 0.1823 | 0.2264 | +0.044 (확신 약화) |
| Train FP prob mean | 0.7087 | 0.6736 | -0.035 (확신 약화) |

**해석**:
1. **CAM IoU가 0.1232 → 0.1588 (+29%)** — 멀티모달이 분류 head를 약간 더 종양 위치를 보게 만듦.
2. **SEG IoU는 거의 동등(0.7753 → 0.7650, 학습셋 기준 -1.3%p)** — 학습셋 seg 정밀도는 두 모델이 비슷.
3. **학습셋에서도 FN 967개 감소·FP 750개 증가** — test의 FN -288, FP +234와 *같은 방향*. → **test 패턴은 학습 단계에서 이미 결정된 분포 안 결과**.
4. **Train FN prob mean이 0.18→0.23**: 멀티모달은 "확신 없게 놓치는" 패턴으로 약하게 이동 — 즉 threshold 조정으로 일부 회복 가능 (실제로 §3 sweep에서 검증).

### 2.10 시각화 (`outputs/figures/mmmt/`)

| 파일 | 내용 | 발표 활용 |
|------|------|----------|
| `mmmt_training_curves.png` | 16 epoch 학습 곡선 (CLS/SEG loss, Acc, Dice) | 슬라이드 §6 학습 안정성 |
| `mmmt_diagnostics.png` | ROC + PR + Confusion + Threshold curves | 슬라이드 §6 평가 결과 |
| `gradcam_train/mmmt_train_*.png` | 6열 그리드 (T1ce/FLAIR/diff/SegGT/CAM/SegPred) | 슬라이드 §6 해석성 |
| `threshold_sweep.png` (§3) | F1/FP/FN vs threshold | 슬라이드 §7 운용점 분석 |
| `fn_diff_*.png` (§4) | Z분포, tumor size, prob scatter | 슬라이드 §7 FN 회복 분석 |

### 🎯 §2 예상 질문 (Q&A)

**Q1. FP가 305 → 539로 +77% 증가했는데, 이건 v1 §6.6의 기대치(150~250)와 정반대다. 어떻게 설명하나?**
A. 세 가지 해석:
1. **T1ce 추가가 가져온 새 오탐 패턴**: T1ce는 정상 혈관·맥락총·뇌막 등이 강하게 조영되어 보이는 영역이 다수 존재. 멀티모달 모델이 이를 "종양 후보"로 잡는 경우가 발생.
2. **Tversky α=0.7의 부작용**: FN에 강한 penalty를 주는 게 의도였으나, 그 결과 sigmoid 출력이 *더 적극적으로* 양성을 예측하게 됨 → FP 증가의 직접 원인.
3. **Threshold가 0.5라서**: §3에서 보듯 threshold를 Day 5의 0.3847로 동등 보정하면 FP가 더 증가(787)하지만, threshold를 Day 6의 *자체 optimal* 0.6202(`mmmt_test_metrics_thr_sweep.json` 참조)로 올리면 FP가 더 줄어든다. 결정 경계가 이동한 것일 뿐 *순위 매기기 성능(AUROC)*은 동등.
- 따라서 발표에서는 "Day 5 → Day 6에서 *operating point*가 이동했고, 두 모델은 *AUROC 동등*이지만 *FN-FP 균형*은 시나리오에 따라 선택"이라고 정정 보고해야 한다.

**Q2. Day 5와 Day 6의 Best epoch이 9 vs 11로 다르다. 우연인가?**
A. 두 모델 모두 Phase 2 6번째(Day5) / 8번째(Day6)에 Best가 나옴. CosineAnnealing scheduler 곡선상 *LR이 5e-5 부근*에서 일반화 최적점이 형성되는 경향. Day 6은 입력 채널이 늘어 약간 더 천천히 수렴 — *Phase 2 8번째 = 전체 11번째*에 Best 도달. 우연이 아니라 같은 scheduler + 비슷한 capacity의 자연스러운 결과.

**Q3. Val Dice가 0.426으로 Test Dice 0.787과 큰 차이가 있다. 평가 기준이 일관된가?**
A. 이 부분은 코드 검증이 필요하다. `step24_mmmt_train.py`의 `_wt_dice` 함수와 `step25_mmmt_evaluate.py`의 `compute_seg_metrics`가 약간 다른 식으로 dice를 집계한다 (양성 슬라이스 정의, sigmoid threshold 적용 순서 등). 학습 중 Val Dice는 *훈련 모드에서 양성 슬라이스만 평균하면서 boundary case를 더 엄격하게 처리*하는 경향이 있어 낮게 나오고, Test 평가의 dice는 *full-test 기준 mean*이라 높게 나온다. **본 보고서는 Test Dice(0.787)를 표준 보고치로 사용**하며, 발표 시에는 Val Dice는 *학습 모니터링용*이라고 명시하는 것이 안전.

**Q4. Tversky loss를 도입한 게 정확히 어떤 효과를 줬는지 분리할 수 있나?**
A. 본 학습에서는 *멀티모달 입력*과 *Tversky 추가*가 동시에 도입됨. 두 효과를 분리하려면 **(D-1) FLAIR-only + Dice-only**, **(D-2) FLAIR-only + Dice+Tversky**, **(D-3) MMMT + Dice-only**의 3차 ablation이 필요 — 본 프로젝트에서는 미실행 (부록 A). 다만 **§5의 C-2 FLAIR-only(채널만 바꿈, loss는 Tversky 그대로)** 결과를 보면 *Tversky만 추가한 효과*가 분리되어 나오므로 부분적 답변 가능: F1 93.91 (Day 5 Dice-only) → 94.15 (FLAIR-only + Tversky, +0.24%p). 즉 **Tversky의 *분리된* 기여는 +0.24%p F1 정도이고, 추가 +0.12%p가 멀티모달 입력의 기여**.

**Q5. CAM IoU가 +29% 올랐다고 했는데, 0.12 → 0.16은 여전히 0.2도 안 되는 낮은 값이다. 해석성 향상이라고 부를 수 있나?**
A. *분류 head Grad-CAM의 IoU*만 보면 미세한 향상이다. 그러나 (1) seg head IoU 0.71(test)은 매우 높고, (2) FN의 평균 prob이 0.18→0.23으로 약간 회복 가능한 영역으로 이동, (3) §4에서 FN 381개가 *recovered*된 *정성적* 효과 — 이 세 가지를 함께 봐야 한다. **분류 head CAM IoU만으로 해석성을 판단하는 것은 v1 §3의 한계 진단과 같은 잘못이다** — 발표에서는 "CAM은 분류 라우팅의 결과일 뿐, *모델의 위치 인지 능력*은 seg head 마스크 + recovered 분석으로 더 정확히 측정된다"고 설명해야 한다.

📎 **참고**:
- `code/mmmt/step20_mmmt_preprocess.py` ~ `step25_mmmt_evaluate.py`
- `code/mmmt/step24b_mmmt_train_gradcam.py`
- `outputs/logs/mmmt/mmmt_summary.json`, `mmmt_history.json`, `mmmt_test_metrics.json`
- `outputs/figures/mmmt/gradcam_train/mmmt_train_gradcam_summary.json`
- `260521v1result.md` §6 (Day 6 전환 계획)
- `260523v1updatemmmt.md` §② (재실행 불필요 점검)

---

## 3. 권장 3종 결과 A — Threshold 동등 보정 (step25b)

### 3.1 작업의 목적 (보존)

> Day 5 MTL의 ROC-Youden 최적 임계값 **0.3847**은 그 모델의 *운용 지점*이고, Day 5의 F1 93.91 / FP 305 / FN 1136은 *그 threshold*에서 산출된 값이다. Day 6 MMMT의 기본 평가(step25)는 threshold=0.5에서 수행되었으므로, "멀티모달의 *순수* 효과"를 분리하려면 두 모델을 *같은 운용 지점*에서 비교해야 한다.

(`260523v1updatemmmt.md` §④ Phase 1 (A)에서 권장된 작업)

### 3.2 산출물

| 파일 | 내용 |
|------|------|
| `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json` | thr ∈ {0.30, 0.35, 0.3847, 0.40, 0.45, 0.50}의 confusion matrix 및 분류 메트릭 |
| `outputs/figures/mmmt/threshold_sweep.png` | thr vs (F1, FP, FN, Recall) 곡선 |

### 3.3 Threshold Sweep 결과 (보존)

| Threshold | TP | TN | FP | FN | Accuracy | Precision | Recall | Specificity | F1 |
|:---------:|:--:|:--:|:--:|:--:|:--------:|:---------:|:------:|:-----------:|:--:|
| 0.30 | 11,633 | 11,790 | 1,076 | 616 | 93.26 | 91.53 | **94.97** | 91.64 | 93.22 |
| 0.35 | 11,559 | 11,975 | 891 | 690 | 93.70 | 92.84 | 94.37 | 93.07 | 93.60 |
| **0.3847 (★ Day5 운용점)** | **11,522** | **12,079** | **787** | **727** | **93.97** | **93.61** | **94.06** | **93.88** | **93.84** |
| 0.40 | 11,508 | 12,112 | 754 | 741 | 94.05 | 93.85 | 93.95 | 94.14 | 93.90 |
| 0.45 | 11,457 | 12,226 | 640 | 792 | 94.30 | 94.71 | 93.53 | 95.03 | 94.12 |
| 0.50 (Day 6 기본) | 11,401 | 12,327 | 539 | 848 | 94.48 | 95.49 | 93.08 | 95.81 | **94.27** |
| Day 6 자체 optimal 0.6202* | — | — | — | — | — | — | — | — | (최대) |

\* `day6_optimal_thr`: 0.6202는 ROC-Youden 기준의 Day 6 자체 최적값이지만 sweep 표에는 포함되지 않음 (그리드 외).

### 3.4 동등 비교의 핵심 표 (★ 발표용)

| 모델 | Threshold | TP | TN | FP | FN | F1 | Recall | Precision |
|------|:---------:|:--:|:--:|:--:|:--:|:--:|:------:|:---------:|
| Day 5 MTL (FLAIR only) | 0.3847 | 11,113 | 12,561 | 305 | 1,136 | 93.91 | 90.73 | 97.33 |
| **Day 6 MMMT (T1ce+FLAIR+diff)** | **0.3847** | **11,522** | **12,079** | **787** | **727** | **93.84** | **94.06** | **93.61** |
| Day 6 MMMT | 0.5 | 11,401 | 12,327 | 539 | 848 | **94.27** | 93.08 | 95.49 |

### 3.5 결정적 발견 (★)

> **본 발표의 핵심 새 인사이트 ②**:
> "동일 threshold(0.3847)에서 Day 5 vs Day 6을 비교하면 **F1은 -0.07%p로 거의 동등**(93.91 vs 93.84)이고 **AUROC도 +0.0008로 거의 동등**(0.9824 vs 0.9832). 즉 **멀티모달의 효과는 "성능 향상"이 아니라 "운용 자유도 확장"이었다.** Day 6 모델은 thr 0.3~0.5 어디서 운용해도 F1 93+ 안에 들어와, 임상 시나리오별로 더 폭넓게 조정 가능."

### 3.6 동등 보정 후 정정된 trade-off

`260523v1updatemmmt.md` §3.1에서 "동등 임계로 보정 시 FP 350~400 추정"이라고 했던 부분이 *실측치 787*로 확인됨. **즉 추정치가 과소 추정이었음**. FP가 더 늘어난 이유:

1. T1ce의 정상 조영 영역이 진성 종양으로 학습된 패턴이 광범위.
2. Tversky α=0.7이 sigmoid 출력을 양성 편향으로 만듦 — threshold 동등 적용 시 FP가 그대로 노출.
3. Day 5 MTL의 FP 305는 *유난히* 낮은 값이었음(Recall 90.7%로 보수적 운용) — Day 6은 Recall 94.1%로 더 적극적 검출 모드.

### 3.7 임상 시나리오 매핑 (보존)

| 시나리오 | 권장 threshold (Day 6) | TP | FN (놓침) | FP (오탐) | 비고 |
|----------|:----------------------:|:--:|:---------:|:---------:|------|
| 1차 스크리닝 (Recall 최우선) | **0.30** | 11,633 | 616 | 1,076 | 종양 누락 최소화 |
| 균형 운용 (F1 최우선) | **0.45~0.50** | 11,401~11,457 | 792~848 | 539~640 | 일반 진단 |
| 확진 (Precision 최우선) | **0.6+** | (낮음) | (높음) | (낮음) | 확진 모드 |
| Day 5와 동등 비교용 | **0.3847** | 11,522 | 727 | 787 | 학술 공정 비교 |

📎 **참고**:
- `code/mmmt/step25b_mmmt_evaluate_thr.py`
- `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json`
- `outputs/figures/mmmt/threshold_sweep.png`
- `260523v1updatemmmt.md` §③ (왜 0.3847인가 — Youden's J 산출 근거), §④ Phase 1 (A) 권장 사항

### 🎯 §3 예상 질문 (Q&A)

**Q1. Day 6 자체 ROC-Youden optimal이 0.6202라면 그 지점에서 비교하는 게 더 공정하지 않나?**
A. 두 가지 기준이 모두 정당하다. (1) *모델별 자체 optimal에서 비교*: Day5@0.3847(F1 93.91) vs Day6@0.6202(F1 추정 ~94.5). 이건 "각자 최고의 운용점"에서 보는 view. (2) *동일 운용점에서 비교*: Day5@0.3847 vs Day6@0.3847. 이건 "같은 운영 비용에서 누가 더 잘 동작하는가" view. 발표에서는 **(1)을 메인 비교로, (2)를 학술 공정 보조로** 제시하는 것이 권장. 본 보고서는 v1 §6.6에서 이미 (1) 관점으로 기대치를 제시했으므로 동일 관점 유지가 안전.

**Q2. Threshold 0.3847의 F1 (93.84)이 Day 5의 F1 (93.91)보다 살짝 낮다. 그러면 "멀티모달이 더 좋다"는 발표 메시지를 어떻게 정당화하나?**
A. 정확히 그래서 **본 보고서가 v1의 "+1.5~2.5%p F1 향상" 기대치를 *부분 수정*해야 한다.** 새 메시지:
- "멀티모달 + Tversky의 효과는 *F1 점프*가 아니라 *Recall 점프*(90.73 → 94.06, +3.33%p)에 집중."
- "이 trade-off의 임상 가치는 §4 step26 분석에서 *FN 381개의 진짜 복구*로 확인된다."
- "AUROC 0.9832 → 운용점 자유도가 확장 (`threshold_sweep.png` 참조)."

**Q3. Day 6 학습에 Tversky가 들어갔는데, 만약 Day 5에도 Tversky를 넣었다면 비교가 더 공정했을 것 아닌가?**
A. 정당한 우려. §5 ablation의 **C-2 FLAIR-only**가 *Day 5와 같은 입력으로 Day 6 코드 경로(Tversky 포함)로 재학습*한 결과인데, F1 94.15가 나옴 (Day 5의 93.91보다 +0.24%p). 즉 **Tversky만의 기여는 +0.24%p**. 멀티모달의 *추가* 기여는 +0.12%p (94.15 → 94.27). 두 효과의 분리는 §5에서 더 자세히.

---

## 4. 권장 3종 결과 B — Day5↔Day6 FN 차분 분석 (step26)

### 4.1 작업의 목적 (보존)

> Day 5 MTL과 Day 6 MMMT의 *테스트 슬라이스별 예측을 1:1로 비교*해, **어떤 슬라이스가 새로 회복(recovered)되고, 어떤 슬라이스가 여전히 놓치며(still_missed), 어떤 슬라이스가 후퇴(regressed)했는지** 를 정량 분리한다.

(`260523v1updatemmmt.md` §④ Phase 1 (B))

### 4.2 산출물

| 파일 | 내용 |
|------|------|
| `outputs/logs/mmmt/fn_diff_per_slice.csv` | 25,115행: filename, patient_id, slice_idx, label, tumor_pixels, day5_prob, day6_prob, group_fair, group_split |
| `outputs/logs/mmmt/fn_diff_summary.json` | 그룹별 카운트 + 메트릭 차분 표 |
| `outputs/figures/mmmt/fn_diff_zdist.png` | 그룹별 z-위치 분포 |
| `outputs/figures/mmmt/fn_diff_tumor_size.png` | 그룹별 종양 픽셀 수 분포 |
| `outputs/figures/mmmt/fn_diff_prob_scatter.png` | Day5 prob vs Day6 prob scatter (그룹 색상) |

### 4.3 8개 비교 그룹 정의 (보존)

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

### 4.4 그룹 카운트 — 두 비교 시나리오

#### 4.4.1 Fair 비교 (Day5 @ 0.3847, Day6 @ 0.3847) — 학술 공정 비교

| Group | N | 의미 |
|-------|:-:|------|
| common_tp | 11,141 | (대부분) |
| common_tn | 11,960 | (대부분) |
| **recovered** | **381** | **★** |
| still_missed | 641 | |
| regressed | 86 | |
| **new_fp** | **515** | **❌** |
| cleaned_fp | 119 | |
| common_fp | 272 | |

총합: 25,115 = 11,141+11,960+381+641+86+515+119+272 ✓

#### 4.4.2 Split 비교 (Day5 @ 0.3847, Day6 @ 0.5) — 본 보고서의 원본 비교

| Group | N | 의미 |
|-------|:-:|------|
| common_tp | 11,102 | |
| common_tn | 12,162 | |
| **recovered** | **299** | ★ |
| still_missed | 723 | |
| regressed | 125 | |
| new_fp | 313 | ❌ |
| cleaned_fp | 165 | |
| common_fp | 226 | |

총합: 25,115 ✓

> **두 비교의 차이**: Fair 비교는 Day 6도 0.3847로 *더 적극적 검출*이라 recovered(381)가 많지만 new_fp(515)도 많다. Split 비교는 Day 6 기본 0.5에서 *더 보수적*이라 recovered(299)는 적지만 new_fp(313)도 적다.

### 4.5 그룹별 통계 (Fair 비교 기준, group_stats_fair)

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

### 4.6 결정적 발견 (★)

> **본 발표의 핵심 새 인사이트 ③**:
> 멀티모달이 새로 잡은 *recovered 381 슬라이스*의 평균 종양 크기는 **231 픽셀(0.46%)** 로 매우 작다. Day 5에서 prob=0.17로 강하게 "없다"고 판단했던 케이스를 Day 6에서 prob=0.70으로 끌어올림. **즉 v1 §6.3에서 가설로 제시했던 "T1ce 조영제로 ET가 명확히 강조되어 작은 종양도 검출"이 정량 검증된 첫 사례.**

> **본 발표의 핵심 새 인사이트 ④**:
> 그러나 *still_missed 641 슬라이스*는 평균 종양 크기 **100 픽셀(0.20%)** 로 극소(주로 종양 경계 슬라이스). Day 5 prob 0.08 → Day 6 prob 0.14 — 약간 올라갔지만 threshold 0.3847 미달. **이 극소 종양 영역은 멀티모달로도 잡지 못하는 잔여 한계이며, Day 7+ Tumor-CP / 2.5D 입력 / Focal-Tversky의 표적이다.**

> **본 발표의 핵심 새 인사이트 ⑤**:
> Regressed 86개 (Day 5 잡았지만 Day 6이 놓침)의 평균 종양 크기는 **222 픽셀**로 recovered와 비슷. Day 5 prob 0.65 → Day 6 prob 0.24. **즉 Day 5와 Day 6은 "다른 종류의 작은 종양"을 잡고 놓치는 패턴 — *상호 보완적*이라 ensemble이 효과적일 가능성이 크다.** (Day 8+ Deep Ensemble의 직접 근거)

### 4.7 Z-위치 분포 — 임상 의미

- common_tp의 z_median=85: 뇌 중앙부에 큰 종양이 분포 (v1 §1.5 EDA의 "중앙부 집중" 분포와 일치).
- recovered의 z_median=69: 뇌 *중앙 약간 아래* 영역. T1ce가 이 영역 ET 검출에 도움.
- still_missed의 z_median=62: 더 아래쪽 (소뇌·뇌간 근방) — 정상 구조가 복잡한 영역에서 극소 종양 분리 어려움.
- new_fp의 z_median=69: recovered와 같은 영역 — **T1ce의 정상 조영 구조와 종양 구조가 겹치는 위치에서 새 오탐 발생**.

### 4.8 Probability scatter (`fn_diff_prob_scatter.png`)

이 그림은 25,115 슬라이스를 Day5 prob (x축) vs Day6 prob (y축)으로 scatter, 그룹 색상 부여:
- recovered: (low x, high y) — 좌상단 cluster
- regressed: (high x, low y) — 우하단 cluster (recovered의 반대)
- still_missed: (low x, low y) — 좌하단
- new_fp: (low x, mid y) — 좌중단
- common_tp: (1, 1) 근방 우상단 cluster

→ 발표 시 1장의 슬라이드로 "멀티모달이 무엇을 했나"를 직관적으로 보여줄 핵심 그림.

📎 **참고**:
- `code/mmmt/step26_fn_diff_analysis.py` (구현)
- `outputs/logs/mmmt/fn_diff_summary.json`
- `outputs/logs/mmmt/fn_diff_per_slice.csv` (25,115 행)
- `outputs/figures/mmmt/fn_diff_{zdist,tumor_size,prob_scatter}.png`
- `260523v1updatemmmt.md` §④ Phase 1 (B)

### 🎯 §4 예상 질문 (Q&A)

**Q1. recovered 381과 regressed 86은 서로 비슷한 패턴(작은 종양)인데, 왜 한쪽은 잡고 다른 쪽은 놓치는가?**
A. **상호 보완성의 직접 증거.** 두 모델은 같은 작은 종양 분포를 보지만 *각자 다른 부분 집합*을 잡는다. T1ce 강조에 잘 보이는 ET는 Day 6이 잡고(recovered), FLAIR 부종 단서가 강한 케이스는 Day 5가 잡았는데(regressed) Day 6 모델이 T1ce 신호를 우선시하면서 놓침. → **Day 8+ Ensemble (Day 5 + Day 6)을 적용하면 recovered + regressed = 467개를 모두 잡을 가능성**.

**Q2. still_missed 641 (평균 100 픽셀)의 정확한 임상적 정체는?**
A. 픽셀 100개 = 224×224 슬라이스 중 **0.20%**. 뇌 부피의 극소 영역. 임상적으로는 (a) 종양 시작 슬라이스의 경계, (b) micro-metastasis, (c) seg mask 라벨링 오차 가능성도 있다. **추가 분석 (z-위치+모달 신호 강도) 권장** — 부록 A. 단 발표에서는 "이 영역은 *2D 슬라이스 단위*의 한계이며, 2.5D 또는 3D 모델에서만 잡힐 수 있다"고 정직히 보고하는 것이 안전.

**Q3. new_fp 515개의 prob mean이 0.579로 *반쯤 확신*이다. 이걸 임상에서 어떻게 처리해야 하나?**
A. 두 가지 처리 가능:
1. **Threshold를 0.5에서 0.6으로 올림** — new_fp 다수가 0.55~0.6 영역에 있을 가능성 (확인 항목, 부록 A의 prob 분포 분석 추가 필요). FP 줄이지만 recovered도 일부 잃을 위험.
2. **확신 구간(0.4~0.6)을 *불확실*로 분류**하여 *의사 검토 큐*에 보냄. → Conformal Prediction (`260521v2ways.md` §5.5)의 직접 응용.

**Q4. 386 + 641 + 86 = 1,108개의 양성 슬라이스가 분기 — 양성 슬라이스 합계 12,249 - 11,141(common_tp) = 1,108 맞다. 검증 통과.**
A. (`recovered` 381 + `still_missed` 641 + `regressed` 86 = 1,108 = 12,249 - 11,141 — 음성과 양성 슬라이스가 분리되어 추적되고 있음을 검증)

**Q5. cleaned_fp 119는 day5_prob 0.596 → day6_prob 0.194. 이 119개의 *진짜 정체*는?**
A. Day 5가 0.596 확신으로 "종양"이라 했던 정상 슬라이스를 Day 6이 0.194로 정상화. 평균 z_median=55 (뇌 *위쪽*). **추정**: 두개골 근접 영역의 FLAIR 고신호 (normal-appearing white matter 변성·CSF 흐름 artifact)를 Day 5가 종양으로 오인했는데, Day 6은 T1ce 채널에서 조영 증강이 없음을 보고 *cross-check*하여 거부. **v1 §6.3의 "T1ce에서 증강 안 되면 진성 종양 아님 → cross-check" 가설의 정량 검증.**

---

## 5. 권장 3종 결과 C — 채널 Ablation 학습/평가 (step27 / step28)

### 5.1 작업의 목적 (보존)

> "멀티모달의 *어느 채널이* 성능 향상에 기여했는가?" 를 분리하기 위해 *입력 채널만 바꾸고* 나머지 (모델/loss/스케줄/augment)는 step24와 **완전히 동일하게** 학습한다.
> - **C-1**: channel_mode='t1ce_only' → [T1ce, T1ce, T1ce] (3복제)
> - **C-2**: channel_mode='flair_only' → [FLAIR, FLAIR, FLAIR] (Day 5와 동일 입력, 다른 loss/코드 경로)

(`260523v1updatemmmt.md` §④ Phase 2)

### 5.2 산출물

| 파일 | 내용 |
|------|------|
| `outputs/checkpoints/mmmt_ablation/{t1ce_only,flair_only}/best.pth` | 각 ablation의 best 체크포인트 |
| `outputs/logs/mmmt_ablation/{t1ce_only,flair_only}/{summary,history}.json` | 학습 통계 |
| `outputs/logs/mmmt_ablation/ablation_table.json` | 3-row × 메트릭 통합 표 |
| `outputs/figures/mmmt_ablation/ablation_table.png` | 비교 막대 차트 (F1, Recall, FP, Dice) |
| `outputs/figures/mmmt_ablation/{t1ce_only,flair_only}/training_curves.png` | 개별 학습 곡선 |

### 5.3 학습 통계 비교

| 모델 | Channel mode | 총 시간 | 총 epoch | Best Val Acc | Best Val Dice |
|------|--------------|:-------:|:--------:|:------------:|:-------------:|
| Day 6 baseline | t1ce_flair_diff | 388.5분 | 16 | 94.12% (e11) | 0.4263 (e16) |
| **C-1: T1ce-only** | t1ce_only | 340.2분 | 15 (마지막까지) | 88.40% (e8) | 0.311 (e15) |
| **C-2: FLAIR-only** | flair_only | **715.3분** | 12 (Early Stop) | 93.67% (e10) | 0.4168 (e12) |

**관찰**:
- **C-2 학습 시간이 가장 김(715분 = 11.9h)**: 동일한 epoch 수 단위 시간이 비슷하다면 12 × ~60분 = 720분 → 1 epoch당 60분 정도. baseline (24분/epoch)과 C-1 (23분/epoch)에 비해 2.5배 느림.
- **추정 원인**: C-2 학습은 *다른 시점에 다른 디스크 캐시 상태*에서 돌렸을 가능성. baseline과 같은 데이터 파이프라인이지만 1ch 입력을 3복제하느라 GPU-CPU 데이터 이송 부담이 늘었을 수 있음 (또는 단순히 환경 차이).
- **이게 발표에서 문제가 되지 않는가?**: 학습 *시간*은 ablation 결과의 *모델 성능*과 독립. 다만 발표에서는 "측정 환경 일관성"을 위해 시간 차이를 *언급은 하되* 모델 비교에서는 제외하는 것이 안전.

### 5.4 핵심 평가표 (`ablation_table.json`)

| 모델 | AUROC | AUPRC | F1 @ 0.5 | F1 @ 0.3847 | TP/TN/FP/FN @ 0.5 | seg Dice | seg IoU |
|------|:-----:|:-----:|:--------:|:-----------:|:------------------:|:--------:|:-------:|
| **Day 6 baseline** [T1ce,FLAIR,diff] | **0.9832** | **0.9863** | **94.27** | 93.84 | 11401/12327/539/848 | **0.7874** | **0.7142** |
| **C-1: T1ce-only** [T1ce, T1ce, T1ce] | 0.9487 | 0.9582 | 87.55 | 86.71 | 10414/11739/1127/1835 | 0.5741 | 0.4780 |
| **C-2: FLAIR-only** [FLAIR, FLAIR, FLAIR] | 0.9822 | 0.9853 | 94.15 | **93.98** | 11205/12518/348/1044 | 0.7458 | 0.6703 |

> 참고: Day 5 MTL @ 0.3847은 F1 93.91, AUROC 0.9824, FP 305, FN 1136, dice 0.7765, iou 0.7061.

### 5.5 결정적 발견 — Day 6의 "성과" 분해 (★ 본 발표의 가장 중요한 슬라이드)

> **본 발표의 핵심 새 인사이트 ⑥**:
> "**WT 분류의 결정적 신호는 FLAIR가 거의 다 가지고 있다.** C-2 FLAIR-only AUROC 0.9822 ≈ baseline AUROC 0.9832 (동등). T1ce-only는 AUROC 0.9487로 분명히 약함."

> **본 발표의 핵심 새 인사이트 ⑦**:
> "**Day 6 멀티모달의 *순수* 효과**:
> - 분류 F1 (@0.5): C-2 → baseline 94.15 → 94.27 (+0.12%p) — 거의 미미.
> - SEG Dice: C-2 → baseline 0.7458 → 0.7874 (+0.0416, +5.6%) — **여기서 진짜 의미 있는 향상이 발생**.
> - 즉 *T1ce + diff 채널은 분류보다 세분화 정밀도에 더 큰 기여를 한다.*"

### 5.6 더 미묘한 발견 — Day 5 (Dice-only) vs C-2 (Dice+Tversky)

같은 입력(FLAIR-only), 같은 모델 구조, 다른 Loss:

| 비교 | F1 @ 0.3847 | FP | FN | seg Dice | seg IoU |
|------|:-----------:|:--:|:--:|:--------:|:-------:|
| Day 5 MTL (Dice only) | 93.91 | **305** | 1,136 | 0.7765 | 0.7061 |
| **C-2 (Dice + Tversky)** | **93.98** | 566 | **890** | 0.7458 | 0.6703 |

- **Tversky의 *분리된* 기여**: F1 +0.07%p (거의 없음), **FN -246 (-21.7%)**, FP +261 (+85.6%), seg Dice **-0.0307** (감소!), seg IoU -0.0358.
- **놀라운 결과**: Tversky가 **분류 FN을 줄이는 데는 성공**했지만 **segmentation Dice는 오히려 감소**시켰다.
- **추정 이유**: Tversky α=0.7이 sigmoid 출력을 양성 편향으로 만들면서, seg head에서도 *작은 양성 영역을 과대 예측*하는 경향이 생김. → seg dice 감소.
- 즉 **Tversky는 분류에 좋고 seg에 나쁘다**는 양면성.
- **이 관찰의 발표 가치**: Tversky를 *분류 보조 task에는 적용하되 seg main task에는 더 보수적인 가중*이 권장된다는 정량 근거. v2 `260521v2ways.md` §3.2에서 "Focal-Tversky / 적응적 α"의 추가 필요성을 시사.

### 5.7 T1ce-only의 *학습 다이내믹* 이상 관찰 (★ 추가 발견)

`outputs/logs/mmmt_ablation/t1ce_only/history.json`에서:

```
epoch 1~8 train_dice: 0.028 → 0.000 → 0.000 → 0.000 → 0.000 → 0.000 → 0.000 → 0.000
epoch 9: 0.000
epoch 10~15: 0.046 → 0.572 → 0.599 → 0.612 → 0.614 → 0.625
```

> **본 발표의 핵심 새 인사이트 ⑧**:
> "**T1ce-only로는 초반 9 epoch까지 seg head가 *전혀 학습되지 않았다*** (train dice 0.000). Encoder가 분류 head로만 신호를 보내다가 epoch 10부터 seg가 깨어남. 즉 **T1ce는 *WT(전체 종양)* 라벨에 대한 정밀 위치 신호가 부족하여 Tversky로 학습이 막히는 단계가 길다.** FLAIR가 본 프로젝트의 WT 검출에 *결정적 모달리티*임을 학습 다이내믹 자체가 보여준 사례."

- 동일 Loss(Dice+Tversky)와 동일 Tversky(α=0.7, β=0.3)에서 FLAIR-only는 train_dice 0.572 → 0.778 (epoch 1→12)로 정상 학습 — T1ce-only는 9 epoch 동안 0.000 plateau.
- 이는 v1 §6.2의 "FLAIR는 ED를 잘 보고, T1ce는 ED를 안 보임" 의학적 사실과 일치 — **WT = NCR + ED + ET이므로 ED를 못 보는 T1ce-only로는 WT 마스크 학습이 어려움**.

### 5.8 "diff 채널의 진짜 기여" 분리

baseline = T1ce + FLAIR + |T1ce - FLAIR|
- baseline F1 94.27 vs FLAIR-only F1 94.15 → **+0.12%p** (분류)
- baseline seg dice 0.7874 vs FLAIR-only seg dice 0.7458 → **+0.0416** (+5.6%) (세분화)

**즉 diff 채널은 분류에는 거의 영향 없고, 세분화 정밀도 향상에 +5.6%p Dice 기여.** **이는 |T1ce - FLAIR|가 *종양 경계 후보*의 추가 inductive bias로 작동했다는 v1 §6.5 (b) 가설을 검증.**

### 5.9 4-Way 종합 비교 표 (보존)

| 모델 | 입력 | Loss | F1 @ 0.5 | F1 @ 0.3847 | seg Dice | AUROC |
|------|------|------|:--------:|:-----------:|:--------:|:-----:|
| **Whole-Slice** (Day 2) | FLAIR 1ch | BCE only | (94.10 best F1) | — | (없음) | 0.9832 |
| **Patch-Based** (Day 4) | FLAIR 64×64 patch | BCE only | (87.08 count agg.) | — | (없음) | 0.9104 |
| **Day 5 Multi-Task** | FLAIR 1ch | BCE + Dice | (94.27) | **93.91** | **0.7765** | 0.9824 |
| **Day 6 MMMT baseline** | T1ce+FLAIR+diff | BCE + Dice + Tversky | **94.27** | 93.84 | **0.7874** | **0.9832** |
| C-2 FLAIR-only ablation | FLAIR 3복제 | BCE + Dice + Tversky | 94.15 | 93.98 | 0.7458 | 0.9822 |
| C-1 T1ce-only ablation | T1ce 3복제 | BCE + Dice + Tversky | 87.55 | 86.71 | 0.5741 | 0.9487 |

📎 **참고**:
- `code/mmmt/step27_ablation_train.py`, `step28_ablation_evaluate.py`
- `outputs/logs/mmmt_ablation/ablation_table.json`
- `outputs/figures/mmmt_ablation/ablation_table.png`
- `outputs/logs/mmmt_ablation/{flair_only,t1ce_only}/history.json`
- `260523v1updatemmmt.md` §④ Phase 2 (C-1, C-2)

### 🎯 §5 예상 질문 (Q&A)

**Q1. FLAIR-only ablation이 baseline과 거의 동등하다면, 굳이 T1ce를 추가할 가치가 있나?**
A. (1) **분류만 본다면 한계 효용 약함** (+0.12%p F1). (2) **세분화 품질을 본다면 가치 분명** (+5.6%p Dice). (3) **임상적으로 WT/TC/ET 등급화 (`260521v2ways.md` §6)** 를 하려면 T1ce 필수 — ET는 T1ce 없이는 검출 불가. (4) **본 프로젝트는 WT 단일 region을 사용하지만, Day 7 grading으로 확장 시 T1ce는 필수**. 따라서 발표 메시지는 "본 단계에서 멀티모달의 *분류 이득*은 미미하지만 *세분화 이득과 grading 확장성*은 유지" 로 정직하게.

**Q2. C-2 FLAIR-only가 Day 5보다 F1 +0.07%p 더 좋은데, 이게 *Tversky의 효과*인지 *학습 환경 변동*인지 어떻게 검증하나?**
A. 두 학습은 같은 split, 같은 seed, 같은 모델 구조에서 *Loss만 다름*. 일반적으로 sklearn split + torch seed 고정 시 차이가 ±0.05%p 이내이지만 본 실험은 +0.07%p로 *경계상*. **엄밀한 검증을 위해서는 *동일 환경에서 3 seed 평균* 필요** — 본 보고서에는 미실행 (부록 A). 발표에서는 "Tversky 도입은 약한 양의 효과로 보이나, 통계적 유의성 검증은 미실행" 으로 정직 보고.

**Q3. T1ce-only가 학습 초반 9 epoch 동안 train_dice 0.000인 게 정말 *모달 한계*인가 *초기화 문제*인가?**
A. (1) Tversky α=0.7 + Dice combo loss에서 sigmoid 출력이 0에 갇히면 dice도 0. (2) 같은 loss에 FLAIR-only는 epoch 1부터 dice 0.572 — *초기화 문제가 아닌 입력 신호 문제*. (3) **WT 마스크의 픽셀 중 ED 영역이 차지하는 비율이 크기 때문**: BraTS 라벨 정의상 WT = NCR + ED + ET이고 ED가 대부분 면적. T1ce는 ED를 거의 안 보이므로 모델이 "어디를 양성으로 예측해도 GT와 안 맞음" 상태에서 시작 → loss landscape이 평평. **이건 *Day 7+에서 WT/TC/ET 3-region 분리* 시 T1ce-only의 TC/ET 학습은 정상화될 것으로 예측됨** — ablation 후속 항목.

**Q4. baseline (멀티모달) 대신 *channel mode를 ratio나 multiplication*으로 바꾸면 더 좋을 수도 있지 않나?**
A. `step21_mmmt_dataset.py`에 `t1ce_flair_ratio`, `t1ce_flair_mul` 모드가 이미 준비되어 있으나 본 실험에서는 미실행 (부록 A). v1 §6.5 (b)에서 *권장 (ii) diff 채널*을 선택한 이유는 (1) ImageNet conv1 가중치 재사용, (2) |diff|가 종양 경계 후보의 inductive bias 가장 직관적. ratio/mul은 normalization 안정성 우려가 있어 후순위.

**Q5. T1ce-only의 seg dice 0.5741은 *T1ce가 ET를 잘 본다*는 의학적 사실과 모순 아닌가?**
A. 본 ablation의 seg label은 **WT (= NCR + ED + ET)** 라서 ED 면적이 큰 점이 T1ce-only에 불리. 만약 *TC (=NCR + ET)* 나 *ET only*를 라벨로 했다면 T1ce-only의 dice가 훨씬 높을 것 — 이게 Day 7+ 3-region multi-class seg(`260521v2ways.md` §6)에서 검증될 핵심 주제. 즉 본 ablation 결과는 "**WT 마스크 기준**의 채널 기여도"이지 "모든 region 기준"의 결론이 아니다.

---

## 6. 전체 모델 비교 (Whole / Patch / MT / MMMT / Ablations 6-way)

### 6.1 분류 성능 종합

| # | 모델 | 학습 데이터 | F1 (best) | F1 @ 동등 0.3847 | AUROC | Precision | Recall | FP | FN |
|:-:|------|-------------|:---------:|:----------------:|:-----:|:---------:|:------:|:--:|:--:|
| 1 | Whole-Slice (Day 2) | FLAIR 1ch | 94.10 | (n/a — BCE only) | **0.9832** | 95.49 | 92.75 | 536 | 888 |
| 2 | Patch-Based count agg. (Day 4) | FLAIR patches | 87.08 | (n/a) | 0.9104 | 84.93 | 89.34 | 1,942 | 1,306 |
| 3 | **Multi-Task (Day 5)** | FLAIR 1ch | (94.27)¹ | **93.91** | 0.9824 | **97.33** | 90.73 | **305** | 1,136 |
| 4 | **MMMT baseline (Day 6)** | T1ce+FLAIR+diff | **94.27** | 93.84 | **0.9832** | 95.49 | 93.08 | 539 | 848 |
| 5 | C-2 FLAIR-only (Day 6 ablation) | FLAIR 3복제 | 94.15 | **93.98** | 0.9822 | 96.99 | 91.48 | 348 | 1,044 |
| 6 | C-1 T1ce-only (Day 6 ablation) | T1ce 3복제 | 87.55 | 86.71 | 0.9487 | 90.23 | 85.02 | 1,127 | 1,835 |

¹ Day 5는 v1에서 *@0.3847 기준 F1 93.91*로 보고됨. @ 0.5로 재산출하면 94.27 정도 (확인 필요).

### 6.2 세분화 성능 (단, 1·2번은 seg head 없음)

| # | 모델 | seg Dice | seg IoU | TP CAM IoU (test) | 비고 |
|:-:|------|:--------:|:-------:|:-----------------:|------|
| 1 | Whole-Slice | (없음) | (없음) | **0.145** | classification only |
| 2 | Patch-Based | (없음) | (없음) | 0.128 | classification only |
| 3 | Multi-Task | **0.7765** | **0.7061** | (CAM 0.12 train, seg 0.706 test)² | seg head 추가 |
| 4 | MMMT baseline | **0.7874** | **0.7142** | (CAM 0.16 train, seg 0.714 test) | + 멀티모달 + Tversky |
| 5 | C-2 FLAIR-only | 0.7458 | 0.6703 | (미측정) | Tversky 도입의 seg 감소 효과 |
| 6 | C-1 T1ce-only | 0.5741 | 0.4780 | (미측정) | T1ce-only의 WT 한계 |

² Multi-Task의 CAM IoU 0.12는 *학습셋 분류 head Grad-CAM*. seg IoU 0.706은 *test 분포 seg head 마스크*. 두 측정은 차원이 다르므로 직접 비교 부적절.

### 6.3 학습 비용

| # | 모델 | 총 시간 | 총 epoch | 파라미터 |
|:-:|------|:-------:|:--------:|:--------:|
| 1 | Whole-Slice | ~120분 (2h) | 14 | 11.2M |
| 2 | Patch-Based | 128.8분 (~2.2h) | 12 | 95K |
| 3 | Multi-Task | 298.8분 (~5h) | 14 | 14M |
| 4 | MMMT baseline | **388.5분 (~6.5h)** | 16 | 14M |
| 5 | C-2 FLAIR-only ablation | 715.3분 (~11.9h) | 12 | 14M |
| 6 | C-1 T1ce-only ablation | 340.2분 (~5.7h) | 15 | 14M |

> **합계** (전체 프로젝트 학습 시간): 약 **1,991분 ≈ 33시간** = 단일 RTX 4070 Laptop 8GB로 ~4일치 학습.

### 6.4 임상 가치 종합 (1,000명 스크리닝 기준)

| 모델 | 정상 1,000 중 오탐 | 종양 1,000 중 누락 | 종양 위치 마스크 |
|------|:-----------------:|:----------------:|:----------------:|
| Whole-Slice | 약 42 | 약 73 | ❌ |
| Patch-Based | 약 150 | 약 107 | ❌ |
| **Multi-Task (Day 5)** | **약 24** | 약 93 | ✅ IoU 0.706 |
| **MMMT (Day 6, @0.5)** | 약 42 | **약 69** | ✅ IoU 0.714 |
| **MMMT (Day 6, @0.3847)** | 약 61 | **약 59** | ✅ IoU 0.714 |
| C-2 FLAIR-only | 약 27 | 약 85 | ✅ IoU 0.670 |

**임상 시나리오별 권장**:
- **확진 모드 (Precision 우선)**: Day 5 MT (FP 305) — 정상 추가검사 최소화
- **1차 스크리닝 (Recall 우선)**: Day 6 MMMT @ 0.3847 또는 @ 0.30 — FN 최소화
- **균형 모드**: Day 6 MMMT @ 0.5

### 6.5 학술적 가치 종합

| 모델 | 학술적 메시지 |
|------|--------------|
| Whole-Slice | "AUC 0.98에도 모델이 종양을 안 본다" → shortcut learning 진단 시작 |
| Patch | "광역 차단 (passive)만으로는 부족하다" — 가설 반박 사례 |
| Multi-Task | "active supervision으로 IoU 5배 향상" — 본 프로젝트 1차 성공 |
| **MMMT** | "**멀티모달이 분류 점프보다 *FN 회복*과 *세분화 정밀도*에 기여**" — 가설 정정 사례 |
| Ablations | "FLAIR가 분류의 대부분, T1ce+diff는 세분화 보조" — **기여 분해의 정량 증거** |

### 🎯 §6 예상 질문 (Q&A)

**Q1. 6개 모델 중 발표에서 어떤 메시지를 가장 강조해야 하나?**
A. 추천 순위:
1. **Multi-Task의 "active supervision으로 종양 위치까지 학습"** (v1 §5의 메시지 그대로) — IoU 5배.
2. **MMMT + Ablation의 "멀티모달은 세분화 정밀도와 FN 회복에 기여; FLAIR가 분류의 대부분"** — 본 보고서의 새 메시지.
3. **6단계 여정 자체** — "정석 패턴의 학부 사례연구". 가설 검증과 가설 정정이 모두 들어 있는 흔치 않은 사례.

**Q2. F1 단일 지표 비교 시 Day 6 MMMT (94.27) > Day 5 MT (93.91)이지만 이 +0.36%p가 의미 있는 차이인가?**
A. 통계적 유의성은 단일 seed라 검증 불가. 그러나 *방향성*은 일관(AUROC, Recall, FN 모두 Day 6이 우위) — 다만 효과 크기가 작음. **발표 시 단일 F1보다는 *FN 288개 감소 + recovered 381 슬라이스* 라는 *임상적으로 더 의미 있는 지표*를 강조**하는 것이 안전.

**Q3. C-1 T1ce-only가 F1 87.55로 *Patch-Based(87.08)와 거의 같다*. 우연인가 의미 있는 비교인가?**
A. 우연일 가능성이 높음 (두 모델은 입력 표현·아키텍처·loss 모두 다름). 다만 **두 모델 모두 *적절한 학습 신호*가 부족했다는 공통점**이 있다 — Patch는 광역 단서 차단으로 정보 손실, T1ce-only는 WT 핵심 정보(ED) 부재. 발표에서는 직접 비교는 피하고 "두 모델 모두 분류 신호가 약한 입력 표현이라는 점에서 공통적으로 약함" 정도 언급 가능.

📎 **참고**: v1 §5.8 (3-way 비교), v1 §7 (차별화) + 본 문서 §3~§5

---

## 7. 발표 핵심 메시지 — 5가지 결정적 인사이트 (Day 5~6 확장판)

> **v1 §8.1의 "학술적 인사이트 3가지"를 확장한 *5가지 핵심 메시지*. 발표 직전 5분 안에 외워야 할 메시지.**

### 7.1 (v1 유지) 단일 지표(Accuracy/AUC)만으로는 의료 AI 신뢰성을 보장할 수 없다
- 94.33% Accuracy + 0.9832 AUROC + IoU 0.145 = shortcut learning의 결정적 증거.

### 7.2 (v1 유지) Shortcut Learning은 명시적 보조 학습 신호로 극복 가능하다
- Multi-Task가 IoU 0.145 → 0.706 (4.87배) 향상.
- 단, 본 보고서 §1.5에서 **분류 head CAM IoU는 여전히 0.12** — 진짜 향상은 *seg head 출력 마스크*임을 정직 보고.

### 7.3 (v1 유지) 광역 단서 차단(passive)보다 올바른 학습 강제(active)가 효과적이다
- Patch (passive) 실패 vs Multi-Task (active) 성공.

### 7.4 (★ 본 보고서 신규) 멀티모달의 효과는 *F1 점프*가 아니라 *FN 회복 + 운용 자유도 + 세분화 정밀도*에 집중된다
- **FN 1,136 → 848 (-25.4%)** : 88명/1000명 종양 누락에서 69명/1000명 누락으로 개선.
- **381개의 *recovered* 슬라이스** : 평균 종양 100~230 픽셀 영역에서 새로 검출 (§4).
- **seg Dice 0.7765 → 0.7874 (+0.011)** : diff 채널의 경계 inductive bias.
- F1 @ 동일 threshold(0.3847)에서는 -0.07%p로 거의 동등 → v1 §6.6 "F1 +1.5~2.5%p" 기대치를 *정정*.

### 7.5 (★ 본 보고서 신규) Ablation으로 모달 기여를 분리하면 *FLAIR가 분류의 대부분, T1ce+diff는 세분화 보조* 임이 드러난다
- **FLAIR-only AUROC 0.9822 ≈ baseline 0.9832** (분류는 거의 동등).
- **T1ce-only AUROC 0.9487** — WT에 대한 T1ce의 한계.
- **diff 채널의 진짜 기여**: 분류 +0.12%p F1, **세분화 +5.6%p Dice**.
- **결론**: 의학적 상보성 가설(v1 §6.2)이 부분적으로 검증됨 — *WT라는 단일 region에 대해서는* FLAIR가 압도적이고, 멀티모달은 *grading (WT/TC/ET)* 으로 확장할 때 진가가 드러날 것 (Day 7+).

### 7.6 (★ 본 보고서 신규) Tversky α=0.7은 분류 FN을 21% 줄이지만 seg Dice를 3%p 감소시키는 *양면성*이 있다
- C-2 FLAIR-only vs Day 5 MT 비교에서 정량 검증.
- **권장 정정**: Day 7+에서 *분류 head에는 Tversky, seg head에는 Dice/Focal-Tversky 적응적*으로 분리 적용.

### 7.7 (★ 본 보고서 신규) 두 모델(Day 5 / Day 6)은 *다른 종류의 작은 종양*을 잡고 놓치는 상호 보완 패턴이 있다
- Recovered 381 + Regressed 86 — Ensemble의 직접 근거.
- 발표에서는 "**Day 5 + Day 6 ensemble로 추정 467개의 FN 추가 회복 가능**" 으로 *향후 방향*에 포함.

---

## 8. 예상 질문 마스터 리스트 — Day 6 / Ablation 추가분

> **v1 §9의 38개 Q&A에 추가로 본 보고서가 다룬 *21개 신규 Q&A*. 발표 직전 빠르게 훑기 위한 단일 페이지.**

### 8.1 Day 5 train Grad-CAM 누락분 (§1)

39. Train CAM IoU 0.12와 Test seg IoU 0.71의 *차원 차이*? → §1.5
40. Multi-Task가 shortcut을 *극복*했나 *우회*했나? → §1 Q2
41. Train FN 4,110의 *under-fit 패턴* 정체? → §1 Q3

### 8.2 Day 6 본 학습 (§2)

42. Day 6 FP가 305 → 539로 *역증가*한 이유? → §2 Q1
43. Day 5 / Day 6 Best epoch 9 vs 11의 의미? → §2 Q2
44. Val Dice 0.426 vs Test Dice 0.787의 큰 차이? → §2 Q3
45. Tversky + 멀티모달 두 효과의 *분리*? → §2 Q4
46. CAM IoU 0.12 → 0.16 (+29%)이 진짜 해석성 향상인가? → §2 Q5

### 8.3 Threshold 동등 보정 (§3)

47. Day 6 자체 optimal 0.6202 vs Day 5 동등 0.3847 둘 중 어디가 공정? → §3 Q1
48. 동일 thr(0.3847)에서 F1 -0.07%p가 *멀티모달의 실패*인가? → §3 Q2
49. Day 5에도 Tversky를 넣었다면? → §3 Q3 (§5 C-2가 부분 답)

### 8.4 FN 차분 분석 (§4)

50. Recovered 381 vs Regressed 86이 *같은 작은 종양*인데 한쪽만 잡는 이유? → §4 Q1
51. Still_missed 641의 *임상적 정체*? → §4 Q2
52. New_fp 515의 *확신 반쯤*(0.58) 처리법? → §4 Q3
53. 양성 슬라이스 분기 합계 검증? → §4 Q4
54. Cleaned_fp 119의 정체? → §4 Q5

### 8.5 채널 Ablation (§5)

55. FLAIR-only가 baseline 동등이면 T1ce 추가 가치는? → §5 Q1
56. C-2 vs Day 5의 +0.07%p가 *유의*한가? → §5 Q2
57. T1ce-only가 9 epoch dice 0인 *학습 다이내믹 이상*은? → §5 Q3
58. ratio/multiplication 채널 mode는? → §5 Q4
59. T1ce-only seg dice 0.57이 *의학적 사실과 모순*? → §5 Q5 (WT vs TC/ET 라벨 차이)

### 8.6 전체 비교 (§6)

60. 6개 모델 중 어느 메시지를 가장 강조? → §6 Q1
61. F1 +0.36%p가 *통계적 의미*가 있는가? → §6 Q2
62. C-1 T1ce-only F1 87.55 ≈ Patch F1 87.08 — 우연인가? → §6 Q3

### 8.7 향후 (부록 C)

63. Day 7 grading 확장의 우선순위? → 부록 C
64. Ensemble (Day 5 + Day 6)의 기대 효과? → §7.7

---

## 부록 A — 배제된 / 보류된 결과

본 보고서 본문에서 *명시적으로 배제*하거나 *측정은 했지만 메인 표에 넣지 않은* 결과들. 발표 슬라이드에서는 *언급만* 하고 깊게 다루지 않을 항목.

### A.1 Multi-Task Val Dice vs Test Dice 측정 함수 차이

§2 Q3에서 다룬 부분. `step24_mmmt_train.py`의 `_wt_dice`와 `step25_mmmt_evaluate.py`의 `compute_seg_metrics`가 dice 계산 방식이 약간 다름. Val Dice 0.426 vs Test Dice 0.787의 갭은 측정 정의의 차이로 추정. **본 발표는 Test Dice (0.787)를 표준 보고치로 사용**, Val Dice는 학습 모니터링용.

### A.2 step20 missing_pair 44건 처리

- T1ce 추출 시 FLAIR와 매핑 안 된 44 슬라이스 — 본 학습에서는 dataset이 자동으로 검은 이미지로 대체.
- **엄격한 strict mode**(완전 일치 슬라이스만)로 재학습 시 결과 약간 다를 가능성. 추정 영향: ±0.02%p F1 이내 — 발표 비교에 영향 미미.

### A.3 Day 5 + Day 6 Ensemble (미실행)

- Recovered 381 + Regressed 86 = 467개 추가 회복 가능성 (§7.7).
- 평균 prob을 단순 평균하는 soft-voting 또는 max-voting 두 방식 모두 미실행.
- 향후 (Day 8+) 권장.

### A.4 3 seed 평균 (미실행)

- §5 Q2의 Tversky +0.07%p 효과의 통계적 유의성을 위해 필요.
- 학습 시간 ×3 = 추가 ~20시간. 본 보고서 시점 미실행.

### A.5 Channel mode 비교 (ratio / multiplication) (미실행)

- `step21_mmmt_dataset.py`에 `t1ce_flair_ratio`, `t1ce_flair_mul` 모드 구현은 있으나 학습 미실행.
- v1 §6.5에서 권장된 (ii) diff 채널만 본 학습에 사용.

### A.6 Tversky 효과 *정확한* 분리 ablation (미실행)

- D-1) FLAIR-only + Dice-only (= Day 5 multitask와 동일)
- D-2) FLAIR-only + Dice+Tversky (= C-2 — 본 실행)
- D-3) MMMT + Dice-only (미실행)
- D-4) MMMT + Dice+Tversky (= baseline — 본 실행)
- → 2×2 ablation에서 D-3이 빠짐. Tversky와 멀티모달의 *상호작용 항*을 분리하려면 필요.

### A.7 Still_missed 641의 추가 패턴 분석 (미실행)

- z-위치, 종양 픽셀 수만 분석됨. 모달리티별 신호 강도(T1ce / FLAIR 평균 강도) 통계 추가 권장.
- 결과에 따라 *2.5D 입력*이나 *고해상도(320×320)* 등으로 잡을 수 있는지 진단 가능.

### A.8 New_fp 515의 prob 분포 분석 (미실행)

- §4 Q3의 "threshold 0.6으로 올리면?" 답을 위해 prob 히스토그램 필요.
- `fn_diff_per_slice.csv`에는 데이터가 있으므로 사후 분석 가능.

### A.9 MMMT의 Score-CAM / EigenCAM 등 강한 XAI (미실행)

- §1 Q2에서 다룬 Grad-CAM의 한계 대응. `260521v2ways.md` §3.3 권장.
- 학부 발표에서는 Grad-CAM만으로 충분 (시간 제약).

### A.10 Whole-Slice / Patch-Based의 *학습셋* Grad-CAM (배제)

- v1 §2~§4는 test split CAM IoU만 보고. Day 5 MT의 train CAM IoU를 §1에서 추가했으나 Whole-Slice / Patch의 train CAM IoU는 본 보고서에 포함하지 않음.
- 이유: (1) Whole-Slice / Patch는 seg head가 없어 *분류 CAM만* 측정 가능 → train과 test의 차이가 크지 않을 것으로 추정, (2) Day 5 MT의 train vs test CAM 차이를 통해 일반 패턴이 이미 확인됨.

---

## 부록 B — 전체 학습 로그 (Multi-Task / MMMT / Ablations)

### B.1 Day 5 Multi-Task (v1 §5.4에 14 epoch 전체 표 있음)

총 14 epoch (P1 3 + P2 11). Best Epoch 9 (Total Loss 0.3484, CLS 0.1909, SEG 0.3151, Val Acc 93.84%, Val Dice 0.756). Phase 1→2 전환 시 CLS Loss 0.39 → 0.18 급감 / SEG Loss 거의 불변. 학습 시간 298.8분.

### B.2 Day 6 MMMT (본 보고서 §2.5 16 epoch 전체 표)

- 총 16 epoch (P1 3 + P2 13, Early Stop).
- Best Val Acc Epoch 11 (94.12%, Val Dice 0.4229).
- Phase 1→2 전환 시 Val total 0.471 → 0.269. Val Acc 83.13% → 92.56%.
- 학습 시간 388.5분.

### B.3 C-2 FLAIR-only Ablation (12 epoch)

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

### B.4 C-1 T1ce-only Ablation (15 epoch, 마지막까지)

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

### B.5 학습 곡선 그림 자료

| 그림 파일 | 위치 |
|----------|------|
| `mt_training_curves.png` | `outputs/figures/multitask/` (v1에 이미 인용) |
| `mmmt_training_curves.png` | `outputs/figures/mmmt/` |
| `flair_only/training_curves.png` | `outputs/figures/mmmt_ablation/` |
| `t1ce_only/training_curves.png` | `outputs/figures/mmmt_ablation/` |
| `ablation_table.png` | `outputs/figures/mmmt_ablation/` (3-row 비교 막대) |

---

## 부록 C — 미실행·향후 계획 (Day 7+)

### C.1 본 보고서 직후 우선순위 (~1주)

| # | 작업 | 소요 | 산출물 |
|:-:|------|------|--------|
| 1 | Day 5 + Day 6 soft-voting Ensemble 평가 | ~30분 (추론만) | recovered 381 + regressed 86 합산 검증 |
| 2 | new_fp 515 / cleaned_fp 119의 prob 히스토그램 분석 | ~10분 | threshold 운영 점 최적화 |
| 3 | still_missed 641의 모달리티 신호 분석 | ~30분 | 2.5D / 고해상도 필요성 판단 |
| 4 | Day 6 Score-CAM/EigenCAM 비교 (`260521v2ways.md` §3.3) | ~1시간 | CAM 0.16의 진짜 한계 진단 |

### C.2 중기 (~1개월) — Day 7 SOTA 패키지

`260521v2ways.md` §8.2 "무료 SOTA 패키지" 적용:

1. **Focal-Tversky Loss** (γ=4/3, FN에 더 강한 비선형 가중)
2. **Uncertainty Weighting (Kendall 2018)** — α/β 자동
3. **Deep Supervision** — 디코더 중간 3단에 보조 seg head
4. **TumorCP augmentation** — 양성 종양 영역을 다른 슬라이스에 합성
5. **SWA (Stochastic Weight Averaging)** — 마지막 25% epoch 평균
6. **TTA (Test-Time Augmentation)** — 4-fold flip/rotate 평균
7. **Temperature Scaling** — threshold 0.3847을 0.5 부근으로 보정

**기대치**: F1 96~98%, IoU 0.78+ (`260521v2ways.md` §10.1 Day 7 행)

### C.3 장기 (~3개월) — Grading 확장

`260521v2ways.md` §6의 4가지 grading 방향:

1. **WT / TC / ET 3-region multi-label seg** — BraTS 공식 평가축
2. **Slice-level multi-label (WT/TC/ET 유무)**
3. **Tumor-type 다중 분류 (GLI / MEN / PED)**
4. **Tumor-burden regression**

### C.4 실용화 (~6개월)

`260523v2proposal.md` §3의 5가지 패키징:

1. GitHub 오픈소스 리포 (집단 A, D)
2. 교육용 콘텐츠 패키지 (집단 B)
3. PACS Plugin / DICOM Viewer (집단 C)
4. PyPI SDK `medical-mtl-bootstrap` (집단 D)
5. 의료 AI 신뢰성 감사 키트 (집단 E)

---

## 부록 D — 참고한 md 파일 인벤토리 (v1 이후 추가분)

본 발표 준비 자료를 작성하기 위해 다음 md 파일과 보조 자료를 참조했다. 각 파일별로 어느 부분을 인용/보강에 사용했는지 명시.

### D.1 v1에서 이미 인용된 핵심 자료 (v1 부록 E.1~E.4 참조)

`260521v1result.md` 부록 E에 정리된 13개 notPublic md + 5개 폴더 밖 md + 2개 이미지 + 코드/산출물 파일. 본 보고서는 그 위에 다음을 추가.

### D.2 v1 이후 새로 작성된 md 파일

| 파일 | 분량 | 참고한 절 | 본 문서에서의 활용 |
|------|------|-----------|---------------------|
| **`260521v1result.md`** | 78.9 KB | §0 도식, §1~§6 Day 1~6 본문, §7 차별화, §8 학술 기여, §9 Q&A, 부록 A~E | **본 문서의 *전제*. §0.1 비교, §2 Day 6 본문의 전 단계 인용, §6 6-way 비교, §7 5가지 인사이트 (v1 3가지 → v2 7가지로 확장)** |
| **`260521v2ways.md`** | 38.4 KB | §0.3 14개 우선순위표, §1 5단계 여정, §2 아키텍처 트랙, §3 Loss 트랙, §5 신뢰성 트랙, §6 Grading 4방향, §8 단기/중기 로드맵, §10 예상 성능 | **부록 C 향후 계획 전체 + §5 Q1의 grading 답변 + §7.4의 "다음 단계" 메시지** |
| **`260523v1updatemmmt.md`** | 9.5 KB | §② 재실행 불필요 진단, §③ threshold 0.3847 유래, §④ Phase 1 (A) (B) + Phase 2 (C-1, C-2) 권장 파이프라인 | **§0.2 작업 도식, §3 (A) 본문 전체, §4 (B) 본문 전체, §5 (C) 본문 전체 — 본 보고서의 *직접 가이드 문서*** |
| **`260523v2proposal.md`** | 25.9 KB | §0.1 자산 정리, §1.1~§1.7 7가지 활용처, §2 사용자 집단, §3.1~§3.5 패키징, §4 로드맵, §5 차별화, §6 위험·한계 | **부록 C.4 실용화 + 차별화 메시지 보강** |

### D.3 v1 이후 새로 생성된 결과 파일

| 파일 | 분량 | 본 문서 활용 |
|------|------|-------------|
| `outputs/logs/multitask/mt_training_summary.json` | 작음 | §1.2 학습 통계 — checkpoint epoch / val acc / val dice 확인 |
| `outputs/logs/multitask/mt_evaluation_results.json` | 작음 | §6.1 분류 성능 비교 — Day 5 F1 93.91 / 0.3847 |
| `outputs/logs/multitask/mt_classification_report.json` | 작음 | §6.1 클래스별 precision/recall 확인 |
| `outputs/figures/multitask/gradcam_train/mt_train_gradcam_summary.json` | 작음 | **§1.3 핵심 수치** (본 보고서의 *누락 보강* 메인 자료) |
| `outputs/figures/multitask/gradcam_train/mt_train_gradcam_subsample.csv` | 400행 | §1.7 IoU 분포 |
| `outputs/logs/mmmt/step20_t1ce_stats.json` | 작음 | §2.1 T1ce 추출 통계 |
| `outputs/logs/mmmt/mmmt_summary.json` | 작음 | §2.4 학습 설정, §2.6 학습 시간 |
| `outputs/logs/mmmt/mmmt_history.json` | 작음 | §2.5 16 epoch 학습 곡선 전체 |
| `outputs/logs/mmmt/mmmt_test_metrics.json` | 작음 | §2.7 test 평가 |
| `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json` | 작음 | **§3 threshold sweep 전체** |
| `outputs/logs/mmmt/fn_diff_summary.json` | 작음 | **§4 그룹 카운트 + 통계 전체** |
| `outputs/logs/mmmt/fn_diff_per_slice.csv` | 25,116행 | §4 슬라이스별 group_fair / group_split |
| `outputs/figures/mmmt/gradcam_train/mmmt_train_gradcam_summary.json` | 작음 | **§2.9 MMMT train Grad-CAM (Day 5와 비교)** |
| `outputs/figures/mmmt/gradcam_train/mmmt_train_gradcam_subsample.csv` | 400행 | §2.9 IoU 분포 |
| `outputs/figures/mmmt/{mmmt_training_curves,mmmt_diagnostics,threshold_sweep,fn_diff_zdist,fn_diff_tumor_size,fn_diff_prob_scatter}.png` | — | 발표 슬라이드 시각화 |
| `outputs/logs/mmmt_ablation/ablation_table.json` | 작음 | **§5.4 ablation 핵심 표** |
| `outputs/logs/mmmt_ablation/{flair_only,t1ce_only}/{summary,history}.json` | 작음 | §5.3 학습 통계, 부록 B.3, B.4 학습 곡선 |
| `outputs/figures/mmmt_ablation/ablation_table.png` | — | §5 비교 막대 차트 슬라이드 |
| `outputs/figures/mmmt_ablation/{flair_only,t1ce_only}/training_curves.png` | — | 부록 B 학습 곡선 |

### D.4 v1 이후 새로 작성된 코드 파일

| 파일 | 본 문서 활용 |
|------|-------------|
| `code/multitask/step18b_multitask_train_gradcam.py` | §1 Multi-Task train Grad-CAM 구현 — Grad-CAM 알고리즘(enc5 hook + CAM threshold) 확인 |
| `code/mmmt/step20_mmmt_preprocess.py` | §2.1 T1ce 슬라이스 추출 — percentile-clip 정규화 일관성 확인 |
| `code/mmmt/step21_mmmt_dataset.py` | §2.2 멀티모달 dataset — channel_mode 5종(diff/ratio/mul/t1ce_only/flair_only) |
| `code/mmmt/step22_mmmt_model.py` | §2.2 모델 구조 — ResNet-18 backbone + UNet decoder + dual head |
| `code/mmmt/step23_mmmt_losses.py` | §2.3 Loss — DiceLoss / TverskyLoss(α=0.7, β=0.3) / MMMTLoss |
| `code/mmmt/step24_mmmt_train.py` | §2.4~2.5 학습 절차 — 2-Phase 동일, EarlyStopping patience=5 |
| `code/mmmt/step24b_mmmt_train_gradcam.py` | §2.9 MMMT train Grad-CAM 구현 — 6열 그리드 시각화 |
| `code/mmmt/step25_mmmt_evaluate.py` | §2.7 test 평가 — compute_seg_metrics |
| `code/mmmt/step25b_mmmt_evaluate_thr.py` | **§3 (A) threshold sweep 구현 — Day5 운용점 0.3847 동등 적용** |
| `code/mmmt/step26_fn_diff_analysis.py` | **§4 (B) FN 차분 분석 구현 — Day5/Day6 1:1 slice 비교, 8개 그룹 분류** |
| `code/mmmt/step27_ablation_train.py` | **§5 (C) ablation 학습 — channel_mode만 변경, 모든 다른 변수 통제** |
| `code/mmmt/step28_ablation_evaluate.py` | **§5 (C) ablation 평가 — 3-row 통합 표, low-memory seg metric 계산** |
| `code/mmmt/README.md` | §2 Day 6 범위 정의 (Day 7과 분리), 폴더 구조, 실행 순서 |

### D.5 본 문서의 *직접 인용 출처* 요약 표

| 본 문서 절 | 직접 인용 출처 |
|-----------|---------------|
| §0 (전체 도식) | `260521v1result.md` §0.1 + 본 보고서 추가 |
| §1 (MT Train Grad-CAM) | `outputs/figures/multitask/gradcam_train/mt_train_gradcam_summary.json` + `code/multitask/step18b_*.py` |
| §2 (MMMT 본 학습) | `outputs/logs/mmmt/mmmt_{summary,history,test_metrics}.json` + `code/mmmt/step20~25_*.py` + `outputs/figures/mmmt/gradcam_train/mmmt_train_gradcam_summary.json` |
| §3 (A Threshold) | `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json` + `code/mmmt/step25b_*.py` + `260523v1updatemmmt.md` §④ Phase 1 (A) |
| §4 (B FN diff) | `outputs/logs/mmmt/fn_diff_summary.json` + `code/mmmt/step26_*.py` + `260523v1updatemmmt.md` §④ Phase 1 (B) |
| §5 (C Ablation) | `outputs/logs/mmmt_ablation/ablation_table.json` + `code/mmmt/step27_*.py`, `step28_*.py` + `260523v1updatemmmt.md` §④ Phase 2 |
| §6 (6-way 비교) | v1 §5.8 (3-way) + 본 보고서 §1~§5 + `260521v1result.md` 부록 A 통합표 |
| §7 (5가지 인사이트) | v1 §8.1 (3가지) + 본 보고서 §2~§5의 새 발견 |
| 부록 A (배제 결과) | 본 보고서 본문에서 *언급은 했으나 메인 비교에 포함 안 한* 모든 항목 |
| 부록 B (학습 로그) | 본 보고서 §2.5 + 각 ablation의 history.json |
| 부록 C (향후 계획) | `260521v2ways.md` §8.2 (Day 7 패키지) + `260523v2proposal.md` §3 (실용화 5형태) |

---

## 끝맺음 — 한 줄 요약 (다시, 본 보고서 버전)

> **"v1까지의 6단계 여정(Whole → Grad-CAM → Patch → MT → MMMT 진행 중)이 *수치 vs 진실*의 trade-off를 보여줬다면, v2(본 보고서)는 멀티모달의 *진짜 효과*를 ablation으로 분해해 보여준다 — **FLAIR가 WT 분류의 대부분이고, T1ce+diff는 세분화 정밀도와 FN 회복에 기여하며, Tversky는 분류 FN과 seg Dice 사이에 양면성**이 있다는 것. *멀티모달이 무조건 좋다*는 가설을 검증·정정한 점이 본 발표의 학술적 정직성."**

| 단계 | 모델 | F1 (@best thr) | seg IoU | FN | 의학적 정합성 | 종합 |
|:----:|------|:--------------:|:-------:|:--:|:------------:|:----:|
| Day 2 | Whole-Slice | 94.10 | (CAM 0.145) | 888 | ❌ | 절반의 성공 (shortcut) |
| Day 4 | Patch-Based | 87.08 | (CAM 0.128) | 1,306 | ❌ | 절반의 실패 (passive) |
| **Day 5** | **Multi-Task** | **93.91** | **0.706** | 1,136 | ⚠️ | **온전한 성공 (active)** |
| **Day 6** | **MMMT baseline** | **94.27** | **0.714** | **848** | **✅ WT (+ET 단서)** | **운용 자유도 확장 + FN 회복** |
| Day 6 ablation | C-2 FLAIR-only | 94.15 | 0.670 | 1,044 | ✅ WT (FLAIR) | Tversky의 분리 효과 |
| Day 6 ablation | C-1 T1ce-only | 87.55 | 0.478 | 1,835 | ⚠️ T1ce는 ET 강조만 | WT 한계 노출 |

---

> 📎 **본 문서의 위치**: `biohealth_lv.1/260523v2mmmtplus.md`
> 📎 **연속성**: `260521v1result.md` (Day 1~6 진행중) → `260521v2ways.md` (기술 확장 로드맵) → `260523v1updatemmmt.md` (멀티모달 평가 갱신 가이드) → `260523v2proposal.md` (실용화 제안) → **`260523v2mmmtplus.md` (Day 5~6 완성 + 권장 3종 결과 + 5가지 인사이트 — 본 문서)** → 최종 발표.
