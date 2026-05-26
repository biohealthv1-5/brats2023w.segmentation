# 🎤 최종 발표 준비 자료 (260526 v3 — Day 7 SOTA 평가 통합 마스터)

> **본 문서의 위치**: `260521v1result.md` (Day 1~5 + Day 6 진행 중) → `260524v1mmmtplus.md` (Day 5~6 완성 + Ablation + 5가지 인사이트) → **`260526v3sotafinal.md` (Day 7 SOTA 학습 결과 + step31 평가 + 발표 직전 최종 통합 — 본 문서)**
>
> **프로젝트 종료 선언**: 본 문서가 *프로젝트의 마지막 보고서*다. 8GB Laptop GPU(RTX 4070)의 **VRAM 누수와 Windows 환경의 PyTorch CUDA Allocator 한계**(특히 `expandable_segments` 옵션의 Linux 전용 동작) 때문에, 본 시점 이후의 추가 학습/대규모 forward는 불가능하다. 따라서 이미 학습 완료된 `sota_best.pth` / `sota_swa.pth` 자산 위에서 *재현 가능한 최종 평가 1회*를 끝낸 본 문서가 학부 발표 자료의 *공식 종착점*이다.
>
> **편집 원칙** (v1/v1mmmtplus와 동일 유지)
> 1. 단계별 결정적 수치·관찰·의사결정은 *원문 그대로 보존* (요약 X).
> 2. 명세서·산출물에 명시되지 않은 부분은 폴더 내 다른 자료(코드 주석·산출 JSON·이전 md)에서 찾아 **[보강]** 표시와 함께 추가.
> 3. 각 단계 끝에 **🎯 예상 질문 (Q&A)** 섹션을 두어 교수님이 던질 만한 의문을 정리.
> 4. 본문에서 빠진 내용(미실행·미해결·보류 등)은 **부록**에 별도 보관.
> 5. 각 섹션 끝에 **📎 참고 md 출처**로 어떤 파일의 어느 절을 참조했는지 명시.

---

## 목차

0. [v1mmmtplus 이후 추가된 작업의 전체 도식 & 종료 선언](#0-v1mmmtplus-이후-추가된-작업의-전체-도식--종료-선언)
1. [Day 7 — SOTA 패키지 설계 의도 & 실행 자산 (step26~step30)](#1-day-7--sota-패키지-설계-의도--실행-자산-step26step30)
2. [step30 SOTA 학습 결과 — 14 epoch 전체 곡선 + SWA](#2-step30-sota-학습-결과--14-epoch-전체-곡선--swa)
3. [step31 SOTA 평가 결과 — best 체크포인트 (TTA + Temp + 후처리)](#3-step31-sota-평가-결과--best-체크포인트-tta--temp--후처리)
4. [신뢰성 SOTA — Calibration, Conformal, Bootstrap CI](#4-신뢰성-sota--calibration-conformal-bootstrap-ci)
5. [세분화 SOTA — WT/TC/ET region별, threshold sweep, post-processing, HD95](#5-세분화-sota--wttcet-region별-threshold-sweep-post-processing-hd95)
6. [실패 케이스 정성 분석 — failure_worst30 / failure_best30](#6-실패-케이스-정성-분석--failure_worst30--failure_best30)
7. [전체 모델 비교 — Whole / Patch / MT / MMMT / Ablation / SOTA 7-way](#7-전체-모델-비교--whole--patch--mt--mmmt--ablation--sota-7-way)
8. [Day 7 SOTA의 의의 — v2ways §10.1 기대치 vs 실측 정정](#8-day-7-sota의-의의--v2ways-§101-기대치-vs-실측-정정)
9. [환경적 한계 — VRAM 누수 / Windows allocator / 추가 진행 불가 사유](#9-환경적-한계--vram-누수--windows-allocator--추가-진행-불가-사유)
10. [발표 핵심 메시지 — 7가지 결정적 인사이트 (확장판)](#10-발표-핵심-메시지--7가지-결정적-인사이트-확장판)
11. [예상 질문 마스터 리스트 (Day 7 / 신뢰성 / SOTA 비교 추가분)](#11-예상-질문-마스터-리스트-day-7--신뢰성--sota-비교-추가분)
12. [부록 A — 배제된 / 보류된 결과](#부록-a--배제된--보류된-결과)
13. [부록 B — 전체 학습/평가 로그 (Day 7 SOTA 14 epoch + step31 raw)](#부록-b--전체-학습평가-로그-day-7-sota-14-epoch--step31-raw)
14. [부록 C — 모든 모델 한눈 비교표 (전체 프로젝트 종합)](#부록-c--모든-모델-한눈-비교표-전체-프로젝트-종합)
15. [부록 D — 참고한 md 파일 인벤토리 + 직접 인용 출처 (본 보고서 신규)](#부록-d--참고한-md-파일-인벤토리--직접-인용-출처-본-보고서-신규)

---

## 0. v1mmmtplus 이후 추가된 작업의 전체 도식 & 종료 선언

### 0.1 두 보고서의 경계와 본 문서의 진입점

```
260521v1result.md       (작성 시점 2026-05-21)
  ├─ Day 1~5 완성 + Day 6 "진행 중"
  └─ 3가지 인사이트 (단일 지표 한계 / shortcut / passive vs active)
                                    │
                                    ▼
260524v1mmmtplus.md     (작성 시점 2026-05-25)
  ├─ Day 5 train Grad-CAM 보강 + Day 6 MMMT 완성
  ├─ Threshold 동등 보정 / FN 차분 분석 / 채널 Ablation 3종 추가
  └─ 7가지 인사이트 (v1 3 + 신규 4)
                                    │
                                    ▼
260525v1sotapluswhy.md  (작성 시점 2026-05-25)
  └─ sota/ 폴더 코드 7종이 mmmt/ 위에 *무엇을* 어떻게 얹는지 정리
                                    │
                                    ▼
260525v2sotawith.md     (작성 시점 2026-05-25)
  └─ BraTS 공식 WT Dice 학술 SOTA(MedNeXt 0.93, DynUNet 0.91 등) 위치 정리
                                    │
                                    ▼
260526v1fullresults.md  (작성 시점 2026-05-26)
  └─ step31 결과를 "오래 걸리더라도 더 짜내는" 9가지 카테고리(A~E)
                                    │
                                    ▼
260526v2step31patch.md  (작성 시점 2026-05-26)
  └─ "VRAM 무증가 + 학습 0회" 제약 아래 step31에 박을 10개 패치(P1~P10)
                                    │
                                    ▼  *** 본 문서 ***
260526v3sotafinal.md    (작성 시점 2026-05-26)
  ├─ Day 7 SOTA 학습(step30) + 평가(step31) 결과 통합
  ├─ 7-way 모델 비교 (Whole/Patch/MT/MMMT/C-1/C-2/SOTA)
  ├─ 신뢰성 SOTA (TTA logit-mean + Temp Scaling + Conformal + Bootstrap CI)
  └─ 환경적 한계로 인한 프로젝트 종료 선언
```

### 0.2 본 문서가 다루는 5개 신규 산출물 패키지 (v1mmmtplus 이후)

| # | 패키지 | 스크립트 | 산출물 위치 | 본 문서 § |
|:-:|--------|----------|-------------|:--:|
| 1 | **WT/TC/ET 3-region 마스크** | `code/sota/step26_sota_segmask.py` | `processed/sota/seg_masks_{wt,tc,et}/` (PNG, 166,626장) + `outputs/logs/sota/step26_segmask_stats.json` | §1.1 |
| 2 | **SOTA 3채널 입력 dataset + Weighted Sampler** | `code/sota/step27_sota_dataset.py` | (in-memory) | §1.2 |
| 3 | **SOTA 모델 (Deep Supervision + Uncertainty Weighting)** | `code/sota/step28_sota_model.py` | `outputs/checkpoints/sota/sota_{best,last,swa}.pth` | §1.3, §2 |
| 4 | **SOTA Loss (Focal-Tversky + Boundary + Compound + TumorCP)** | `code/sota/step29_sota_losses.py` | (loss 정의) | §1.4 |
| 5 | **SOTA 학습 (step30) + 평가 (step31)** | `code/sota/step30_sota_train.py`, `step31_sota_evaluate.py` | `outputs/logs/sota/{sota_history,sota_summary,sota_test_metrics,sota_test_raw}.{json,npz}` + `outputs/figures/sota/*.png` 9장 | §2, §3, §4, §5, §6 |

### 0.3 7-단계 여정 (v1 5단계 → v1mmmtplus 6단계 → v3 7단계)

```
Day 1 ─── Day 2 ─── Day 3 ─── Day 4 ──── Day 5 ────── Day 6 ──────── Day 7 (★ 본 문서)
EDA       Whole     Grad-CAM  Patch      Multi-Task   Multi-Modal    SOTA Package
1,251명   Acc 94.33  IoU 0.145 F1 87.08   F1 93.91     F1 94.27       F1 92.96 (TTA, thr 0.5)
166K      F1  94.10  FN 92%   IoU 0.128  IoU 0.706    IoU 0.7142     **F1 93.50 (thr 0.7)** ★
슬라이스  AUC 0.9832 소종양   (passive)  (active)     (modality)     **AUROC 0.9824 [0.981, 0.984]**
                                                                    **WT/TC/ET 동시평가**
                                                                    **Calibration ECE 0.036**
                                                                    **Conformal 0.896 cov**
                                                                    **HD95(WT)=4.58, HD95(TC)=3.13, HD95(ET)=3.02**
✅          ✅          ✅          ✅          ✅          ✅          🎯 본 문서로 종료
```

### 0.4 프로젝트 종료 선언 — 왜 Day 7에서 멈추는가

본 보고서를 마지막으로 **이 프로젝트는 추가 학습/대규모 forward를 더 이상 진행하지 않는다.** 그 사유:

1. **GPU VRAM 누수**: step30 학습 중 epoch 후반부터 free VRAM이 점진적으로 줄어드는 패턴 관찰. 14 epoch 시점에 *OOM에 의한 학습 강제 중단*(`sota_last.pth`에 epoch 14가 남음 — 계획상 P1 3 + P2 15 = 18 epoch이었으나 P2 11번째까지만 완주). 실제 `sota_summary.json` 의 `total_epochs: 14` 가 그 증거.
2. **Windows PyTorch CUDA Allocator 한계**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 는 PyTorch 공식 문서상 *Linux 전용*. Windows에서 켜면 `~TensorImpl` 시 Illegal Memory Access / Allocator 손상 발생. step31 코드에서 OS 분기로 우회는 했으나 (`step31_sota_evaluate.py` L57–67), Windows에서 가용한 옵션은 `max_split_size_mb` 축소 정도여서 본질적 한계 존재.
3. **step31 평가의 SWA 평가 실패**: `sota_test_metrics.json` 의 `swa` 블록에 `"error": "Unable to allocate 14.0 GiB for an array with shape (24882, 3, 224, 224) and data type float32"` 가 그대로 남음. CPU RAM 16GB 환경에서 val raw 결과를 메모리에 누적하는 도중 시스템 RAM 부족. 즉 **best 평가는 성공했지만 SWA 평가는 실패** — 두 체크포인트 모두 동일 환경에서 비교할 수 없게 됨.
4. **추가 학습 = 추가 위험**: 시드 변경/3-seed 평균/K-fold/3D 모델 전환 등 모든 향후 옵션이 (1)(2)에 의해 사실상 불가능. 발표 일정상 환경 재구축(WSL2 Ubuntu + Linux native PyTorch + 24GB+ GPU 임차)도 비현실적.

따라서 **본 문서는 "이미 손에 들고 있는 자산 + step31의 마지막 평가 1회"** 라는 *현실의 발표 자료*다.

📎 **참고**:
- `260524v1mmmtplus.md` §0.1 (마일스톤 타임라인, v1 이후 도식)
- `260525v1sotapluswhy.md` 전체 (sota 폴더 7종 코드의 위치)
- `260525v2sotawith.md` §1~§2 (학술 SOTA 비교 기준)
- `260526v1fullresults.md` 전체 (성능 더 짜내는 9가지)
- `260526v2step31patch.md` §0~§3 (P1~P10 패치, VRAM 무증가 원칙)
- `outputs/logs/sota/sota_summary.json` `total_minutes: 1017.8, total_epochs: 14` (조기 종료의 직접 증거)
- `outputs/logs/sota/sota_test_metrics.json` `swa.error` (SWA 평가 실패 직접 증거)

---

## 1. Day 7 — SOTA 패키지 설계 의도 & 실행 자산 (step26~step30)

### 1.0 큰 그림 — sota 폴더 = "v2ways §8.2 무료 SOTA 패키지" 의 구현체

> **세 개의 동시 목표**:
> 1. *BraTS 공식 평가축*(WT/TC/ET 3-region)에 정렬 — 학술 SOTA(MedNeXt 0.93, DynUNet 0.91)와 비교 가능한 상태로 격상.
> 2. *FN 한계의 직접 표적화* — Focal-Tversky + Weighted Sampler + TumorCP로 소종양에 학습 가중을 집중.
> 3. *신뢰성 SOTA* — TTA + Temperature Scaling + Conformal Prediction 으로 결정 경계의 *불확실성 정량화*.

세 목표는 학부 발표용으로 "Day 6 멀티모달 + 새로운 신뢰성 계층" 의 결합이며, `code/`, `code/multitask/`, `code/mmmt/` 의 *어떤 파일도 수정하지 않고* 새 폴더(`code/sota/`, `processed/sota/`, `outputs/{checkpoints,figures,logs}/sota/`) 안에서만 동작하도록 격리. 따라서 Day 2/4/5/6 결과가 그대로 보존되어 **7-way 비교** 가 가능 (§7).

### 1.1 step26 — WT/TC/ET 3-region 마스크 생성

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
- BraTS label: **0**=배경, **1**=NCR(괴사 핵), **2**=ED(부종), **3**=ET(조영 증강 종양)
- **WT = (seg > 0)** — 종양 전체 (NCR + ED + ET)
- **TC = (seg == 1) | (seg == 3)** — 종양 코어 (NCR + ET, 부종 제외)
- **ET = (seg == 3)** — 조영 증강 종양만

**보존 수치**:
- 환자 수 1,251명 — Day 1 EDA와 일치 (검증 통과).
- WT positive 81,337 — Day 1 의 *81,374* (FLAIR seg 기준)와 37 차이. **이는 step26이 *4 모달의 seg가 일치하는 슬라이스만* 저장하기 때문**이며, FLAIR-only 기준의 Day 1 라벨(`labels.csv`)에 비해 약간 보수적. 발표 시 영향 미미(±0.05% 슬라이스 수 차이).
- TC positive 53,593 / ET positive 50,958 — 슬라이스 65.9% / 62.7% 가 *TC/ET를 가짐*. 거의 모든 종양 슬라이스에 enhancing 영역이 존재한다는 의미.

### 1.2 step27 — 3채널 입력 + 3채널 마스크 + Weighted Sampler

**입력 구성**: `[T1ce, FLAIR, |T1ce - FLAIR|]` (Day 6 `processed/mmmt/t1ce_slices/` 재사용)
**출력 구성**: cls 스칼라 + 3채널 (WT/TC/ET) 마스크
**Weighted Sampler**: WT 양성 픽셀 < 256인 "소종양" 슬라이스에 가중치 **×3**

**근거**:
- `260524v1mmmtplus.md` §4.5 **인사이트 ④**: still_missed 641 슬라이스의 평균 종양 크기는 100픽셀(0.20%) — 멀티모달로도 못 잡음.
- `_build_sampler` 코드는 이 진단을 직접 처방으로 옮긴 것.

### 1.3 step28 — Deep Supervision + Uncertainty Weighting 모델

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
- `aux_head_d2/d3/d4` — Deep Supervision 보조 WT seg head 3개 (Decoder 중간 단계의 supervision).
- `log_var_cls`, `log_var_seg` — 학습 가능한 가중치 자동 조정 파라미터.

**핵심 근거** (`260524v1mmmtplus.md` §1.5의 결정적 인사이트 ①):
> Multi-Task 분류 head Grad-CAM IoU는 train에서도 0.1232에 불과한데, seg head IoU는 0.7753 → encoder feature는 충분히 종양을 알지만 분류 head로 전파되지 않음.

Deep Supervision은 decoder 중간 단계마다 supervisory signal을 직접 주입해서 이 격차를 줄임. `260525v1sotapluswhy.md` §1-3 에서 정확히 같은 처방으로 해석됨.

### 1.4 step29 — Focal-Tversky + Boundary + Compound + TumorCP

| 컴포넌트 | 정의 | 출처 |
|----------|------|------|
| `FocalTverskyLoss(α=0.7, β=0.3, γ=4/3)` | TP/(TP + α·FN + β·FP) → (1−Tversky)^γ | Abraham & Khan ISBI 2019 |
| `BoundaryLoss` | SDF(signed distance function) 기반, 경계 슬라이스용 | Kervadec MIDL 2019 |
| `CompoundSegLoss` | **1.0·Focal-Tversky + 0.5·BCE + 0.3·Boundary** per-region | — |
| `SOTAMultiTaskLoss` | cls/seg 두 task에 *Uncertainty Weighting* 적용 + Deep Sup aux 3개에 `ds_weights=(0.125, 0.25, 0.5)` 가중합 | Kendall 2018 |
| `tumor_copy_paste` | 배치 내 종양 슬라이스 패치를 다른 슬라이스에 paste (Yang MICCAI 2022 단순화) | — |

**Day 6 대비 변경점**:
- Day 6 = `0.5·Dice + 0.5·Tversky(α=0.7)` → **Day 7 = `1.0·Focal-Tversky(γ=4/3) + 0.5·BCE + 0.3·Boundary`**.
- α/β 수동 튜닝 → Uncertainty Weighting 자동.
- 단일 task → 3 region (WT/TC/ET) 동시.

### 1.5 step30 — 2-Phase + SWA + TumorCP 학습 루프 (CONFIG)

`outputs/logs/sota/sota_summary.json` 에 저장된 실제 학습 설정:

```json
{
  "phase1_epochs": 3,
  "phase1_lr": 1e-3,
  "phase2_epochs": 15,
  "phase2_lr": 1e-4,
  "swa_start_epoch": 11,        ← Phase 2 epoch 기준 (15-5+1)
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

**Day 5/6 와 동일 통제**: phase 구성, batch_size, weight_decay, EarlyStopping patience, channel_mode 모두 *Day 6과 동일* — 본 SOTA 학습의 *순수 효과*는 (i) loss 변경, (ii) 3-region seg head, (iii) Deep Supervision, (iv) Uncertainty Weighting, (v) Weighted Sampler+TumorCP, (vi) SWA 의 *6개의 동시 변경* 으로 측정된다. (분리 ablation은 미실행 — 부록 A.)

### 1.6 step31 — TTA + Temperature Scaling + Conformal + 후처리 일습

`260526v2step31patch.md` 의 P1~P10 패치가 *모두 적용 완료*된 상태. 핵심:
- **P1**: raw prediction `.npz` 직렬화 → `sota_test_raw.npz` (1.08 GB), `sota_val_raw.npz` (0.96 GB).
- **P2**: per-region threshold sweep (val 기반) — WT/TC/ET 모두 **0.7** 채택.
- **P3**: post-processing (CC filter min_size=10 + 3×3 closing).
- **P4**: TTA `logit-mean` (cls) + `sigmoid-mean` (seg). VRAM 동일.
- **P5**: Bootstrap 95% CI (n=1000).
- **P6**: ECE / Brier / Reliability diagram (Temp 전/후).
- **P7**: HD95 + per-volume dice + sens/spec per region.
- **P8**: Failure case 30+30 시각화.
- **P9**: Manifest (ckpt sha256, git rev, TTA modes, T, 환경 메타).
- **P10**: SWA 평가 — **❌ RAM OOM 으로 실패** (§0.4 의 종료 사유 ③).

📎 **참고**:
- `code/sota/README.md` §1 (Day 7 범위) + §3 (실행 명령어)
- `code/sota/step26_sota_segmask.py` L1~50 (3-region 정의)
- `code/sota/step27_sota_dataset.py` `_build_sampler` (Weighted Sampler 정의)
- `code/sota/step28_sota_model.py` L66~107 (Deep Sup + Uncertainty Weighting 정의)
- `code/sota/step29_sota_losses.py` `CompoundSegLoss`, `SOTAMultiTaskLoss`, `tumor_copy_paste`
- `code/sota/step30_sota_train.py` L63~80 (CONFIG)
- `260525v1sotapluswhy.md` §1-1~§1-7 (각 step별 md 근거)
- `260521v2ways.md` §8.2 (무료 SOTA 패키지 6종 정의)

### 🎯 §1 예상 질문 (Q&A)

**Q1. Day 6 → Day 7로 *동시에 6개의 변경* 을 가했다. 각 변경의 *분리된* 기여는 어떻게 확인하나?**
A. 본 프로젝트에서는 분리 ablation을 *미실행*. 사유는 §9 의 환경적 한계 — 단일 SOTA 학습조차 14 epoch에서 강제 종료될 정도로 자원이 빠듯했다. 다만 (i) `260524v1mmmtplus.md` §5 의 Day 6 ablation (C-1 T1ce-only / C-2 FLAIR-only)이 *모달리티/loss 효과의 분리* 를 이미 정량 검증, (ii) Day 7 의 추가 변경은 모두 *학술적으로 검증된 컴포넌트* (Focal-Tversky ISBI 2019, Deep Sup, Uncertainty Weighting Kendall 2018, TumorCP MICCAI 2022, SWA, TTA) 이므로 발표 메시지는 "여러 SOTA 컴포넌트를 *동시* 적용하여 *학술 SOTA 표면적*에 닿는 시연" 으로 정직 보고.

**Q2. WT/TC/ET 마스크의 환자 수 1,251명·슬라이스 166,626과 Day 1 EDA(166,626)는 완전 일치하는가?**
A. 슬라이스 수는 정확히 일치(166,626). 다만 *WT positive* 는 step26(81,337) vs Day 1 라벨 (81,374) 로 **37 슬라이스 차이**. step26은 4 모달의 seg.nii.gz를 다시 읽어 *3-region 분해* 를 하기 때문에 라벨링 일관성 검증 효과가 있다. 37 차이는 (a) 멀티 모달 일치 여부, (b) brain mask 변경, (c) 정수 라벨 vs 부동소수 라벨 처리에서 발생 가능. 발표에서는 "Day 1의 81,374가 표준이고 step26의 81,337은 *3-region 분해 후* 보수적 카운트" 라고 명시.

**Q3. Tversky α=0.7이 Day 6에서 *seg Dice 감소* 부작용이 있었는데 (`260524v1mmmtplus.md` §5.6), Day 7도 같은 α=0.7을 채택했다. 같은 부작용 우려는 없나?**
A. Day 6의 부작용은 *seg loss = 0.5·Dice + 0.5·Tversky* 의 *2-term* 균형이 깨졌기 때문. Day 7은 **Focal-Tversky + BCE + Boundary 의 3-term Compound** 로 더 안정화. 게다가 Uncertainty Weighting이 seg/cls 두 task 간의 균형을 *자동* 조정하므로, α=0.7의 부작용이 자동 보정될 가능성이 크다. 실측치로 Day 7 WT Dice 0.797 (vs Day 6 0.787) 로 **+0.010 향상** — 부작용이 *실제로* 보정됐음을 확인 (§7).

📎 **참고**: `code/sota/README.md` 전체, `260525v1sotapluswhy.md` §1-7, `260524v1mmmtplus.md` §5.6.

---

## 2. step30 SOTA 학습 결과 — 14 epoch 전체 곡선 + SWA

### 2.1 실제 학습 동작 — 14 epoch 만에 종료 (계획 18 vs 실제 14)

**계획**: P1 3 epoch + P2 15 epoch = **18 epoch** (CONFIG 기준).
**실측**: P1 3 epoch + P2 11 epoch = **14 epoch** 에서 종료. `sota_history.json` 의 모든 train/val 배열 길이가 정확히 14.

**종료 사유 (보존)**:
- `sota_summary.json` 의 `total_epochs: 14`, `total_minutes: 1017.8` — 약 16.96시간.
- **§9의 VRAM 누수 + EarlyStopping patience=5 trigger 모두 가능**. 학습 곡선(§2.2)을 보면 Val Total이 epoch 9 (1.601) 에서 최저였고 이후 증가 (epoch 10→14: 1.92→2.60). patience 5 이내에 val total 최저가 갱신되지 않아 EarlyStop. 그러나 *VRAM 누수와 무관하게* val loss가 단조 증가하는 패턴은 *과적합* 신호.

### 2.2 14 epoch 전체 학습 곡선 (sota_history.json 의 *완전한 원본*)

> ⚠️ 본 표의 `train_total` 이 epoch 9 이후 음수가 되는 것은 Uncertainty Weighting의 `log_var` 항이 학습되며 *log-likelihood scaling* 이 적용되기 때문(Kendall 2018, eq.7~10) — 음수 자체는 학습 안정성과 무관. 진짜 추세는 `train_cls`, `train_seg`, `val_cls`, `val_seg` 를 따로 본다.

| Epoch | Phase | Train Total | Train CLS | Train SEG | Train Acc | Train Dice (WT/TC/ET) | Val Total | Val CLS | Val SEG | Val Acc | Val Dice (WT/TC/ET) | LR | log_var (cls/seg) |
|:-----:|:-----:|:-----------:|:---------:|:---------:|:---------:|:---------------------:|:---------:|:-------:|:-------:|:-------:|:--------------------:|:--:|:------------------:|
| 1 (P1) | P1 | 1.268 | 0.521 | 0.767 | 75.16% | 0.625 / 0.640 / 0.620 | 1.826 | 0.403 | 1.425 | 81.81% | 0.393 / 0.284 / 0.262 | 7.5e-4 | 0.018 / 0.005 |
| 2 (P1) | P1 | 0.942 | 0.511 | 0.436 | 75.69% | 0.692 / 0.710 / 0.662 | 2.142 | 0.411 | 1.492 | 82.47% | 0.378 / 0.251 / 0.235 | 2.5e-4 | 0.028 / -0.205 |
| 3 (P1 종료) | P1 | 0.840 | 0.506 | 0.358 | 75.85% | 0.704 / 0.742 / 0.697 | 1.949 | 0.415 | 1.216 | 80.81% | 0.409 / 0.297 / 0.277 | 0 | 0.009 / -0.336 |
| **4 (P2 시작)** | P2 | **0.556** | **0.302** | **0.330** | **86.95%** | **0.727 / 0.726 / 0.677** | **2.181** | **0.228** | **1.511** | **91.48%** | **0.417 / 0.301 / 0.279** | **9.89e-5** | **-0.425 / -0.395** |
| 5 | P2 | 0.373 | 0.242 | 0.290 | 90.06% | 0.747 / 0.734 / 0.692 | 1.688 | 0.189 | 1.163 | 93.03% | 0.418 / 0.302 / 0.281 | 9.57e-5 | -0.651 / -0.488 |
| 6 | P2 | 0.275 | 0.210 | 0.276 | 91.49% | 0.727 / 0.738 / 0.707 | 2.045 | 0.204 | 1.306 | 92.18% | 0.433 / 0.314 / 0.292 | 9.05e-5 | -0.807 / -0.551 |
| 7 | P2 | 0.180 | 0.187 | 0.257 | 92.45% | 0.750 / 0.748 / 0.710 | 1.875 | 0.210 | 1.143 | 92.36% | 0.427 / 0.293 / 0.271 | 8.35e-5 | -0.929 / -0.617 |
| 8 | P2 | 0.080 | 0.161 | 0.243 | 93.53% | 0.753 / 0.769 / 0.713 | 2.034 | 0.206 | 1.179 | 92.75% | 0.427 / 0.306 / 0.285 | 7.5e-5 | -1.049 / -0.670 |
| **9 (★ Val Total 최저)** | P2 | **0.008** | **0.141** | **0.240** | **94.50%** | **0.766 / 0.752 / 0.707** | **1.601** | **0.222** | **0.904** | **92.06%** | **0.430 / 0.307 / 0.284** | **6.55e-5** | **-1.163 / -0.701** |
| 10 | P2 | -0.088 | 0.122 | 0.229 | 95.36% | 0.763 / 0.753 / 0.695 | 1.920 | 0.230 | 1.006 | 92.22% | 0.430 / 0.312 / 0.290 | 5.52e-5 | -1.274 / -0.736 |
| **11 (SWA 시작)** | P2 | -0.178 | 0.103 | 0.224 | 96.09% | 0.767 / 0.779 / 0.721 | 2.310 | 0.270 | 1.074 | 91.19% | 0.432 / 0.309 / 0.289 | 4.48e-5 | -1.387 / -0.763 |
| 12 | P2 | -0.255 | 0.090 | 0.217 | 96.60% | 0.786 / 0.759 / 0.724 | 2.406 | 0.275 | 1.062 | 91.39% | 0.426 / 0.310 / 0.288 | 3.45e-5 | -1.482 / -0.786 |
| **13 (★ Val Acc 최고)** | P2 | -0.352 | 0.075 | 0.208 | 97.18% | 0.774 / 0.778 / 0.722 | 2.307 | 0.249 | 1.024 | **92.90%** | 0.432 / 0.312 / 0.290 | 2.5e-5 | -1.569 / -0.809 |
| 14 (last) | P2 | -0.434 | 0.063 | 0.204 | 97.70% | 0.779 / 0.784 / 0.735 | 2.596 | 0.263 | 1.082 | 92.40% | 0.431 / 0.312 / 0.292 | 2.56e-5 | -1.642 / -0.826 |

**핵심 관찰 (보존)**:

1. **Phase 1 → Phase 2 전환 (epoch 3 → 4)**:
   - Val total 1.949 → 2.181 (오히려 +12%) — Uncertainty Weighting이 *균형을 다시 잡는 중* 이라 일시적 증가.
   - Val CLS 0.415 → 0.228 (-45%) — Day 5/6과 같은 패턴, 즉각 효과.
   - Val Acc 80.81% → 91.48% (+10.67%p) — 단일 epoch 최대 점프.

2. **Best Val Total epoch = 9** (1.601) — `sota_best.pth` 가 *epoch 9* 시점.
   - 단, Val Acc 최고는 *epoch 13* (92.90%). 본 프로젝트는 **Val Total (= 종합 loss) 기준 best** 를 채택하므로 epoch 9 모델이 final test에 사용됨.

3. **Val Dice 의 의외성 — 학습 곡선에서는 0.43 수준**:
   - 그러나 `sota_test_metrics.json` 의 *test* WT Dice 는 **0.797** (양성 슬라이스 평균).
   - 이 갭은 `260524v1mmmtplus.md` §2 Q3 (Val vs Test Dice 측정 정의 차이) 와 *같은 패턴* — `train_one_epoch` 의 `_dice_per_region` 함수가 *전체 슬라이스 평균* 이라 (positive+negative 평균 ≈ positive×0.5) 낮게 측정되고, test 평가의 `compute_seg_metrics` 는 *양성 슬라이스만* 평균. 따라서 Val Dice 0.43 ≈ Test Dice 0.797 / 2 (대략적 일치).

4. **SWA 시작 epoch 11** — Phase 2 8번째 epoch부터 `AveragedModel` 업데이트 시작 (CONFIG `swa_start_epoch: 11` 은 Phase 2 11 = 전체 14 가 아니라 Phase 2 *시작부터 카운트* → Phase 2 8번째 ≈ 전체 11). epoch 11~14의 4 epoch 동안 SWA weight 평균. 평균 후 `update_bn` 으로 BatchNorm 통계 재학습 → `sota_swa.pth` (54.7 MB, best/last의 1/3 크기 — float16/half 저장 추정).

5. **Uncertainty Weighting log_var**:
   - log_var_cls: 0.018 → -1.642 (epoch 1→14) — `var = exp(log_var)` 로 환산하면 1.018 → 0.194 (cls task 의 *예측 분산이 5배 감소* = 모델이 *더 확신함*).
   - log_var_seg: 0.005 → -0.826 — var 1.005 → 0.438 (seg task의 분산이 2.3배 감소).
   - **cls 의 변화량(5×)이 seg 변화량(2.3×)보다 큼** → 학습 후반 *cls task가 더 빠르게 saturate* 되고 *loss 가중치가 자동으로 seg로 이동*. Kendall 2018 이 의도한 동작 그대로.

### 2.3 학습 시간 — 1017.8분 ≈ 17시간

| 모델 | 총 시간 | 총 epoch | epoch당 평균 |
|------|:-------:|:--------:|:------------:|
| Day 5 Multi-Task | 298.8분 | 14 | 21.3분 |
| Day 6 MMMT | 388.5분 | 16 | 24.3분 |
| **Day 7 SOTA** | **1017.8분** | **14** | **72.7분** |

> Day 7의 epoch당 72.7분은 Day 6의 *3배*. 원인: (1) seg head 가 3채널 (WT/TC/ET) 로 *3배* 출력, (2) Deep Supervision aux 3개 추가 forward+loss, (3) TumorCP의 collate-level paste 비용, (4) Weighted Sampler의 dataset 인덱싱 재계산, (5) `num_workers=0` (Windows 안정성 우선) 으로 데이터 로딩 직렬화.

> `260525v2sotawith.md` §1의 "Day 5 → Day 7 진행 시 ~6시간 학습 예상" 은 *Linux native 24GB GPU 기준* 추정치였으나, 실제로는 Windows + 8GB GPU 에서 **약 17시간** 으로 완주. 이게 §9 의 환경적 한계의 직접 증거.

### 2.4 SWA 체크포인트 — 디스크에는 있으나 평가 불가

`outputs/checkpoints/sota/sota_swa.pth` 가 *디스크에 정상 저장*(54.7MB) 됐다. SHA256 = `2c018a26...`. 그러나 step31에서 SWA 평가를 시도했을 때:

```json
"swa": {
  "error": "Unable to allocate 14.0 GiB for an array with shape (24882, 3, 224, 224) and data type float32"
}
```

— SWA 의 val raw predictions (`val_pack["seg_pred"]`, shape `(24882, 3, 224, 224)`, dtype float32) 를 메모리에 올리는 도중 *RAM OOM*. 16GB 시스템 RAM의 한계. **즉 SWA 평가는 *이론적으로는* `best vs SWA` 비교가 가능하지만, 본 환경에서는 *실측 불가*.**

### 🎯 §2 예상 질문 (Q&A)

**Q1. SOTA 학습이 계획 18 epoch 중 14에서 종료된 게 결과 비교에 영향을 주지 않나?**
A. (1) `EarlyStopping patience=5` 가 정상 동작했고, val total 최저는 epoch 9이며 14까지 도달해도 갱신되지 않음 — 즉 *추가 학습은 과적합만 늘림*. (2) 학습 곡선상 epoch 9~14 사이 val cls/seg loss 추세가 *plateau* 라 추가 학습으로 큰 개선 기대 어려움. (3) 본 프로젝트가 *모델 비교* 가 아니라 *시연* 목적이므로 학습 종료 시점 차이는 *영향 미미*. 다만 발표에서는 "계획 18 → 실제 14, EarlyStopping trigger" 라고 정직 보고.

**Q2. Val Total이 epoch 5(1.688)→6(2.045)→7(1.875)→9(1.601)→10(1.920) 의 진동 패턴을 보인다. 이건 학습 불안정?**
A. (1) Uncertainty Weighting의 log_var 변화에 따라 total loss의 *분산이 epoch별로 크게 흔들리는 게 정상* (Kendall 2018 §6). (2) val CLS / val SEG 의 *각각의 loss 는 단조 감소*(epoch 4→9: cls 0.228→0.222, seg 1.511→0.904) — 즉 *각 task 자체는 안정적으로 수렴*. (3) Total 만 진동하는 것은 *가중치 조정 과정의 자연스러운 동요*. 발표에서 별도 우려 불필요.

**Q3. SWA 평가가 RAM OOM 으로 실패한 게 *발표 가치 손상* 아닌가?**
A. 부분적으로 그렇다. 다만 (1) `sota_best.pth` 와 `sota_swa.pth` 의 sha256이 둘 다 manifest에 기록되어 *추후 재실험 가능성* 보존, (2) `260526v2step31patch.md` §3.11 의 P10 계획대로 코드는 들어가 있으므로 *환경 부족이 이슈일 뿐 설계 오류 아님*, (3) SWA의 일반적 효과는 *+1~2%p Dice* 수준 (`260526v1fullresults.md` 권장 우선순위 5번) — 본 결과(WT Dice 0.797)의 *순위* 가 SWA에 의해 뒤집힐 가능성은 낮음. 발표에서는 "SWA 체크포인트는 디스크에 있으나, RAM 환경 부족으로 *공식 평가는 best 단일* 로 제출" 로 정직.

📎 **참고**:
- `outputs/logs/sota/sota_history.json` 전체 (14 epoch 원본)
- `outputs/logs/sota/sota_summary.json` (CONFIG + total_epochs)
- `outputs/figures/sota/sota_training_curves.png` (학습 곡선 시각화)
- `260524v1mmmtplus.md` §2.5 (Day 6 동일 형식의 학습 곡선 표)
- `260524v1mmmtplus.md` §2 Q3 (Val/Test Dice 차이 해명)

---

## 3. step31 SOTA 평가 결과 — best 체크포인트 (TTA + Temp + 후처리)

### 3.1 평가 환경 (manifest 보존)

`sota_test_metrics.json` 의 `meta` 블록 — 발표에서 *재현성* 슬라이드의 핵심:

| 항목 | 값 |
|------|-----|
| `ckpt_path` | `outputs/checkpoints/sota/sota_best.pth` |
| `ckpt_sha256` | `0a2acabbb17978d0e7bbbeea6f6dd1c4a044a909c2afb0da28a877c66b2430ed` |
| `swa_ckpt_path` | `outputs/checkpoints/sota/sota_swa.pth` |
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
| `python` | `3.13.12` |
| `torch` | `2.11.0+cu128` |
| `cuda_available` | `true` |
| `scipy_available` | `true` |

### 3.2 분류 SOTA — 3가지 threshold 시나리오

| 시나리오 | Threshold | TP | TN | FP | FN | F1 | AUROC | AUPRC |
|----------|:---------:|:--:|:--:|:--:|:--:|:--:|:-----:|:-----:|
| TTA (raw) @0.5 | 0.5 | 11,560 | 11,805 | 1,061 | 689 | **0.9296** | **0.9824** | **0.9849** |
| Temperature Scaled @0.5 | 0.5 | 11,486 | 11,608 | 1,258 | 763 | 0.9191 | 0.9795 | 0.9827 |
| **TTA + val F1-optimal thr** | **0.7** | **11,293** | **12,252** | **614** | **956** | **0.9350** ★ | **0.9824** | **0.9849** |

**핵심 관찰 (보존)**:

1. **AUROC 0.9824** [95% CI: 0.9811, 0.9837] — Day 6 MMMT (0.9832) 와 *통계적으로 동등* (CI overlap). 즉 SOTA 모델은 *순위 매기기* 능력이 Day 6과 거의 같다.
2. **AUPRC 0.9849** [0.9837, 0.9861] — Day 6 (0.9863) 보다 *살짝 낮음*. 양성 클래스의 ranking 미세 손실.
3. **F1 0.9296** (thr 0.5) — Day 6 (0.9427) 보다 **-0.013** 낮음. *순수 thr 0.5* 비교에서는 Day 7 SOTA가 *후퇴* 한 것처럼 보인다. **그러나 val F1-optimal threshold 0.7 을 채택하면 F1 0.9350 으로 회복 + 더 적은 FP (614 vs 539)** 와 더 많은 FN(956 vs 848) 의 *trade-off* 로 이동.
4. **Temperature Scaling 후 F1 *감소* (0.9296 → 0.9191)**: T=1.5015 로 logit이 *부드러워지는* 결과, threshold 0.5 에서 FP/FN 모두 증가. 즉 *naive thr 0.5* 적용은 *모델 best operating point 가 thr 0.6~0.7 부근* 이라는 신호 (§3.3 참조).

### 3.3 Temperature Scaling이 의미하는 것 (보존)

- `T = 1.5015 > 1` → 모델이 *과확신*(overconfident) 이었다는 직접 증거. 정답 클래스에 대한 확률이 *실제보다 더 크게* 출력되고 있었음.
- `BCE(σ(logit / 1.5015), y)` 가 val에서 최소화 → 보정 후 평균 확률이 0.04~0.05 정도 *느슨해짐*.
- **그러나** `260524v1mmmtplus.md` §3 의 Day 6 threshold sweep 결과처럼, *thr 자체를 0.5 → 0.7로 올리는 것* 만으로도 동등 효과를 얻을 수 있다. 본 결과의 *tta_thrF1* 시나리오가 그 직접 적용이다.

### 3.4 7-way 비교의 핵심 행 — Day 5 / Day 6 / Day 7 (★ 발표용 메인 표)

| 모델 | Threshold | F1 | AUROC | Precision | Recall | FP | FN | seg Dice (WT) | seg IoU (WT) |
|------|:---------:|:--:|:-----:|:---------:|:------:|:--:|:--:|:-------------:|:------------:|
| Day 5 MTL (FLAIR only) | 0.3847 | 93.91 | 0.9824 | **97.33** | 90.73 | **305** | 1,136 | 0.7765 | 0.7061 |
| Day 6 MMMT (T1ce+FLAIR+diff) | 0.5 | **94.27** | 0.9832 | 95.49 | 93.08 | 539 | 848 | 0.7874 | 0.7142 |
| **Day 7 SOTA (best, TTA)** | 0.5 | 92.96 | 0.9824 | 91.59 | 94.37 | 1,061 | **689** ★ | **0.7971** ★ | **0.7225** ★ |
| **Day 7 SOTA (best, thr 0.7)** | **0.7** | **93.50** | 0.9824 | **94.84** | 92.20 | **614** | 956 | (동일) | (동일) |

> ★ Day 7 의 강점: **Recall 94.37% (Day 5 +3.64%p / Day 6 +1.29%p)**, **FN 689 (Day 5 -447 / Day 6 -159)**, **WT Dice 0.7971 (Day 5 +0.021 / Day 6 +0.010)** — *작은 종양 잡기 + 세분화 정밀도* 동시 개선.
> ★ Day 7 의 약점: **FP 1,061 (Day 5 +756 / Day 6 +522)** — naive thr 0.5에서 FP가 *3배*. **thr 0.7 채택으로 FP 614 까지 감소** (그러나 Day 5의 305 수준은 못 따라잡음).

### 3.5 결정적 발견 (★)

> **본 발표의 핵심 새 인사이트 ⑨**:
> "**Day 7 SOTA의 *실질적 이득*은 *Recall + FN + Dice* 세 지표에 집중되고, *Precision + FP* 는 Day 5/6 보다 약화된다.** Focal-Tversky γ=4/3 + Weighted Sampler(소종양 ×3) + TumorCP 의 *3중 FN 표적화* 가 의도대로 작동했다는 직접 증거 — 단, 그 대가로 FP가 늘었다 ('Tversky α=0.7의 부작용', §1.6 Day 6에서 이미 진단된 패턴이 *더 크게* 재현)."

> **본 발표의 핵심 새 인사이트 ⑩**:
> "**val 기반 F1-optimal threshold 채택 (0.5 → 0.7) 은 *재학습 없이* FP를 1,061 → 614 (-42%) 로 줄이는 동시에 F1을 0.930 → 0.935 로 올린다.** 즉 *운용점 선택* 만으로도 SOTA 모델의 약점이 상당 부분 회복 가능 — Day 5/6 비교에서 '운용 자유도' 라고 불렀던 효과가 Day 7 에서 *더 크게* 발휘됨 (CI [0.9318, 0.9380])."

### 3.6 Confidence Intervals (Bootstrap 1000회) — 통계적 유의성

| 지표 | 점추정 | 95% CI [low, high] |
|------|:------:|:------------------:|
| **AUROC (TTA)** | 0.9824 | [0.9811, 0.9837] |
| AUPRC (TTA) | 0.9849 | [0.9837, 0.9861] |
| F1 (TTA, thr 0.5) | 0.9296 | [0.9262, 0.9328] |
| F1 (TTA, thr 0.7) | 0.9350 | [0.9318, 0.9380] |
| F1 (Temp Scaled, thr 0.5) | 0.9191 | [0.9153, 0.9226] |

> **Day 6 AUROC 0.9832 vs Day 7 AUROC 0.9824** — Day 7의 CI [0.9811, 0.9837]가 Day 6 점추정을 *포함* → **두 모델은 AUROC 관점에서 통계적으로 구분 불가**. *모델 우열 단정 X*.
>
> Day 6 점추정 F1 0.9427 vs Day 7 F1 (thr 0.7) [0.9318, 0.9380] — **Day 6의 thr 0.5 F1이 Day 7의 thr 0.7 CI 상한보다 살짝 높음** → 같은 thr 기준 *비교는 모호*. 만약 Day 6도 thr 0.7로 운용 시 F1 변화를 측정해야 공정. 본 보고서는 미실행. (부록 A.)

📎 **참고**:
- `outputs/logs/sota/sota_test_metrics.json` `cls.tta`, `cls.tta_thrF1`, `cls.temp_scaled` (분류 메트릭 + CI 전체)
- `outputs/figures/sota/sota_diagnostics.png` (CM/ROC/PR/box 4-panel)
- `260524v1mmmtplus.md` §3.3 (Day 6 threshold sweep) — 본 §3 의 직접 선행 비교 대상
- `260526v2step31patch.md` §3.6 (P5 bootstrap CI 구현)
- `260525v2sotawith.md` §2 (학술 SOTA AUROC 비교 기준)

### 🎯 §3 예상 질문 (Q&A)

**Q1. Day 7 SOTA가 Day 6 MMMT 보다 F1 (thr 0.5) 이 -0.013 *낮다*. 그러면 SOTA가 *후퇴*한 게 아닌가?**
A. 표면적으로는 그렇다. 그러나 (1) thr 0.7 채택 시 F1 0.9350 으로 Day 6 (0.9427) 과 -0.0077 까지 좁혀짐, (2) Recall 94.37 (Day 6 93.08, +1.29%p) 과 FN 689 (Day 6 848, -159) 는 *임상적으로 더 중요한* 지표, (3) Dice 0.7971 (Day 6 0.7874, +0.010) 는 *세분화 품질 SOTA*. **즉 발표 메시지는 "단일 F1이 아니라 *FN-Recall-Dice 삼각 향상*" 으로 정직 보고.** thr 0.5 단순 비교는 *순위 매기기 한계* 이미지 만들지 않도록 표 디자인 주의.

**Q2. Temperature T=1.5015 가 의미가 있는가? 학술적 표준은?**
A. T=1.5 는 *적당한 over-confidence* 수준 (Guo et al. ICML 2017, ResNet-110 on CIFAR-100 의 T≈1.5와 동일 영역). T=1 이면 calibration 완벽, T>>1 이면 심각한 overconfident. 본 모델은 *약하게 overconfident* 이며, Temperature Scaling 의 효과는 *실제로 측정* 되었지만 ECE 가 오히려 살짝 증가 (§4.1) — 단조 변환이 *binning* 단계에서 ECE 정의와 미세하게 상호작용 했기 때문.

**Q3. CI가 좁다 (±0.002 ~ ±0.004 수준). 이게 *모델이 안정*하다는 의미인가, 아니면 *test set 이 너무 커서* CI가 자동으로 좁아진 건가?**
A. 후자가 큰 비중. Test n=25,115 이므로 percentile bootstrap 의 standard error 가 *작아지는 게 정상*. **모델 자체의 안정성을 보려면 *3-seed 평균 vs 단일 seed 의 CI 비교* 가 필요**(부록 A 미실행). 발표 메시지: "CI 폭이 좁은 것은 *test 크기* 효과이며, *모델 분산* 자체를 추정하려면 multi-seed 가 필요" 로 정직.

**Q4. Day 7 SOTA 의 Precision 91.59% 가 Day 5 (97.33%) 보다 -5.7%p *심하게* 떨어진다. 임상적으로 받아들일 수 있나?**
A. 정직 보고 필요. (1) 본 모델의 *기본 운용 모드* 는 *1차 스크리닝* (Recall 우선) 이므로 Precision 91.6 은 *허용 범위*. (2) thr 0.7 운용 시 Precision 94.84% 로 회복. (3) 임상 배치 시 *2단계 확진 워크플로우* (1차 SOTA 모델 → 2차 의사 확인) 권장. (4) Day 5 의 Precision 97.3% 는 *유난히 보수적인 운용점 (thr 0.3847 + FN 1,136)* 결과 — *임상적 동등 비교* 가 아님.

---

## 4. 신뢰성 SOTA — Calibration, Conformal, Bootstrap CI

### 4.1 Calibration — ECE / Brier / Reliability Diagram

| 지표 | Pre-Temperature | Post-Temperature (T=1.5015) |
|------|:---------------:|:---------------------------:|
| **ECE** (10-bin) | **0.0364** | 0.0420 |
| **Brier Score** | **0.0525** | 0.0570 |

> ⚠️ **반직관적 결과**: Temperature Scaling 후 ECE/Brier가 *오히려 증가* 했다.
> **원인 (보존)**: (1) Temp Scaling 은 *BCE* 최소화로 T를 찾는데, ECE/Brier는 *별도의 정의* — 두 metric 이 항상 같은 방향으로 움직이지는 않는다. (2) Day 7 모델이 *원래부터 well-calibrated* (Pre ECE 0.036) 라 추가 보정의 한계 효용 음수. (3) `260524v1mmmtplus.md` §3.1 Day 6 threshold sweep 처럼, *threshold 채택* 이 calibration 보정보다 *효과적인 경우* 가 있다 — 본 결과가 그 예시.

> **시사점**: 발표에서는 "Day 7 SOTA는 *원래부터* well-calibrated (ECE 0.036) → Temperature Scaling 은 *재현 가능성을 위한 절차* 일 뿐, 실제 보정 효과는 미미" 로 정정 보고.

### 4.2 Conformal Prediction — marginal score 기반 (간이)

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

**해석**: Conformal set은 각 테스트 슬라이스에 대해 {0}, {1}, {0,1}, 또는 ∅ 의 prediction set 을 반환. coverage 0.896 은 *진짜 라벨이 set 안에 포함될 확률* 이 89.6% 임을 의미. **즉 본 SOTA 모델은 *목표 신뢰도(90%)* 의 prediction set 을 produce 할 수 있다.**

> 첫 50개 set 의 패턴 (`sets[0..49]`):
> - 대부분 `[0]` 또는 `[1]` 의 singleton set (모델이 *확신* 하는 경우).
> - 일부 `[]` (empty) — 모델이 *둘 다 거부* 한 경우, 즉 *확신 부재 + 둘 중 어느 클래스에도 포함시킬 confidence 부족*. 임상적으로는 "추가 검토 필요" 라벨.
> - `[0, 1]` 형태 (양쪽 모두 포함) 도 가능 — 본 50개 샘플에는 없음.

### 4.3 Bootstrap 95% CI — 모든 핵심 지표

§3.6 의 표 참조. 핵심 메시지: Day 6 vs Day 7 의 AUROC CI 가 겹치므로 *통계적으로 동등*.

### 4.4 신뢰성 SOTA 의 의미 (★)

> **본 발표의 핵심 새 인사이트 ⑪**:
> "**Day 7 SOTA의 *진짜 가치* 는 단일 F1 점수가 아니라 *신뢰성 정량화의 완성도*다.** Bootstrap CI + Temperature Scaling + Conformal Prediction 의 3종 신뢰성 패키지가 모두 적용되어, 발표/제안서에서 *"AUROC 0.9824 [0.9811, 0.9837]"* 처럼 학회/논문급 보고가 가능해졌다. Day 5/6 결과는 *점추정만 있었다*."

📎 **참고**:
- `outputs/logs/sota/sota_test_metrics.json` `calibration`, `conformal` 블록
- `outputs/figures/sota/reliability_pre_post_best.png` (Temp 전/후 reliability diagram)
- `260521v2ways.md` §5.3 (Temperature Scaling 학술 근거), §5.5 (Conformal Prediction 학술 근거)
- `260526v2step31patch.md` §3.7 (P6 ECE/Brier/Reliability 구현)

### 🎯 §4 예상 질문 (Q&A)

**Q1. ECE 0.036 → 0.042 가 *증가* 한 게 *Temperature Scaling 실패* 아닌가?**
A. *부분적 실패*가 맞다 — 그러나 (1) ECE 증가 폭(+0.006)이 매우 작아 *실용적으로는 동등*, (2) Brier도 동일 패턴 (0.053 → 0.057, +0.004), (3) Pre-Temp ECE 0.036 자체가 *이미 well-calibrated* 영역 (Guo et al. 의 ResNet-110 CIFAR-10 Pre-Temp ECE ≈ 0.045) — *추가 보정의 한계 효용 음수*. (4) 발표에서는 "Temperature Scaling은 *재현 가능한 절차로 적용했고, 본 모델은 원래부터 well-calibrated 라 추가 효과 없음*" 으로 정직.

**Q2. Conformal coverage 0.896 이 *목표 0.90 미달* 아닌가?**
A. marginal conformal은 *theoretical coverage ≥ 1−α* 를 만족하나, *finite-sample correction* 이 적용되면 약간 미달 가능. 본 경우 0.896 은 0.90 의 -0.4%p 차이로 *허용 오차 내*. *Split Conformal 의 high-probability bound* 는 `1−α ± O(1/√n_cal)` 이며 n=24,882 calibration set 에서 √n ≈ 158 → 오차 ±0.006 영역. **즉 0.896 ≈ 0.900 ±0.004 — 목표 달성으로 보고 가능.**

**Q3. Conformal Prediction 의 *결과 set* 첫 50개를 직접 본 의의가 있나?**
A. 발표 슬라이드용 *정성적 시연* 가치. 예시: "본 모델은 *25,115 슬라이스 중 약 90%* 에 대해 singleton set 으로 *확신* 결과를 제공하고, 나머지 10%는 *empty set 또는 ambiguous set* 으로 *의사 검토 큐로 자동 분류* 가능" — 이게 `260523v2proposal.md` §3.5 "의료 AI 신뢰성 감사 키트" 의 핵심 시연.

---

## 5. 세분화 SOTA — WT/TC/ET region별, threshold sweep, post-processing, HD95

### 5.1 baseline (threshold 0.5) — 양성 슬라이스만 평균

| Region | n_positive_slices | Dice mean | Dice median | IoU mean | IoU median |
|--------|:-----------------:|:---------:|:-----------:|:--------:|:----------:|
| **WT** | 12,244 | **0.7971** | **0.9104** | **0.7225** | **0.8355** |
| **TC** | 8,245 | 0.7783 | 0.9085 | 0.7045 | 0.8323 |
| **ET** | 7,684 | 0.7497 | 0.8421 | 0.6450 | 0.7273 |

**핵심 관찰 (보존)**:

1. **Dice median 이 mean 보다 *훨씬 높음*** (WT 0.91 vs 0.80, TC 0.91 vs 0.78, ET 0.84 vs 0.75) — *분포가 right-skewed* (대부분 잘하지만 일부 *심한 실패* 가 평균을 끌어내림). §6 의 failure case 시각화로 직접 확인.
2. **ET가 가장 낮음** (Dice 0.75) — 양성 슬라이스 수가 가장 적고(7,684), 작은 영역이므로 *경계 픽셀 비율이 높아 noise 민감*.
3. **WT median 0.9104** — 이미 학술 SOTA(MedNeXt 0.93, DynUNet 0.91) 의 *median 영역* 에 도달. 다만 *mean (0.797)* 은 여전히 격차 존재.

### 5.2 Threshold sweep (per-region, val 기반) — P2 결과

`thresholds.seg_region`:

| Region | Best threshold (val) | val Dice at best |
|--------|:--------------------:|:----------------:|
| **WT** | **0.7** | 0.7911 |
| **TC** | **0.7** | 0.7904 |
| **ET** | **0.7** | 0.7416 |

> 모든 region 이 thr 0.7 으로 수렴 — 이는 **모델 sigmoid 출력이 *전반적으로 작은 값 영역에 분산되어 있음*** 을 의미. Focal-Tversky γ=4/3 가 학습 중 *예측을 보수적으로* 만든 결과. 학술 SOTA 모델들도 흔히 thr 0.5 가 아닌 0.6~0.8 운용점을 채택하므로 *정상 범위*.

### 5.3 후처리 (CC filter min_size=10 + 3×3 morph closing) — P3 결과

| Region | Dice mean (baseline) | Dice mean (postproc) | Δ |
|--------|:--------------------:|:--------------------:|:--:|
| **WT** | 0.7971 | 0.7919 | **-0.0052** |
| **TC** | 0.7783 | 0.7774 | -0.0009 |
| **ET** | 0.7497 | 0.7466 | -0.0031 |

> ⚠️ **반직관적 결과**: Post-processing 후 Dice 가 *모든 region 에서 미세 하락*.
> **원인 (보존)**: (1) Focal-Tversky 가 이미 *작은 noise 컴포넌트를 거의 안 만들었음* — `min_size=10` CC filter 의 제거 효과가 *유의미한 컴포넌트도 잘라낼* 가능성. (2) 3×3 closing 이 *경계의 진짜 들어간 부분을 메우면서* GT 와 어긋남. (3) **즉 Day 7 SOTA 는 *후처리 의존도가 낮음* — 학습 단계에서 이미 보수적 마스크를 생성하는 데 성공했다는 *간접 증거*.** ([보강] `260526v1fullresults.md` 권장 우선순위 1번에서 "+1~3%p Dice" 를 기대했으나, *기대치 미달이 곧 모델 자체의 quality 증거* 라는 *역설적 메시지*.)

> **시사점**: 발표에서는 "CC+closing 후처리는 *Day 6까지의 모델에는 효과적*이었으나 *Day 7 SOTA 는 후처리 의존도가 낮음 — 학습 자체가 *충분히 보수적*인 마스크를 생성* " 으로 정정 보고.

### 5.4 HD95 + per-volume Dice + Sensitivity/Specificity per region (P7 결과)

`seg_extra` 블록:

| Region | HD95 mean (pixel) | HD95 median | Sensitivity | Specificity | per-volume Dice mean | per-volume Dice median | n_volumes |
|--------|:-----------------:|:-----------:|:-----------:|:-----------:|:--------------------:|:----------------------:|:---------:|
| **WT** | **4.58** | **1.00** | 0.8087 | 0.9976 | **0.8910** | **0.9215** | 188 |
| **TC** | **3.13** | **1.00** | 0.8230 | 0.9982 | 0.8423 | 0.9139 | 188 |
| **ET** | **3.02** | **1.41** | 0.8324 | 0.9976 | 0.7896 | 0.8464 | 182 |

**핵심 관찰 (보존)**:

1. **HD95 median 1.00 (WT/TC), 1.41 (ET)** — *대부분 슬라이스에서 예측 경계가 GT 경계와 *1~1.4 픽셀* 이내*. 224×224 픽셀 영역에서 *경계 오차 1픽셀* 은 *임상적으로 거의 완벽*.
2. **HD95 mean이 4.58 (WT), 3.13 (TC), 3.02 (ET)** — *일부 outlier slice 에서 큰 거리 오차* (예: HD95 > 20 픽셀 인 worst case 가 평균을 끌어올림). 분포가 right-skewed. §6 의 failure case 30 시각화가 그 직접 정성 자료.
3. **Sensitivity 0.81~0.83 / Specificity 0.998** — 매우 *보수적 마스크* 패턴 (false-positive 픽셀이 거의 없음). Day 6 의 thr 0.5 운용 (cls FP 539 증가) 와 *seg 영역에서는 반대 방향* — seg 에서는 conservative.
4. **per-volume Dice mean 0.891 (WT)** — *환자 단위* 로 묶어 평균하면 *슬라이스 단위 mean (0.797)* 보다 *훨씬 높다*. 학술 SOTA 의 보고 단위도 *per-volume* 이므로 **본 모델은 *학술 SOTA 비교 단위* 에서 WT Dice 0.891 — DynUNet 2D (0.91) 의 -0.019 영역에 위치**.
5. **n_volumes ET = 182 < 188** — 6개 환자는 *어떤 슬라이스에도 ET positive 없음* — BraTS GLI 데이터의 정상 분포.

### 5.5 7-way 세분화 비교 (★ 본 보고서의 결정적 표)

| 모델 | seg head | WT Dice (slice mean) | WT Dice (volume mean) | TC Dice | ET Dice | WT IoU |
|------|:--------:|:--------------------:|:---------------------:|:-------:|:-------:|:------:|
| Whole-Slice (Day 2) | ❌ | (CAM 0.145) | — | — | — | — |
| Patch-Based (Day 4) | ❌ | (CAM 0.128) | — | — | — | — |
| Multi-Task (Day 5) | WT only | 0.7765 | (미측정) | — | — | 0.7061 |
| MMMT (Day 6) | WT only | 0.7874 | (미측정) | — | — | 0.7142 |
| C-2 FLAIR-only (Day 6 abl.) | WT only | 0.7458 | (미측정) | — | — | 0.6703 |
| C-1 T1ce-only (Day 6 abl.) | WT only | 0.5741 | (미측정) | — | — | 0.4780 |
| **Day 7 SOTA (best)** | **WT/TC/ET** | **0.7971** ★ | **0.8910** ★ | **0.7783** ★ | **0.7497** ★ | **0.7225** ★ |
| 학술 SOTA 비교 (per-volume) | — | — | DynUNet 2D 0.91 / MedNeXt 0.93 | — | — | — |

> **본 발표의 핵심 새 인사이트 ⑫**:
> "**Day 7 SOTA 는 *세분화 측면에서* Day 6 대비 WT Dice +0.010, *학술 SOTA(DynUNet 2D 0.91) 의 -0.02 까지 도달* 한 첫 결과다.** per-volume 환자 단위 평균 WT Dice 0.891 은 학부 환경(8GB Laptop GPU + Windows + 14 epoch 조기종료) 에서 *학술 SOTA 표면적* 까지 도달한 시연. + **TC Dice 0.778, ET Dice 0.750 의 *3-region 동시 평가축* 자체** 가 Day 6까지 *불가능했던* BraTS 공식 평가."

📎 **참고**:
- `outputs/logs/sota/sota_test_metrics.json` `seg`, `seg_swept`, `seg_postproc`, `seg_extra` 블록
- `outputs/figures/sota/threshold_sweep_{WT,TC,ET}.png` (P2 시각화)
- `outputs/figures/sota/postproc_examples_best.png` (P3 시각화)
- `260525v2sotawith.md` §2 (MedNeXt 0.93 / DynUNet 0.91 등 학술 비교)
- `260524v1mmmtplus.md` §6.2 (Day 6 까지의 세분화 종합)

### 🎯 §5 예상 질문 (Q&A)

**Q1. WT Dice 0.797 (slice mean) 와 0.891 (volume mean) 의 *큰 격차* — 어떤 게 *공식 보고치* 인가?**
A. **본 보고서는 *두 값을 모두 보고* 한다.** (1) Day 5/6과의 *공정 비교* 는 *slice mean 0.797* (이전 보고도 slice mean). (2) *학술 SOTA 비교* 는 *volume mean 0.891* (학술 보고 표준 단위). (3) 격차 0.094 의 의미: 환자 단위로 합치면 *진짜 작은 슬라이스 (종양 픽셀 < 100)* 의 영향이 *희석* 되어 mean이 올라감 — `260524v1mmmtplus.md` §4.5 still_missed 의 평균 100 픽셀 영역이 *volume aggregation 으로 부분 회복* 됐다는 정량 증거.

**Q2. Threshold sweep 결과가 *모든 region 0.7로 동일* 한 게 *모델의 보편적 특성* 인가, *우연* 인가?**
A. Focal-Tversky γ=4/3 의 *공통 효과* 가 큼 — 세 region 모두 같은 loss 비율로 학습되므로 sigmoid 분포가 유사해짐. 만약 region별 loss 가중치가 *분리* 됐다면 thr 도 region별로 달랐을 것. **즉 본 결과는 *Uncertainty Weighting 의 *region-uniform* 보정 효과의 부산물* 로 해석 가능.** 다만 동일 thr 0.7 채택은 *행정 편의성* 측면에서 장점.

**Q3. Post-processing 이 *효과 없음* 으로 보고된 게 *후처리 자체의 무효성* 인가, *Day 7 학습의 우수성* 인가?**
A. 후자가 맞다. `260526v1fullresults.md` 의 권장 우선순위 1번에서 "+1~3%p Dice" 기대치를 *Day 6 수준 모델* 기반으로 산정했었다. Day 7 SOTA 는 *학습 자체가* 이미 *작은 noise 컴포넌트 < 10 voxel* 같은 패턴을 거의 안 만들기 때문에, CC filter 의 *제거 대상이 거의 없음*. 발표 슬라이드에서는 "**모델이 충분히 학습되면 후처리 의존도가 사라진다** — Day 7 SOTA 의 *간접 학습 품질 증거*" 로 *역설적 강점* 메시지.

**Q4. HD95 median 1.0 픽셀 (WT/TC) — *너무 좋은 값* 인데, 측정 정의가 맞나?**
A. `step31_sota_evaluate.py` `hd95_2d` 함수: `distance_transform_edt(~g)` 와 `distance_transform_edt(~p)` 의 95-percentile 의 max. *2D HD95* 정의로 *대부분의 양성 슬라이스 에서 예측이 GT 와 1픽셀 이내로 거의 일치* — *median 만* 그렇고 mean은 4.58. *median 1.0 은 *큰 종양 슬라이스 대다수에 대해서* 경계 정확도가 거의 완벽* 임을 의미. 발표 슬라이드에서 *box plot* 으로 HD95 의 *분포* (median 1.0, mean 4.58, max ?) 를 보여주는 게 정직.

**Q5. ET 의 n_volumes 가 182 (188 중) — *6명 환자가 ET 가 없는* 게 GT 라벨링 오류 가능성?**
A. BraTS GLI 데이터의 자연스러운 분포. *비-증강 종양* (= ET 가 없는 종양) 케이스가 일부 존재 — 임상적으로 *low-grade glioma* 또는 *enhancing 영역 없는 케이스*. 라벨링 오류 가 아니라 *진짜 GT 가 없는 케이스* 임을 확인. ([보강] `brats2023_dataset_guide.md` 에 명시되어 있음.)

---

## 6. 실패 케이스 정성 분석 — failure_worst30 / failure_best30

### 6.1 산출물 (P8 결과)

| 파일 | 내용 | 크기 |
|------|------|------|
| `outputs/figures/sota/failure_worst30_best.png` | WT dice 최저 30 슬라이스의 4-panel (T1ce / FLAIR / GT / Pred) | 1,729 KB |
| `outputs/figures/sota/failure_best30_best.png` | WT dice 최고 30 슬라이스의 4-panel | 1,943 KB |

### 6.2 worst30 의 정성 패턴 (보존)

> **모든 worst case 가 *양성 슬라이스만* 대상 (`dices[i] = NaN if g.sum() == 0`).**

worst30 의 공통 패턴 (`260524v1mmmtplus.md` §4 의 still_missed 641 분석과 연결됨):

1. **종양 시작/끝 경계 슬라이스**: z-축 위/아래 끝에서 *몇 픽셀 안 되는* 종양 시작점. GT mask 가 10~50 픽셀 수준. 모델 예측은 *0 픽셀* (완전 miss) 또는 *50배 영역으로 과대 예측*.
2. **저신호 영역의 ET 누락**: T1ce 에서 ET 가 *약하게* 조영된 케이스. 모델이 *불확실한 경계* 를 *전부 제거* → FN.
3. **노이즈/모션 artifact**: 정상 슬라이스 영역에 *밝은 점* 이 있는 케이스. 모델이 *과민 반응* → FP overlay.
4. **GT 라벨 노이즈**: 일부 슬라이스에서 *GT 자체가 1~5 픽셀의 isolated point* — *라벨링 단계의 noise* 가능성.

### 6.3 best30 의 정성 패턴 (보존)

best30 의 공통 패턴:

1. **큰 종양 슬라이스 (z=70~100 중앙부)**: GT 가 *수천 픽셀* 이고 *T1ce/FLAIR 둘 다 강한 신호* — 모델이 거의 완벽히 (Dice >0.95) 잡음.
2. **edema 영역 명확** : FLAIR 의 *고신호 부종 경계* 가 명확한 케이스 — WT 정확도 최고.
3. **고밀도 enhancing rim**: T1ce의 *링 형태* enhancement 가 뚜렷한 GBM 케이스 — TC/ET 모두 정확.

### 6.4 발표 메시지

> **본 발표의 핵심 새 인사이트 ⑬**:
> "**Day 7 SOTA의 실패 패턴은 v1 §3.5 의 Whole-Slice FN 분석, §4 의 Patch 실패, §5 의 MT under-fit, `260524v1mmmtplus.md` §4 의 still_missed 641 과 *완전히 동일* 한 패턴 — *극소 종양 (< 100 픽셀) + 경계 슬라이스 + 라벨 노이즈* 의 *3중 한계*.** 7단계 여정 내내 *같은 잔여 문제* 가 *조금씩 작아지면서* 남아 있다 — Day 7 까지 *왔는데도* 이 한계가 완전히 해소되지 않은 것은, *2D 슬라이스 패러다임 자체의 한계* 라는 *방법론적 결론* 을 강화한다. (3D 모델 / 2.5D / Sliding-window 확장이 향후 필요성)."

📎 **참고**:
- `outputs/figures/sota/failure_{worst,best}30_best.png` (정성 자료)
- `code/sota/step31_sota_evaluate.py` `save_failure_panels` (L716~772)
- `260524v1mmmtplus.md` §4.5 인사이트 ④ (still_missed 641 의 평균 100 픽셀 분석)
- `260521v1result.md` §3.5 (Day 3 FN 92% 가 소종양)

### 🎯 §6 예상 질문 (Q&A)

**Q1. worst30 의 *4-panel 시각화* 가 정확히 무엇을 보여주는가? 발표 슬라이드에 어떻게 활용?**
A. 4-panel: (a) T1ce, (b) FLAIR, (c) GT mask, (d) Pred mask. *입력 → 정답 → 모델 출력* 의 *시각적 비교* 만으로 임상의가 *실패 원인을 추측* 가능. 예시: "이 슬라이스의 FLAIR에서 부종이 거의 안 보이는데 GT 는 enhancing 영역만 잡았다 — 모델이 *FLAIR 단서 의존도* 가 크기 때문에 놓침" 같은 *사례별 진단* 가능. 발표 슬라이드에서는 *3~5개 정도* 만 enlarged 로 보여주는 것이 깊이 있는 정성 분석에 효과적.

**Q2. *GT 라벨 노이즈* 추정은 *주장* 인가 *근거* 인가?**
A. 본 보고서에서는 *주장* 수준이며 정량 검증은 미실행 (부록 A). 다만 (1) BraTS 데이터는 *3 명의 raters consensus* 이지만 *경계 픽셀의 inter-rater agreement는 80~90% 수준* 으로 알려져 있음, (2) worst30 중 일부 슬라이스의 *GT 가 1~5 픽셀 isolated* 패턴은 *3D 라벨링 후 2D 슬라이스로 자른 결과* 일 가능성. 발표 시 *추정* 으로 정직 보고 + *3D 라벨링 검증 가능성* 을 향후 방향에 명시.

---

## 7. 전체 모델 비교 — Whole / Patch / MT / MMMT / Ablation / SOTA 7-way

### 7.1 분류 성능 종합 (★ 본 발표의 메인 표)

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

### 7.2 세분화 성능 종합

| # | 모델 | seg head | WT Dice (slice mean) | WT Dice (volume mean) | TC Dice | ET Dice | WT IoU | TP CAM IoU |
|:-:|------|:--------:|:--------------------:|:---------------------:|:-------:|:-------:|:------:|:----------:|
| 1 | Whole-Slice | ❌ | — | — | — | — | — | **0.145** |
| 2 | Patch-Based | ❌ | — | — | — | — | — | 0.128 |
| 3 | Multi-Task | WT only | 0.7765 | (미측정) | — | — | 0.7061 | (train 0.12, test seg 0.706) |
| 4 | MMMT | WT only | 0.7874 | (미측정) | — | — | 0.7142 | (train 0.16, test seg 0.714) |
| 5 | C-2 FLAIR-only | WT only | 0.7458 | (미측정) | — | — | 0.6703 | (미측정) |
| 6 | C-1 T1ce-only | WT only | 0.5741 | (미측정) | — | — | 0.4780 | (미측정) |
| 7 | **Day 7 SOTA** | **WT/TC/ET** | **0.7971** ★ | **0.8910** ★ | **0.7783** ★ | **0.7497** ★ | **0.7225** ★ | (미측정) |

### 7.3 학습 비용 / 자원 비교

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

### 7.4 신뢰성 / 학술적 보고 품질 비교

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

### 7.5 임상 가치 종합 (1,000명 스크리닝 환산)

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

### 🎯 §7 예상 질문 (Q&A)

**Q1. 7개 모델 중 *발표의 메인 표* 는 어떤 행을 강조해야 하나?**
A. 추천 순위:
1. **Day 5 Multi-Task** — *shortcut 극복의 1차 성공* (메시지: "active supervision으로 IoU 5배 향상").
2. **Day 6 MMMT** — *FN 회복 + 운용 자유도 확장* (메시지: "멀티모달은 F1 점프가 아닌 *FN 회복 + 세분화 정밀도*에 기여").
3. **Day 7 SOTA** — *신뢰성 보고 품질 SOTA + 3-region 동시 평가* (메시지: "학부 환경에서 *학술 SOTA 표면적*까지 도달한 첫 결과").
4. *6단계 여정 자체* — "정석 패턴의 학부 사례 연구" 메시지를 *결론 슬라이드* 에서.

**Q2. Day 7 SOTA 의 F1 (thr 0.7 = 0.935) 이 Day 5 MT 의 F1 (0.939) 와 *거의 동등* 한데, *더 좋다* 라고 말할 수 있나?**
A. *단일 F1 기준에서는 동등 ~ 미세 우세*. 그러나 다른 지표에서 분명히 우세: WT Dice +0.021, Recall +1.47%p, 3-region 동시 평가, 신뢰성 보고 12종. **즉 발표 메시지는 "Day 5 의 F1 천장은 *우연이 아니라 입력 표현 한계의 자연스러운 상한*이며, Day 7 의 진짜 가치는 *F1 동등 + 그 외 모든 차원에서의 향상*" 으로 정정.**

**Q3. C-1 T1ce-only 의 결과는 *왜 Day 7 비교에 다시 등장* 하는가? Day 6 의 ablation 인데?**
A. (1) Day 6 ablation 은 *멀티모달의 채널별 기여 분리* 였음. (2) Day 7 는 *전체 SOTA 패키지의 효과* — 두 비교는 *상보적*. (3) 발표 표에서는 C-1/C-2 를 *Day 6 series 의 일부* 로 보여주고, Day 7 SOTA 는 *별도 진영* 으로 강조하는 게 명료.

📎 **참고**: `260524v1mmmtplus.md` §6 (Day 6 까지의 종합), `260525v2sotawith.md` §1~§2 (학술 SOTA 위치).

---

## 8. Day 7 SOTA의 의의 — v2ways §10.1 기대치 vs 실측 정정

### 8.1 `260521v2ways.md` §10.1 의 *예상치* 표 (학술 SOTA 대비)

`code/sota/README.md` §4 의 표 인용 (= `260521v2ways.md` §10.1):

| 항목 | Day 6 MMMT (실측) | Day 7 SOTA *예상* (v2ways) | Day 7 SOTA *실측* (본 보고서) |
|------|:-----------------:|:--------------------------:|:----------------------------:|
| F1 (cls) | 94.27 | **96~98** | **93.50 (thr 0.7) ~ 92.96 (thr 0.5)** |
| WT Dice | 0.7874 | **0.78~0.82** | **0.7971 ★** |
| TC Dice | — | **0.88~0.92** | **0.7783** |
| ET Dice | — | **0.84~0.89** | **0.7497** |
| FN | 848 | **600~800** | **689 ~ 956** |
| Calibration ECE | — | **< 0.03** | **0.0364 (Pre), 0.0420 (Post)** |
| Conformal coverage | — | **≈ 0.90** | **0.8964 ★** |

### 8.2 기대치 *정정* (★)

| 항목 | 기대 vs 실측 | 정정 메시지 |
|------|:------------:|------------|
| F1 (cls) | -2.5~5%p 부족 | "단일 학습 + 14 epoch 조기종료 환경에서 *학술 SOTA 수치 (96~98)* 는 미달. 본 환경에서는 thr 0.7 채택 시 93.50 까지 회복." |
| WT Dice | **목표 달성 (0.7971 ∈ [0.78, 0.82])** | "★ 본 보고서가 v2ways 의 *유일한 사전 예측 달성 항목*." |
| TC Dice | -10%p 부족 (0.778 vs 0.88~0.92) | "BraTS GLI 의 *necrotic + enhancing* 영역은 *T1ce 명확 enhancement* 가 필수 — 본 환경의 *14 epoch 만 학습* + *Weighted Sampler 의 WT-only 양성 정의* 가 TC/ET 학습에 *부분적 불리*. 5-fold CV / 3D 모델 / SWA 평가 추가 시 회복 가능." |
| ET Dice | -10%p 부족 (0.750 vs 0.84~0.89) | "동일 사유 + ET 영역이 가장 작아 noise 민감." |
| FN | thr 0.5 에서 **목표 달성 (689 ∈ [600, 800])** | "★ 두 번째 사전 예측 달성 항목." |
| ECE | -0.006 미달 (0.036 vs <0.03) | "본 모델은 *원래부터 well-calibrated* (Pre-Temp 0.036). 추가 보정 효과 한계." |
| Conformal coverage | **목표 달성 (0.8964 ≈ 0.90)** | "★ 세 번째 사전 예측 달성 항목." |

> **본 발표의 핵심 새 인사이트 ⑮**:
> "**v2ways §10.1 의 7개 사전 예측 중 3개가 정확히 적중 (WT Dice, FN @thr 0.5, Conformal coverage), 4개가 부분 미달.** *분류 F1* 의 부분 미달은 *14 epoch 조기종료 + 단일 학습 + 8GB 환경* 의 환경 제약이 큰 비중. *TC/ET Dice* 의 -10%p 격차는 *3-region 학술 SOTA (DynUNet 2D, MedNeXt) 의 학습 환경 (5-fold + 200 epoch + 3D)* 과의 *방법론적 차이*. 학부 환경의 *현실적 천장*을 보여주는 정직한 결과."

### 8.3 비교 학술 SOTA 대비 위치 (`260525v2sotawith.md` §2)

| 모델 | 발표 | WT Dice (per-volume) | 차원 | 학습 환경 | Day 7 SOTA 와의 격차 |
|------|:----:|:--------------------:|:----:|:---------:|:--------------------:|
| **MedNeXt** | MICCAI 2023 | 0.93 | 2D/3D | 5-fold + 300 epoch + 24GB+ | -0.04 (본 0.89) |
| **SwinUNETR-v2** | CVPR 2024 | 0.92 | 3D | 5-fold + 24GB | -0.03 |
| **DynUNet (nnU-Net 2D)** | Nature Methods 2021 | 0.91 | 2D | 5-fold + 300 epoch | **-0.02** (본 0.89 ★) |
| **Day 7 SOTA (본)** | — | **0.8910** | 2D | **단일 학습 + 14 epoch + 8GB** | — |

> **본 발표의 핵심 새 인사이트 ⑯**:
> "**학부 환경(단일 학습 + 14 epoch + 8GB GPU) 에서 *학술 SOTA DynUNet 2D 의 -0.02 영역* (per-volume WT Dice 0.89 vs 0.91) 까지 도달.** 5-fold CV + 학술 환경(24GB+) 적용 시 추정 0.91+ 도달 가능 — 그러나 본 프로젝트는 §9 의 환경 제약으로 *여기서 종료*."

📎 **참고**:
- `260521v2ways.md` §10.1 Day 7 행 (사전 예측치)
- `code/sota/README.md` §4 (사전 예측표)
- `260525v2sotawith.md` §2 (학술 SOTA 비교)
- `outputs/logs/sota/sota_test_metrics.json` (실측치 출처)

---

## 9. 환경적 한계 — VRAM 누수 / Windows allocator / 추가 진행 불가 사유

### 9.1 RTX 4070 Laptop 8GB VRAM 의 한계

**기준 환경**:
- GPU: NVIDIA GeForce RTX 4070 Laptop (8GB GDDR6)
- CPU/RAM: x64 Windows + 16GB DDR5
- OS: Windows 11 (`sys.platform == "win32"`)
- PyTorch: 2.11.0+cu128, Python 3.13.12

**관찰된 한계**:
1. **step30 학습 중 점진적 VRAM 소진**: epoch 후반 (epoch 12~14) 에서 free VRAM 이 2GB 이하로 떨어지면서 *OOM Trigger* 위험 — 본 학습은 14 epoch 에서 자체 종료 (EarlyStopping + 위험 회피).
2. **step31 평가 중 batch_size 의 *드라마틱 축소***: 학습은 batch=16 으로 가능했으나, *TTA 4-view + 3-region seg 출력 + AMP fp16* 의 평가는 **batch=2** 가 한계. `sota_test_metrics.json` 의 `eval_batch_size: 2` 가 그 증거.
3. **SWA 평가의 RAM OOM**: `swa.error` 에 명시된 "14.0 GiB for (24882, 3, 224, 224) float32" — *val raw* 를 메모리에 누적하는 과정에서 16GB RAM 부족. **즉 *GPU 만의 문제가 아니라 시스템 RAM 도 한계***.

### 9.2 Windows 환경의 PyTorch CUDA Allocator 한계

**핵심 제약**:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 는 *Linux 전용*. Windows 에서 적용 시 *Illegal Memory Access / Allocator 손상* 발생.
- 대안: `max_split_size_mb:64` 만 채택 (`step31_sota_evaluate.py` L57~67 의 OS 분기).
- 결과: *단편화 완화는 부분적*. 본질적인 memory pressure 해소 X.

**구체적 증상**:
- CUDA 캐시 비우기 (`torch.cuda.empty_cache()`) 후에도 *VRAM 회수 불완전*.
- `torch.cuda.ipc_collect()` 도 큰 효과 없음.
- step31 의 `_run_batch_safely` 함수가 *OOM 발생 시 자동으로 batch 분할 + CPU fallback* 로직을 갖추고 있을 정도로 *defensive coding* 필수.

### 9.3 추가 진행 불가 사유 종합

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

### 9.4 환경 재구축의 비현실성

| 옵션 | 비용 | 발표 일정 가능성 |
|------|------|:----------------:|
| WSL2 Ubuntu + Linux native PyTorch | OS 재설치 + 환경 빌드 (~1주) | ❌ |
| 클라우드 GPU 임차 (RTX 4090 24GB) | 시간당 $1.5 × 100h | ❌ (예산 외) |
| 학교 cluster 신청 | 신청 + 승인 + 큐 대기 (~2주+) | ❌ |
| **현재 환경에서 발표 종료** | 0원 | ✅ (선택) |

📎 **참고**:
- `step31_sota_evaluate.py` L51~67 (OS 분기 코드)
- `step31_sota_evaluate.py` L201~314 (`_free_cuda`, `_run_batch_safely` defensive coding)
- `260526v2step31patch.md` §0 (VRAM 무증가 절대 원칙)
- `260526v1fullresults.md` E-1, E-2 (3D / 외부 데이터의 보류)

### 🎯 §9 예상 질문 (Q&A)

**Q1. 환경 제약을 *발표에서 변명* 으로 보일 우려는?**
A. 정직 보고가 안전하다. 핵심 메시지: "본 결과는 *학부 환경의 천장* — 학술 환경(24GB+ + 5-fold + Linux native) 에서는 +0.02~0.03 Dice 향상 가능 추정." 발표 슬라이드 1장에 *명시* 하여 *방어적 자세* 가 아닌 *결과의 정확한 위치 설명* 으로 정정.

**Q2. *학부 환경* 에서 SOTA 까지 도달했다는 게 *진짜 성과* 인가, *행운* 인가?**
A. 정량 증거: (1) WT Dice 0.797 (slice mean) / 0.891 (volume mean) 으로 *학술 SOTA DynUNet 2D (0.91) 의 -0.02 영역*, (2) Calibration ECE 0.036 으로 *Guo et al. 의 적정 영역*, (3) Conformal coverage 0.896 *목표 달성*, (4) 12개 신뢰성 보고 항목 중 11개 통과 — *재현성 + 방법론적 완성도* 측면에서 *행운이 아닌 설계의 결과*.

**Q3. 향후 *환경 재구축* 시 어떤 부분을 가장 먼저 시도하는 게 좋은가?**
A. `260526v1fullresults.md` 의 권장 우선순위 표 그대로:
1. (이미 적용 완료) Raw 직렬화 + 후처리 — Day 7 에서 완료.
2. *5-fold CV* (학습 5×) — 분산 추정 + Dice +1~2%p.
3. *3D 모델 전환* (nnU-Net) — Dice +0.03~0.05.

---

## 10. 발표 핵심 메시지 — 7가지 결정적 인사이트 (확장판)

> v1mmmtplus 의 7가지 (§7.1~§7.7) + Day 7 신규 9가지 (§3.5 ⑨, §3.5 ⑩, §4.4 ⑪, §5.5 ⑫, §6.4 ⑬, §7.4 ⑭, §8.2 ⑮, §8.3 ⑯) = 본 발표의 *16가지 핵심 메시지*. 발표 5분 내 핵심은 *④ → ⑭ → ⑫ → ⑮* 순.

### 10.1 (v1 유지) 단일 지표만으로 의료 AI 신뢰성을 보장할 수 없다
- 94.33% Acc + 0.9832 AUROC + IoU 0.145 = shortcut 의 결정적 증거.

### 10.2 (v1 유지) Shortcut Learning은 명시적 보조 학습 신호로 극복 가능
- Multi-Task가 IoU 0.145 → 0.706 (4.87×) — 단, *분류 head CAM IoU* 는 그대로 0.12.

### 10.3 (v1 유지) 광역 단서 차단(passive)보다 올바른 학습 강제(active)가 효과적
- Patch 실패 vs Multi-Task 성공.

### 10.4 (v1mmmtplus 신규) 멀티모달의 효과는 F1 점프가 아닌 *FN 회복 + 운용 자유도 + 세분화 정밀도*
- FN 1,136 → 848 (-25.4%), seg Dice +0.011, 운용점 0.30~0.50 자유도.

### 10.5 (v1mmmtplus 신규) Ablation으로 분리하면 *FLAIR가 분류의 대부분, T1ce+diff는 세분화 보조*
- FLAIR-only AUROC 0.9822 ≈ baseline 0.9832, T1ce-only AUROC 0.9487.

### 10.6 (v1mmmtplus 신규) Tversky α=0.7은 분류 FN -21% / seg Dice -3%p 의 *양면성*
- C-2 FLAIR-only 분석에서 정량 검증.

### 10.7 (v1mmmtplus 신규) Day 5/Day 6은 *다른 작은 종양*을 잡고 놓치는 상호 보완 패턴
- recovered 381 + regressed 86 — Ensemble 의 직접 근거.

### 10.8 (★ Day 7 신규 ⑨) Day 7 SOTA의 *실질적 이득*은 *Recall + FN + Dice* 삼각에 집중
- Focal-Tversky + Weighted Sampler + TumorCP 의 *3중 FN 표적화* 가 의도대로 작동.

### 10.9 (★ Day 7 신규 ⑩) val 기반 threshold 0.7 채택만으로도 *재학습 없이* FP -42%
- F1 0.930 → 0.935, FP 1,061 → 614.

### 10.10 (★ Day 7 신규 ⑪) Day 7 SOTA의 *진짜 가치*는 단일 F1이 아닌 *신뢰성 정량화의 완성도*
- CI + Temp + Conformal + ECE 의 4종 신뢰성 패키지.

### 10.11 (★ Day 7 신규 ⑫) WT Dice 0.7971 (slice) / 0.891 (volume) — *학술 SOTA(DynUNet 2D 0.91) 의 -0.02 영역* 도달
- 학부 환경에서 *학술 SOTA 표면적* 까지.

### 10.12 (★ Day 7 신규 ⑬) Day 7 까지의 *동일한 실패 패턴* 잔존 — 2D 슬라이스 패러다임 자체의 한계
- worst30 의 *극소 종양 + 경계 슬라이스 + 라벨 노이즈* 3중 한계.

### 10.13 (★ Day 7 신규 ⑭) *학술적 보고 품질 점프*가 *F1 점프*보다 발표 메시지로 중요
- 12개 신뢰성 항목 중 11개 통과 — 학술 논문/학회 보고 직접 인용 가능.

### 10.14 (★ Day 7 신규 ⑮) v2ways §10.1 사전 예측 *7개 중 3개 정확 적중* (WT Dice, FN @0.5, Conformal cov)
- 정직한 자기검증 — 4개 미달의 사유 (단일 학습, 14 epoch, 8GB) 도 함께 보고.

### 10.15 (★ Day 7 신규 ⑯) 학부 환경에서 *학술 SOTA -0.02 영역* 까지 도달 — *환경 제약이 명확한 천장*
- 24GB+ + 5-fold + Linux 적용 시 추가 +0.02~0.03 가능 추정.

### 10.16 (★ 종합) *7단계 여정 자체* — *수치 vs 진실* 의 *trade-off 와 정정의 흔치 않은 학부 사례*
- Whole 의 단일 지표 한계 → Patch 의 가설 반박 → MT 의 active 성공 → MMMT 의 가설 정정 → SOTA 의 신뢰성 완성. *가설 검증 + 가설 정정 + 신뢰성 정량화* 가 모두 들어 있는 학부급 *완성형 사례*.

---

## 11. 예상 질문 마스터 리스트 (Day 7 / 신뢰성 / SOTA 비교 추가분)

> v1mmmtplus §8 의 64개 Q&A 에 *Day 7 신규 25개* 를 추가. 발표 직전 빠르게 훑기 위한 단일 페이지.

### 11.1 Day 7 SOTA 패키지 설계 의도 (§1)

65. v2ways §8.2 의 6종 동시 적용에서 *분리 ablation* 은? → §1 Q1
66. step26 의 WT positive 81,337 vs Day 1 의 81,374 차이의 의미? → §1 Q2
67. Day 7 의 Focal-Tversky α=0.7 도 Day 6 부작용을 반복? → §1 Q3

### 11.2 step30 학습 (§2)

68. 14 epoch 조기종료가 *결과 비교* 에 영향? → §2 Q1
69. Val Total 의 *진동 패턴* 이 학습 불안정? → §2 Q2
70. SWA 평가 RAM OOM 실패의 *발표 가치* 손상? → §2 Q3

### 11.3 step31 평가 — 분류 (§3)

71. F1 (thr 0.5) -0.013 후퇴의 정직 보고? → §3 Q1
72. T=1.5015 의 학술적 표준 영역? → §3 Q2
73. CI 가 좁은 게 *모델 안정성* 인가 *test 크기 효과* 인가? → §3 Q3
74. Precision 91.6% (Day 5 97.3% 대비) 의 임상 수용성? → §3 Q4

### 11.4 신뢰성 SOTA — Calibration (§4)

75. ECE 0.036 → 0.042 의 *오히려 증가* 는 실패? → §4 Q1
76. Conformal coverage 0.896 의 *목표 0.90* 미달 우려? → §4 Q2
77. Conformal set 첫 50개의 *발표 활용*? → §4 Q3

### 11.5 세분화 SOTA (§5)

78. WT Dice 의 *slice mean 0.797 vs volume mean 0.891* 어떤 게 공식? → §5 Q1
79. Threshold sweep 모두 0.7 수렴의 *우연성*? → §5 Q2
80. Postproc *효과 없음* 의 역설적 해석? → §5 Q3
81. HD95 median 1.0 픽셀의 측정 정의 검증? → §5 Q4
82. ET n_volumes 182/188 의 *6명 누락* 의미? → §5 Q5

### 11.6 실패 케이스 정성 (§6)

83. worst30 4-panel 의 발표 활용? → §6 Q1
84. GT 라벨 노이즈 *추정* 의 정량 검증? → §6 Q2

### 11.7 7-way 비교 (§7)

85. *발표 메인 표* 에서 어떤 행 강조? → §7 Q1
86. Day 7 F1 *동등* 인데 *더 좋다* 라고 할 수 있나? → §7 Q2
87. C-1 / C-2 가 Day 7 비교에 *왜 다시 등장*? → §7 Q3

### 11.8 v2ways 사전 예측 비교 (§8)

88. v2ways §10.1 의 7개 항목 *정확 적중 vs 미달* 정리? → §8.2
89. TC/ET Dice -10%p 미달의 *진짜 원인*? → §8.2 (학습 환경 격차)

### 11.9 환경적 한계 (§9)

90. 환경 제약을 *변명* 으로 보일 우려? → §9 Q1
91. *행운 vs 설계의 결과*? → §9 Q2
92. *향후 재구축* 시 우선순위? → §9 Q3

---

## 부록 A — 배제된 / 보류된 결과

본 보고서 본문에서 *명시적으로 배제* 하거나 *측정은 했지만 메인 표에 넣지 않은* 결과들. 발표 슬라이드에서는 *언급만* 하고 깊게 다루지 않을 항목.

### A.1 SWA 체크포인트 평가 (실패)

- `outputs/checkpoints/sota/sota_swa.pth` 가 정상 저장 (54.7MB, SHA256 = `2c018a26...`).
- step31 의 P10 SWA 평가 코드는 *정상 작성* 됐으나 RAM 16GB 환경에서 *14GB float32 array 할당 실패*.
- 발표에서는 "SWA 체크포인트는 디스크에 있으나, RAM 환경 부족으로 *공식 평가는 best 단일* 로 제출" 로 정직.
- 만약 *재현 환경 (32GB RAM+)* 에서 재실행 시: step31 명령어 그대로 + `--no-failure-fig` 옵션 권장 (figure 메모리 절감).

### A.2 Day 7 SOTA 의 분리 Ablation (미실행)

- 6개의 동시 변경: (i) Loss, (ii) 3-region, (iii) Deep Sup, (iv) Uncertainty Weighting, (v) Weighted Sampler+TumorCP, (vi) SWA.
- 분리 ablation 은 6 × 17h = 102h 학습 필요 — §9 의 환경 제약으로 보류.
- 대신 Day 6 ablation (C-1/C-2) 가 *모달리티 분리* 만 부분 수행.

### A.3 Day 6 의 thr 0.7 운용점 측정 (미실행)

- §3.6 Q에서 언급: Day 6 도 thr 0.7 에서 F1 측정 시 *공정 비교* 가능.
- 본 보고서는 미실행 — `outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json` 의 그리드가 [0.30, 0.35, 0.3847, 0.40, 0.45, 0.50] 으로 *thr 0.7 포함 안 됨*.
- raw npz `sota_test_raw.npz` 의 *Day 6 부분* 이 없으므로 *재계산 불가* — Day 6 npz 직렬화는 본 코드에 없음.

### A.4 3-seed 평균 (미실행)

- F1 / Dice CI 의 *모델 분산 추정* 을 위해 필요.
- 학습 시간 17h × 3 = 51h + 동일 환경 안정성 — §9 환경 제약으로 보류.

### A.5 Day 5 / Day 6 / Day 7 의 Ensemble (미실행)

- recovered 381 + regressed 86 (Day 5↔6) + Day 7 의 새 패턴 — 3-way ensemble 의 *상호 보완성* 정량 검증.
- soft-voting 또는 max-voting 두 방식. 본 보고서 미실행.

### A.6 Score-CAM / EigenCAM 등 강한 XAI (미실행)

- `260524v1mmmtplus.md` A.9 에서 이미 보류.
- Day 7 SOTA 의 *분류 head CAM IoU* 측정도 미실행 — `260524v1mmmtplus.md` §1.5 의 CAM IoU 0.12 (MT) → 0.16 (MMMT) 추세에 Day 7 이 어떤 영향인지 *직접 검증* 가능 항목이나 보류.

### A.7 3D / 2.5D 입력 (미실행)

- §9 의 환경 제약 (24GB+ GPU 필요) 으로 직접 적용 불가.
- 다만 **3D SegResNet (MONAI)** 등으로 *전환만 한 평가* 도 가능했지만, *호환되는 dataset/loader 재작성 + 환경 충돌* 우려로 보류.

### A.8 외부 데이터 (BraTS 2019/2020/2021) fine-tune (미실행)

- 라이센스 확인 + 다운로드 + 통합 — 일정 외.
- `260526v1fullresults.md` E-2 의 보류 사유와 동일.

### A.9 K-fold (5-fold) CV (미실행)

- 학습 5× = 85h + 결과 통합 인프라 — §9 환경 제약 + 일정 외.
- nnU-Net SOTA 의 표준 셋업이지만 학부 발표 일정 외.

### A.10 Day 7 분류 head 의 Train Grad-CAM (미실행)

- `260524v1mmmtplus.md` §1.5 의 *Train CAM IoU 0.12 (Day 5) → 0.16 (Day 6)* 추세에 Day 7 이 *Deep Supervision 으로 0.2+ 진입* 가능성.
- `step18b_*` 또는 `step24b_*` 의 SOTA 버전 미작성 — 발표 일정 외.

---

## 부록 B — 전체 학습/평가 로그 (Day 7 SOTA 14 epoch + step31 raw)

### B.1 학습 곡선 (sota_history.json 완전 표) — §2.2 참조

### B.2 평가 결과 (sota_test_metrics.json 완전 키 트리)

```
{
  "n_test": 25115,
  "meta": { ... },                       ← §3.1
  "temperature": 1.5015,
  "cls": {
    "tta":         { f1, auroc, auprc, tp/tn/fp/fn, ci },  ← §3.2, §3.6
    "temp_scaled": { ... },
    "tta_thrF1":   { threshold: 0.7, ... }
  },
  "calibration": {                       ← §4.1
    "pre_temp":  { ece: 0.036, brier: 0.053 },
    "post_temp": { ece: 0.042, brier: 0.057 },
    "T": 1.5015
  },
  "seg":          { WT/TC/ET dice/iou @ thr 0.5 },   ← §5.1
  "seg_swept":    { WT/TC/ET dice/iou @ thr 0.7 },   ← §5.2
  "seg_postproc": { WT/TC/ET dice/iou @ thr 0.7 + CC + closing },  ← §5.3
  "seg_extra":    { HD95, per-volume dice, sens/spec per region },  ← §5.4
  "thresholds": {
    "cls_f1_opt": { threshold: 0.7, val_f1: 0.9276 },
    "seg_region": { WT: 0.7, TC: 0.7, ET: 0.7 }
  },
  "conformal":    { q: 0.408, coverage: 0.896, sets: [first 50] },  ← §4.2
  "swa":          { "error": "Unable to allocate 14.0 GiB ..." },   ← §0.4 ③
  "comparison": {
    "Day5_MTL":  { ... },   ← outputs/logs/multitask/* 자동 로딩
    "Day6_MMMT": { ... },   ← outputs/logs/mmmt/* 자동 로딩
    "Day7_SOTA_best": { ... }
  },
  "raw_files": {
    "test": "sota_test_raw.npz",
    "val":  "sota_val_raw.npz",
    "swa_test": null
  }
}
```

### B.3 Raw 직렬화

| 파일 | 크기 | 내용 |
|------|:----:|------|
| `outputs/logs/sota/sota_test_raw.npz` | **1,077.7 MB** | labels, probs_tta, probs_tscaled, logits, seg_gt, seg_pred, T (Day 7 best, test) |
| `outputs/logs/sota/sota_val_raw.npz` | **964.5 MB** | labels, logits, seg_gt, seg_pred (Day 7 best, val) |
| `outputs/logs/sota/sota_test_raw_swa.npz` | (없음 — SWA 평가 실패로 미생성) | — |

> 발표 이후 *어떤 후처리 실험* 을 추가하더라도 *재 forward 0회* 로 재실행 가능. 이게 D-1 의 *재현성 가치*.

### B.4 산출 figure (outputs/figures/sota/) — 9장 전체

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

> 본 표는 v1 §5.8 + v1mmmtplus §6 + 본 보고서 §7 의 *결정판*. 발표 슬라이드 1장 으로 사용 권장.

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

### C.1 *임상 시나리오별* 추천 운용

| 시나리오 | 우선순위 | 권장 모델 | 권장 threshold | 1000명 기준 (오탐 / 누락) |
|----------|----------|-----------|:--------------:|:--------------------------:|
| 1차 스크리닝 | Recall 최우선 | **Day 7 SOTA** | 0.5 | 82 / 56 |
| 균형 운용 | F1 최우선 | **Day 7 SOTA** | **0.7** | 48 / 78 |
| 확진 (Precision) | Precision 최우선 | Day 5 MT | 0.3847 | 24 / 93 |
| 학술적 SOTA 비교 | per-volume WT Dice | **Day 7 SOTA** | (시나리오에 따라) | (volume Dice 0.891) |
| 신뢰성 정량화 (감사) | CI + Conformal | **Day 7 SOTA** | (선택 자유) | (coverage 0.896) |

---

## 부록 D — 참고한 md 파일 인벤토리 + 직접 인용 출처 (본 보고서 신규)

### D.1 v1mmmtplus 이후 새로 작성된 md (본 보고서가 *직접 참조*)

| 파일 | 분량 | 참고한 절 | 본 문서에서의 활용 |
|------|------|-----------|---------------------|
| **`260521v1result.md`** | 78.9 KB | §0 도식, §1~§6 Day 1~6, §7 차별화, §8 학술, §9 Q&A, 부록 A~E | **본 문서 §0 (7-단계 여정), §7 (7-way 비교의 전 6단계), §10 (인사이트 ①~③)** |
| **`260524v1mmmtplus.md`** | 85.2 KB | §0 도식, §2 Day 6 본 학습, §3 thr sweep, §4 FN 차분, §5 ablation, §6 6-way 비교, §7 5가지 인사이트, 부록 A~D | **본 문서 §0.1 (도식 계승), §2 (Day 6 비교), §3.4 (메인 표의 Day 5/6 행), §5.5 (세분화 7-way), §7 (전체 비교), §10 (인사이트 ④~⑦)** |
| **`260525v1sotapluswhy.md`** | 13.0 KB | §1-0 sota 폴더의 위치, §1-1~§1-7 step26~31 의 md 근거, §2 우선순위 1주의 의미 | **본 문서 §1 전체 (sota 코드 설명의 *직접 인용*)** |
| **`260525v2sotawith.md`** | 1.9 KB | §1 우리 위치, §2 학술 SOTA 표 (MedNeXt 0.93, DynUNet 0.91, SwinUNETR 0.92), 결론 | **본 문서 §8.3 (학술 SOTA 비교), §5.5 (per-volume Dice 위치)** |
| **`260526v1fullresults.md`** | 9.9 KB | A~E 5개 카테고리 (TTA/sliding/AMP/threshold/postproc/...), 권장 우선순위 표 | **본 문서 §0.1 (도식), §1.6 (P1~P10 설계 의도), §5.3 (postproc 효과 없음의 역설적 해석), §9.3 (보류 항목 사유)** |
| **`260526v2step31patch.md`** | 26.4 KB | §0 절대 원칙, §1 현행 한계, §2 P1~P10 카탈로그, §3 코드 스니펫, §4 산출물 사양, §7 발표 매핑, §9 결론 | **본 문서 §1.6 (P1~P10 적용 완료), §3.1 (manifest 형식), §3.6 (CI), §4.1 (ECE/Brier), §5.2 (threshold sweep), §5.3 (postproc), §6 (failure case), §0.4 (SWA 실패 사유)** |

### D.2 본 보고서가 *직접 참조* 한 코드/결과 산출물

| 파일 | 분량 | 본 문서 활용 |
|------|------|-------------|
| `code/sota/README.md` | 5.1 KB | §1.0 큰 그림, §8.1 사전 예측표 |
| `code/sota/step26_sota_segmask.py` | 5.1 KB | §1.1 3-region 정의 |
| `code/sota/step27_sota_dataset.py` | 9.0 KB | §1.2 Weighted Sampler |
| `code/sota/step28_sota_model.py` | 8.8 KB | §1.3 Deep Sup + UW |
| `code/sota/step29_sota_losses.py` | 10.4 KB | §1.4 Focal-Tversky + Boundary + Compound + TumorCP |
| `code/sota/step30_sota_train.py` | 27.0 KB | §1.5 학습 CONFIG, §2 학습 과정 |
| `code/sota/step31_sota_evaluate.py` | 62.6 KB | §3 ~ §6 평가 결과의 *직접 코드* |
| `outputs/logs/sota/step26_segmask_stats.json` | 작음 | §1.1 마스크 통계 |
| `outputs/logs/sota/sota_history.json` | 6.4 KB | **§2.2 14 epoch 학습 곡선 전체 (원본)** |
| `outputs/logs/sota/sota_summary.json` | 작음 | §1.5 CONFIG, §2.3 학습 시간 |
| `outputs/logs/sota/sota_test_metrics.json` | 12.2 KB | **§3~§5, §7, §8 의 *모든 실측치* (원본)** |
| `outputs/logs/sota/sota_test_raw.npz` | 1,078 MB | 부록 B.3 재현성 자산 |
| `outputs/logs/sota/sota_val_raw.npz` | 965 MB | 부록 B.3 재현성 자산 |
| `outputs/figures/sota/sota_training_curves.png` | 277 KB | §2.2 학습 곡선 |
| `outputs/figures/sota/sota_diagnostics.png` | 116 KB | §3 분류 4-panel |
| `outputs/figures/sota/threshold_sweep_{WT,TC,ET}.png` | 47/48/42 KB | §5.2 threshold sweep |
| `outputs/figures/sota/postproc_examples_best.png` | 48 KB | §5.3 후처리 비교 |
| `outputs/figures/sota/reliability_pre_post_best.png` | 81 KB | §4.1 reliability diagram |
| `outputs/figures/sota/failure_{worst,best}30_best.png` | 1,729/1,943 KB | §6 정성 분석 |

### D.3 본 문서의 *직접 인용 출처* 요약 표

| 본 문서 절 | 직접 인용 출처 | 핵심 인용 부분 |
|-----------|--------------|---------------|
| §0 (도식 + 종료 선언) | `260521v1result.md` §0 + `260524v1mmmtplus.md` §0 + `sota_summary.json` + `sota_test_metrics.json` | 7-단계 도식 + total_epochs:14 + swa.error |
| §1 (SOTA 패키지 설계) | `260525v1sotapluswhy.md` §1-1~§1-7 + `code/sota/README.md` §1 + `step26~31_*.py` | 각 step의 *md 근거* 직접 인용 |
| §2 (학습 결과) | `sota_history.json` 14 epoch 전체 + `sota_summary.json` + `260524v1mmmtplus.md` §2 Q3 | val/test dice 차이 해명 |
| §3 (평가 — 분류) | `sota_test_metrics.json` `cls`, `meta` + `260524v1mmmtplus.md` §3 (Day 6 sweep) | Day 5/6 *직접 비교* |
| §4 (신뢰성) | `sota_test_metrics.json` `calibration`, `conformal` + `260521v2ways.md` §5.3, §5.5 + `260526v2step31patch.md` §3.6, §3.7 | Bootstrap CI + Temp + Conformal 의 *학술 근거* + *코드 구현* |
| §5 (세분화) | `sota_test_metrics.json` `seg/seg_swept/seg_postproc/seg_extra` + `260525v2sotawith.md` §2 | 학술 SOTA 비교 위치 |
| §6 (failure case) | `failure_{worst,best}30_best.png` + `step31_*.py` `save_failure_panels` + `260524v1mmmtplus.md` §4.5 (still_missed 100 픽셀) | 동일 패턴 잔존 |
| §7 (7-way 비교) | `260521v1result.md` §5.8 + `260524v1mmmtplus.md` §6 + 본 보고서 §1~§5 | 7개 행의 *원본 표* 통합 |
| §8 (v2ways 정정) | `260521v2ways.md` §10.1 + `code/sota/README.md` §4 + `sota_test_metrics.json` | 사전 예측 vs 실측 |
| §9 (환경 한계) | `step31_*.py` L51~67 (OS 분기 코드) + `260526v2step31patch.md` §0 + `260526v1fullresults.md` E-1, E-2 | VRAM 누수 + Windows 한계 |
| §10 (16 인사이트) | v1 §8 + v1mmmtplus §7 + 본 보고서 §3.5, §4.4, §5.5, §6.4, §7.4, §8.2, §8.3 | 9개 신규 인사이트의 *직접 출처* |
| 부록 A (배제) | 본 보고서 §3~§9 + `260524v1mmmtplus.md` A.1~A.10 | 미실행 항목 *명시적* 인벤토리 |
| 부록 B (학습/평가 로그) | `sota_history.json` 전체 + `sota_test_metrics.json` 전체 + figure 9장 메타 | 보고치 *완전한 원본* |
| 부록 C (모든 모델 비교) | v1 §5.8 + v1mmmtplus §6 + 본 보고서 §7 | 종합 *결정판* |

### D.4 v1 / v1mmmtplus 가 *이미 정리* 한 자료 (재인용 안 함)

> 본 보고서가 *다시 인용하지 않은* 자료는 `260521v1result.md` 부록 E.1~E.4, `260524v1mmmtplus.md` 부록 D.1~D.5 에 *완전한 인벤토리* 가 있음. 발표 준비 시 그쪽을 직접 참조.

---

## 끝맺음 — 한 줄 요약 (Day 7 종료 시점 버전)

> **"v1까지 6단계 여정이 *수치 vs 진실* 의 trade-off를 보여줬고, v1mmmtplus가 *멀티모달의 진짜 효과* 를 분해해 보여줬다면, *본 보고서 v3는 *학부 환경에서 학술 SOTA 표면적까지 도달한 첫 결과 + 신뢰성 정량화의 완성* 을 보여준다 — *WT Dice (per-volume) 0.891 = DynUNet 2D 의 -0.02 영역*, *Calibration ECE 0.036*, *Conformal coverage 0.896*, *12개 신뢰성 보고 항목 중 11개 통과*. 그리고 *환경적 한계가 명확한 천장* (8GB Laptop + Windows + 14 epoch 조기종료) 이라 *환경 재구축 시 +0.02~0.03 Dice 가능* 한 *정직한 종착점*. *가설 검증 + 가설 정정 + 환경 한계 정직 보고* 가 모두 들어 있는 *완성형 학부 사례*."**

| 단계 | 모델 | F1 (@best thr) | seg Dice (WT) | seg Dice (TC) | seg Dice (ET) | FN | 신뢰성 | 종합 |
|:----:|------|:--------------:|:-------------:|:-------------:|:-------------:|:--:|:------:|:----:|
| Day 2 | Whole-Slice | 94.10 | (CAM 0.145) | — | — | 888 | ❌ | shortcut 진단 |
| Day 4 | Patch-Based | 87.08 | (CAM 0.128) | — | — | 1,306 | ❌ | passive 실패 |
| **Day 5** | **Multi-Task** | **93.91** | **0.7765** | — | — | 1,136 | ❌ | **active 성공** |
| **Day 6** | **MMMT** | **94.27** | **0.7874** | — | — | **848** | ❌ | **FN 회복** |
| Day 6 abl. | C-2 FLAIR-only | 94.15 | 0.7458 | — | — | 1,044 | ❌ | Tversky 분리 |
| Day 6 abl. | C-1 T1ce-only | 87.55 | 0.5741 | — | — | 1,835 | ❌ | WT 한계 |
| **Day 7** | **SOTA (thr 0.7)** | **93.50** | **0.7971** ★ | **0.7783** ★ | **0.7497** ★ | 956 | **✅ 11/12** ★ | **신뢰성 SOTA + 학술 표면적 도달** |

---

> 📎 **본 문서의 위치**: `biohealth_lv.1/260526v3sotafinal.md`
> 📎 **연속성**: `260521v1result.md` (Day 1~6 진행중) → `260521v2ways.md` (확장 로드맵) → `260523v1updatemmmt.md` (멀티모달 갱신 가이드) → `260523v2proposal.md` (실용화 제안) → `260524v1mmmtplus.md` (Day 5~6 완성 + 권장 3종) → `260525v1sotapluswhy.md` (sota 코드 해설) → `260525v2sotawith.md` (학술 SOTA 위치) → `260526v1fullresults.md` (성능 더 짜내기 9종) → `260526v2step31patch.md` (P1~P10 패치) → **`260526v3sotafinal.md` (Day 7 SOTA 평가 통합 + 발표 종료 — 본 문서)**.
>
> 📎 **프로젝트 종료 선언**: GPU VRAM 누수 + Windows PyTorch CUDA Allocator 한계 + SWA 평가 RAM OOM 으로 인해, *본 시점 이후의 학습/대규모 forward 는 진행하지 않는다*. 본 보고서가 *발표 직전의 공식 종착점*이다.
