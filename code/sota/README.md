# Day 7 — SOTA Package (sota/)

> 베이스라인: `mmmt/` (Day 6 멀티모달 MM-MTL, F1 ~95%, WT 0.74~0.78 예상)
> Day 7 목표: **v2ways.md §8.2 "무료 SOTA 패키지"** 를 Day 6 위에 한 번에 통합
> 가이드: `260521v2ways.md` §2~§5 + §8.2

## 1. Day 7 범위 (v2ways §8.2 그대로)

| # | 항목 | 적용 코드 | v2ways § |
|:-:|------|--------|:---:|
| A | Deep Supervision (decoder aux head 3개) | `step28` `aux_head_d2/d3/d4` | §2.2 |
| B | Focal-Tversky + Boundary Loss + BCE Compound | `step29` `FocalTverskyLoss` / `BoundaryLoss` / `CompoundSegLoss` | §3.2~§3.3 |
| C | Uncertainty Weighting (α/β 자동) | `step28` `log_var_*` + `step29` `SOTAMultiTaskLoss` | §3.5 |
| D | WT/TC/ET 3-region seg head 확장 | `step26` 마스크 + `step28` `seg_classes=3` | §3.4 |
| E | TumorCP (소종양 oversampling) | `step29` `tumor_copy_paste` | §4.3 |
| F | Weighted sampler (소종양 ×3) | `step27` `_build_sampler` | §4.5 |
| G | SWA (Phase 2 후반 5 epoch) | `step30` `AveragedModel` + `SWALR` | §5.2 |
| H | TTA (4-view) | `step31` `predict_with_tta` | §5.1 |
| I | Temperature Scaling (LBFGS) | `step31` `TemperatureScaler` | §5.3 |
| J | Conformal Prediction (간이 marginal) | `step31` `try_conformal` | §5.5 |

## 2. 새 폴더 구조

```
biohealth_lv.1/
├── code/sota/                    ← 본 폴더 (Day 7 코드)
│   ├── step26_sota_segmask.py       : WT/TC/ET 3-region 마스크 생성
│   ├── step27_sota_dataset.py       : 3채널 입력 + 3채널 마스크 Dataset
│   ├── step28_sota_model.py         : Deep Sup + Uncertainty Weighting 모델
│   ├── step29_sota_losses.py        : Focal-Tversky + Boundary + Compound + TumorCP
│   ├── step30_sota_train.py         : 2-Phase + SWA + TumorCP
│   └── step31_sota_evaluate.py      : TTA + Temp Scaling + Conformal + 다중 비교
│
├── processed/sota/               ← 새 데이터 (step26 결과)
│   ├── seg_masks_wt/                : WT 마스크
│   ├── seg_masks_tc/                : TC 마스크
│   └── seg_masks_et/                : ET 마스크
│       (T1ce 슬라이스는 processed/mmmt/t1ce_slices/ 재사용)
│
└── outputs/sota/                 ← 새 결과 (실행 시 자동 생성)
    ├── checkpoints/                 : sota_best.pth, sota_last.pth, sota_swa.pth
    ├── figures/                     : sota_training_curves.png, sota_diagnostics.png
    └── logs/                        : sota_history.json, sota_summary.json,
                                       sota_test_metrics.json, step26_segmask_stats.json
```

## 3. 실행 명령어 (Day 6 mmmt 완료 후)

> 사전: Day 6 (`code/mmmt/step20_mmmt_preprocess.py`) 가 실행되어
> `processed/mmmt/t1ce_slices/` 가 존재해야 합니다.

```bash
# (선택) Day 7 추가 라이브러리
pip install scipy seaborn
# (선택) Conformal Prediction 정식 라이브러리
# pip install mapie==0.8.3

# ── 1) WT/TC/ET 3-region 마스크 (≈ 8~12분)
python code/sota/step26_sota_segmask.py
#   → processed/sota/seg_masks_{wt,tc,et}/*.png

# ── 2) Dataset 점검 (선택)
python code/sota/step27_sota_dataset.py

# ── 3) 모델 forward 점검 (선택)
python code/sota/step28_sota_model.py

# ── 4) Loss 점검 (선택)
python code/sota/step29_sota_losses.py

# ── 5) 학습 (RTX 4070 8GB 기준 ≈ 5~6시간)
python code/sota/step30_sota_train.py
#   → outputs/sota/checkpoints/sota_best.pth
#     outputs/sota/checkpoints/sota_swa.pth
#     outputs/sota/logs/sota_history.json
#     outputs/sota/figures/sota_training_curves.png

# ── 6) 평가 (TTA + Temp + Conformal + 다중 비교)
python code/sota/step31_sota_evaluate.py
#   → outputs/sota/logs/sota_test_metrics.json
#     outputs/sota/figures/sota_diagnostics.png
```

## 4. 예상 결과 (v2ways §10.1 — Day 7 칸)

| 항목 | Day 6 MMMT | Day 7 SOTA |
|------|:--:|:--:|
| F1 (cls) | 95~97 | **96~98** |
| WT Dice | 0.74~0.78 | **0.78~0.82** |
| TC Dice | — | **0.88~0.92** |
| ET Dice | — | **0.84~0.89** |
| FN | 700~900 | **600~800** |
| Calibration ECE | — | < 0.03 (Temp Scaling 후) |
| Conformal coverage | — | ≈ 0.90 (alpha=0.1) |

## 5. 기존 코드 변경 여부

- `code/`, `code/multitask/`, `code/mmmt/` 의 *어떤 파일도 수정하지 않습니다.*
- Day 7은 새 폴더 (`code/sota/`, `processed/sota/`, `outputs/sota/`) 안에서만 동작.
- 따라서 Day 5/6 결과가 그대로 보존되어 **5-way 비교** (Whole / Patch / MTL / MMMT / SOTA) 가 가능합니다.

## 6. 주의 사항

1. **VRAM**: ResNet-18 backbone(11M) + decoder(3M) + aux heads. batch 16/224/fp32 기준 약 5~6 GB. OOM 시 `CONFIG["batch_size"]` 8.
2. **TumorCP**: 배치 내 paste만 (간이 구현). 풀-스케일 (외부 슬라이스 풀)은 향후 확장.
3. **Conformal**: mapie 없이 marginal score 기반 자체 구현. 정식 평가는 mapie 설치 후 보강.
4. **MEN/PED tumor-type 다중 분류**: v2ways §6.3, Day 8 이후 권장 — 현재 sota/ 는 GLI 단일 코호트만.
