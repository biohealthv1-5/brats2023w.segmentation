# Day 6 — Multi-Modal Multi-Task Learning (mmmt/)

> 베이스라인: `multitask/` (T2-FLAIR 단일 모달, F1 93.91 / IoU 0.706)
> Day 6 목표: **T1ce + T2-FLAIR 멀티모달 입력만** (단일 변경, 분리 평가)
> 가이드: `260521v2ways.md` §8.1 (Day 6 = 입력 표현 강화에만 집중)

## 1. Day 6 범위 정의 (Day 7과의 분리)

본 `mmmt/` 패키지는 **순수 Day 6 작업** 만 명세합니다.
revise_analysis.md §1~§3.1 + v2ways.md §8.1 기반:

| 단계 | Day 6 (본 폴더) | Day 7 (`code/sota/`) |
|------|----------------|---------------------|
| 입력 | **T1ce+FLAIR+diff** (3채널) | 동일 |
| Seg head | **WT 1-region** (multitask와 동일) | WT/TC/ET 3-region |
| Loss (cls) | BCE + pos_weight | 동일 |
| Loss (seg) | **0.5·Dice + 0.5·Tversky(α=0.7,β=0.3)** | + Focal-Tversky + Boundary + Compound |
| MTL 가중 | β=0.5 수동 | Uncertainty Weighting (자동) |
| Aux head | — | Deep Supervision (3개) |
| Aug | spatial(flip/rot) | + TumorCP |
| Train | 2-Phase | + SWA |
| Eval | 단일 forward | + TTA + Temperature + Conformal |

**Day 6 의 한 가지 변화 = "T2-FLAIR 단일 모달 → T1ce+FLAIR 멀티모달".**
모델/Loss/학습 절차는 multitask Day 5 와 *최대한 동일*하게 유지하여 멀티모달
효과만 분리 검증합니다.

## 2. 새 폴더 구조

```
biohealth_lv.1/
├── code/mmmt/                    ← 본 폴더 (Day 6 코드)
│   ├── step20_mmmt_preprocess.py    : T1ce 슬라이스 추출
│   ├── step21_mmmt_dataset.py       : 멀티모달 Dataset/DataLoader (3ch 입력, WT 1ch)
│   ├── step22_mmmt_model.py         : MM-MTL 모델 (ResNet-18 + UNet decoder)
│   ├── step23_mmmt_losses.py        : BCE + Dice + Tversky
│   ├── step24_mmmt_train.py         : 2-Phase 학습
│   └── step25_mmmt_evaluate.py      : 평가 + 4-way 비교
│
├── processed/mmmt/               ← 새 데이터 (step20 결과)
│   └── t1ce_slices/                 : T1ce 224x224 PNG
│       (WT 마스크는 기존 processed/seg_masks/ 재사용)
│
└── outputs/mmmt/                 ← 새 결과 (실행 시 자동 생성)
    ├── checkpoints/                 : mmmt_best.pth, mmmt_last.pth
    ├── figures/                     : mmmt_training_curves.png, mmmt_diagnostics.png
    └── logs/                        : mmmt_history.json, mmmt_summary.json,
                                       mmmt_test_metrics.json, step20_t1ce_stats.json
```

## 3. 실행 명령어 (순서대로)

> Windows / PowerShell 또는 git-bash 어디서나 동일.
> 사전: `step1_preprocess.py`, `multitask/step15_prep_segmask.py` 가 이미 실행되어
> `processed/slices/`, `processed/seg_masks/`, `processed/splits.csv`, `labels.csv` 가
> 존재해야 합니다.

```bash
# (선택) 추가 라이브러리는 보통 이미 있음
pip install nibabel opencv-python pandas tqdm scikit-learn matplotlib seaborn

# ── 1) T1ce 슬라이스 추출 (≈ 7~10분)
python code/mmmt/step20_mmmt_preprocess.py
#   → processed/mmmt/t1ce_slices/*.png
#     outputs/mmmt/logs/step20_t1ce_stats.json

# ── 2) Dataset/DataLoader 동작 점검 (선택)
python code/mmmt/step21_mmmt_dataset.py

# ── 3) 모델 forward pass 점검 (선택)
python code/mmmt/step22_mmmt_model.py

# ── 4) Loss 자가 테스트 (선택)
python code/mmmt/step23_mmmt_losses.py

# ── 5) 학습 (RTX 4070 8GB 기준 ≈ 5~6시간)
python code/mmmt/step24_mmmt_train.py
#   → outputs/mmmt/checkpoints/mmmt_best.pth
#     outputs/mmmt/checkpoints/mmmt_last.pth
#     outputs/mmmt/logs/mmmt_history.json
#     outputs/mmmt/figures/mmmt_training_curves.png

# ── 6) 평가 + 4-way 비교
python code/mmmt/step25_mmmt_evaluate.py
#   → outputs/mmmt/logs/mmmt_test_metrics.json
#     outputs/mmmt/figures/mmmt_diagnostics.png
```

## 4. 예상 결과 (v2ways §10.1 — Day 6 칸)

| 항목 | Day 5 MTL (FLAIR only) | Day 6 MM-MTL (T1ce+FLAIR) |
|------|:--:|:--:|
| F1 (cls) | 93.91 | **95 ~ 97** |
| WT Dice | 0.706 | **0.74 ~ 0.78** |
| FN | 1,136 | **700 ~ 900** |
| FP | 305 | **150 ~ 250** |

## 5. 기존 코드 변경 여부

- 기존 `code/`, `code/multitask/`, `processed/slices/`, `processed/seg_masks/`,
  `processed/labels.csv`, `processed/splits.csv` 는 **전혀 수정하지 않습니다.**
- Day 6는 새 폴더 (`code/mmmt/`, `processed/mmmt/`, `outputs/mmmt/`) 안에서만
  동작 → Day 5 MTL 결과 (`outputs/multitask/`) 는 그대로 보존되어 4-way 비교 가능.

## 6. 주의 사항

1. **VRAM**: ResNet-18 backbone(11M) + decoder(3M). batch 16 / 224×224 / fp32
   기준 약 4~5 GB. OOM 시 `CONFIG["batch_size"]` 를 8 로 줄이세요.
2. **WT 마스크**: 기존 `processed/seg_masks/` (multitask Day 5 산출) 를 그대로
   재사용하므로 별도 마스크 생성 단계가 없습니다.
3. **Day 7 으로 진입 시점**: 본 Day 6 학습/평가가 끝나 4-way 비교가 완료되면,
   `code/sota/` 로 이동하여 WT/TC/ET 3-region + SOTA 패키지 도입을 진행합니다.
