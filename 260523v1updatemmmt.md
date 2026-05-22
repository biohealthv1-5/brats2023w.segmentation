코드와 문서를 모두 확인했습니다. 다음과 같이 답변드립니다.

---

## ② step19 / step25 재실행 필요 여부

**결론: 재실행 불필요합니다.**

근거 (코드 레벨로 확인):

| 항목 | step18b (`step18b_multitask_train_gradcam.py`) | step24b (`step24b_mmmt_train_gradcam.py`) |
|---|---|---|
| 체크포인트 로드 방식 | `torch.load(..., weights_only=True)` → `model.load_state_dict(...)` → `model.eval()` | 동일 |
| 가중치 저장 호출 | **없음** (`torch.save` / `save_checkpoint` 없음) | **없음** |
| 출력물 | `outputs/figures/multitask/gradcam_train/*.png`, `mt_train_gradcam_summary.json`, `mt_train_gradcam_subsample.csv` | `outputs/figures/mmmt/gradcam_train/*.png`, `mmmt_train_gradcam_summary.json`, `mmmt_train_gradcam_subsample.csv` |
| 모델 파일에 대한 부수효과 | 없음 (hook 등록은 인스턴스 메모리 한정, 프로세스 종료 시 소멸) | 없음 |

즉 18b/24b는 **읽기 전용 분석 스크립트**이고, `mt_best_model.pth` / `mmmt_best.pth` 자체에는 손을 대지 않습니다. step19/step25가 사용하는 가중치는 동일하므로 결과가 바뀔 이유가 없습니다.

**유일한 예외 케이스**: 만약 18b/24b 실행 중 *우연히* checkpoint 파일을 덮어썼거나(코드를 수정해 디버깅하다 실수로 저장), 입력 PNG들(`processed/slices`, `processed/masks/wt`, `processed/slices_t1ce`)을 사이에 재생성했다면 그때만 재실행이 필요합니다. 현재 코드 상태로는 둘 다 해당 없음.

---

## ③ 왜 하필 threshold = 0.3847인가 — 추론

`step19_multitask_evaluate.py`의 ROC 곡선 처리 부분이 핵심 단서입니다:

```python
def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr            # Youden's J = TPR − FPR
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]
    ...
results = { ..., "optimal_threshold": round(float(best_thr), 4), ... }
```

→ **Day 5(MTL) 평가에서 ROC-Youden 최적 임계값이 정확히 0.3847로 산출되어 `mt_evaluation_results.json["optimal_threshold"]`에 저장**된 값입니다. 260522mmmt.md §3.1·§5.2·§4 핵심 인사이트 5번에서 "Day 5의 강력한 보수적 임계(0.3847)→Day 6 표준(0.5) 전환과 결합된 결과"라고 표현하는 그 숫자.

따라서 0.3847은 **임의로 고른 보수적 값이 아니라, Day 5 MTL 모델의 ROC 곡선 위에서 Youden's J(=Sensitivity + Specificity − 1)를 최대화하는 점**입니다. 의미는:

1. **공정 비교 원칙** — Day 5와 Day 6은 *모델 변경 + threshold 변경* 두 가지가 섞여 있습니다(0.3847 → 0.5). 멀티모달의 순수 효과만 분리하려면 **동일 결정 경계**에서 둘을 비교해야 합니다. 그 비교 기준을 *Day 5가 선택했던 그 임계값*에 맞추는 것.
2. **FP +234 해석의 정정** — §3.1·§5.2에서 "동등 임계로 보정 시 FP 350~400 추정"이라고 한 근거가 바로 이 비교. threshold 0.5에서 FP=539인데, 0.3847에서는 양성 판정 영역이 더 넓어져도 *Day 5와 같은 운용 지점*이라는 점이 의미가 있습니다(Precision-Recall trade-off 곡선 자체는 AUROC=0.9832로 거의 동일하므로).
3. **AUROC ≈ 동일, threshold만 이동했다는 §3.1 주장의 직접 검증** — "모델의 *순위 매기기* 성능은 거의 무차이, 결정 경계 위치만 이동"이라는 명제를 숫자로 보이려면 Day 5의 결정 경계에서 Day 6을 재계산해 봐야 합니다.

> 정리: **0.3847은 Day 5 MTL의 Youden-J optimal threshold라서**, 그리고 **그 값이 그대로 Day 5 보고치(F1 93.91, FP 305, FN 1136)를 산출한 운용 지점**이라서 — 두 실험을 같은 운용 지점에서 비교하는 유일한 공정 기준이기 때문입니다. 다른 0.4, 0.45 같은 값을 쓰면 "Day 5의 운용 지점"과 일치하지 않아 비교의 닻이 사라집니다.

---

## ④ 6.1 권장 3종 — 동시 수행 가능성 + 파이프라인

### 종속성 분석

| 작업 | 새 학습 필요? | 입력 의존 | 산출물 | GPU 부담 | 예상 시간 |
|---|:--:|---|---|:--:|:--:|
| **(A)** Threshold 동등 보정 (step25 재실행 @ 0.3847) | ❌ | `mmmt_best.pth` (있음) | `mmmt_test_metrics_thr0.3847.json` | 추론만 (≤1GB) | ~5분 |
| **(B)** FN 회복 패턴 ablation (Day 5↔Day 6 슬라이스 단위 차분) | ❌ | `mt_best_model.pth` + `mmmt_best.pth` + `splits.csv` | 슬라이스 ID별 비교 CSV + 시각화 | 추론만 (≤2GB) | ~10–15분 |
| **(C)** 채널 ablation (T1ce-only / FLAIR-only 재학습) | ✅ | `processed/slices*` 전체 | 새 best.pth 2개 + 평가 | **풀 학습** (~8GB) | 약 6.5시간 × 2 = 13시간 |

→ **(A)와 (B)는 서로 완전히 독립**(읽는 파일이 겹치지 않음), 그리고 둘 다 (C)와도 산출물 의존성이 없음. 다만 (C)는 GPU·디스크 I/O를 단독으로 점유해야 안정적입니다(RTX 4070 Laptop 8GB 환경 기준).

### 권장 파이프라인

**Phase 1 — 즉시·병렬 (≤ 30분, 학습 없음)**

```
   ┌──────────────────────────────────────────────────────┐
   │  (A) step25 재평가 (threshold=0.3847)                 │
   │      입력  : mmmt_best.pth                            │
   │      변경점: collect_predictions 결과를 threshold만   │
   │              바꿔 cls metric 재계산                    │
   │      목적  : 4-way 표의 Day 6 행을 동일 운용 지점에서 │
   │              재산출 (FP 추정 350~400 검증)             │
   └──────────────────────────────────────────────────────┘
                       ║  (서로 독립, 동시 실행 OK)
   ┌──────────────────────────────────────────────────────┐
   │  (B) FN 차분 분석                                     │
   │      입력  : mt_best_model.pth + mmmt_best.pth        │
   │              splits.csv (test=25,115)                 │
   │      산출  : Day5 FN ∩ Day6 TP (= 회복된 슬라이스)    │
   │              Day5 FN ∩ Day6 FN (= 여전히 놓침)         │
   │              + 각 그룹의 슬라이스 위치(z), 종양 픽셀수 │
   │              분포 시각화                               │
   │      비고  : 0.3847 vs 0.5 비교가 의미 있으려면 (A)의 │
   │              결과(threshold 보정 후 confusion)도 함께 │
   │              차분 — 사실상 (A) 완료 후 (B) 시작을      │
   │              권장 (5분 차이라 큰 문제 아님)            │
   └──────────────────────────────────────────────────────┘
```

**핵심**: (A)+(B)는 *추론만* 하므로 메모리 점유가 작아 한 GPU에서 동시 실행도 가능합니다. 다만 (B)의 해석에 (A)의 보정 결과를 쓰는 게 자연스러워, *완전 병렬*보다는 **(A) 끝 → (B) 시작**의 짧은 직렬화를 추천합니다(전체 30분 이내).

**Phase 2 — 장기·순차 (~13시간, GPU 학습)**

```
   (C-1) T1ce-only 학습  ─►  step25 평가
         step20 산출물(T1ce PNG) 그대로 사용
         step21 dataset의 channel_mode를 신규 't1ce_only'로
         만들어 ch0=T1ce, ch1=T1ce, ch2=T1ce 로 3복제
         (또는 1ch 입력 + conv1 가중치 평균)
         ─ step24 학습 6.5h
         ─ step25 평가 ~5분
              │
              ▼
   (C-2) FLAIR-only 학습 ─►  step25 평가
         channel_mode='flair_only'
         ─ Day 5 MTL과 거의 동일하지만 동일 코드 경로/
           dataset/loss로 통제 → "코드 차이"를 제외한
           순수 모달 효과만 비교 가능
         ─ step24 학습 6.5h
         ─ step25 평가 ~5분
              │
              ▼
   최종 ablation 표 (3-row × 동일 메트릭):
     │ Day6 [T1ce, FLAIR, |diff|]  │ baseline(=실측)
     │ Day6 [T1ce, T1ce, T1ce]     │ T1ce-only 기여
     │ Day6 [FLAIR, FLAIR, FLAIR]  │ FLAIR-only 기여
   → "diff 채널이 더한 값"이 정확히 분리됨
```

**Phase 2의 병렬화는 비추천**: GPU 1장이면 (C-1)·(C-2)를 동시 학습하면 OOM/스로틀 위험. 순차로 돌리거나, (C-2)는 다른 머신에 위탁.

### 통합 파이프라인 한 줄 요약

```
[즉시·30분]   (A)threshold=0.3847 재평가  →  (B)FN 슬라이스 차분
[당일/익일]   (C-1)T1ce-only 학습+평가  →  (C-2)FLAIR-only 학습+평가
[정리]        4-way 표 → 6-way 표(+ 두 ablation) 갱신,
              260522mmmt.md §3.1/§5.2 추정치(FP 350~400) 실측치로 대체
```

이렇게 하면 **30분 안에 가장 ROI 높은 비교 결과(Day 5↔Day 6 공정 비교 + FN 회복 슬라이스 식별)** 가 손에 들어오고, 그 다음에 시간 비용이 큰 채널 ablation을 돌려 멀티모달의 *기여도 분해*를 완성할 수 있습니다.