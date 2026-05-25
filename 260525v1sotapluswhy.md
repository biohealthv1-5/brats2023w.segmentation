이제 부록 C.1의 우선순위 항목들을 이해하기 위해 관련 맥락을 더 살펴보겠습니다.이제 충분한 맥락을 확보했습니다. 두 작업을 모두 정리하여 답변을 작성하겠습니다.---

# 1. `code/sota/` 폴더 코드의 역할 — md 문서와 연결지어 설명

## 1-0. 큰 그림 — sota 폴더는 "Day 7 무료 SOTA 패키지"의 구현체

- `code/sota/README.md` 첫 문장이 명시: **베이스라인은 Day 6 `mmmt/` (F1 ~95%, WT 0.74~0.78)이며, Day 7 목표는 `260521v2ways.md` §8.2 "무료 SOTA 패키지"를 Day 6 위에 한 번에 통합**.
- `260521v2ways.md` §8.2 표(위에서 확인)는 정확히 (A)~(F) 6종 패키지를 5시간 코딩 + 5~6시간 학습으로 "WT F1 95.5~97%, IoU 0.75~0.80, FN 600~800, TC F1 88~92%, ET F1 84~89%"까지 끌어올린다고 약속함.
- `260524v1mmmtplus.md` 부록 C.2도 같은 패키지를 **중기(~1개월) 계획**으로 재확인.
- 기존 `code/`, `code/multitask/`, `code/mmmt/`를 **전혀 손대지 않고** 새 폴더 `code/sota/`, `processed/sota/`, `outputs/sota/` 안에서만 동작하도록 격리되어 있어, 5-way 비교(Whole / Patch / MTL / MMMT / SOTA) 자산이 그대로 보존됨.

## 1-1. `step26_sota_segmask.py` — WT/TC/ET 3-region 마스크 생성

**역할**: Day 5 multitask가 만들었던 "WT 단일 바이너리 마스크"를 BraTS 공식 평가축인 **WT(Whole Tumor) / TC(Tumor Core) / ET(Enhancing Tumor)** 3-region으로 분해해서 `processed/sota/seg_masks_{wt,tc,et}/`에 저장.

**md 근거**:
- `260521v2ways.md` §3.4 "Region-aware Loss" — BraTS 공식 평가 단위로 정렬.
- BraTS 2023 라벨 규약(`brats2023_dataset_guide.md`): label 1=NCR, 2=ED, 3=ET.
- 3-region 정의는 `WT = seg>0`, `TC = (seg==1)|(seg==3)`, `ET = seg==3`.

**왜 필요한가**:
- `260524v1mmmtplus.md` §2.8 비교에서 Day 6는 WT 단일 채널만 측정 가능 — TC/ET 평가축이 없어 BraTS SOTA 표와 직접 비교가 불가했음.
- TC/ET가 들어와야 §2.8 한계가 메워지고, 부록 C.3 "장기 — Grading 확장 1번(WT/TC/ET 3-region multi-label seg)"이 1단계 달성됨.

## 1-2. `step27_sota_dataset.py` — 3채널 입력 + 3채널 마스크 + Weighted Sampler

**역할**:
1. 입력: 3채널 `[T1ce, FLAIR, |T1ce-FLAIR|]` (Day 6 `processed/mmmt/t1ce_slices/` 재사용)
2. 출력: cls 스칼라 + 3채널 (WT/TC/ET) 마스크
3. `_build_sampler` — WT 양성 픽셀 < 256인 "소종양" 슬라이스에 가중치 ×3.

**md 근거**:
- `README.md` 표의 F항 = `260521v2ways.md` §4.5 "Weighted sampler".
- `260524v1mmmtplus.md` §4.5의 인사이트 ④: **still_missed 641 슬라이스 평균 종양 크기 100픽셀(0.20%)** — 극소 종양은 멀티모달도 못 잡음 → "Day 7+ Tumor-CP / 2.5D 입력 / Focal-Tversky의 표적".
- 즉 `_build_sampler`는 §4.5의 진단을 직접 처방으로 옮긴 코드.

## 1-3. `step28_sota_model.py` — Deep Supervision + Uncertainty Weighting 모델

**역할**: Day 6 `MMMTBrainNet`의 확장.
- ResNet-18 encoder + U-Net 스타일 decoder는 유지.
- `seg_classes 1 → 3` (WT/TC/ET 멀티채널 seg head).
- `aux_head_d2/d3/d4` — Deep Supervision 보조 WT seg head 3개.
- `log_var_cls`, `log_var_seg` — Kendall 2018 Uncertainty Weighting의 학습 파라미터.

**md 근거**:
- `README.md` 표 A/C/D = `260521v2ways.md` §2.2 Deep Supervision, §3.5 Uncertainty Weighting, §3.4 3-region.
- `260524v1mmmtplus.md` §1.5의 **결정적 인사이트 ①**: "Multi-Task 분류 head Grad-CAM IoU는 train에서도 0.1232에 불과한데, seg head IoU는 0.7753" → encoder feature는 충분히 종양을 알지만 분류 head로 전파되지 않음. Deep Supervision은 decoder 중간 단계마다 supervisory signal을 직접 주입해서 이 격차를 줄임.
- Uncertainty Weighting은 §2 Q4의 "Tversky α/β 수동 튜닝 부작용"(FP 증가 — §2.8 "Tversky α=0.7의 부작용"에서 언급) 문제를 자동화하려는 시도.

## 1-4. `step29_sota_losses.py` — Focal-Tversky + Boundary + Compound + TumorCP

**역할**:
- `FocalTverskyLoss`: α=0.7, β=0.3, γ=4/3 — FN에 비선형 강패널티(Abraham & Khan ISBI 2019).
- `BoundaryLoss`: SDF(signed distance function) 기반, 종양 경계 슬라이스 잡기 목적(Kervadec MIDL 2019).
- `CompoundSegLoss`: `1.0·Focal-Tversky + 0.5·BCE + 0.3·Boundary` per-region.
- `SOTAMultiTaskLoss`: cls/seg 두 task에 Uncertainty Weighting 적용 + Deep Supervision aux 3개에 ds_weights=(0.125, 0.25, 0.5) 가중합.
- `tumor_copy_paste`: 배치 내 종양 슬라이스 패치를 다른 슬라이스에 paste(Yang MICCAI 2022 단순화).

**md 근거**:
- `README.md` B = `260521v2ways.md` §3.2~§3.3, E = §4.3.
- `260524v1mmmtplus.md` 부록 C.2의 7개 항목 중 1, 4번이 직접 매칭.
- §4 Q1 "recovered 381 vs regressed 86 상호 보완성" — 두 모델이 *각자 다른* 작은 종양을 잡는다는 진단의 해결책으로 TumorCP가 작은 종양을 더 자주 보게 만듦.
- §4 인사이트 ④의 "극소 종양 100픽셀" 한계 → Boundary Loss가 경계 신호를, Focal-Tversky가 FN 비선형 가중을 제공.

## 1-5. `step30_sota_train.py` — 2-Phase + SWA + TumorCP 학습 루프

**역할**:
- Phase 1 (3 epoch): encoder freeze, head만 학습 (lr=1e-3).
- Phase 2 (15 epoch): full fine-tune (lr=1e-4) + Cosine LR.
- Phase 2 epoch 11~15 (마지막 5 epoch): SWA `AveragedModel` 업데이트 + `SWALR`(swa_lr=5e-5).
- 학습 끝에 `update_bn`으로 SWA BatchNorm 통계 갱신 후 `sota_swa.pth` 저장.
- TumorCP는 매 batch 50% 확률로 적용.

**md 근거**:
- `README.md` G = `260521v2ways.md` §5.2 "SWA".
- Day 5/Day 6 모두 2-Phase 학습이었음(`260524v1mmmtplus.md` §2.4) — 동일 프로토콜로 fair comparison 유지.
- SWA는 §5.2 "마지막 25% epoch 평균"에 정확히 부합(15 epoch 중 마지막 5 = 33%, 비슷).

## 1-6. `step31_sota_evaluate.py` — TTA + Temperature Scaling + Conformal + 다중 비교

**역할**:
- `predict_with_tta`: identity / hflip / vflip / rot180 4-view 평균.
- `fit_temperature` (LBFGS): val set logit으로 Temperature T 학습 → BCE calibration.
- `try_conformal`: marginal score 기반 conformal set 생성, coverage 측정.
- `compute_seg_metrics`: WT/TC/ET별 Dice / IoU.
- 마지막에 Day 5 multitask / Day 6 mmmt 결과 json을 읽어 **5-way 비교**.

**md 근거**:
- `README.md` H/I/J = `260521v2ways.md` §5.1 TTA, §5.3 Temperature Scaling, §5.5 Conformal.
- `260524v1mmmtplus.md` §3.1: "Day 5의 threshold 0.3847은 그 모델의 *운용 지점*. Day 6의 기본 0.5는 다른 지점" → Temperature Scaling이 이 미스매치를 **자동 보정**.
- 부록 C.2의 7번 항목 "Temperature Scaling — threshold 0.3847을 0.5 부근으로 보정"이 step31에서 정확히 실행됨.

## 1-7. 5-way 비교 자산 보존

`README.md` §5: "기존 코드 변경 여부 — 어떤 파일도 수정하지 않습니다. Day 5/6 결과가 그대로 보존되어 5-way 비교(Whole / Patch / MTL / MMMT / SOTA)가 가능". 이는 `260524v1mmmtplus.md` 부록 D.5의 "6-way 비교 표" 작성 의도와 정확히 같은 노선.

---

# 2. 부록 C.1 "본 보고서 직후 우선순위 (~1주)"의 이유 및 기대 효과

원문(라인 1086~1093):

| # | 작업 | 소요 | 산출물 |
|:-:|------|------|--------|
| 1 | Day 5 + Day 6 soft-voting Ensemble 평가 | ~30분 (추론만) | recovered 381 + regressed 86 합산 검증 |
| 2 | new_fp 515 / cleaned_fp 119의 prob 히스토그램 분석 | ~10분 | threshold 운영 점 최적화 |
| 3 | still_missed 641의 모달리티 신호 분석 | ~30분 | 2.5D / 고해상도 필요성 판단 |
| 4 | Day 6 Score-CAM/EigenCAM 비교 (`260521v2ways.md` §3.3) | ~1시간 | CAM 0.16의 진짜 한계 진단 |

## 왜 이 4개가 "최우선 1주"인가 — 공통 원리

- **이미 학습된 모델만 사용** — 새 학습 없이 분석/추론만으로 가능 → 총 소요 **~2시간 10분**, 가성비 최상.
- **본 보고서 §4의 4가지 인사이트(recovered 381, still_missed 641, new_fp 515, CAM 0.16)를 *주장*에서 *검증된 사실*로 격상**시키는 작업들.
- Day 7 SOTA(부록 C.2, 학습 ~6시간) 전에 *어떤 처방이 정말 필요한지*를 먼저 확정해 두어야 5~6시간짜리 학습을 헛되이 쓰지 않을 수 있음.

## 항목별 이유와 기대 효과

### ① Day 5 + Day 6 soft-voting Ensemble 평가 (~30분)

- **이유**: §4 Q1에서 이미 가설로 제시됨 — "Day 5는 FLAIR 부종 단서 강한 케이스(regressed 86)를 잡고, Day 6은 T1ce 조영제로 ET 강조된 케이스(recovered 381)를 잡는다. 즉 *상호 보완*". 학습 비용 0(이미 있는 두 체크포인트만 사용).
- **기대 효과**:
  - **recovered 381 + regressed 86 = 467 슬라이스를 모두 살릴 수 있는지** 즉답 가능.
  - 만약 ensemble이 둘 다 잡는다면 → 보고서의 "상호 보완성" 주장이 정량 입증되고 §6의 5/6-way 비교 표에 즉시 한 행이 추가됨.
  - 만약 ensemble이 별 효과 없다면 → 두 모델이 같은 특징을 본다는 뜻이므로, Day 7 SOTA에서 *다른* augmentation 전략(TumorCP, Boundary)에 더 무게를 둬야 함을 의미.

### ② new_fp 515 / cleaned_fp 119의 prob 히스토그램 분석 (~10분)

- **이유**: §4 Q3 "new_fp 515의 prob mean이 0.579로 *반쯤 확신* — 어떻게 처리해야 하나?"의 직접 후속. `fn_diff_per_slice.csv`(25,116행)에 이미 데이터가 있음(부록 A.8). 새 추론 불필요, 단순 통계.
- **기대 효과**:
  - prob 분포가 0.5 부근 좁게 모여 있다면 → threshold를 0.5→0.6 정도로 올리는 것만으로도 FP 다수 제거 가능(operating point 최적화).
  - 분포가 0.5~0.95에 넓게 퍼져 있다면 → threshold만으론 못 잡고, Day 7 Temperature Scaling(부록 C.2 7번) 또는 Focal-Tversky의 FP 패널티 강화가 필수.
  - **즉 Day 7 패키지의 어떤 컴포넌트가 *진짜* 필요한지 사전 진단** — 5~6시간 학습 낭비 방지.

### ③ still_missed 641의 모달리티 신호 분석 (~30분)

- **이유**: §4 인사이트 ④와 §4 Q2 — "픽셀 100개의 극소 종양, 멀티모달로도 못 잡음". 부록 A.7에 명시: "z-위치, 종양 픽셀 수만 분석됨. **모달리티별 신호 강도(T1ce / FLAIR 평균 강도) 통계 추가 권장** → 결과에 따라 2.5D 입력이나 고해상도(320×320) 필요성 판단 가능".
- **기대 효과**:
  - T1ce/FLAIR 모두 신호가 *원래 약하다*면 → 2D 슬라이스의 정보 부족 → **2.5D(인접 z 동시 입력)** 또는 **고해상도 320×320** 필요. 이는 step27 dataset과 step28 model의 input shape 변경을 의미하는 큰 결정.
  - 신호가 충분히 있는데 모델이 못 잡는다면 → 모델 capacity 문제이므로 단순한 Loss 변경(Focal-Tversky)으로 해결될 가능성.
  - **부록 C.2의 Day 7 패키지를 그대로 적용할 것인지, 추가로 입력 표현(2.5D)까지 변경할 것인지의 분기점**.

### ④ Day 6 Score-CAM/EigenCAM 비교 (~1시간)

- **이유**: §2 Q5와 §1 Q2 — "CAM IoU 0.12→0.16 (+29%)이 진짜 해석성 향상인가? 여전히 0.2도 안 됨". 부록 A.9: "Grad-CAM의 한계 대응. `260521v2ways.md` §3.3 권장". 학부 발표에선 Grad-CAM만으로 충분하나, **CAM 0.16이 모델 한계인지 Grad-CAM 알고리즘 한계인지** 구분이 필수.
- **기대 효과**:
  - Score-CAM/EigenCAM에서 IoU가 0.3+로 뛴다면 → **모델은 종양을 보고 있지만 Grad-CAM이 못 잡아낸 것** → 발표에서 "해석성은 충분, XAI 도구 선택 문제"로 정정. 모델 변경 불필요.
  - 여전히 0.16 수준이면 → §1.5의 인사이트 ① ("분류 head는 광역 단서에 의존") 결정적 확정 → Day 7에서 Deep Supervision(step28의 `aux_head_d*`)가 *반드시* 필요한 처방임이 입증됨.
  - 어느 쪽이든 발표 핵심 슬라이드의 메시지를 강화함.

## 종합 — 우선순위 1주가 끝났을 때 얻는 것

1. **모든 §4 인사이트가 *정량 검증된 사실*로 격상** → 발표·논문 신뢰도 상승.
2. **Day 7 SOTA 패키지의 *진짜로 필요한* 컴포넌트 확정** — 5~6시간짜리 학습을 가장 효과적인 설정으로 1회만 돌릴 수 있음(부록 C.2의 시간 효율 극대화).
3. **무료 ensemble 한 줄 추가**로 5-way → 6-way 비교 표 즉시 확장 가능 → 학술적 기여 보강.
4. **모델 변경 vs 입력 변경 vs XAI 변경**의 의사결정 분기점이 모두 사전에 정리되므로, 본 보고서 직후 1주 안에 *Day 7 학습 1회*가 곧바로 "최선의 SOTA 결과"로 이어짐.

요컨대 C.1은 **"새로 학습하지 말고, 이미 가진 자산으로 *주장을 사실로 굳히고* Day 7 학습을 *겨냥*하는" 사전 정찰(reconnaissance) 단계**입니다.