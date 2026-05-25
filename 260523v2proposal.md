# 🎯 260523 v2 — 프로젝트 실용화 제안서 (Project Practical Proposal)

> **작성일**: 2026-05-23
> **참고 문서**: `260521v1result.md` (Day 1~6 종합 결과), `260521v2ways.md` (SOTA 확장 로드맵), `260523v1updatemmmt.md` (멀티모달 평가 갱신)
> **문서 목적**: 본 프로젝트(BraTS-GLI 기반 멀티모달 Multi-Task 뇌 MRI 종양 분류·세분화 시스템)의 **실용적 활용처를 정의**하고, **그것을 필요로 하는 사용자 집단별로 어떤 형태(제품/서비스/오픈소스)로 제공해야 하는지** 구체화한다.

---

## 0. Executive Summary

### 0.1 프로젝트 자산 정리 (현재까지의 성과)

| 자산 | 구체 내용 | 실용화 가치 |
|------|----------|-----------|
| **Multi-Task 모델 (Day 5)** | F1 93.91%, IoU 0.706, FP 305 (Whole 대비 -43%) | 분류 + 위치 동시 출력의 **단일 14M 모델** |
| **Multi-Modal Multi-Task (Day 6)** | T1ce + FLAIR + |diff| 3채널, F1 95~97% (목표) | 의학적 상보성 기반 임상 정렬 |
| **Shortcut Learning 진단 방법론** | Grad-CAM IoU 0.145 → 0.706 정량 입증 | 모든 의료 AI 모델에 적용 가능한 **감사(audit) 절차** |
| **6단계 여정 자체** | Naive → 진단 → Passive 시도 → Active 해결 → 입력 강화 | 의료 AI 연구·교육의 **정석 사례연구(case study)** |
| **운영 도구 일습** | Grad-CAM, Youden's J threshold (0.3847/0.5), TTA, Temperature Scaling, FN 차분 분석 | 신뢰성·해석성 SOTA 패키지 |
| **확장된 라벨 인프라** | WT/TC/ET seg mask, T1ce 슬라이스, splits.csv 875/188/188 | 추가 학습 없이 grading 확장 가능 |

### 0.2 한 줄 결론

> **"본 프로젝트는 단순한 학부 과제가 아니라, *의료 AI를 임상에 안전하게 배포하기 위한 4단계 신뢰성 점검 파이프라인(분류 → 위치 → 종류 → 신뢰구간)*을 학부 환경에서 재현 가능한 형태로 구현한 자산이며, 5개 사용자 집단에게 5가지 다른 형태로 제공될 수 있다."**

---

## 1. 어디에 쓸 수 있는가 — 7가지 실용 활용처

### 1.1 ① 의료 AI 1차 스크리닝 보조 (Triage Assistant)

**상황**: 종합병원 영상의학과에서 매일 수십~수백 건의 뇌 MRI가 들어오고, 그중 종양 의심 케이스를 빠르게 판별해야 함.

**우리 모델의 강점**:
- **FP 43% 감소** (정상 1,000명 스크리닝 시 18명 불필요 추가검사 회피) — `260521v1result.md §5.10`
- **분류 확률 + 위치 마스크 동시 출력** → 판독의가 모델 결과를 1초 안에 검증 가능
- **TP IoU 0.706** — Whole-Slice (0.145) 대비 5배 높아 "모델이 왜 이렇게 판단했는가" 신뢰 가능

**한계**:
- 1차 스크리닝에 한정 (확진은 의사의 판단 필수)
- FN 28% 증가 trade-off는 1차 스크리닝의 *민감도 우선* 시나리오와 충돌 → Tversky Loss + 멀티모달로 Day 6~7 단계에서 해결 중 (`260521v2ways.md §3.2`)

### 1.2 ② 학부·대학원·전공의 교육용 사례연구 (Case Study)

**가장 강력한 활용처**. 의료 AI 교육에서 가장 어렵게 가르치는 두 가지가 다음이다:

1. "왜 단일 지표(Accuracy/AUC)를 믿으면 안 되는가"
2. "Shortcut Learning은 실제 어떻게 일어나고, 어떻게 진단·해결하는가"

이 두 개념을 **정량 증거(IoU 0.145 → 0.706, FP 536 → 305)와 함께 6단계 여정으로 보여주는 자료는 매우 드물다.** 본 프로젝트는:

- Day 2 Acc 94.33%의 "허위 성과" → Day 3 Grad-CAM 폭로 → Day 4 Patch 실패 → Day 5 MTL 성공
- 각 단계마다 학습 곡선, confusion matrix, 임상 비용 모델, Q&A 38개가 정리되어 있음 (`260521v1result.md §9`)

**활용 형태**: Jupyter Notebook + 슬라이드 + 사전학습 모델 = "**한 학기 강의의 1주차 사례연구**"

### 1.3 ③ 의료 AI 신뢰성 감사(Audit) 템플릿

**상황**: 병원·규제 기관·헬스케어 스타트업이 외부에서 구매한 의료 AI를 배포 전 검증해야 함. 그러나 현재는 *Accuracy / Sensitivity / Specificity* 같은 단일 지표만 제시되는 경우가 많고, 모델이 *왜* 그렇게 판단했는지는 블랙박스.

**본 프로젝트가 제공하는 감사 파이프라인**:

```
[Step 1] 분류 성능 측정 (Accuracy/F1/AUC/PR)
   ↓
[Step 2] Grad-CAM IoU 측정 — 모델이 진짜 종양을 보는가? (Day 3)
   ↓
[Step 3] FN 슬라이스 패턴 분석 — 어떤 종양을 놓치는가? (소종양/큰종양/특정 위치) (Day 3 §3.5)
   ↓
[Step 4] FP 슬라이스 패턴 분석 — 무엇을 종양으로 오인하는가? (Day 3 §3.6)
   ↓
[Step 5] Threshold 운영 곡선 + 임상 비용 모델 (Day 2 §2.5, Day 5 §5.10)
   ↓
[Step 6] Calibration (Temperature Scaling) + Conformal Prediction (uncertainty) (`260521v2ways.md §5.3, §5.5`)
```

→ 이 6-Step 감사 파이프라인은 **본 프로젝트 코드 그대로 적용 가능**하며, 어떤 의료 AI 모델(외부 구매든 자체 개발이든)에도 부착할 수 있다.

### 1.4 ④ 헬스케어 스타트업의 분류 친화 베이스라인 (Bootstrap)

**상황**: 헬스케어 AI 스타트업이 새 모달리티(예: 안저 영상, 흉부 X-ray, 피부 병변)에서 분류 모델을 만들려는 초기 단계. 보통 ResNet/EfficientNet 분류 모델로 시작하지만 → Day 3에서 본 우리처럼 Shortcut에 빠짐.

**본 프로젝트의 아키텍처는 모달리티 독립적**:
- Shared Encoder (ResNet/MedNeXt) + Cls Head + Seg Decoder (U-Net)
- Loss: `α·BCE + β·(Dice + BCE)`, α=1.0, β=0.5 (자동화 시 Uncertainty Weighting)
- Input shape, segmentation label만 갈아 끼우면 다른 도메인에 즉시 이식 가능

**활용 예시**:
| 도메인 | 분류 task | 보조 seg task |
|--------|----------|---------------|
| 흉부 X-ray | 결핵/폐렴 유무 | 병변 영역 |
| 안저 영상 | 당뇨망막병증 유무 | 미세혈관류 영역 |
| 피부 병변 | 악성/양성 | 병변 경계 |
| 유방 mammography | 종괴 유무 | 종괴 영역 |
| 병리 슬라이드 | 암종 유무 | 암 영역 |

→ **"Multi-Task Bootstrap Template"** 로 패키징할 수 있다.

### 1.5 ⑤ BraTS Challenge / Kaggle 베이스라인 코드

BraTS 챌린지(매년 MICCAI)는 학계 표준이지만 진입장벽이 높다. 본 프로젝트는:

- **2D 슬라이스 단위로 단순화** → 3D 부담 없이 시작 가능
- **단일 RTX 4070 Laptop 8GB로 5~6시간 학습** — 학부 GPU 친화
- **MedNeXt / DynUNet (`260521v2ways.md §2.3, §2.5`)** 비교 코드 포함 예정 → SOTA 정렬

→ 입문자가 "BraTS를 어떻게 시작해야 하는지" 모르는 상태에서 **24시간 안에 첫 결과를 내도록 만드는 부트스트랩 키트**가 될 수 있다.

### 1.6 ⑥ 임상 의사결정 지원 시스템(CDSS) 프로토타입

**시나리오**: 비전공 의사(가정의학과·응급의학과 1년차 등)가 외래에서 뇌 MRI를 1차로 검토해야 할 때, *"의심소견이 있으니 신경외과 협진 권유"* 같은 의사결정을 보조.

**본 모델이 제공하는 출력의 임상 친화성**:
- 슬라이스별 종양 확률 + 종양 위치 마스크 → 보고서 자동 초안
- Multi-modal MTL: T1ce / FLAIR 각각의 기여도 시각화 가능 → "왜 그렇게 판단했는가" 설명
- Threshold 운영 곡선 → 병원·과별로 *민감도 우선 / 특이도 우선* 운영점 조정 가능

→ DICOM viewer (OHIF, Slicer) plugin 형태로 통합 가능.

### 1.7 ⑦ 종양 등급화(Grading) 확장 — 임상 가치의 다음 단계

`260521v2ways.md §6`에 정리된 4가지 grading 방향:

1. **WT / TC / ET 3-region multi-label seg** — BraTS 공식 평가축 정렬
2. **Slice-level multi-label** (WT 유무 + TC 유무 + ET 유무)
3. **Tumor-type (GLI / MEN / PED)** — 이미 폴더 존재
4. **Tumor-burden regression** — pixel ratio 기반 침범도

→ "종양 있나?" → "어디?" → "어떤 종류?" → "얼마나 심각?" 의 **임상 진단 4단계를 단일 forward pass로 제공**.

이것이 본 프로젝트의 진짜 임상 가치다. 단순한 이진 분류기는 PACS에 부착할 동기가 약하지만, **4단계 정보를 한 번에 제공하면 판독 시간 단축·report 자동 초안 작성 등 직접 가치 발생**.

---

## 2. 누가 필요로 하는가 — 5개 사용자 집단

### 2.1 사용자 집단 매핑

| 집단 | 핵심 니즈 | 우리 자산의 적합도 | 우선순위 |
|------|----------|:------------------:|:--------:|
| **(A) 의료 AI 연구자 / 대학원생** | 재현 가능한 베이스라인 + 비교 가능한 ablation | ★★★★★ | 🥇 1순위 |
| **(B) 의대생 / 전공의 / 학부생** | "Shortcut Learning이 뭔지" 실감 나는 사례 | ★★★★★ | 🥇 1순위 |
| **(C) 영상의학과 / 신경외과 의사** | 판독 시간 단축, report 자동 초안, 누락 방지 | ★★★ (확진 정확도 부족, 추가 검증 필요) | 🥈 2순위 |
| **(D) 헬스케어 AI 스타트업 / 엔지니어** | 분류+세분화 멀티태스크 부트스트랩 템플릿 | ★★★★ | 🥈 2순위 |
| **(E) 의료 AI 규제·인증 기관 / 병원 IT** | 외부 AI 감사(audit) 절차 + 신뢰성 SOTA 모음 | ★★★★ | 🥉 3순위 |

### 2.2 각 집단의 결정적 통점(Pain Point)과 우리 자산이 해결하는 부분

#### (A) 의료 AI 연구자 / 대학원생

**통점**:
- BraTS, MTL, Multi-Modal, Grading 등 키워드별로 코드는 흩어져 있고 SOTA끼리 직접 비교 불가
- 단일 RTX 4070 8GB 같은 학부 자원에서 SOTA 재현이 불가능한 경우 많음
- "분류 + 세분화 + 등급화"를 단일 모델로 다룬 깔끔한 베이스라인이 드물다

**우리 자산**:
- 6단계 여정 + 4-way 비교 (Whole/Patch/MT/MM-MTL) + ablation 코드 (`code/mmmt/step27_ablation_train.py`, `step28_ablation_evaluate.py`)
- 학부 GPU 5~6시간으로 재현 가능한 코드 일습
- 향후 grading 확장 시 단일 encoder + 4 head 구조 그대로 사용 가능

#### (B) 의대생 / 전공의 / 학부생

**통점**:
- 의료 AI 관련 강의에서 "AUC만 보면 안 된다"고 말로는 배우지만 직접 경험하기 어려움
- Shortcut Learning, Multi-Task Learning, Calibration, Conformal Prediction 같은 개념을 *연결된 하나의 사례*로 보기 어려움

**우리 자산**:
- Day 1~6의 단계별 학습 곡선, confusion matrix, Grad-CAM 시각화 — *전 단계의 결과가 다음 단계의 동기*인 인과 서사
- Q&A 38개 (`260521v1result.md §9`) — 교수님 질문 사례까지 정리됨
- Notebook 1개로 핵심 결과 재현 (`run.ipynb` 존재 확인됨)

#### (C) 영상의학과 / 신경외과 의사

**통점**:
- 매일 들어오는 MRI 양 대비 판독 시간 부족 → 트리아지 도구가 필요
- 시판 의료 AI는 블랙박스 + 비싼 라이선스 + 병원 PACS 통합 부담
- "정확도 95%"라고 광고하지만 *어떤 케이스에서 틀리는지* 알 수 없음

**우리 자산**:
- 분류 확률 + 위치 마스크 동시 제공 (의사 검토에 즉시 사용 가능)
- FN/FP 패턴이 정량 공개되어 있어 "우리 병원 케이스 분포에서는 이렇게 동작할 것" 추정 가능
- (단점) 임상 인증 부재 — *연구·교육 용도*임을 명시 필수

#### (D) 헬스케어 AI 스타트업 / 엔지니어

**통점**:
- 새 모달리티에서 MVP를 빨리 만들어야 하는데 ResNet 분류 한 줄로는 부족 (Shortcut 위험)
- 분류 + Segmentation 같이 학습하는 표준 코드가 없음 (직접 짜면 1~2주 소요)
- Loss 가중치, threshold 같은 하이퍼파라미터 튜닝에 막대한 시간 낭비

**우리 자산**:
- ResNet-18 + U-Net decoder + dual head + Uncertainty Weighting (자동 가중) — 한 줄 import로 사용 가능한 형태로 패키징 가능
- α/β/threshold 자동 보정 (Temperature Scaling, Uncertainty Weighting) — 튜닝 시간 절약
- 모달리티 독립적 — channel 수와 segmentation label만 갈아 끼우면 됨

#### (E) 의료 AI 규제·인증 기관 / 병원 IT

**통점**:
- 외부 AI 도입 전 검증할 표준 절차 부재
- "단일 지표가 부족하다"는 건 알지만 *대안 절차*가 정립되지 않음

**우리 자산**:
- 6-Step 감사 파이프라인 (위 §1.3) — 그대로 다른 AI에 부착 가능
- Conformal Prediction 등 신뢰성 SOTA 모음 (`260521v2ways.md §5`)
- FN/FP 차분 분석 도구 (`code/mmmt/step26_fn_diff_analysis.py` 존재)

---

## 3. 어떤 형태로 제공해야 하는가 — 5가지 제공 모델

각 사용자 집단에 다른 형태로 제공해야 한다. 같은 자산을 다섯 가지 포장으로 출시하는 전략.

### 3.1 [형태 1] **GitHub 오픈소스 리포지토리** — 집단 (A), (D) 대상

**구성**:
```
biohealth-mtl/
├── README.md (영문 + 한글)
├── docs/
│   ├── shortcut-learning-case-study.md  (Day 1~6 여정 영문판)
│   ├── architecture.md
│   ├── api.md
│   └── reproduce.md
├── src/
│   ├── models/        ResNet/MedNeXt + UNet decoder + dual/quad head
│   ├── losses/        BCE/Dice/Tversky/Focal-Tversky/Boundary/Compound
│   ├── data/          BraTS-GLI dataset wrappers
│   ├── train/         Uncertainty Weighting + GradNorm + SWA
│   ├── eval/          TTA + Temperature Scaling + Conformal
│   └── interpret/     Grad-CAM + Seg overlay + FN 차분 분석
├── configs/           Hydra YAML
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_shortcut_learning_demo.ipynb (★ 시그니처 자료)
│   └── 03_multimodal_grading.ipynb
├── checkpoints/       (HuggingFace Model Hub에 별도 호스팅)
└── benchmark/         BraTS leaderboard 비교
```

**라이선스**: MIT (코드) + CC-BY-4.0 (문서). 모델 가중치는 BraTS 라이선스 준수.

**채널**:
- GitHub Repository
- Hugging Face Model Hub (사전학습 체크포인트)
- PyPI 패키지 `pip install biohealth-mtl` (라이브러리 모드)
- Papers with Code 등재 (BraTS Classification 항목)

**핵심 시그니처**: `02_shortcut_learning_demo.ipynb` — "당신의 분류 모델이 종양을 진짜로 보는가?" 5분 데모. **바이럴 가능 자료**.

### 3.2 [형태 2] **교육용 콘텐츠 패키지** — 집단 (B) 대상

**구성**:
```
medical-ai-shortcut-bootcamp/
├── slides/
│   ├── 01_intro_brats.pptx
│   ├── 02_resnet_baseline.pptx
│   ├── 03_gradcam_revelation.pptx  (★ 메인 슬라이드)
│   ├── 04_patch_failure.pptx
│   ├── 05_multitask_solution.pptx
│   ├── 06_multimodal_extension.pptx
│   └── 07_calibration_uncertainty.pptx
├── notebooks/        (Colab 호환, 무료 T4 GPU에서 동작하도록 라이트화)
├── handout/
│   ├── qa-master-list.pdf  (38개 Q&A)
│   └── timeline-infographic.png
├── video/            (5분 데모 영상 6개)
└── assessment/
    ├── quiz.pdf
    └── project-template.md
```

**채널**:
- 학부/대학원 강의 자료 (CC-BY-NC-4.0)
- Coursera / edX / fast.ai 강좌 콘텐츠로 기증
- 의대·간호대 의료정보학 수업 사례연구
- 의공학회 / 영상의학회 워크숍 자료

**가격 모델**: 무료 (의료 AI 교육 확산이 미션)

### 3.3 [형태 3] **PACS Plugin / DICOM Viewer 통합 데모** — 집단 (C) 대상

**구성**:
```
brats-mtl-viewer/
├── frontend/         OHIF Viewer plugin (TypeScript)
├── backend/          FastAPI + 사전학습 모델
├── docker/           docker-compose up 한 줄 실행
└── demo/
    ├── dicom-samples/ (de-identified 샘플 10건)
    └── README.md
```

**기능**:
1. DICOM 시리즈 업로드 → 슬라이스별 종양 확률 + 위치 마스크 자동 표시
2. Threshold slider (0.3~0.7) 실시간 조정
3. Grad-CAM overlay toggle
4. (Day 6+) WT/TC/ET 3-region 분리 표시
5. (Day 7+) 모달리티 누락 강건성 — T1ce 없이 / FLAIR 없이 어떤 결과인지 비교

**중요한 면책**:
> "본 시스템은 *연구·교육 용도*이며 의료기기로 인증되지 않았다. 임상 의사결정은 반드시 인가된 의사가 수행해야 한다."

**채널**:
- GitHub 별도 리포 (`brats-mtl-viewer`)
- Docker Hub 이미지
- 영상의학회 / 신경외과학회 데모 부스
- 병원 IT 부서 PoC (PoC 환경에 한해)

### 3.4 [형태 4] **헬스케어 AI 부트스트랩 SDK** — 집단 (D) 대상

**구성**:
```python
# pip install medical-mtl-bootstrap

from medical_mtl import MultiTaskBootstrap

# 새 도메인에 1줄로 적용
model = MultiTaskBootstrap(
    backbone='resnet18',       # or 'mednext', 'segformer'
    num_classes=1,              # 분류 출력 수
    num_seg_channels=3,         # WT/TC/ET 또는 도메인별 영역
    loss='compound',            # 'bce_dice' / 'focal_tversky' / 'compound'
    mtl_weighting='uncertainty' # 자동 가중치
)
model.fit(train_loader, val_loader, epochs=20)
model.evaluate_with_calibration(test_loader)  # Temperature Scaling 자동
model.predict_with_conformal(image, alpha=0.1) # 90% coverage
```

**API 설계 원칙**:
- `model.fit()` 한 줄로 학습 (PyTorch Lightning 내장)
- Modality-agnostic 입력 (n_channels 파라미터)
- 결과는 dataclass로 반환 (cls_prob, seg_mask, uncertainty, calibrated_prob)
- 분류 head / seg head / type head / regression head 모듈 조합 가능 (LEGO 식)

**채널**:
- PyPI 라이브러리
- 헬스케어 컨퍼런스 (HIMSS, RSNA AI Lounge) 부스
- 의료 AI 스타트업 인큐베이터 협업

**비즈니스 모델**: 오픈소스 코어 + 유료 지원·맞춤형 컨설팅 (B2B)

### 3.5 [형태 5] **의료 AI 신뢰성 감사 키트** — 집단 (E) 대상

**구성**:
```
medical-ai-audit-kit/
├── audit-checklist.pdf        (6-Step 표준 절차)
├── tools/
│   ├── shortcut_diagnosis.py  (Grad-CAM IoU 자동 측정)
│   ├── fn_pattern_analysis.py (FN 슬라이스 패턴 분석)
│   ├── fp_pattern_analysis.py
│   ├── calibration_check.py   (ECE, Brier, Reliability Diagram)
│   ├── conformal_report.py    (Coverage report)
│   └── modality_robustness.py (모달리티 누락 강건성)
├── report-template/           (감사 리포트 자동 생성)
└── case-studies/
    └── brats-our-model.md     (본 프로젝트가 감사 대상으로 자기 보고서)
```

**핵심 차별점**:
- AI 모델을 *블랙박스로 받아서* 6-Step 감사 수행 가능 (모델 구조 알 필요 없음)
- PDF 리포트 자동 생성 — 규제 기관 제출 형식
- 본 프로젝트 자체가 *감사 대상 케이스 1번*으로 첨부됨

**채널**:
- 식약처 / FDA 의료 AI 가이드라인 작성 참고 자료
- 병원 IT 부서·CIO 대상 SaaS (B2B)
- 의료 AI 인증 컨설팅 회사 협업

**비즈니스 모델**: 오픈소스 + 인증 컨설팅 유료 서비스

---

## 4. 우선순위 로드맵 — 어디부터 시작할 것인가

### 4.1 단기 (1~2개월) — 학부 프로젝트 완성과 직결

| # | 작업 | 형태 | 산출물 | 기대 가치 |
|:-:|------|:----:|--------|---------|
| 1 | `260523v1updatemmmt.md`의 (A)+(B) 완료 | 분석 | threshold 0.3847 비교 + FN 차분 결과 | Day 6 보고서 완성 |
| 2 | Day 7 "무료 SOTA 패키지" 통합 (`v2ways §8.2`) | 학습 | F1 96~98% / IoU 0.78+ 달성 | 발표·논문 핵심 결과 |
| 3 | GitHub 리포 정리 (형태 1) | 오픈소스 | README + Quickstart Notebook 1개 | 즉시 외부 공유 가능 |
| 4 | `02_shortcut_learning_demo.ipynb` 작성 | 교육 | 5분 데모 노트북 | 바이럴 시드 콘텐츠 |

### 4.2 중기 (3~6개월) — 실용화 첫걸음

| # | 작업 | 형태 | 산출물 | 기대 가치 |
|:-:|------|:----:|--------|---------|
| 5 | Day 8 Grading 확장 (WT/TC/ET + Type) | 학습 | 4-task 모델 + ablation | BraTS 챌린지 정렬, 임상 가치 4단계 |
| 6 | 교육용 슬라이드/비디오 패키지 | 교육 | 7개 슬라이드 + 6개 비디오 | 강좌·학회 워크숍 자료 |
| 7 | PyPI 라이브러리 `medical-mtl-bootstrap` v0.1 | SDK | pip install + Quickstart | 헬스케어 스타트업 활용 |
| 8 | 6-Step 감사 키트 v0.1 | 도구 | shortcut_diagnosis.py 등 6개 모듈 | 다른 의료 AI에 적용 |

### 4.3 장기 (6~12개월) — 임상·산업 확산

| # | 작업 | 형태 | 산출물 | 기대 가치 |
|:-:|------|:----:|--------|---------|
| 9 | OHIF Viewer plugin | PACS 통합 | DICOM 업로드 → 결과 표시 | 의사 사용자 PoC |
| 10 | Self-supervised pretraining + MedSAM 연동 | 연구 | SimMIM BraTS pretrain | 데이터 적은 도메인으로 일반화 |
| 11 | 식약처·규제 기관 가이드라인 제안서 | 정책 | 6-Step 감사 절차 문서화 | 의료 AI 인증 표준화 기여 |
| 12 | MICCAI / RSNA 논문 투고 | 학술 | 1편 (Shortcut → MTL → Grading) | 학술적 인증 |

---

## 5. 핵심 차별화 메시지 — 왜 우리 자산이어야 하는가

### 5.1 시장의 다른 자산들과의 비교

| 비교 대상 | 강점 | 본 프로젝트 대비 약점 |
|-----------|------|---------------------|
| **nnU-Net** | 절대 Dice SOTA (0.92) | 분류 친화성·해석성·재현성·자원 부담 (3D, GPU-day) |
| **MedSAM** | Universal segmentation | 분류 task 부재, 도메인별 fine-tune 비용 |
| **BiomedCLIP** | 멀티모달 feature | Segmentation 출력 없음, 보조 task 없음 |
| **시판 의료 AI (icobrain, Quantib, AIDoc 등)** | 임상 인증, 통합된 워크플로우 | 블랙박스, 비쌈, 외부 검증 불가, Shortcut 진단 부재 |
| **GitHub의 기타 BraTS 리포** | 다양 | 6단계 여정·Shortcut 진단·Grading 통합 부재 |

### 5.2 본 프로젝트의 고유 가치 (Unique Selling Point)

> **"학부 GPU에서 5~6시간이면 재현되는 의료 AI 6단계 여정 + Shortcut Learning 정량 진단 + Multi-Task 5배 해석성 향상 + 4-task Grading 확장 + 신뢰성 SOTA 패키지 — 이 다섯 가지가 한 리포에 묶인 것은 본 프로젝트가 유일하다."**

특히 차별점은:

1. **인과 서사가 있다** — "왜 MTL을 도입했는가"가 Day 3 Grad-CAM IoU 0.145라는 정량 근거에서 출발. 다른 리포는 결과만 있고 동기가 없다.
2. **학부 자원 친화** — RTX 4070 8GB / 5~6시간으로 완주. 대학원·연구소 자원이 없어도 재현 가능.
3. **교육·연구·산업 동시 활용** — 단일 자산을 5가지 형태로 패키징 가능 (`§3`).
4. **Grading 확장의 확실성** — BraTS-MEN, BraTS-PED 폴더가 이미 있고, WT/TC/ET label도 처리됨. 데이터 추가 수집 없이 종양 종류·침범도까지 확장 가능.

---

## 6. 위험·한계와 대응

### 6.1 기술적 한계

| 한계 | 영향 | 대응 |
|------|------|------|
| Day 5 MTL의 FN 28% 증가 | 종양 누락 위험 | Day 6 멀티모달 + Day 7 Tversky Loss로 해결 중 |
| 절대 Dice가 SOTA 미달 (0.78 vs 0.92) | 임상 인증 어려움 | "분류 친화·해석성 중심" 포지셔닝, Dice 경쟁 회피 |
| 2D 슬라이스 단위 → 3D 맥락 부재 | 종양 부피 측정 불가 | 2.5D 입력 (z±1) 또는 후처리 통합 (`v2ways §4.2`) |
| 단일 데이터셋 (BraTS-GLI) | 분포 외 강건성 불확실 | MEN/PED zero-shot 평가 (`v2ways §5.6`) |

### 6.2 비기술적 한계

| 한계 | 대응 |
|------|------|
| 임상 인증 부재 | 모든 공개 자료에 *"연구·교육 용도"* 명시 |
| 의료 데이터 라이선스 | BraTS 라이선스 준수, 환자 단위 de-identification |
| 책임 소재 (FN 발생 시) | 의사 최종 판단 강제, 모델 출력은 *제안*에 한정 |
| 학부 프로젝트 신뢰성 | 동료 검토(peer review) + 의료진 자문 + 학회 발표 |

---

## 7. 한 줄 결론 (다시)

> **"본 프로젝트는 *연구·교육·산업·임상·규제* 다섯 영역에 동시 활용 가능한 *드문 자산*이며, 다섯 가지 형태(GitHub 오픈소스 / 교육 패키지 / PACS Plugin / SDK / 감사 키트)로 패키징하여 각 사용자 집단에 별도 제공할 때 그 가치가 극대화된다."**

| 사용자 | 형태 | 우선순위 | 우리 단계 |
|--------|------|:--------:|:--------:|
| 의료 AI 연구자 | GitHub 오픈소스 + 사전학습 모델 | 🥇 | 1~2개월 |
| 학부생 / 의대생 | 교육용 슬라이드 + Colab 노트북 | 🥇 | 1~3개월 |
| 헬스케어 스타트업 | PyPI SDK + Bootstrap 템플릿 | 🥈 | 3~6개월 |
| 영상의학과 의사 | OHIF Viewer plugin (PoC) | 🥈 | 6~12개월 |
| 규제 기관 / 병원 IT | 6-Step 감사 키트 | 🥉 | 6~12개월 |

---

## 8. 다음 단계 권장 행동 (Actionable Next Steps)

1. **Day 6 마무리** — `260523v1updatemmmt.md`의 (A)+(B) 30분 작업으로 4-way 표 완성.
2. **Day 7 "무료 SOTA 패키지" 1회 학습** — Focal-Tversky + Uncertainty Weighting + Deep Supervision + TumorCP 동시 적용 (`v2ways §8.2`).
3. **GitHub 리포 정리** — 현재 코드(`code/whole/`, `code/patch/`, `code/multitask/`, `code/mmmt/`)를 `src/`로 재배치, README 영문화.
4. **Shortcut Learning Demo Notebook 작성** — `02_shortcut_learning_demo.ipynb` 5분 분량. (가장 강력한 바이럴 시드.)
5. **학회·강의 발표 자료 정리** — 본 프로젝트 v1·v2·v2-proposal 3종을 발표 자료로 합성.

---

> 📎 **본 문서의 위치**: `biohealth_lv.1/260523v2proposal.md`
> 📎 **연속성**: `260521v1result.md` (결과 정리) → `260521v2ways.md` (기술 확장 로드맵) → `260523v1updatemmmt.md` (멀티모달 평가 갱신) → **`260523v2proposal.md` (실용화 제안 — 본 문서)** → 최종 발표 자료 / GitHub 공개.
