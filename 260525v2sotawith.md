# 📊 `260521v2ways.md`에 언급된 학술적 베이스라인 성능 정리

문서를 자세히 검토했습니다. **우리 비교 대상**의 위치를 한눈에 파악할 수 있도록 정리해드립니다.

---

## 1. 우리의 현재 위치 (출발점)

| 단계 | 모델 | F1 (WT) | IoU (WT) | FN | FP |
|:----:|:----:|:-------:|:--------:|:--:|:--:|
| **Day 5 (현재 baseline)** | T2-FLAIR Multi-Task | **93.91%** | **0.706** | 1,136 | 305 |

> 우리는 **이진 분류 + WT(Whole Tumor) 세분화**를 동시에 하는 2D Multi-Task 모델이며, IoU 측면에서 학술 SOTA Dice 0.91~0.93 대비 아직 격차가 큰 상태(IoU 0.706 ≈ Dice 0.83 수준)입니다.

---

## 2. 세분화 (Segmentation) 직접 비교 SOTA — §7.1

BraTS 데이터셋 기준 **WT(Whole Tumor) Dice 점수**:

| 모델 | 발표 | WT Dice | 차원 | 적합도 | 본 프로젝트와의 격차 |
|------|:----:|:-------:|:----:|:------:|:--------------------|
| **MedNeXt** | MICCAI 2023 | **0.93** | 2D/3D | ⭐⭐⭐⭐⭐ | +0.10 (현재 IoU 0.706 ≈ Dice 0.83 대비) |
| **SwinUNETR-v2** | CVPR 2024 | **0.92** | 3D | ⭐⭐⭐ | +0.09 |
| **DynUNet (nnU-Net 2D)** | Nature Methods 2021 | **0.91** | 2D | ⭐⭐⭐⭐⭐ | +0.08 (Day 9 ablation 기준선) |
...
### 핵심 결론
> **MedNeXt(0.93)이 2D BraTS의 천장**, **DynUNet 2D(0.91)이 학부 환경에서 따라잡을 수 있는 현실적 기준선**입니다. 우리 현재 위치(Dice ≈ 0.83)는 **이 기준선에서 약 8%p 아래**이며, v2 로드맵의 Day 7~8 통합 적용 시 **DynUNet 수준에 도달**하는 것이 정량적 목표입니다.

> 참고로 §7.5의 **TumorCP(MICCAI 2022)는 단독으로 +1.5%p Dice** 효과가 있다고 명시되어 있어, augmentation 트랙도 격차 축소에 유효한 카드입니다.
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...