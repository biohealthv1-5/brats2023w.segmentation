# BraTS 2023 데이터셋 가이드

## 개요

**BraTS (Brain Tumor Segmentation) Challenge 2023**은 ASNR-MICCAI에서 주관하는 뇌종양 분할(segmentation) 대회입니다. 다운로드한 데이터는 Synapse 플랫폼(syn64952532)에서 제공되는 공식 데이터셋입니다.

각 환자 케이스는 **4가지 MRI 모달리티**로 구성됩니다:

| 모달리티 | 파일 접미사 | 설명 |
|---------|-----------|------|
| T1 | `*-t1n.*` | T1-weighted (네이티브) — 뇌 해부학적 구조 |
| T1CE | `*-t1c.*` | T1-weighted + 조영제(Contrast Enhanced) — 종양 강화 영역 강조 |
| T2 | `*-t2w.*` | T2-weighted — 부종(edema) 영역 강조 |
| FLAIR | `*-t2f.*` | T2-FLAIR — 부종과 종양 경계 강조 |
| Seg | `*-seg.*` | Segmentation label (Training에만 포함) |

---

## 다운로드된 파일 상세 설명

### 🧠 1. GLI (Glioma, 성인 교모세포종/신경교종)

| 파일 | 설명 |
|------|------|
| `asnr-miccai-brats2023-gli-challenge-trainingdata.zip` | GLI **훈련 데이터** — MRI 4종 + segmentation label 포함 |
| `asnr-miccai-brats2023-gli-challenge-validationdata.zip` | GLI **검증 데이터** — MRI 4종만 포함 (label 없음, 모델 평가용) |

> [!NOTE]
> Glioma는 BraTS Challenge의 가장 핵심적인 종양 유형으로, 성인에게 가장 흔한 원발성 악성 뇌종양입니다.

### 🔵 2. MEN (Meningioma, 수막종)

| 파일 | 설명 |
|------|------|
| `asnr-miccai-brats2023-men-challenge-trainingdata.zip` | 수막종 **훈련 데이터** |
| `asnr-miccai-brats2023-men-challenge-validationdata.zip` | 수막종 **검증 데이터** |
| `brats-men-train-fix-v4.zip` | 수막종 훈련 데이터의 **수정 패치** (일부 케이스의 label 오류 수정) |

> [!IMPORTANT]
> `brats-men-train-fix-v4.zip`은 원본 MEN 훈련 데이터의 **버그픽스**입니다. 수막종 데이터를 사용할 경우, 원본 압축 해제 후 이 패치를 덮어씌워야 합니다.

### 👶 3. PED (Pediatric, 소아 뇌종양)

| 파일 | 설명 |
|------|------|
| `asnr-miccai-brats2023-ped-challenge-trainingdata.zip` | 소아 뇌종양 **훈련 데이터** |
| `asnr-miccai-brats2023-ped-challenge-validationdata.zip` | 소아 뇌종양 **검증 데이터** |

### 📊 4. 보조 파일

| 파일 | 설명 |
|------|------|
| `brats2023_2017_gli_mapping.xlsx` | BraTS 2023 ↔ BraTS 2017 GLI 케이스 ID **매핑 테이블**. 이전 대회 결과와 비교할 때 사용 |
| `meningioma supplementary clinical data...xlsx` | 수막종 환자의 **임상 메타데이터** (나이, 성별, 종양 등급 등) |
| `readme.md` | 데이터셋 공식 README |

### 🧪 5. Sample Dataset (MLCubes용 샘플)

| 파일 | 설명 |
|------|------|
| `brats-gli-fastlane.tar.gz` | GLI **소규모 샘플** — MLCube 파이프라인 테스트용 |
| `brats-local-synthesis-fastlane.tar.gz` | Local Synthesis/Inpainting **소규모 샘플** — 종양 영역 복원 태스크 테스트용 |

> [!TIP]
> fastlane 파일들은 전체 데이터셋을 사용하기 전에 파이프라인이 올바르게 동작하는지 빠르게 테스트할 수 있는 소량의 샘플 데이터입니다.

---

## 정리 후 디렉토리 구조

```
biohealth_lv.1/
└── data/
    ├── BraTS-GLI/
    │   ├── training/          ← gli training zip 해제
    │   └── validation/        ← gli validation zip 해제
    ├── BraTS-MEN/
    │   ├── training/          ← men training zip 해제 + fix-v4 패치 적용
    │   └── validation/        ← men validation zip 해제
    ├── BraTS-PED/
    │   ├── training/          ← ped training zip 해제
    │   └── validation/        ← ped validation zip 해제
    ├── supplementary/
    │   ├── brats2023_2017_gli_mapping.xlsx
    │   ├── meningioma_clinical_data.xlsx
    │   └── readme.md
    └── sample/
        ├── brats-gli-fastlane/        ← tar.gz 해제
        └── brats-local-synthesis/     ← tar.gz 해제
```

---

## Segmentation Label 값 (Training data)

| Label 값 | 의미 |
|----------|------|
| 0 | 배경 (정상 조직) |
| 1 | NCR (Necrotic Core, 괴사 핵) |
| 2 | ED (Peritumoral Edema, 종양 주변 부종) |
| 3 | ET (Enhancing Tumor, 조영 증강 종양) |
