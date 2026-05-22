# -*- coding: utf-8 -*-
"""
Step 20 (Day 6): T1ce 모달리티 슬라이스 추출
=============================================
Day 6 = "T2-FLAIR 단일 모달 → T1ce + T2-FLAIR 멀티모달" 전환 (revise_analysis §1~§2).
기존 `processed/slices/` (T2-FLAIR) 와 1:1 대응하는 T1ce 슬라이스 PNG를
`processed/mmmt/t1ce_slices/` 에 저장합니다.

- 기존 step1_preprocess.py 의 정규화/리사이즈 로직과 동일하게 처리하여
  T1ce와 FLAIR의 정규화 일관성을 유지합니다.
- 기존 labels.csv / slices/ 는 변경하지 않습니다 (참조만 합니다).
- 환자 단위 875/188/188 split 유지 (splits.csv 재사용).

본 단계는 Day 6 작업의 *입력 표현* 부분만 수행합니다 — Day 7의 3-region 마스크
(WT/TC/ET) 생성은 별도 폴더 `code/sota/` 에서 수행됩니다.

사용법:
    python code/mmmt/step20_mmmt_preprocess.py
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "BraTS-GLI"
    / "training"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
SLICE_DIR_FLAIR = PROJECT_DIR / "processed" / "slices"          # 참조용 (기존 FLAIR)
LABELS_CSV = PROJECT_DIR / "processed" / "labels.csv"
T1CE_OUT_DIR = PROJECT_DIR / "processed" / "mmmt" / "t1ce_slices"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt"

# ─── 설정 (기존 step1과 동일) ────────────────────────────────
MODALITY = "t1c"     # T1ce
IMAGE_SIZE = 224
MIN_BRAIN_FRACTION = 0.01


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """step1과 동일한 percentile-clip + minmax → uint8"""
    brain_mask = slice_2d > 0
    if not np.any(brain_mask):
        return np.zeros_like(slice_2d, dtype=np.uint8)
    p1, p99 = np.percentile(slice_2d[brain_mask], [1, 99])
    clipped = np.clip(slice_2d, p1, p99)
    if p99 - p1 < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    normalized = (clipped - p1) / (p99 - p1)
    return (normalized * 255).astype(np.uint8)


def main():
    print("=" * 60)
    print("  Step 20 (Day 6 / mmmt): T1ce 슬라이스 추출")
    print("=" * 60)
    print(f"\n  데이터 경로:   {DATA_DIR}")
    print(f"  출력 경로:     {T1CE_OUT_DIR}")
    print(f"  모달리티:      {MODALITY}")
    print(f"  이미지 크기:   {IMAGE_SIZE}x{IMAGE_SIZE}")

    if not DATA_DIR.exists():
        print(f"\n[ERROR] 데이터 경로가 존재하지 않습니다: {DATA_DIR}")
        return
    if not LABELS_CSV.exists():
        print(f"\n[ERROR] labels.csv 없음: {LABELS_CSV}")
        print("  step1_preprocess.py 먼저 실행 필요.")
        return

    # 기존 슬라이스 목록 → 1:1 매칭 기준
    labels_df = pd.read_csv(LABELS_CSV)
    existing_slices = set(labels_df["filename"].tolist())
    print(f"  기존 FLAIR 슬라이스: {len(existing_slices):,}")

    T1CE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"  환자 수: {len(patient_dirs)}")

    stats = {"patients": 0, "saved": 0, "skipped_patient": 0,
             "skipped_slice": 0, "missing_pair": 0, "errors": 0}

    print(f"\n{'-' * 60}\n  T1ce 추출 시작...\n{'-' * 60}\n")

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        patient_id = patient_dir.name
        t1c_file = patient_dir / f"{patient_id}-{MODALITY}.nii.gz"
        if not t1c_file.exists():
            stats["skipped_patient"] += 1
            continue
        try:
            img_data = nib.load(str(t1c_file)).get_fdata().astype(np.float32)
            num_slices = img_data.shape[2]
            for z in range(num_slices):
                slice_filename = f"{patient_id}_z{z:03d}.png"
                # 기존 FLAIR 슬라이스에 존재하는 경우에만 저장 (1:1 매칭)
                if slice_filename not in existing_slices:
                    stats["skipped_slice"] += 1
                    continue
                img_slice = img_data[:, :, z]
                if np.mean(img_slice > 0) < MIN_BRAIN_FRACTION:
                    # 일관성: FLAIR에서 통과한 슬라이스도 T1ce 뇌 영역이 부족하면
                    # 검은 PNG로 저장 (페어 누락 방지)
                    norm = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                    stats["missing_pair"] += 1
                else:
                    norm = normalize_slice(img_slice)
                    norm = cv2.resize(norm, (IMAGE_SIZE, IMAGE_SIZE),
                                      interpolation=cv2.INTER_LINEAR)
                out_path = T1CE_OUT_DIR / slice_filename
                success, encoded = cv2.imencode(".png", norm)
                if success:
                    encoded.tofile(str(out_path))
                    stats["saved"] += 1
            stats["patients"] += 1
        except Exception as e:
            print(f"  [ERROR] {patient_id}: {e}")
            stats["errors"] += 1

    print(f"\n{'=' * 60}\n  T1ce 추출 완료\n{'=' * 60}")
    print(f"  처리 환자:        {stats['patients']}")
    print(f"  저장 슬라이스:    {stats['saved']:,}")
    print(f"  스킵 환자:        {stats['skipped_patient']}")
    print(f"  스킵 슬라이스:    {stats['skipped_slice']:,}")
    print(f"  페어 누락(0충전): {stats['missing_pair']:,}")
    print(f"  에러:             {stats['errors']}")
    if stats["saved"] == len(existing_slices):
        print(f"\n  ✅ FLAIR 슬라이스와 1:1 매칭 완료!")
    else:
        diff = len(existing_slices) - stats["saved"]
        print(f"\n  ⚠ {diff:,}개 슬라이스 매칭 누락 (다음 단계에서 영(0) 처리됨)")

    with open(LOG_DIR / "step20_t1ce_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  통계 저장: {LOG_DIR / 'step20_t1ce_stats.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
