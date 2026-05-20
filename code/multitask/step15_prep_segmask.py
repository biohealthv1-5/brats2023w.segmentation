# -*- coding: utf-8 -*-
"""
Step 15: Segmentation 마스크 전처리
====================================
기존 슬라이스와 1:1 대응하는 seg 마스크를 224×224 바이너리 PNG로 추출합니다.
기존 코드(step1~14)에는 어떤 변경도 가하지 않습니다.

사용법:
    python code/multitask/step15_prep_segmask.py
"""

import os
import sys
import json
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path
from tqdm import tqdm

# Windows 콘솔 UTF-8 출력 설정
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
SLICE_DIR = PROJECT_DIR / "processed" / "slices"              # 기존 슬라이스 (참조용)
MASK_DIR = PROJECT_DIR / "processed" / "seg_masks"             # 새로 생성
LABELS_CSV = PROJECT_DIR / "processed" / "labels.csv"
LOG_DIR = PROJECT_DIR / "outputs" / "multitask" / "logs"

# ─── 설정 ────────────────────────────────────────────────────
MODALITY = "t2f"
IMAGE_SIZE = 224
MIN_BRAIN_FRACTION = 0.01  # step1과 동일


def main():
    print("=" * 60)
    print("  Step 15: Segmentation 마스크 전처리")
    print("=" * 60)
    print(f"\n  데이터 경로: {DATA_DIR}")
    print(f"  마스크 출력: {MASK_DIR}")

    # 데이터 경로 확인
    if not DATA_DIR.exists():
        print(f"\n[ERROR] 데이터 경로가 존재하지 않습니다: {DATA_DIR}")
        return

    # 기존 labels.csv 확인
    if not LABELS_CSV.exists():
        print(f"\n[ERROR] labels.csv가 없습니다: {LABELS_CSV}")
        print("  step1_preprocess.py를 먼저 실행하세요.")
        return

    # 기존 labels.csv 로드 → 어떤 슬라이스가 추출되었는지 확인
    import pandas as pd
    labels_df = pd.read_csv(LABELS_CSV)
    existing_slices = set(labels_df["filename"].tolist())
    print(f"  기존 슬라이스 수: {len(existing_slices):,}")

    # 출력 디렉토리 생성
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 환자 디렉토리 목록
    patient_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"  환자 수: {len(patient_dirs)}")

    # 통계
    stats = {
        "patients_processed": 0,
        "masks_saved": 0,
        "masks_with_tumor": 0,
        "masks_empty": 0,
        "skipped": 0,
        "errors": 0,
    }

    print(f"\n{'─' * 60}")
    print("  마스크 추출 시작...")
    print(f"{'─' * 60}\n")

    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name

        # 파일 경로
        modality_file = patient_dir / f"{patient_id}-{MODALITY}.nii.gz"
        seg_file = patient_dir / f"{patient_id}-seg.nii.gz"

        if not modality_file.exists() or not seg_file.exists():
            stats["skipped"] += 1
            continue

        try:
            # NIfTI 로드
            img = nib.load(str(modality_file))
            seg = nib.load(str(seg_file))

            img_data = img.get_fdata().astype(np.float32)
            seg_data = seg.get_fdata().astype(np.int16)

            num_slices = img_data.shape[2]

            for z in range(num_slices):
                # 기존 step1과 동일한 필터링 로직
                img_slice = img_data[:, :, z]
                brain_fraction = np.mean(img_slice > 0)
                if brain_fraction < MIN_BRAIN_FRACTION:
                    continue

                slice_filename = f"{patient_id}_z{z:03d}.png"

                # 기존 슬라이스에 존재하는 경우에만 마스크 생성
                if slice_filename not in existing_slices:
                    continue

                # seg 마스크 처리
                seg_slice = seg_data[:, :, z]
                # 바이너리 마스크: 종양 픽셀 > 0 → 255, 그 외 → 0
                binary_mask = (seg_slice > 0).astype(np.uint8) * 255

                # 224×224로 리사이즈 (nearest neighbor로 마스크 보간)
                resized_mask = cv2.resize(
                    binary_mask, (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_NEAREST
                )

                # 저장 (한글 경로 지원)
                mask_path = MASK_DIR / slice_filename
                success, encoded = cv2.imencode(".png", resized_mask)
                if success:
                    encoded.tofile(str(mask_path))

                stats["masks_saved"] += 1
                if np.any(seg_slice > 0):
                    stats["masks_with_tumor"] += 1
                else:
                    stats["masks_empty"] += 1

            stats["patients_processed"] += 1

        except Exception as e:
            print(f"  [ERROR] {patient_id}: {e}")
            stats["errors"] += 1

    # 결과 요약
    print(f"\n{'=' * 60}")
    print(f"  마스크 추출 완료!")
    print(f"{'=' * 60}")
    print(f"  처리된 환자 수: {stats['patients_processed']}")
    print(f"  건너뛴 환자 수: {stats['skipped']}")
    print(f"  오류 발생 수: {stats['errors']}")
    print(f"{'─' * 60}")
    print(f"  총 마스크 수: {stats['masks_saved']:,}")
    print(f"    종양 포함: {stats['masks_with_tumor']:,}")
    print(f"    빈 마스크: {stats['masks_empty']:,}")
    print(f"{'─' * 60}")
    print(f"  마스크 저장 경로: {MASK_DIR}")

    # 검증: 기존 슬라이스 수와 일치하는지 확인
    if stats["masks_saved"] == len(existing_slices):
        print(f"\n  ✅ 기존 슬라이스와 1:1 매칭 완료!")
    else:
        diff = len(existing_slices) - stats["masks_saved"]
        print(f"\n  ⚠️ {diff:,}개 슬라이스와 매칭되지 않음")

    # 통계 저장
    with open(LOG_DIR / "step15_segmask_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  통계 저장: {LOG_DIR / 'step15_segmask_stats.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
