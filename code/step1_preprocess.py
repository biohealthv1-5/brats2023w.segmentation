# -*- coding: utf-8 -*-
"""
Step 1: 데이터 전처리 - 3D NIfTI → 2D 슬라이스 추출
=====================================================
BraTS-GLI 3D MRI 볼륨을 2D axial 슬라이스로 변환하고,
segmentation 마스크에서 종양 유무 라벨을 자동 생성합니다.

사용법:
    python code/step1_preprocess.py
"""

import os
import sys
import csv
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "BraTS-GLI"
    / "training"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
OUTPUT_DIR = PROJECT_DIR / "processed"
SLICE_DIR = OUTPUT_DIR / "slices"
LABELS_CSV = OUTPUT_DIR / "labels.csv"

# ─── 설정 ────────────────────────────────────────────────────
MODALITY = "t2f"  # T2-FLAIR (종양 경계를 가장 잘 보여줌)
IMAGE_SIZE = 224  # ResNet 표준 입력 크기
MIN_BRAIN_FRACTION = 0.01  # 뇌 영역이 이 비율 이하인 슬라이스는 제외 (빈 슬라이스 필터링)


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """2D 슬라이스를 0~255로 정규화 (uint8)"""
    # 비-뇌 영역 제외하고 정규화
    brain_mask = slice_2d > 0
    if not np.any(brain_mask):
        return np.zeros_like(slice_2d, dtype=np.uint8)

    # percentile clipping (1~99%)으로 이상치 제거
    brain_values = slice_2d[brain_mask]
    p1, p99 = np.percentile(brain_values, [1, 99])
    clipped = np.clip(slice_2d, p1, p99)

    # min-max 정규화
    min_val = p1
    max_val = p99
    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    normalized = (clipped - min_val) / (max_val - min_val)
    normalized = (normalized * 255).astype(np.uint8)
    return normalized


def resize_slice(slice_2d: np.ndarray, target_size: int) -> np.ndarray:
    """2D 슬라이스를 target_size × target_size로 리사이즈"""
    import cv2

    resized = cv2.resize(
        slice_2d, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )
    return resized


def process_patient(patient_dir: Path, writer, stats: dict):
    """
    한 환자의 3D 볼륨을 처리하여 2D 슬라이스를 저장합니다.

    Args:
        patient_dir: 환자 디렉토리 경로
        writer: CSV writer
        stats: 통계 딕셔너리
    """
    patient_id = patient_dir.name

    # 파일 경로 구성
    modality_file = patient_dir / f"{patient_id}-{MODALITY}.nii.gz"
    seg_file = patient_dir / f"{patient_id}-seg.nii.gz"

    if not modality_file.exists() or not seg_file.exists():
        stats["skipped"] += 1
        return

    try:
        # NIfTI 로드
        img = nib.load(str(modality_file))
        seg = nib.load(str(seg_file))

        img_data = img.get_fdata().astype(np.float32)
        seg_data = seg.get_fdata().astype(np.int16)

        # 볼륨 shape: (H, W, D) - axial 슬라이스는 [:, :, z]
        num_slices = img_data.shape[2]

        patient_positive = 0
        patient_negative = 0

        for z in range(num_slices):
            img_slice = img_data[:, :, z]
            seg_slice = seg_data[:, :, z]

            # 빈 슬라이스 필터링 (뇌 영역이 너무 적으면 제외)
            brain_fraction = np.mean(img_slice > 0)
            if brain_fraction < MIN_BRAIN_FRACTION:
                continue

            # 라벨 생성: 종양 픽셀이 1개라도 있으면 → Positive (1)
            label = 1 if np.any(seg_slice > 0) else 0

            # 정규화 & 리사이즈
            normalized = normalize_slice(img_slice)
            resized = resize_slice(normalized, IMAGE_SIZE)

            # 저장 파일명: {patient_id}_z{slice_idx:03d}.png
            slice_filename = f"{patient_id}_z{z:03d}.png"
            slice_path = SLICE_DIR / slice_filename

            import cv2

            # 한글 경로 지원을 위해 imencode + tofile 사용
            success, encoded = cv2.imencode(".png", resized)
            if success:
                encoded.tofile(str(slice_path))

            # CSV에 기록
            writer.writerow(
                {
                    "filename": slice_filename,
                    "patient_id": patient_id,
                    "slice_idx": z,
                    "label": label,
                    "brain_fraction": f"{brain_fraction:.4f}",
                }
            )

            if label == 1:
                patient_positive += 1
            else:
                patient_negative += 1

        stats["patients_processed"] += 1
        stats["total_positive"] += patient_positive
        stats["total_negative"] += patient_negative

    except Exception as e:
        print(f"  [ERROR] {patient_id}: {e}")
        stats["errors"] += 1


def main():
    print("=" * 60)
    print("  Step 1: 3D NIfTI → 2D 슬라이스 추출")
    print("=" * 60)
    print(f"\n  데이터 경로: {DATA_DIR}")
    print(f"  출력 경로: {OUTPUT_DIR}")
    print(f"  모달리티: {MODALITY}")
    print(f"  이미지 크기: {IMAGE_SIZE}×{IMAGE_SIZE}")

    # 데이터 경로 확인
    if not DATA_DIR.exists():
        print(f"\n[ERROR] 데이터 경로가 존재하지 않습니다: {DATA_DIR}")
        return

    # 출력 디렉토리 생성
    SLICE_DIR.mkdir(parents=True, exist_ok=True)

    # 환자 디렉토리 목록
    patient_dirs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir()]
    )
    print(f"  환자 수: {len(patient_dirs)}")

    # 통계 초기화
    stats = {
        "patients_processed": 0,
        "skipped": 0,
        "errors": 0,
        "total_positive": 0,
        "total_negative": 0,
    }

    # CSV 파일 생성 & 처리
    print(f"\n{'─' * 60}")
    print("  슬라이스 추출 시작...")
    print(f"{'─' * 60}\n")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "patient_id", "slice_idx", "label", "brain_fraction"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
            process_patient(patient_dir, writer, stats)

    # 결과 요약
    total_slices = stats["total_positive"] + stats["total_negative"]
    print(f"\n{'=' * 60}")
    print(f"  전처리 완료!")
    print(f"{'=' * 60}")
    print(f"  처리된 환자 수: {stats['patients_processed']}")
    print(f"  건너뛴 환자 수: {stats['skipped']}")
    print(f"  오류 발생 수: {stats['errors']}")
    print(f"{'─' * 60}")
    print(f"  총 슬라이스 수: {total_slices:,}")
    print(f"    Positive (종양 존재): {stats['total_positive']:,} ({stats['total_positive']/max(total_slices,1)*100:.1f}%)")
    print(f"    Negative (종양 없음): {stats['total_negative']:,} ({stats['total_negative']/max(total_slices,1)*100:.1f}%)")
    print(f"{'─' * 60}")
    print(f"  슬라이스 저장 경로: {SLICE_DIR}")
    print(f"  라벨 CSV 경로: {LABELS_CSV}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
