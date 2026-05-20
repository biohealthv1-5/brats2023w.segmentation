# -*- coding: utf-8 -*-
"""
Step 8: 패치 좌표 추출 및 라벨 생성
=====================================================
기존 224×224 슬라이스를 기반으로 64×64 패치의 좌표와 라벨을 생성합니다.
원본 NIfTI seg 마스크를 사용하여 패치별 종양 비율을 계산합니다.

패치 이미지는 별도 저장하지 않고, 학습 시 원본 슬라이스에서 on-the-fly로 크롭합니다.
(~300만 개 개별 PNG 저장 → 디스크 비효율 방지)

출력:
    processed/patch_labels.csv          — 패치 좌표 + 라벨
    outputs/logs/step8_stats.json       — 통계 요약 (다음 단계 참조용)

사용법:
    python code/step8_patch_extract.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from pathlib import Path
from tqdm import tqdm

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_DIR / "data" / "BraTS-GLI" / "training"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
PROCESSED_DIR = PROJECT_DIR / "processed"
SPLITS_CSV = PROCESSED_DIR / "splits.csv"
PATCH_LABELS_CSV = PROCESSED_DIR / "patch_labels.csv"
STATS_JSON = PROJECT_DIR / "outputs" / "logs" / "step8_stats.json"

# ─── 패치 설정 ───────────────────────────────────────────────
PATCH_SIZE = 64
STRIDE = 32
IMAGE_SIZE = 224
MODALITY = "t2f"
TUMOR_THRESHOLD = 0.05   # 종양 비율 ≥ 5% → Positive
BRAIN_THRESHOLD = 0.10   # 뇌 영역 비율 < 10% → 제외

# 패치 그리드: 6×6 = 36 패치/슬라이스
POSITIONS = list(range(0, IMAGE_SIZE - PATCH_SIZE + 1, STRIDE))  # [0,32,64,96,128,160]


def main():
    print("=" * 60)
    print("  Step 8: 패치 좌표 추출 및 라벨 생성")
    print("=" * 60)
    print(f"  패치: {PATCH_SIZE}×{PATCH_SIZE}, stride={STRIDE}")
    print(f"  그리드: {len(POSITIONS)}×{len(POSITIONS)} = {len(POSITIONS)**2}개/슬라이스")
    print(f"  종양 임계값: {TUMOR_THRESHOLD*100}%, 뇌 임계값: {BRAIN_THRESHOLD*100}%")

    # 경로 확인
    if not DATA_DIR.exists():
        print(f"\n[ERROR] 데이터 경로 없음: {DATA_DIR}")
        return
    if not SPLITS_CSV.exists():
        print(f"\n[ERROR] splits.csv 없음: {SPLITS_CSV}")
        return

    df = pd.read_csv(SPLITS_CSV)
    print(f"\n  슬라이스: {len(df):,}개, 환자: {df['patient_id'].nunique()}명")

    # 환자별 그룹
    patient_groups = df.groupby("patient_id")
    patient_ids = sorted(patient_groups.groups.keys())

    # 통계 카운터
    total_extracted = 0
    total_filtered = 0
    positive_count = 0
    negative_count = 0
    patients_ok = 0
    patients_skip = 0
    errors = []
    split_counts = {"train": {"pos": 0, "neg": 0}, "val": {"pos": 0, "neg": 0}, "test": {"pos": 0, "neg": 0}}

    # 출력 디렉토리
    STATS_JSON.parent.mkdir(parents=True, exist_ok=True)

    # CSV 작성 (스트리밍)
    print(f"\n{'─'*60}\n  패치 추출 시작...\n{'─'*60}\n")

    columns = [
        "slice_filename", "patient_id", "slice_idx",
        "patch_idx", "patch_row", "patch_col", "x_start", "y_start",
        "patch_label", "tumor_ratio", "brain_ratio",
        "slice_label", "split"
    ]

    with open(PATCH_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")

        for patient_id in tqdm(patient_ids, desc="Patients"):
            patient_df = patient_groups.get_group(patient_id)
            patient_dir = DATA_DIR / patient_id
            seg_file = patient_dir / f"{patient_id}-seg.nii.gz"
            img_file = patient_dir / f"{patient_id}-{MODALITY}.nii.gz"

            if not seg_file.exists() or not img_file.exists():
                patients_skip += 1
                continue

            try:
                seg_vol = nib.load(str(seg_file)).get_fdata().astype(np.int16)
                img_vol = nib.load(str(img_file)).get_fdata().astype(np.float32)
            except Exception as e:
                errors.append(f"{patient_id}: {e}")
                patients_skip += 1
                continue

            patients_ok += 1

            for _, row in patient_df.iterrows():
                z = int(row["slice_idx"])
                if z >= seg_vol.shape[2] or z >= img_vol.shape[2]:
                    continue

                seg_slice = seg_vol[:, :, z]
                img_slice = img_vol[:, :, z]

                # 224×224로 리사이즈
                seg_224 = cv2.resize(seg_slice.astype(np.float32),
                                     (IMAGE_SIZE, IMAGE_SIZE),
                                     interpolation=cv2.INTER_NEAREST).astype(np.int16)
                img_224 = cv2.resize(img_slice, (IMAGE_SIZE, IMAGE_SIZE),
                                     interpolation=cv2.INTER_LINEAR)

                pidx = 0
                for r, y in enumerate(POSITIONS):
                    for c, x in enumerate(POSITIONS):
                        seg_patch = seg_224[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                        img_patch = img_224[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                        brain_ratio = float(np.mean(img_patch > 0))
                        if brain_ratio < BRAIN_THRESHOLD:
                            total_filtered += 1
                            pidx += 1
                            continue

                        tumor_ratio = float(np.mean(seg_patch > 0))
                        plabel = 1 if tumor_ratio >= TUMOR_THRESHOLD else 0

                        line = (
                            f"{row['filename']},{patient_id},{z},"
                            f"{pidx},{r},{c},{x},{y},"
                            f"{plabel},{tumor_ratio:.6f},{brain_ratio:.4f},"
                            f"{int(row['label'])},{row['split']}\n"
                        )
                        f.write(line)

                        total_extracted += 1
                        if plabel == 1:
                            positive_count += 1
                        else:
                            negative_count += 1
                        split_counts[row["split"]]["pos" if plabel == 1 else "neg"] += 1
                        pidx += 1

    # ─── 통계 저장 ────────────────────────────────────────────
    stats = {
        "patients_processed": patients_ok,
        "patients_skipped": patients_skip,
        "total_patches_extracted": total_extracted,
        "total_patches_filtered_brain": total_filtered,
        "positive_patches": positive_count,
        "negative_patches": negative_count,
        "pos_neg_ratio": round(positive_count / max(negative_count, 1), 4),
        "split_counts": split_counts,
        "settings": {
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "tumor_threshold": TUMOR_THRESHOLD,
            "brain_threshold": BRAIN_THRESHOLD,
        },
        "errors": errors[:20],
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ─── 결과 출력 ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  패치 추출 완료!")
    print(f"{'='*60}")
    print(f"  처리 환자: {patients_ok}, 건너뜀: {patients_skip}")
    print(f"  추출 패치: {total_extracted:,}")
    print(f"  필터링 패치 (뇌<{BRAIN_THRESHOLD*100}%): {total_filtered:,}")
    print(f"  Positive: {positive_count:,} ({positive_count/max(total_extracted,1)*100:.1f}%)")
    print(f"  Negative: {negative_count:,} ({negative_count/max(total_extracted,1)*100:.1f}%)")
    print(f"  Pos:Neg 비율 = 1:{negative_count/max(positive_count,1):.1f}")
    print(f"\n  Split별:")
    for s in ["train", "val", "test"]:
        sp, sn = split_counts[s]["pos"], split_counts[s]["neg"]
        print(f"    {s:5s}: Pos={sp:,}, Neg={sn:,}, Total={sp+sn:,}")
    print(f"\n  CSV: {PATCH_LABELS_CSV}")
    print(f"  Stats: {STATS_JSON}")
    if errors:
        print(f"  에러: {len(errors)}건 (상세는 JSON 참조)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
