# -*- coding: utf-8 -*-
"""
Step 26 (Day 7 / sota): BraTS 3-region (WT/TC/ET) 마스크 생성
==============================================================
v2ways.md §3.4 "Region-aware Loss" — BraTS 공식 평가축 (WT/TC/ET)에 정렬하기
위한 3-region 마스크 분해.

BraTS 라벨 규약 (BraTS 2023):
  - label 1: NCR (Necrotic + Non-enhancing tumor core)
  - label 2: ED  (Peritumoral edema)
  - label 3: ET  (Enhancing tumor)

3-region 정의:
  - WT (Whole Tumor)        = (seg > 0)         (= label 1 | 2 | 3)
  - TC (Tumor Core)         = (seg == 1) | (seg == 3)
  - ET (Enhancing Tumor)    = (seg == 3)

기존 `processed/seg_masks/` (WT 바이너리, multitask Day 5) 는 건드리지 않고,
새 폴더 `processed/sota/seg_masks_wt|tc|et/` 에 3개 마스크를 분리 저장합니다.

사용법:
    python code/sota/step26_sota_segmask.py
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

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = (
    PROJECT_DIR / "data" / "BraTS-GLI" / "training"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
LABELS_CSV = PROJECT_DIR / "processed" / "labels.csv"
OUT_BASE = PROJECT_DIR / "processed" / "sota"
WT_DIR = OUT_BASE / "seg_masks_wt"
TC_DIR = OUT_BASE / "seg_masks_tc"
ET_DIR = OUT_BASE / "seg_masks_et"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "sota"

IMAGE_SIZE = 224
MIN_BRAIN_FRACTION = 0.01
MODALITY_REF = "t2f"   # step1과 동일하게 FLAIR 기준으로 슬라이스 유효성 판단


def main():
    print("=" * 60)
    print("  Step 26 (Day 7 / sota): WT/TC/ET 3-region 마스크 생성")
    print("=" * 60)
    if not LABELS_CSV.exists():
        print(f"[ERROR] {LABELS_CSV} 없음. step1 먼저.")
        return
    for d in (WT_DIR, TC_DIR, ET_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(LABELS_CSV)
    existing_slices = set(labels_df["filename"].tolist())
    print(f"  기준 슬라이스 수: {len(existing_slices):,}")

    patient_dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    stats = {"patients": 0, "saved": 0,
             "wt_pos": 0, "tc_pos": 0, "et_pos": 0,
             "skipped": 0, "errors": 0}

    for patient_dir in tqdm(patient_dirs, desc="Patients"):
        pid = patient_dir.name
        mod_file = patient_dir / f"{pid}-{MODALITY_REF}.nii.gz"
        seg_file = patient_dir / f"{pid}-seg.nii.gz"
        if not mod_file.exists() or not seg_file.exists():
            stats["skipped"] += 1
            continue
        try:
            img_data = nib.load(str(mod_file)).get_fdata().astype(np.float32)
            seg_data = nib.load(str(seg_file)).get_fdata().astype(np.int16)
            num_slices = img_data.shape[2]
            for z in range(num_slices):
                fname = f"{pid}_z{z:03d}.png"
                if fname not in existing_slices:
                    continue
                if np.mean(img_data[:, :, z] > 0) < MIN_BRAIN_FRACTION:
                    continue
                seg = seg_data[:, :, z]
                # 3-region 분해
                wt = (seg > 0).astype(np.uint8) * 255
                tc = (((seg == 1) | (seg == 3))).astype(np.uint8) * 255
                et = (seg == 3).astype(np.uint8) * 255
                wt = cv2.resize(wt, (IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_NEAREST)
                tc = cv2.resize(tc, (IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_NEAREST)
                et = cv2.resize(et, (IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_NEAREST)
                for arr, outdir in [(wt, WT_DIR), (tc, TC_DIR), (et, ET_DIR)]:
                    ok, enc = cv2.imencode(".png", arr)
                    if ok:
                        enc.tofile(str(outdir / fname))
                stats["saved"] += 1
                if wt.max() > 0: stats["wt_pos"] += 1
                if tc.max() > 0: stats["tc_pos"] += 1
                if et.max() > 0: stats["et_pos"] += 1
            stats["patients"] += 1
        except Exception as e:
            print(f"  [ERROR] {pid}: {e}")
            stats["errors"] += 1

    print(f"\n{'=' * 60}\n  3-region 마스크 생성 완료\n{'=' * 60}")
    print(f"  처리 환자:    {stats['patients']}")
    print(f"  저장 슬라이스: {stats['saved']:,}")
    print(f"  WT 양성:      {stats['wt_pos']:,}")
    print(f"  TC 양성:      {stats['tc_pos']:,}")
    print(f"  ET 양성:      {stats['et_pos']:,}")
    print(f"  에러:         {stats['errors']}")
    with open(LOG_DIR / "step26_segmask_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  통계 저장: {LOG_DIR / 'step26_segmask_stats.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
