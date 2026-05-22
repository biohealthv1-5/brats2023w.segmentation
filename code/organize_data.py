# 현재 폴더 밖에 있어야 하는 파일입니다. 
# 이 파일은 Synapse 캐시에서 데이터를 정리하여 프로젝트 폴더로 복사하는 역할을 합니다.


# -*- coding: utf-8 -*-
"""
BraTS 2023 데이터 정리 스크립트
================================
Synapse 캐시(~/.synapseCache)에서 프로젝트 폴더(data/)로
파일을 복사하고 압축을 해제합니다.

사용법:
    python organize_data.py
"""

import os
import sys
import shutil
import zipfile
import tarfile
from pathlib import Path

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 설정 ───────────────────────────────────────────────
SYNAPSE_CACHE = Path(os.path.expanduser("~")) / ".synapseCache"
# 이 스크립트가 code/ 아래로 이동되었으므로 .parent.parent 로 프로젝트 루트를 가리킨다
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# ─── 캐시 내 파일 매핑 (Synapse ID 기반 폴더 → 파일명) ───────
CACHE_FILES = {
    # GLI
    "453/124663453/asnr-miccai-brats2023-gli-challenge-trainingdata.zip": {
        "dest": "BraTS-GLI/training",
        "action": "unzip",
    },
    "768/124662768/asnr-miccai-brats2023-gli-challenge-validationdata.zip": {
        "dest": "BraTS-GLI/validation",
        "action": "unzip",
    },
    # MEN
    "483/126637483/asnr-miccai-brats2023-men-challenge-trainingdata.zip": {
        "dest": "BraTS-MEN/training",
        "action": "unzip",
    },
    "597/126633597/asnr-miccai-brats2023-men-challenge-validationdata.zip": {
        "dest": "BraTS-MEN/validation",
        "action": "unzip",
    },
    "507/125817507/brats-men-train-fix-v4.zip": {
        "dest": "BraTS-MEN/training",  # 패치: training 위에 덮어쓰기
        "action": "unzip",
    },
    # PED
    "426/125238426/asnr-miccai-brats2023-ped-challenge-trainingdata.zip": {
        "dest": "BraTS-PED/training",
        "action": "unzip",
    },
    "145/126632145/asnr-miccai-brats2023-ped-challenge-validationdata.zip": {
        "dest": "BraTS-PED/validation",
        "action": "unzip",
    },
    # 보조 파일
    "905/125241905/brats2023_2017_gli_mapping.xlsx": {
        "dest": "supplementary",
        "action": "copy",
    },
    "465/139834465/meningioma supplementary clinical data and imaging parameters for training and validation sets.xlsx": {
        "dest": "supplementary",
        "action": "copy",
        "rename": "meningioma_clinical_data.xlsx",
    },
    "669/158521669/readme.md": {
        "dest": "supplementary",
        "action": "copy",
    },
    # Sample datasets (MLCubes)
    "89/127811089/brats-gli-fastlane.tar.gz": {
        "dest": "sample/brats-gli-fastlane",
        "action": "untar",
    },
    "94/127811094/brats-local-synthesis-fastlane.tar.gz": {
        "dest": "sample/brats-local-synthesis",
        "action": "untar",
    },
}


def extract_zip(src: Path, dest: Path):
    """ZIP 파일을 dest 디렉토리에 해제"""
    print(f"  [ZIP] 압축 해제 중: {src.name} -> {dest}")
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(dest)
    item_count = len(list(dest.rglob("*")))
    print(f"  [OK] 완료 ({item_count} 항목)")


def extract_tar(src: Path, dest: Path):
    """tar.gz 파일을 dest 디렉토리에 해제"""
    print(f"  [TAR] 압축 해제 중: {src.name} -> {dest}")
    with tarfile.open(src, "r:gz") as tf:
        tf.extractall(dest)
    item_count = len(list(dest.rglob("*")))
    print(f"  [OK] 완료 ({item_count} 항목)")


def copy_file(src: Path, dest: Path, rename: str = None):
    """파일을 dest 디렉토리로 복사"""
    target_name = rename if rename else src.name
    target = dest / target_name
    print(f"  [COPY] 복사 중: {src.name} -> {target}")
    shutil.copy2(src, target)
    print(f"  [OK] 완료")


def main():
    print("=" * 60)
    print("  BraTS 2023 데이터 정리 스크립트")
    print("=" * 60)
    print(f"\n  캐시 경로: {SYNAPSE_CACHE}")
    print(f"  대상 경로: {DATA_DIR}\n")

    if not SYNAPSE_CACHE.exists():
        print("[ERROR] Synapse 캐시 디렉토리를 찾을 수 없습니다.")
        print(f"   경로: {SYNAPSE_CACHE}")
        return

    # 처리 결과 추적
    success_count = 0
    skip_count = 0
    fail_count = 0

    for rel_path, config in CACHE_FILES.items():
        src = SYNAPSE_CACHE / rel_path
        dest = DATA_DIR / config["dest"]
        action = config["action"]

        print(f"\n{'─' * 50}")
        print(f"  파일: {Path(rel_path).name}")

        if not src.exists():
            print(f"  [SKIP] 캐시에 파일이 없습니다")
            print(f"     ({src})")
            skip_count += 1
            continue

        # 대상 디렉토리 생성
        dest.mkdir(parents=True, exist_ok=True)

        try:
            if action == "unzip":
                extract_zip(src, dest)
            elif action == "untar":
                extract_tar(src, dest)
            elif action == "copy":
                copy_file(src, dest, config.get("rename"))
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] 오류: {e}")
            fail_count += 1

    # 요약
    print(f"\n{'=' * 60}")
    print(f"  결과 요약")
    print(f"{'=' * 60}")
    print(f"  성공: {success_count}")
    print(f"  건너뜀: {skip_count}")
    print(f"  실패: {fail_count}")
    print(f"\n  데이터 경로: {DATA_DIR}")

    # 최종 디렉토리 구조 출력
    print(f"\n{'─' * 50}")
    print("  생성된 디렉토리 구조:")
    print(f"{'─' * 50}")
    if DATA_DIR.exists():
        for dirpath, dirnames, filenames in os.walk(DATA_DIR):
            level = len(Path(dirpath).relative_to(DATA_DIR).parts)
            indent = "  " + "  | " * level
            folder_name = Path(dirpath).name
            if level == 0:
                print(f"  data/")
            else:
                print(f"{indent}+-- {folder_name}/")
            # 파일 수만 표시 (너무 많으므로)
            if filenames and level >= 1:
                sub_indent = "  " + "  | " * (level + 1)
                nifti_count = sum(
                    1 for f in filenames if f.endswith((".nii.gz", ".nii"))
                )
                other_count = len(filenames) - nifti_count
                if nifti_count:
                    print(f"{sub_indent}({nifti_count} NIfTI files)")
                if other_count:
                    for f in filenames[:5]:
                        if not f.endswith((".nii.gz", ".nii")):
                            print(f"{sub_indent}+-- {f}")
                    if other_count > 5:
                        print(f"{sub_indent}... +{other_count - 5} more")
    else:
        print("  (디렉토리가 생성되지 않았습니다)")

    print(f"\n{'=' * 60}")
    print("  완료!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
