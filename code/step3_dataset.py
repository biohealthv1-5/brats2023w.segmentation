# -*- coding: utf-8 -*-
"""
Step 3: 데이터 분할 및 PyTorch Dataset/DataLoader 구현
=======================================================
환자 수준으로 Train/Val/Test를 분할하고,
PyTorch Dataset 및 DataLoader를 구현합니다.

사용법:
    python code/step3_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
LABELS_CSV = PROJECT_DIR / "processed" / "labels.csv"
SLICE_DIR = PROJECT_DIR / "processed" / "slices"
SPLITS_CSV = PROJECT_DIR / "processed" / "splits.csv"

# ─── 설정 ────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet 정규화 파라미터
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── 데이터 분할 ─────────────────────────────────────────────
def split_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    환자 수준으로 Train/Val/Test를 분할합니다.
    같은 환자의 모든 슬라이스가 같은 split에 속하게 합니다.
    """
    # 환자 ID 목록
    patient_ids = df["patient_id"].unique()
    np.random.seed(RANDOM_SEED)

    # 1차 분할: Train vs (Val+Test)
    train_patients, temp_patients = train_test_split(
        patient_ids,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
    )

    # 2차 분할: Val vs Test
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_SEED,
    )

    # split 컬럼 추가
    df = df.copy()
    df["split"] = "train"
    df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_id"].isin(test_patients), "split"] = "test"

    return df


# ─── PyTorch Dataset ─────────────────────────────────────────
class BrainTumorSliceDataset(Dataset):
    """
    뇌 MRI 2D 슬라이스 이진 분류 Dataset

    Args:
        dataframe: 'filename', 'label' 컬럼을 포함하는 DataFrame
        slice_dir: 슬라이스 이미지가 저장된 디렉토리
        transform: torchvision 변환
    """

    def __init__(self, dataframe: pd.DataFrame, slice_dir: Path, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.slice_dir = slice_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row["filename"]
        label = row["label"]

        # 이미지 로드 (한글 경로 지원: np.fromfile + cv2.imdecode)
        img_path = self.slice_dir / filename
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # 오류 시 빈 이미지 반환
            img = np.zeros((224, 224), dtype=np.uint8)

        # 3채널로 복제 (ResNet 호환)
        img_3ch = np.stack([img, img, img], axis=-1)  # (H, W, 3)

        # PIL Image로 변환 (torchvision transforms 호환)
        from PIL import Image

        img_pil = Image.fromarray(img_3ch)

        # 변환 적용
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        # 라벨을 float tensor로 (BCEWithLogitsLoss 호환)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor


# ─── Transform 정의 ──────────────────────────────────────────
def get_train_transform():
    """학습용 데이터 변환 (Data Augmentation 포함)"""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform():
    """검증/테스트용 데이터 변환 (Augmentation 없음)"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ─── DataLoader 생성 함수 ────────────────────────────────────
def create_dataloaders(
    splits_csv: Path = SPLITS_CSV,
    slice_dir: Path = SLICE_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Train/Val/Test DataLoader를 생성합니다.

    Returns:
        train_loader, val_loader, test_loader, pos_weight
    """
    df = pd.read_csv(splits_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Dataset 생성
    train_dataset = BrainTumorSliceDataset(train_df, slice_dir, get_train_transform())
    val_dataset = BrainTumorSliceDataset(val_df, slice_dir, get_val_transform())
    test_dataset = BrainTumorSliceDataset(test_df, slice_dir, get_val_transform())

    # pos_weight 계산 (클래스 불균형 보정)
    n_positive = train_df["label"].sum()
    n_negative = len(train_df) - n_positive
    pos_weight = torch.tensor([n_negative / max(n_positive, 1)], dtype=torch.float32)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, pos_weight


# ─── 메인: 데이터 분할 실행 ──────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: 데이터 분할 및 Dataset 구현")
    print("=" * 60)

    # 라벨 CSV 확인
    if not LABELS_CSV.exists():
        print(f"\n[ERROR] 라벨 파일이 없습니다: {LABELS_CSV}")
        print("  step1_preprocess.py를 먼저 실행하세요.")
        return

    # 데이터 로드
    df = pd.read_csv(LABELS_CSV)
    print(f"\n  총 슬라이스 수: {len(df):,}")
    print(f"  환자 수: {df['patient_id'].nunique()}")

    # 환자 수준 분할
    print(f"\n{'─' * 60}")
    print("  환자 수준 데이터 분할 중...")
    df_split = split_by_patient(df)

    # 분할 결과 저장
    df_split.to_csv(SPLITS_CSV, index=False, encoding="utf-8")
    print(f"  분할 CSV 저장: {SPLITS_CSV}")

    # 분할 결과 출력
    print(f"\n{'─' * 60}")
    print("  분할 결과:")
    print(f"{'─' * 60}")

    for split_name in ["train", "val", "test"]:
        split_df = df_split[df_split["split"] == split_name]
        n_patients = split_df["patient_id"].nunique()
        n_slices = len(split_df)
        n_pos = split_df["label"].sum()
        n_neg = n_slices - n_pos

        print(f"\n  [{split_name.upper()}]")
        print(f"    환자 수: {n_patients}")
        print(f"    슬라이스 수: {n_slices:,}")
        print(f"    Positive: {n_pos:,} ({n_pos/n_slices*100:.1f}%)")
        print(f"    Negative: {n_neg:,} ({n_neg/n_slices*100:.1f}%)")

    # DataLoader 테스트 (PyTorch가 설치된 경우)
    print(f"\n{'─' * 60}")
    print("  DataLoader 테스트...")
    try:
        train_loader, val_loader, test_loader, pos_weight = create_dataloaders(
            batch_size=4, num_workers=0
        )
        batch_imgs, batch_labels = next(iter(train_loader))
        print(f"  ✓ 배치 이미지 shape: {batch_imgs.shape}")
        print(f"  ✓ 배치 라벨: {batch_labels}")
        print(f"  ✓ pos_weight: {pos_weight.item():.4f}")
        print(f"  ✓ DataLoader 정상 동작!")
    except Exception as e:
        print(f"  [WARNING] DataLoader 테스트 실패: {e}")
        print("  PyTorch가 설치되지 않았을 수 있습니다.")

    print(f"\n{'=' * 60}")
    print(f"  데이터 분할 및 Dataset 구현 완료!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
