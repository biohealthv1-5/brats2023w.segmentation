# -*- coding: utf-8 -*-
"""
Step 16: Multi-Task Dataset / DataLoader
=========================================
기존 슬라이스 + seg 마스크를 함께 반환하는 Dataset 구현.
기존 step3_dataset.py의 코드 구조를 유지하면서 마스크를 추가.

사용법:
    from step16_multitask_dataset import create_multitask_dataloaders
"""

import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SLICE_DIR = PROJECT_DIR / "processed" / "slices"
MASK_DIR = PROJECT_DIR / "processed" / "seg_masks"
SPLITS_CSV = PROJECT_DIR / "processed" / "splits.csv"

# ImageNet 정규화 파라미터
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MultiTaskBrainDataset(Dataset):
    """
    Multi-Task 학습용 Dataset
    - 이미지 (224×224, 3ch) + 분류 라벨 + seg 마스크 (224×224, 1ch) 반환
    """

    def __init__(self, dataframe: pd.DataFrame, slice_dir: Path,
                 mask_dir: Path, transform=None, augment_spatial=None):
        """
        Args:
            dataframe: 'filename', 'label' 컬럼을 포함하는 DataFrame
            slice_dir: 슬라이스 이미지 디렉토리
            mask_dir: seg 마스크 디렉토리
            transform: 이미지에 적용할 색상/정규화 변환
            augment_spatial: 이미지+마스크에 동시 적용할 공간 변환 (flip, rotate 등)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.slice_dir = slice_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment_spatial = augment_spatial

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row["filename"]
        label = row["label"]

        # ─── 이미지 로드 ─────────────────────────────────────
        img_path = self.slice_dir / filename
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)

        # ─── 마스크 로드 ─────────────────────────────────────
        mask_path = self.mask_dir / filename
        if mask_path.exists():
            mask_array = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((224, 224), dtype=np.uint8)
        else:
            mask = np.zeros((224, 224), dtype=np.uint8)

        # 마스크를 0/1 바이너리로 변환
        mask = (mask > 127).astype(np.float32)  # (H, W)

        # ─── 공간 증강 (이미지+마스크 동시) ───────────────────
        if self.augment_spatial is not None:
            img, mask = self._apply_spatial_augment(img, mask)

        # ─── 이미지 → 3ch tensor ─────────────────────────────
        img_3ch = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        from PIL import Image
        img_pil = Image.fromarray(img_3ch)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        # ─── 마스크 → tensor ─────────────────────────────────
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # ─── 분류 라벨 ───────────────────────────────────────
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor, mask_tensor

    def _apply_spatial_augment(self, img, mask):
        """이미지와 마스크에 동일한 공간 변환 적용"""
        # Random Horizontal Flip
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random Vertical Flip
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random Rotation (±15°)
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                              flags=cv2.INTER_NEAREST)

        return img, mask


# ─── Transform 정의 ──────────────────────────────────────────
def get_multitask_train_transform():
    """학습용 변환 (색상/정규화만, 공간 변환은 Dataset 내부에서 처리)"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_multitask_val_transform():
    """검증/테스트용 변환"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── DataLoader 생성 함수 ────────────────────────────────────
def create_multitask_dataloaders(
    splits_csv: Path = SPLITS_CSV,
    slice_dir: Path = SLICE_DIR,
    mask_dir: Path = MASK_DIR,
    batch_size: int = 16,
    num_workers: int = 4,
):
    """
    Multi-Task Train/Val/Test DataLoader를 생성합니다.

    Returns:
        train_loader, val_loader, test_loader, pos_weight
    """
    df = pd.read_csv(splits_csv)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Dataset 생성
    train_dataset = MultiTaskBrainDataset(
        train_df, slice_dir, mask_dir,
        transform=get_multitask_train_transform(),
        augment_spatial=True,  # 학습 시 공간 증강
    )
    val_dataset = MultiTaskBrainDataset(
        val_df, slice_dir, mask_dir,
        transform=get_multitask_val_transform(),
        augment_spatial=None,  # 검증 시 증강 없음
    )
    test_dataset = MultiTaskBrainDataset(
        test_df, slice_dir, mask_dir,
        transform=get_multitask_val_transform(),
        augment_spatial=None,
    )

    # pos_weight 계산 (분류 손실용)
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


# ─── 메인: DataLoader 테스트 ─────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 16: Multi-Task Dataset 테스트")
    print("=" * 60)

    # 마스크 디렉토리 확인
    if not MASK_DIR.exists():
        print(f"\n[ERROR] 마스크 디렉토리가 없습니다: {MASK_DIR}")
        print("  step15_prep_segmask.py를 먼저 실행하세요.")
        return

    mask_count = len(list(MASK_DIR.glob("*.png")))
    print(f"\n  마스크 파일 수: {mask_count:,}")

    # DataLoader 테스트
    print(f"\n{'─' * 60}")
    print("  DataLoader 테스트...")
    try:
        train_loader, val_loader, test_loader, pos_weight = create_multitask_dataloaders(
            batch_size=4, num_workers=0
        )
        batch_imgs, batch_labels, batch_masks = next(iter(train_loader))
        print(f"  ✓ 이미지 shape: {batch_imgs.shape}")    # (4, 3, 224, 224)
        print(f"  ✓ 라벨 shape:   {batch_labels.shape}")   # (4,)
        print(f"  ✓ 마스크 shape:  {batch_masks.shape}")    # (4, 1, 224, 224)
        print(f"  ✓ 라벨 값:      {batch_labels.tolist()}")
        print(f"  ✓ 마스크 범위:   [{batch_masks.min():.1f}, {batch_masks.max():.1f}]")
        print(f"  ✓ pos_weight:   {pos_weight.item():.4f}")

        # 마스크 내 종양 픽셀 수 확인
        for i in range(len(batch_labels)):
            tumor_pixels = (batch_masks[i] > 0.5).sum().item()
            label = int(batch_labels[i].item())
            print(f"    샘플 {i}: label={label}, 종양 픽셀={tumor_pixels:,}")

        print(f"\n  ✓ DataLoader 정상 동작!")

        # 통계
        print(f"\n{'─' * 60}")
        print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
        print(f"  Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
        print(f"  Test:  {len(test_loader.dataset):,} samples ({len(test_loader)} batches)")

    except Exception as e:
        print(f"  [ERROR] DataLoader 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
