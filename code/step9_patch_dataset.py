# -*- coding: utf-8 -*-
"""
Step 9: 패치 단위 PyTorch Dataset / DataLoader
================================================
patch_labels.csv를 읽어 기존 224×224 PNG에서 패치를 on-the-fly 크롭합니다.
클래스 불균형 대응을 위해 WeightedRandomSampler를 제공합니다.

사용법:
    python code/step9_patch_dataset.py          # 단독 테스트
    from step9_patch_dataset import create_patch_dataloaders  # import
"""

import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
SLICE_DIR = PROJECT_DIR / "processed" / "slices"
PATCH_CSV = PROJECT_DIR / "processed" / "patch_labels.csv"

# ImageNet 정규화
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PATCH_SIZE = 64


# ─── Dataset ─────────────────────────────────────────────────
class BrainTumorPatchDataset(Dataset):
    """패치 단위 Dataset — 224×224 PNG에서 64×64 패치를 on-the-fly 크롭"""

    def __init__(self, dataframe: pd.DataFrame, slice_dir: Path, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.slice_dir = slice_dir
        self.transform = transform
        # 슬라이스 이미지 캐시 (같은 슬라이스에서 여러 패치 추출 → 로드 1회)
        self._cache_fname = None
        self._cache_img = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["slice_filename"]
        x, y = int(row["x_start"]), int(row["y_start"])
        label = int(row["patch_label"])

        # 슬라이스 로드 (캐시 활용)
        if fname != self._cache_fname:
            img_path = self.slice_dir / fname
            buf = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            self._cache_fname = fname
            self._cache_img = img
        else:
            img = self._cache_img

        # 패치 크롭
        patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

        # 3채널 복제 + PIL 변환
        patch_3ch = np.stack([patch, patch, patch], axis=-1)
        patch_pil = Image.fromarray(patch_3ch)

        if self.transform:
            patch_tensor = self.transform(patch_pil)
        else:
            patch_tensor = transforms.ToTensor()(patch_pil)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return patch_tensor, label_tensor


# ─── Transforms ──────────────────────────────────────────────
def get_patch_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_patch_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── DataLoader 생성 ─────────────────────────────────────────
def create_patch_dataloaders(
    patch_csv: Path = PATCH_CSV,
    slice_dir: Path = SLICE_DIR,
    batch_size: int = 128,
    num_workers: int = 4,
    neg_subsample_ratio: float = 3.0,
):
    """
    패치 DataLoader를 생성합니다.

    Args:
        neg_subsample_ratio: Negative를 Positive의 몇 배까지 허용할지 (3.0 = 1:3)
                             None이면 서브샘플링 없이 WeightedRandomSampler만 사용

    Returns:
        train_loader, val_loader, test_loader, pos_weight
    """
    df = pd.read_csv(patch_csv)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # ─── Train: Negative 서브샘플링 ──────────────────────────
    if neg_subsample_ratio is not None:
        pos_train = train_df[train_df["patch_label"] == 1]
        neg_train = train_df[train_df["patch_label"] == 0]
        max_neg = int(len(pos_train) * neg_subsample_ratio)
        if len(neg_train) > max_neg:
            neg_train = neg_train.sample(n=max_neg, random_state=42)
        train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42)

    # pos_weight 계산
    n_pos = train_df["patch_label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    # WeightedRandomSampler (학습용)
    labels = train_df["patch_label"].values
    class_counts = np.bincount(labels, minlength=2)
    weights = np.where(labels == 1,
                       1.0 / max(class_counts[1], 1),
                       1.0 / max(class_counts[0], 1))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_df), replacement=True)

    # Datasets
    train_ds = BrainTumorPatchDataset(train_df, slice_dir, get_patch_train_transform())
    val_ds = BrainTumorPatchDataset(val_df, slice_dir, get_patch_val_transform())
    test_ds = BrainTumorPatchDataset(test_df, slice_dir, get_patch_val_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, pos_weight


# ─── 단독 테스트 ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 9: 패치 Dataset / DataLoader 테스트")
    print("=" * 60)

    df = pd.read_csv(PATCH_CSV)
    print(f"\n  전체 패치: {len(df):,}")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        p = sub["patch_label"].sum()
        print(f"  {s:5s}: {len(sub):,} (Pos={p:,}, Neg={len(sub)-p:,})")

    print(f"\n{'─'*60}")
    print("  DataLoader 생성 (neg_subsample 1:3)...")
    train_loader, val_loader, test_loader, pw = create_patch_dataloaders(
        batch_size=64, num_workers=0, neg_subsample_ratio=3.0
    )

    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  Test batches:  {len(test_loader):,}")
    print(f"  pos_weight:    {pw.item():.4f}")

    # 배치 테스트
    imgs, labels = next(iter(train_loader))
    print(f"\n  배치 shape:  {imgs.shape}")
    print(f"  배치 labels: {labels[:16].tolist()}")
    print(f"  Positive비율: {labels.mean():.2%}")
    print(f"\n{'='*60}")
    print("  ✓ DataLoader 정상!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
