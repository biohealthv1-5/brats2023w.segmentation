# -*- coding: utf-8 -*-
"""
Step 27 (Day 7 / sota): Multi-Modal SOTA Dataset / DataLoader
==============================================================
v2ways.md §3.4 + §4.5 — 3-region (WT/TC/ET) 마스크 + 소종양 가중 sampler.

- 입력:  3채널 [T1ce, FLAIR, |T1ce - FLAIR|]     (224x224)
- 출력:
    1) 분류 라벨           (scalar)
    2) Seg 마스크 (3채널)  [WT, TC, ET]          (3x224x224)

T1ce 슬라이스: Day 6 결과 (`processed/mmmt/t1ce_slices/`) 재사용.
WT/TC/ET 마스크: `processed/sota/seg_masks_{wt,tc,et}/`.

사용법:
    from step27_sota_dataset import create_sota_dataloaders
"""

import sys
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
FLAIR_DIR = PROJECT_DIR / "processed" / "slices"
T1CE_DIR = PROJECT_DIR / "processed" / "mmmt" / "t1ce_slices"   # Day 6 산출물 재사용
WT_DIR = PROJECT_DIR / "processed" / "sota" / "seg_masks_wt"
TC_DIR = PROJECT_DIR / "processed" / "sota" / "seg_masks_tc"
ET_DIR = PROJECT_DIR / "processed" / "sota" / "seg_masks_et"
SPLITS_CSV = PROJECT_DIR / "processed" / "splits.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _imread_gray(path: Path):
    if not Path(path).exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


class SOTADataset(Dataset):
    """Multi-Modal SOTA Dataset (3채널 입력 + 3채널 WT/TC/ET 마스크)"""

    def __init__(self, dataframe: pd.DataFrame,
                 flair_dir=FLAIR_DIR, t1ce_dir=T1CE_DIR,
                 wt_dir=WT_DIR, tc_dir=TC_DIR, et_dir=ET_DIR,
                 augment_spatial=False, normalize=True,
                 channel_mode="t1ce_flair_diff"):
        self.df = dataframe.reset_index(drop=True)
        self.flair_dir = Path(flair_dir)
        self.t1ce_dir = Path(t1ce_dir)
        self.wt_dir = Path(wt_dir)
        self.tc_dir = Path(tc_dir)
        self.et_dir = Path(et_dir)
        self.augment_spatial = augment_spatial
        self.normalize = normalize
        self.channel_mode = channel_mode
        if normalize:
            self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _compose_3ch(self, t1ce, flair):
        t1ce_f = t1ce.astype(np.float32) / 255.0
        flair_f = flair.astype(np.float32) / 255.0
        if self.channel_mode == "t1ce_flair_diff":
            c3 = np.abs(t1ce_f - flair_f)
        elif self.channel_mode == "t1ce_flair_ratio":
            c3 = t1ce_f / (flair_f + 1e-3)
            c3 = np.clip(c3, 0.0, 1.0)
        elif self.channel_mode == "t1ce_flair_mul":
            return np.stack([t1ce_f * flair_f, t1ce_f, flair_f], axis=-1)
        else:
            raise ValueError(f"unknown channel_mode {self.channel_mode}")
        return np.stack([t1ce_f, flair_f, c3], axis=-1)

    def _load_masks(self, fname):
        masks = []
        for d in (self.wt_dir, self.tc_dir, self.et_dir):
            m = _imread_gray(d / fname)
            if m is None:
                m = np.zeros((224, 224), dtype=np.uint8)
            masks.append((m > 127).astype(np.float32))
        return np.stack(masks, axis=0)  # (3, H, W)

    def _apply_spatial(self, img_3ch, mask_3ch):
        if np.random.rand() > 0.5:
            img_3ch = np.ascontiguousarray(img_3ch[:, ::-1, :])
            mask_3ch = np.ascontiguousarray(mask_3ch[:, :, ::-1])
        if np.random.rand() > 0.5:
            img_3ch = np.ascontiguousarray(img_3ch[::-1, :, :])
            mask_3ch = np.ascontiguousarray(mask_3ch[:, ::-1, :])
        angle = np.random.uniform(-15, 15)
        h, w = img_3ch.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_3ch = cv2.warpAffine(img_3ch, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        new_masks = np.zeros_like(mask_3ch)
        for i in range(3):
            new_masks[i] = cv2.warpAffine(mask_3ch[i], M, (w, h),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0, flags=cv2.INTER_NEAREST)
        return img_3ch, new_masks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]
        label = float(row["label"])
        flair = _imread_gray(self.flair_dir / fname)
        t1ce = _imread_gray(self.t1ce_dir / fname)
        if flair is None: flair = np.zeros((224, 224), dtype=np.uint8)
        if t1ce is None: t1ce = np.zeros((224, 224), dtype=np.uint8)
        img_3ch = self._compose_3ch(t1ce, flair)
        masks = self._load_masks(fname)
        if self.augment_spatial:
            img_3ch, masks = self._apply_spatial(img_3ch, masks)
        if self.normalize:
            img_3ch = (img_3ch - self.mean) / self.std
        img_tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1).astype(np.float32))
        mask_tensor = torch.from_numpy(masks.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label_tensor, mask_tensor


# ─── 소종양 가중 sampler (v2ways §4.5) ───────────────────────
def _build_sampler(train_df, small_tumor_weight=3.0, wt_dir=WT_DIR):
    weights = []
    for _, row in train_df.iterrows():
        if int(row["label"]) == 0:
            weights.append(1.0)
            continue
        mask = _imread_gray(wt_dir / row["filename"])
        if mask is None:
            weights.append(1.0)
            continue
        n_pos = int((mask > 127).sum())
        if 0 < n_pos < 256:
            weights.append(float(small_tumor_weight))
        else:
            weights.append(1.0)
    weights = np.array(weights, dtype=np.float64)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    return sampler


def create_sota_dataloaders(
    splits_csv: Path = SPLITS_CSV,
    batch_size: int = 16,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    channel_mode: str = "t1ce_flair_diff",
):
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = SOTADataset(train_df, augment_spatial=True, channel_mode=channel_mode)
    val_ds = SOTADataset(val_df, augment_spatial=False, channel_mode=channel_mode)
    test_ds = SOTADataset(test_df, augment_spatial=False, channel_mode=channel_mode)

    sampler = None
    if use_weighted_sampler:
        print("  [Sampler] Building WeightedRandomSampler (small-tumor x3)...")
        sampler = _build_sampler(train_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    n_pos = int(train_df["label"].sum())
    n_neg = int(len(train_df) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    return train_loader, val_loader, test_loader, pos_weight


def main():
    print("=" * 60)
    print("  Step 27 (Day 7 / sota): SOTA Dataset 테스트")
    print("=" * 60)
    if not T1CE_DIR.exists() or len(list(T1CE_DIR.glob("*.png"))) == 0:
        print(f"[ERROR] T1ce 슬라이스 없음: {T1CE_DIR}")
        print("  Day 6 mmmt/step20_mmmt_preprocess.py 먼저 실행 필요.")
        return
    if not WT_DIR.exists():
        print(f"[ERROR] 3-region 마스크 없음: {WT_DIR}")
        print("  step26_sota_segmask.py 먼저 실행 필요.")
        return
    train_l, val_l, test_l, pw = create_sota_dataloaders(
        batch_size=4, num_workers=0, use_weighted_sampler=False,
    )
    imgs, labels, masks = next(iter(train_l))
    print(f"  img tensor:    {imgs.shape}  range[{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"  label tensor:  {labels.shape}  values={labels.tolist()}")
    print(f"  mask tensor:   {masks.shape}  unique={torch.unique(masks).tolist()}")
    print(f"  pos_weight:    {pw.item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
