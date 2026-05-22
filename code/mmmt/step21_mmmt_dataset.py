# -*- coding: utf-8 -*-
"""
Step 21 (Day 6 / mmmt): Multi-Modal Multi-Task Dataset / DataLoader
====================================================================
Day 6 = 입력 표현을 단일 모달 → 멀티모달로 확장. seg head는 *기존과 동일하게
WT 1채널* 만 사용 (3-region 확장은 Day 7 sota/ 에서 진행).

- 입력:  3채널 [T1ce, FLAIR, |T1ce - FLAIR|]     (224x224)
- 출력:
    1) 분류 라벨           (scalar)
    2) Seg 마스크 (1채널)  [WT]                  (1x224x224)

기존 코드는 변경하지 않으며, 기존 multitask/seg_masks (WT 바이너리)와 step20
에서 만든 T1ce 슬라이스만 사용합니다.

사용법:
    from step21_mmmt_dataset import create_mm_dataloaders
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
T1CE_DIR = PROJECT_DIR / "processed" / "mmmt" / "t1ce_slices"
WT_DIR = PROJECT_DIR / "processed" / "seg_masks"      # 기존 multitask WT 재사용
SPLITS_CSV = PROJECT_DIR / "processed" / "splits.csv"

# ImageNet 정규화 — 3채널 모두에 일관 적용
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _imread_gray(path: Path):
    """한글 경로 안전 grayscale 로더"""
    if not Path(path).exists():
        return None
    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


class MMMTDataset(Dataset):
    """Multi-Modal Multi-Task BraTS Dataset (Day 6 — WT 단일 region)"""

    def __init__(self, dataframe: pd.DataFrame,
                 flair_dir=FLAIR_DIR, t1ce_dir=T1CE_DIR,
                 wt_dir=WT_DIR,
                 augment_spatial=False, normalize=True,
                 channel_mode="t1ce_flair_diff"):
        """
        Args:
            channel_mode:
                - 't1ce_flair_diff'  : [T1ce, FLAIR, |T1ce - FLAIR|]   (권장)
                - 't1ce_flair_ratio' : [T1ce, FLAIR, T1ce/(FLAIR+eps)]
                - 't1ce_flair_mul'   : [T1ce*FLAIR, T1ce, FLAIR]
        """
        self.df = dataframe.reset_index(drop=True)
        self.flair_dir = Path(flair_dir)
        self.t1ce_dir = Path(t1ce_dir)
        self.wt_dir = Path(wt_dir)
        self.augment_spatial = augment_spatial
        self.normalize = normalize
        self.channel_mode = channel_mode
        if normalize:
            self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            self.std = np.array(IMAGENET_STD, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _compose_3ch(self, t1ce: np.ndarray, flair: np.ndarray) -> np.ndarray:
        """uint8 (H,W) 두 장 -> float32 (H,W,3)"""
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

    def _load_wt_mask(self, fname: str) -> np.ndarray:
        """(1, H, W) WT 바이너리 마스크"""
        m = _imread_gray(self.wt_dir / fname)
        if m is None:
            m = np.zeros((224, 224), dtype=np.uint8)
        m_bin = (m > 127).astype(np.float32)
        return m_bin[None, :, :]   # (1, H, W)

    def _apply_spatial(self, img_3ch: np.ndarray, mask_1ch: np.ndarray):
        """img: (H,W,3), mask: (1,H,W) 동시 변환"""
        # flip H
        if np.random.rand() > 0.5:
            img_3ch = np.ascontiguousarray(img_3ch[:, ::-1, :])
            mask_1ch = np.ascontiguousarray(mask_1ch[:, :, ::-1])
        # flip V
        if np.random.rand() > 0.5:
            img_3ch = np.ascontiguousarray(img_3ch[::-1, :, :])
            mask_1ch = np.ascontiguousarray(mask_1ch[:, ::-1, :])
        # rotate ±15°
        angle = np.random.uniform(-15, 15)
        h, w = img_3ch.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img_3ch = cv2.warpAffine(img_3ch, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        new_mask = cv2.warpAffine(mask_1ch[0], M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0, flags=cv2.INTER_NEAREST)
        mask_1ch = new_mask[None, :, :]
        return img_3ch, mask_1ch

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["filename"]
        label = float(row["label"])

        flair = _imread_gray(self.flair_dir / fname)
        t1ce = _imread_gray(self.t1ce_dir / fname)
        if flair is None: flair = np.zeros((224, 224), dtype=np.uint8)
        if t1ce is None: t1ce = np.zeros((224, 224), dtype=np.uint8)

        img_3ch = self._compose_3ch(t1ce, flair)   # (H, W, 3) float [0,1]
        mask = self._load_wt_mask(fname)           # (1, H, W) float {0,1}

        if self.augment_spatial:
            img_3ch, mask = self._apply_spatial(img_3ch, mask)

        if self.normalize:
            img_3ch = (img_3ch - self.mean) / self.std

        # (H,W,3) -> (3,H,W)
        img_tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1).astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return img_tensor, label_tensor, mask_tensor


# ─── DataLoader 생성 ─────────────────────────────────────────
def create_mm_dataloaders(
    splits_csv: Path = SPLITS_CSV,
    batch_size: int = 16,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,    # Day 6는 기본 sampler (균형 잡힌 split)
    channel_mode: str = "t1ce_flair_diff",
):
    df = pd.read_csv(splits_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = MMMTDataset(train_df, augment_spatial=True,
                           channel_mode=channel_mode)
    val_ds = MMMTDataset(val_df, augment_spatial=False,
                         channel_mode=channel_mode)
    test_ds = MMMTDataset(test_df, augment_spatial=False,
                          channel_mode=channel_mode)

    sampler = None
    if use_weighted_sampler:
        # 단순 binary 클래스 균형 sampler (소종양 가중은 Day 7 sota에서 도입)
        n_pos = int(train_df["label"].sum())
        n_neg = len(train_df) - n_pos
        w_pos = 1.0 / max(n_pos, 1)
        w_neg = 1.0 / max(n_neg, 1)
        weights = train_df["label"].map(lambda x: w_pos if x == 1 else w_neg).values
        sampler = WeightedRandomSampler(weights.astype(np.float64),
                                        num_samples=len(weights),
                                        replacement=True)

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


# ─── 테스트 ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 21 (Day 6 / mmmt): MM-MTL Dataset 테스트")
    print("=" * 60)
    if not T1CE_DIR.exists() or len(list(T1CE_DIR.glob("*.png"))) == 0:
        print(f"[ERROR] T1ce 슬라이스 없음: {T1CE_DIR}")
        print("  step20_mmmt_preprocess.py 먼저 실행 필요.")
        return
    if not WT_DIR.exists():
        print(f"[ERROR] WT 마스크 없음 (기존 multitask 결과): {WT_DIR}")
        print("  multitask/step15_prep_segmask.py 먼저.")
        return
    train_l, val_l, test_l, pw = create_mm_dataloaders(
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
