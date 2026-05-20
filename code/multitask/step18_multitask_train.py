# -*- coding: utf-8 -*-
"""
Step 18: Multi-Task 모델 학습
==============================
MultiTaskBrainNet을 2-Phase로 학습합니다.
  Phase 1: Encoder 동결, Classification + Segmentation Head만 학습
  Phase 2: 전체 Fine-tuning

Loss = α × L_cls (BCEWithLogitsLoss) + β × L_seg (Dice + BCE)

사용법:
    python code/multitask/step18_multitask_train.py
"""

import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 프로젝트 모듈 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step16_multitask_dataset import create_multitask_dataloaders
from step17_multitask_model import (
    create_multitask_model, freeze_encoder, unfreeze_encoder, get_model_summary
)

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "multitask" / "checkpoints"
LOG_DIR = PROJECT_DIR / "outputs" / "multitask" / "logs"
FIGURES_DIR = PROJECT_DIR / "outputs" / "multitask" / "figures"

# ─── 하이퍼파라미터 ──────────────────────────────────────────
CONFIG = {
    # Phase 1: Encoder 동결, Head만 학습
    "phase1_epochs": 3,
    "phase1_lr": 1e-3,
    # Phase 2: 전체 Fine-tuning
    "phase2_epochs": 15,
    "phase2_lr": 1e-4,
    # 공통
    "batch_size": 16,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "early_stopping_patience": 5,
    # Multi-Task Loss 가중치
    "alpha_cls": 1.0,     # 분류 Loss 가중치 (주 태스크)
    "beta_seg": 0.5,      # 세분화 Loss 가중치 (보조 태스크)
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ─── Loss 함수 ──────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss = α × L_cls + β × L_seg
    L_cls = BCEWithLogitsLoss (분류)
    L_seg = DiceLoss + BCEWithLogitsLoss (세분화)
    """

    def __init__(self, alpha=1.0, beta=0.5, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # 분류 Loss
        self.cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight if pos_weight is not None else None
        )

        # 세분화 Loss
        self.dice_loss = DiceLoss()
        self.seg_bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, cls_out, seg_out, cls_target, seg_target):
        # Classification Loss
        l_cls = self.cls_loss(cls_out.squeeze(1), cls_target)

        # Segmentation Loss (Dice + BCE)
        l_dice = self.dice_loss(seg_out, seg_target)
        l_seg_bce = self.seg_bce_loss(seg_out, seg_target)
        l_seg = l_dice + l_seg_bce

        # Total
        total = self.alpha * l_cls + self.beta * l_seg

        return total, l_cls, l_seg, l_dice


# ─── Early Stopping ─────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


# ─── 학습 함수 ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    correct = 0
    total = 0
    dice_sum = 0.0
    dice_count = 0

    for batch_idx, (images, labels, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward
        cls_out, seg_out = model(images)
        total_loss, l_cls, l_seg, l_dice = criterion(cls_out, seg_out, labels, masks)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 통계
        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size
        running_cls_loss += l_cls.item() * batch_size
        running_seg_loss += l_seg.item() * batch_size

        # Classification accuracy
        preds = (torch.sigmoid(cls_out.squeeze(1)) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += batch_size

        # Dice score (배치 평균)
        with torch.no_grad():
            seg_pred = (torch.sigmoid(seg_out) >= 0.5).float()
            for i in range(batch_size):
                pred_flat = seg_pred[i].view(-1)
                target_flat = masks[i].view(-1)
                intersection = (pred_flat * target_flat).sum().item()
                union = pred_flat.sum().item() + target_flat.sum().item()
                if union > 0:
                    dice_sum += 2.0 * intersection / union
                    dice_count += 1

        # 진행 상황 출력
        if (batch_idx + 1) % 500 == 0:
            batch_acc = correct / total * 100
            batch_loss = running_loss / total
            print(f"      batch {batch_idx+1}/{len(loader)} | "
                  f"loss: {batch_loss:.4f} | acc: {batch_acc:.1f}%")

    epoch_loss = running_loss / total
    epoch_cls_loss = running_cls_loss / total
    epoch_seg_loss = running_seg_loss / total
    epoch_acc = correct / total * 100
    epoch_dice = dice_sum / max(dice_count, 1)

    return epoch_loss, epoch_cls_loss, epoch_seg_loss, epoch_acc, epoch_dice


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    correct = 0
    total = 0
    dice_sum = 0.0
    dice_count = 0

    for images, labels, masks in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        cls_out, seg_out = model(images)
        total_loss, l_cls, l_seg, l_dice = criterion(cls_out, seg_out, labels, masks)

        batch_size = images.size(0)
        running_loss += total_loss.item() * batch_size
        running_cls_loss += l_cls.item() * batch_size
        running_seg_loss += l_seg.item() * batch_size

        preds = (torch.sigmoid(cls_out.squeeze(1)) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += batch_size

        seg_pred = (torch.sigmoid(seg_out) >= 0.5).float()
        for i in range(batch_size):
            pred_flat = seg_pred[i].view(-1)
            target_flat = masks[i].view(-1)
            intersection = (pred_flat * target_flat).sum().item()
            union = pred_flat.sum().item() + target_flat.sum().item()
            if union > 0:
                dice_sum += 2.0 * intersection / union
                dice_count += 1

    epoch_loss = running_loss / total
    epoch_cls_loss = running_cls_loss / total
    epoch_seg_loss = running_seg_loss / total
    epoch_acc = correct / total * 100
    epoch_dice = dice_sum / max(dice_count, 1)

    return epoch_loss, epoch_cls_loss, epoch_seg_loss, epoch_acc, epoch_dice


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_dice, filepath):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_dice": val_dice,
    }, filepath)


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs, phase_name, early_stopping, history):
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_dice = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # 학습
        t_loss, t_cls, t_seg, t_acc, t_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        v_loss, v_cls, v_seg, v_acc, v_dice = validate(
            model, val_loader, criterion, device
        )

        # 스케줄러
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # 로그 출력
        print(
            f"  [{phase_name}] Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s)\n"
            f"    Train | Loss: {t_loss:.4f} (cls:{t_cls:.4f} seg:{t_seg:.4f}) "
            f"Acc: {t_acc:.2f}% Dice: {t_dice:.4f}\n"
            f"    Val   | Loss: {v_loss:.4f} (cls:{v_cls:.4f} seg:{v_seg:.4f}) "
            f"Acc: {v_acc:.2f}% Dice: {v_dice:.4f} "
            f"LR: {current_lr:.2e}"
        )

        # 히스토리 기록
        history["train_loss"].append(t_loss)
        history["train_cls_loss"].append(t_cls)
        history["train_seg_loss"].append(t_seg)
        history["train_acc"].append(t_acc)
        history["train_dice"].append(t_dice)
        history["val_loss"].append(v_loss)
        history["val_cls_loss"].append(v_cls)
        history["val_seg_loss"].append(v_seg)
        history["val_acc"].append(v_acc)
        history["val_dice"].append(v_dice)
        history["lr"].append(current_lr)

        # Best model 저장
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc = v_acc
            best_val_dice = v_dice
            save_checkpoint(
                model, optimizer, epoch, v_loss, v_acc, v_dice,
                CHECKPOINT_DIR / "mt_best_model.pth",
            )
            print(f"    ★ Best model 저장 (val_loss: {v_loss:.4f}, "
                  f"acc: {v_acc:.2f}%, dice: {v_dice:.4f})")

        # Early Stopping
        if early_stopping(v_loss):
            print(f"    ⚠ Early Stopping! (patience: {early_stopping.patience})")
            break

    return best_val_loss, best_val_acc, best_val_dice


# ─── 학습 곡선 시각화 ────────────────────────────────────────
def plot_training_curves(history):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    phase1_end = CONFIG["phase1_epochs"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    # 1. Total Loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=3)
    axes[0, 0].plot(epochs, history["val_loss"], "r-o", label="Val", markersize=3)
    axes[0, 0].axvline(x=phase1_end + 0.5, color="gray", ls="--", alpha=0.7, label="P1→P2")
    axes[0, 0].set_title("Total Loss", fontsize=13, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Classification Loss
    axes[0, 1].plot(epochs, history["train_cls_loss"], "b-o", label="Train CLS", markersize=3)
    axes[0, 1].plot(epochs, history["val_cls_loss"], "r-o", label="Val CLS", markersize=3)
    axes[0, 1].axvline(x=phase1_end + 0.5, color="gray", ls="--", alpha=0.7)
    axes[0, 1].set_title("Classification Loss", fontsize=13, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Segmentation Loss
    axes[0, 2].plot(epochs, history["train_seg_loss"], "b-o", label="Train SEG", markersize=3)
    axes[0, 2].plot(epochs, history["val_seg_loss"], "r-o", label="Val SEG", markersize=3)
    axes[0, 2].axvline(x=phase1_end + 0.5, color="gray", ls="--", alpha=0.7)
    axes[0, 2].set_title("Segmentation Loss", fontsize=13, fontweight="bold")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Accuracy
    axes[1, 0].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1, 0].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1, 0].axvline(x=phase1_end + 0.5, color="gray", ls="--", alpha=0.7)
    axes[1, 0].set_title("Classification Accuracy", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Dice Score
    axes[1, 1].plot(epochs, history["train_dice"], "b-o", label="Train Dice", markersize=3)
    axes[1, 1].plot(epochs, history["val_dice"], "r-o", label="Val Dice", markersize=3)
    axes[1, 1].axvline(x=phase1_end + 0.5, color="gray", ls="--", alpha=0.7)
    axes[1, 1].set_title("Segmentation Dice Score", fontsize=13, fontweight="bold")
    axes[1, 1].set_ylabel("Dice")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Learning Rate
    axes[1, 2].plot(epochs, history["lr"], "g-o", label="LR", markersize=3)
    axes[1, 2].axvline(x=phase1_end + 0.5, color="gray", ls="--", alpha=0.7)
    axes[1, 2].set_title("Learning Rate", fontsize=13, fontweight="bold")
    axes[1, 2].set_yscale("log")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Multi-Task Learning 학습 곡선", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mt_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 학습 곡선 저장: {FIGURES_DIR / 'mt_training_curves.png'}")


# ─── 메인 ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 18: Multi-Task 모델 학습")
    print("=" * 60)
    print(f"\n  Device: {CONFIG['device']}")
    if CONFIG["device"] == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Loss: α={CONFIG['alpha_cls']} × L_cls + β={CONFIG['beta_seg']} × L_seg")

    # 디렉토리 생성
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ─── DataLoader ──────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Multi-Task DataLoader 생성 중...")
    train_loader, val_loader, _, pos_weight = create_multitask_dataloaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  pos_weight: {pos_weight.item():.4f}")

    # ─── 모델 생성 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  MultiTaskBrainNet 생성 중...")
    device = torch.device(CONFIG["device"])
    model = create_multitask_model(pretrained=True)
    model = model.to(device)
    get_model_summary(model)

    # ─── 손실 함수 ───────────────────────────────────────────
    criterion = MultiTaskLoss(
        alpha=CONFIG["alpha_cls"],
        beta=CONFIG["beta_seg"],
        pos_weight=pos_weight.to(device),
    )

    # ─── 히스토리 ────────────────────────────────────────────
    history = {
        "train_loss": [], "train_cls_loss": [], "train_seg_loss": [],
        "train_acc": [], "train_dice": [],
        "val_loss": [], "val_cls_loss": [], "val_seg_loss": [],
        "val_acc": [], "val_dice": [],
        "lr": [],
    }

    total_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Encoder 동결
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Phase 1: Encoder 동결 — Classification + Segmentation Head 학습")
    print(f"{'═' * 60}")

    freeze_encoder(model)

    optimizer_p1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["phase1_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=CONFIG["phase1_epochs"])
    early_stop_p1 = EarlyStopping(patience=999)  # Phase 1은 전체 실행

    train_phase(
        model, train_loader, val_loader, criterion, optimizer_p1, scheduler_p1,
        device, CONFIG["phase1_epochs"], "Phase1", early_stop_p1, history,
    )

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 전체 Fine-tuning
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Phase 2: 전체 Fine-tuning")
    print(f"{'═' * 60}")

    unfreeze_encoder(model)

    optimizer_p2 = AdamW(
        model.parameters(),
        lr=CONFIG["phase2_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=CONFIG["phase2_epochs"])
    early_stop_p2 = EarlyStopping(patience=CONFIG["early_stopping_patience"])

    best_val_loss, best_val_acc, best_val_dice = train_phase(
        model, train_loader, val_loader, criterion, optimizer_p2, scheduler_p2,
        device, CONFIG["phase2_epochs"], "Phase2", early_stop_p2, history,
    )

    total_time = time.time() - total_start

    # ─── 마지막 모델 저장 ────────────────────────────────────
    save_checkpoint(
        model, optimizer_p2, len(history["train_loss"]),
        history["val_loss"][-1], history["val_acc"][-1], history["val_dice"][-1],
        CHECKPOINT_DIR / "mt_last_model.pth",
    )

    # ─── 히스토리 저장 ───────────────────────────────────────
    with open(LOG_DIR / "mt_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  히스토리 저장: {LOG_DIR / 'mt_training_history.json'}")

    # ─── 학습 곡선 ───────────────────────────────────────────
    plot_training_curves(history)

    # ─── 최종 결과 ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Multi-Task 학습 완료!")
    print(f"{'=' * 60}")
    print(f"  총 학습 시간: {total_time / 60:.1f}분")
    print(f"  Best Val Loss:  {best_val_loss:.4f}")
    print(f"  Best Val Acc:   {best_val_acc:.2f}%")
    print(f"  Best Val Dice:  {best_val_dice:.4f}")
    print(f"  체크포인트: {CHECKPOINT_DIR}")
    print(f"{'=' * 60}")

    # ─── 요약 저장 ───────────────────────────────────────────
    summary = {
        "total_time_minutes": round(total_time / 60, 1),
        "total_epochs": len(history["train_loss"]),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(best_val_acc, 2),
        "best_val_dice": round(best_val_dice, 4),
        "config": CONFIG,
    }
    # device는 JSON 직렬화 불가하므로 문자열로 변환
    summary["config"]["device"] = str(CONFIG["device"])
    with open(LOG_DIR / "mt_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  요약 저장: {LOG_DIR / 'mt_training_summary.json'}")


if __name__ == "__main__":
    main()
