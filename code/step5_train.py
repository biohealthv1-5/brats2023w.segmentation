# -*- coding: utf-8 -*-
"""
Step 5: 모델 학습
==================
ResNet-18 모델을 2단계로 학습합니다.
  Phase 1: FC layer만 학습 (3 epochs)
  Phase 2: 전체 Fine-tuning (12 epochs)

사용법:
    python code/step5_train.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
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
from step3_dataset import create_dataloaders
from step4_model import create_model, freeze_backbone, unfreeze_backbone, get_model_summary

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
LOG_DIR = PROJECT_DIR / "outputs" / "logs"

# ─── 하이퍼파라미터 ──────────────────────────────────────────
CONFIG = {
    # Phase 1: FC layer만 학습
    "phase1_epochs": 3,
    "phase1_lr": 1e-3,
    # Phase 2: 전체 Fine-tuning
    "phase2_epochs": 12,
    "phase2_lr": 1e-4,
    # 공통
    "batch_size": 32,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "early_stopping_patience": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


class EarlyStopping:
    """Early Stopping 구현"""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
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


def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 진행 상황 출력 (매 200 배치마다)
        if (batch_idx + 1) % 200 == 0:
            batch_acc = correct / total * 100
            batch_loss = running_loss / total
            print(f"      batch {batch_idx+1}/{len(loader)} | "
                  f"loss: {batch_loss:.4f} | acc: {batch_acc:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, filepath):
    """체크포인트 저장"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        filepath,
    )


def train_phase(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, num_epochs, phase_name, early_stopping, history
):
    """학습 Phase 실행"""
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 스케줄러 업데이트
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        # 로그
        print(
            f"  [{phase_name}] Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        # 히스토리 기록
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                CHECKPOINT_DIR / "best_model.pth",
            )
            print(f"    ★ Best model 저장 (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")

        # Early Stopping
        if early_stopping(val_loss):
            print(f"    ⚠ Early Stopping! (patience: {early_stopping.patience})")
            break

    return best_val_loss, best_val_acc


def main():
    print("=" * 60)
    print("  Step 5: 모델 학습")
    print("=" * 60)
    print(f"\n  Device: {CONFIG['device']}")
    if CONFIG["device"] == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 디렉토리 생성
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ─── DataLoader 생성 ─────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  DataLoader 생성 중...")
    train_loader, val_loader, test_loader, pos_weight = create_dataloaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"  Train: {len(train_loader.dataset):,} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset):,} samples ({len(val_loader)} batches)")
    print(f"  pos_weight: {pos_weight.item():.4f}")

    # ─── 모델 생성 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  모델 생성 중...")
    device = torch.device(CONFIG["device"])
    model = create_model(pretrained=True)
    model = model.to(device)
    get_model_summary(model)

    # ─── 손실 함수 ───────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ─── 학습 히스토리 ───────────────────────────────────────
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    # ═══════════════════════════════════════════════════════════
    # Phase 1: FC layer만 학습
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Phase 1: FC layer만 학습")
    print(f"{'═' * 60}")

    freeze_backbone(model)

    optimizer_p1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["phase1_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=CONFIG["phase1_epochs"])

    # Phase 1은 Early Stopping 없이 전체 실행
    early_stop_p1 = EarlyStopping(patience=999)

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

    unfreeze_backbone(model)

    optimizer_p2 = AdamW(
        model.parameters(),
        lr=CONFIG["phase2_lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=CONFIG["phase2_epochs"])
    early_stop_p2 = EarlyStopping(patience=CONFIG["early_stopping_patience"])

    best_val_loss, best_val_acc = train_phase(
        model, train_loader, val_loader, criterion, optimizer_p2, scheduler_p2,
        device, CONFIG["phase2_epochs"], "Phase2", early_stop_p2, history,
    )

    # ─── 마지막 모델 저장 ────────────────────────────────────
    save_checkpoint(
        model, optimizer_p2, len(history["train_loss"]),
        history["val_loss"][-1], history["val_acc"][-1],
        CHECKPOINT_DIR / "last_model.pth",
    )

    # ─── 히스토리 저장 ───────────────────────────────────────
    with open(LOG_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  학습 히스토리 저장: {LOG_DIR / 'training_history.json'}")

    # ─── 학습 곡선 시각화 ────────────────────────────────────
    plot_training_curves(history)

    # ─── 최종 결과 ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  학습 완료!")
    print(f"{'=' * 60}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val Acc:  {best_val_acc:.2f}%")
    print(f"  체크포인트: {CHECKPOINT_DIR}")
    print(f"{'=' * 60}")


def plot_training_curves(history: dict):
    """학습 곡선 시각화"""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Loss 곡선
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    # Phase 경계선
    phase1_end = 3
    axes[0].axvline(x=phase1_end + 0.5, color="gray", linestyle="--",
                     alpha=0.7, label="Phase 1→2")
    axes[0].set_title("Loss 곡선", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # 2. Accuracy 곡선
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].axvline(x=phase1_end + 0.5, color="gray", linestyle="--",
                     alpha=0.7, label="Phase 1→2")
    axes[1].set_title("Accuracy 곡선", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3. Learning Rate
    axes[2].plot(epochs, history["lr"], "g-o", label="Learning Rate", markersize=4)
    axes[2].axvline(x=phase1_end + 0.5, color="gray", linestyle="--",
                     alpha=0.7, label="Phase 1→2")
    axes[2].set_title("Learning Rate 스케줄", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 학습 곡선 저장: {FIGURES_DIR / '06_training_curves.png'}")


if __name__ == "__main__":
    main()
