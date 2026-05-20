# -*- coding: utf-8 -*-
"""
Step 11: 패치 분류 모델 학습
=============================
PatchCNN을 학습하고 체크포인트 · 로그 · 학습곡선을 자동 저장합니다.

출력:
    outputs/checkpoints/patch_best_model.pth
    outputs/logs/step11_train_log.json    ← 에포크별 메트릭
    outputs/figures/11_patch_training_curves.png

사용법:
    python code/step11_patch_train.py
"""

import sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
LOG_DIR  = PROJECT_DIR / "outputs" / "logs"
FIG_DIR  = PROJECT_DIR / "outputs" / "figures"
for d in [CKPT_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BEST_MODEL = CKPT_DIR / "patch_best_model.pth"
LAST_MODEL = CKPT_DIR / "patch_last_model.pth"
TRAIN_LOG  = LOG_DIR  / "step11_train_log.json"
CURVE_FIG  = FIG_DIR  / "11_patch_training_curves.png"

# ─── 하이퍼파라미터 ──────────────────────────────────────────
EPOCHS          = 20
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
BATCH_SIZE      = 128
NUM_WORKERS     = 4
PATIENCE        = 5
NEG_SUBSAMPLE   = 3.0   # Negative를 Positive의 3배까지만

# ─── 학습 함수 ───────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total * 100


def plot_curves(history, save_path):
    """학습 곡선 그래프 저장"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="Train")
    axes[0].plot(epochs, [h["val_loss"]   for h in history], "s-", label="Val")
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [h["train_acc"] for h in history], "o-", label="Train")
    axes[1].plot(epochs, [h["val_acc"]   for h in history], "s-", label="Val")
    axes[1].set_title("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # LR
    axes[2].plot(epochs, [h["lr"] for h in history], "D-", color="green")
    axes[2].set_title("Learning Rate", fontsize=14)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Patch CNN Training Curves", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  학습 곡선 저장: {save_path}")


# ─── 메인 ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 11: PatchCNN 학습")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 데이터 로드 ──────────────────────────────────────────
    # code 디렉토리를 sys.path에 추가하여 정상 import (Windows multiprocessing 호환)
    code_dir = str(Path(__file__).resolve().parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from step9_patch_dataset import create_patch_dataloaders
    from step10_patch_model import create_patch_model

    print(f"\n  DataLoader 생성 (batch={BATCH_SIZE}, subsample 1:{NEG_SUBSAMPLE:.0f})...")
    train_loader, val_loader, _, pos_weight = create_patch_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        neg_subsample_ratio=NEG_SUBSAMPLE,
    )
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  pos_weight:    {pos_weight.item():.4f}")

    # ── 모델 ─────────────────────────────────────────────────
    print(f"\n  모델 생성...")
    model = create_patch_model(device)

    # ── Loss / Optimizer / Scheduler ─────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── 학습 루프 ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  학습 시작: {EPOCHS} epochs, patience={PATIENCE}")
    print(f"{'─'*60}\n")

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t_ep

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 2),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 2),
            "lr":         round(current_lr, 8),
            "time_sec":   round(elapsed, 1),
        }
        history.append(record)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL)
            marker = " ★ Best"
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:2d}/{EPOCHS} │ "
            f"Loss {train_loss:.4f}/{val_loss:.4f} │ "
            f"Acc {train_acc:.1f}%/{val_acc:.1f}% │ "
            f"LR {current_lr:.2e} │ {elapsed:.0f}s{marker}"
        )

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    total_time = time.time() - t_start
    torch.save(model.state_dict(), LAST_MODEL)

    # ── 로그 저장 ────────────────────────────────────────────
    log_data = {
        "hyperparams": {
            "epochs_run": len(history),
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "neg_subsample": NEG_SUBSAMPLE,
            "patience": PATIENCE,
        },
        "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"],
        "best_val_loss": round(best_val_loss, 4),
        "best_val_acc": round(
            min(history, key=lambda h: h["val_loss"])["val_acc"], 2
        ),
        "total_time_min": round(total_time / 60, 1),
        "history": history,
    }
    with open(TRAIN_LOG, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # ── 학습 곡선 ────────────────────────────────────────────
    plot_curves(history, CURVE_FIG)

    # ── 요약 ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"{'='*60}")
    print(f"  Best epoch: {log_data['best_epoch']}")
    print(f"  Best val loss: {log_data['best_val_loss']}")
    print(f"  Best val acc:  {log_data['best_val_acc']}%")
    print(f"  총 소요 시간: {log_data['total_time_min']:.1f}분")
    print(f"\n  체크포인트: {BEST_MODEL}")
    print(f"  로그: {TRAIN_LOG}")
    print(f"  곡선: {CURVE_FIG}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
