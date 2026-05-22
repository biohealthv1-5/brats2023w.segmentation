# -*- coding: utf-8 -*-
"""
Step 24 (Day 6 / mmmt): Multi-Modal Multi-Task Training
=========================================================
- 입력: [T1ce, FLAIR, |T1ce-FLAIR|] 3채널
- 출력: cls (binary) + seg (WT 1-region)
- Loss: BCE(cls) + 0.5 * (0.5 Dice + 0.5 Tversky)(seg)
- 학습: 2-Phase (Phase 1 encoder freeze, Phase 2 fine-tune)
- 결과: outputs/{checkpoints,figures,logs}/mmmt/

Day 6 작업의 핵심은 *입력 표현 (단일 모달 → 멀티모달)* 의 효과만 분리 평가하는
것이므로 Augmentation/Loss/Train 절차는 multitask Day 5 와 동일하게 유지합니다.
SOTA 패키지 (Deep Sup, Uncertainty Weighting, SWA, TumorCP, TTA 등)는
모두 Day 7 sota/ 에서 도입됩니다.

사용법:
    python code/mmmt/step24_mmmt_train.py
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step21_mmmt_dataset import create_mm_dataloaders
from step22_mmmt_model import (
    create_mm_model, freeze_encoder, unfreeze_encoder, get_model_summary
)
from step23_mmmt_losses import MMMTLoss

# ─── 경로 (mmmt 결과물 폴더) ─────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt"
FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "mmmt"

# ─── 하이퍼파라미터 ──────────────────────────────────────────
CONFIG = {
    "phase1_epochs": 3,
    "phase1_lr": 1e-3,
    "phase2_epochs": 15,
    "phase2_lr": 1e-4,
    "batch_size": 16,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "early_stopping_patience": 5,
    "use_weighted_sampler": False,
    "channel_mode": "t1ce_flair_diff",
    # Loss
    "beta_seg": 0.5,
    "tversky_alpha": 0.7, "tversky_beta": 0.3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# ─── EarlyStopping ───────────────────────────────────────────
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


# ─── 평가 헬퍼 ──────────────────────────────────────────────
@torch.no_grad()
def _wt_dice(seg_logits: torch.Tensor, seg_target: torch.Tensor):
    """WT Dice (양성 슬라이스만 평균)"""
    p = (torch.sigmoid(seg_logits) >= 0.5).float()
    scores = []
    for b in range(p.shape[0]):
        tgt = seg_target[b, 0].flatten()
        prd = p[b, 0].flatten()
        if tgt.sum() == 0:
            continue
        inter = (prd * tgt).sum().item()
        denom = prd.sum().item() + tgt.sum().item()
        if denom > 0:
            scores.append(2.0 * inter / denom)
    return float(np.mean(scores)) if scores else 0.0


# ─── 학습/검증 루프 ─────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = {"total": 0.0, "cls": 0.0, "seg": 0.0}
    n_samples = 0
    correct = 0
    dice_scores = []
    for batch_idx, (imgs, labels, masks) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        cls_out, seg_out = model(imgs)
        total_loss, l_cls, l_seg = criterion(cls_out, seg_out, labels, masks)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        bsz = imgs.size(0)
        n_samples += bsz
        running["total"] += total_loss.item() * bsz
        running["cls"] += l_cls.item() * bsz
        running["seg"] += l_seg.item() * bsz

        preds = (torch.sigmoid(cls_out.squeeze(1)) >= 0.5).float()
        correct += (preds == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            dice_scores.append(_wt_dice(seg_out, masks))

        if (batch_idx + 1) % 500 == 0:
            print(f"      batch {batch_idx + 1}/{len(loader)} | "
                  f"loss={running['total']/n_samples:.4f} "
                  f"acc={correct/n_samples*100:.1f}%")

    avg = {k: v / n_samples for k, v in running.items()}
    avg["acc"] = correct / n_samples * 100
    avg["dice"] = float(np.mean(dice_scores)) if dice_scores else 0.0
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = {"total": 0.0, "cls": 0.0, "seg": 0.0}
    n_samples = 0
    correct = 0
    dice_scores = []
    for imgs, labels, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cls_out, seg_out = model(imgs)
        total_loss, l_cls, l_seg = criterion(cls_out, seg_out, labels, masks)
        bsz = imgs.size(0)
        n_samples += bsz
        running["total"] += total_loss.item() * bsz
        running["cls"] += l_cls.item() * bsz
        running["seg"] += l_seg.item() * bsz
        preds = (torch.sigmoid(cls_out.squeeze(1)) >= 0.5).float()
        correct += (preds == labels).sum().item()
        dice_scores.append(_wt_dice(seg_out, masks))
    avg = {k: v / n_samples for k, v in running.items()}
    avg["acc"] = correct / n_samples * 100
    avg["dice"] = float(np.mean(dice_scores)) if dice_scores else 0.0
    return avg


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                       else v) for k, v in CONFIG.items()},
    }, filepath)


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, n_epochs, phase_name,
                early_stopping, history):
    best_val_loss = float("inf")
    best_metrics = None
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion,
                                  optimizer, device)
        val_m = validate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
        else:
            lr_now = optimizer.param_groups[0]["lr"]

        dt = time.time() - t0
        print(
            f"  [{phase_name}] Epoch {epoch}/{n_epochs} ({dt:.1f}s)\n"
            f"    Train | loss={train_m['total']:.4f} cls={train_m['cls']:.4f} "
            f"seg={train_m['seg']:.4f} acc={train_m['acc']:.2f}% "
            f"WT={train_m['dice']:.3f}\n"
            f"    Val   | loss={val_m['total']:.4f} cls={val_m['cls']:.4f} "
            f"seg={val_m['seg']:.4f} acc={val_m['acc']:.2f}% "
            f"WT={val_m['dice']:.3f} LR={lr_now:.2e}"
        )

        for k in ("total", "cls", "seg", "acc", "dice"):
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])
        history["lr"].append(lr_now)

        # Best 저장
        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            best_metrics = val_m
            save_checkpoint(model, optimizer, epoch, val_m,
                            CKPT_DIR / "mmmt_best.pth")
            print(f"    ★ best 저장 (val_loss={val_m['total']:.4f}, "
                  f"WT={val_m['dice']:.3f})")

        if early_stopping(val_m["total"]):
            print(f"    ⚠ Early stop (patience={early_stopping.patience})")
            break
    return best_val_loss, best_metrics


# ─── 학습 곡선 ──────────────────────────────────────────────
def plot_curves(history):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_total"], "b-o", label="train", ms=3)
    axes[0, 0].plot(epochs, history["val_total"], "r-o", label="val", ms=3)
    axes[0, 0].set_title("Total Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, history["train_cls"], "b-o", label="train cls", ms=3)
    axes[0, 1].plot(epochs, history["val_cls"], "r-o", label="val cls", ms=3)
    axes[0, 1].plot(epochs, history["train_seg"], "b--s", label="train seg", ms=3)
    axes[0, 1].plot(epochs, history["val_seg"], "r--s", label="val seg", ms=3)
    axes[0, 1].set_title("Cls / Seg Loss"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, history["train_acc"], "b-o", label="train acc", ms=3)
    axes[1, 0].plot(epochs, history["val_acc"], "r-o", label="val acc", ms=3)
    axes[1, 0].set_title("Cls Accuracy"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, history["train_dice"], "b-o", label="train WT Dice", ms=3)
    axes[1, 1].plot(epochs, history["val_dice"], "r-o", label="val WT Dice", ms=3)
    axes[1, 1].set_title("WT Dice"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "mmmt_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── 메인 ───────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 24 (Day 6 / mmmt): MM-MTL Training")
    print("=" * 60)
    print(f"  device: {CONFIG['device']}")
    print(f"  channel_mode: {CONFIG['channel_mode']}")
    if CONFIG["device"] == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 결과물 폴더 생성 (이 코드가 실행될 때 자동 생성)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoader ──────────────────────────────────────────
    print(f"\n  building dataloaders ...")
    train_l, val_l, _, pos_weight = create_mm_dataloaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        use_weighted_sampler=CONFIG["use_weighted_sampler"],
        channel_mode=CONFIG["channel_mode"],
    )
    print(f"  Train: {len(train_l.dataset):,}  ({len(train_l)} batches)")
    print(f"  Val:   {len(val_l.dataset):,}")
    print(f"  pos_weight: {pos_weight.item():.4f}")

    # ── Model ───────────────────────────────────────────────
    device = torch.device(CONFIG["device"])
    print(f"\n  building model ...")
    model = create_mm_model(pretrained=True, in_ch=3, seg_classes=1).to(device)
    get_model_summary(model)

    # ── Loss ────────────────────────────────────────────────
    criterion = MMMTLoss(
        pos_weight=pos_weight.to(device),
        beta_seg=CONFIG["beta_seg"],
        alpha_tv=CONFIG["tversky_alpha"], beta_tv=CONFIG["tversky_beta"],
    )

    # ── History dict ────────────────────────────────────────
    keys_metric = ["total", "cls", "seg", "acc", "dice"]
    history = {f"train_{k}": [] for k in keys_metric}
    history.update({f"val_{k}": [] for k in keys_metric})
    history["lr"] = []

    start_all = time.time()

    # Phase 1: encoder freeze
    print(f"\n{'=' * 60}\n  Phase 1: encoder freeze\n{'=' * 60}")
    freeze_encoder(model)
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=CONFIG["phase1_lr"], weight_decay=CONFIG["weight_decay"])
    sch1 = CosineAnnealingLR(opt1, T_max=CONFIG["phase1_epochs"])
    es1 = EarlyStopping(patience=999)
    train_phase(model, train_l, val_l, criterion, opt1, sch1, device,
                CONFIG["phase1_epochs"], "Phase1", es1, history)

    # Phase 2: full fine-tune
    print(f"\n{'=' * 60}\n  Phase 2: full fine-tune\n{'=' * 60}")
    unfreeze_encoder(model)
    opt2 = AdamW(model.parameters(),
                 lr=CONFIG["phase2_lr"], weight_decay=CONFIG["weight_decay"])
    sch2 = CosineAnnealingLR(opt2, T_max=CONFIG["phase2_epochs"])
    es2 = EarlyStopping(patience=CONFIG["early_stopping_patience"])
    train_phase(model, train_l, val_l, criterion, opt2, sch2, device,
                CONFIG["phase2_epochs"], "Phase2", es2, history)

    # 마지막 저장
    save_checkpoint(model, opt2, len(history["train_total"]),
                    {k: history[f"val_{k}"][-1] for k in keys_metric},
                    CKPT_DIR / "mmmt_last.pth")

    with open(LOG_DIR / "mmmt_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  history: {LOG_DIR / 'mmmt_history.json'}")

    plot_curves(history)
    print(f"  curves:  {FIG_DIR / 'mmmt_training_curves.png'}")

    total_time = (time.time() - start_all) / 60
    print(f"\n{'=' * 60}")
    print(f"  완료. 총 학습 시간: {total_time:.1f}분")
    print(f"  ckpt: {CKPT_DIR}")
    print("=" * 60)

    summary = {
        "total_minutes": round(total_time, 1),
        "total_epochs": len(history["train_total"]),
        "final_val": {k: history[f"val_{k}"][-1] for k in keys_metric},
        "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                       else v) for k, v in CONFIG.items()},
    }
    with open(LOG_DIR / "mmmt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
