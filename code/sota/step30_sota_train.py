# -*- coding: utf-8 -*-
"""
Step 30 (Day 7 / sota): SOTA Multi-Modal Multi-Task Training
=============================================================
v2ways.md §8.2 (Day 7 "무료 SOTA 패키지") 전면 적용:

- 입력: [T1ce, FLAIR, |T1ce-FLAIR|] 3채널
- 출력: cls (binary) + seg 3-region (WT/TC/ET)
- Loss: Focal-Tversky + BCE + Boundary  per region
        + Deep Supervision (WT aux x3)
        + Uncertainty Weighting (Kendall 2018, learnable log_var)
- Augmentation: Dataset 내부 공간 증강 + TumorCP collate-level paste
- 2-Phase 학습 (Phase 1 encoder freeze, Phase 2 fine-tune)
- SWA (Phase 2 마지막 5 epoch)
- 결과: outputs/{checkpoints,figures,logs}/sota/

사용법:
    python code/sota/step30_sota_train.py

[Resume 기능]
- 매 epoch 끝에서 outputs/checkpoints/sota/sota_latest.pth 에
  model / optimizer / scheduler / swa_model / history / phase / epoch 등
  전체 상태를 원자적으로 저장한다.
- 실행이 강제 종료되어도 동일 명령으로 다시 실행하면 latest 체크포인트를
  자동 감지하여 동일 phase / 동일 epoch 다음 차례부터 이어서 학습한다.
- 학습이 정상적으로 끝나면 latest 체크포인트는 자동 삭제되고,
  sota_best.pth / sota_last.pth / sota_swa.pth 만 남는다.
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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step27_sota_dataset import create_sota_dataloaders
from step28_sota_model import (
    create_sota_model, freeze_encoder, unfreeze_encoder, get_model_summary
)
from step29_sota_losses import SOTAMultiTaskLoss, tumor_copy_paste

# ─── 경로 (sota 결과물 폴더) ─────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints" / "sota"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "sota"
FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "sota"

# ─── Resume 체크포인트 경로 ───────────────────────────────────
# 학습이 중단되어도 이 파일이 있으면 자동으로 이어서 진행한다.
LATEST_CKPT = CKPT_DIR / "sota_latest.pth"

# ─── 하이퍼파라미터 ──────────────────────────────────────────
CONFIG = {
    "phase1_epochs": 3,
    "phase1_lr": 1e-3,
    "phase2_epochs": 15,
    "phase2_lr": 1e-4,
    "swa_start_epoch": 11,         # Phase 2 epoch 기준 (15-5+1)
    "swa_lr": 5e-5,
    "batch_size": 16,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "early_stopping_patience": 5,
    "use_weighted_sampler": True,
    "channel_mode": "t1ce_flair_diff",
    "tumorcp_prob": 0.5,
    "w_tv": 1.0, "w_bce": 0.5, "w_b": 0.3,
    "tversky_alpha": 0.7, "tversky_beta": 0.3, "tversky_gamma": 4.0 / 3.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


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

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "counter": self.counter,
            "best_loss": self.best_loss,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, sd):
        self.patience = sd.get("patience", self.patience)
        self.min_delta = sd.get("min_delta", self.min_delta)
        self.counter = sd.get("counter", 0)
        self.best_loss = sd.get("best_loss", None)
        self.should_stop = sd.get("should_stop", False)


@torch.no_grad()
def _dice_per_region(seg_logits: torch.Tensor, seg_target: torch.Tensor):
    """(B, 3, H, W) → {WT, TC, ET}별 Dice (양성 슬라이스만 평균)"""
    out = {}
    p = (torch.sigmoid(seg_logits) >= 0.5).float()
    for r, name in enumerate(["WT", "TC", "ET"]):
        scores = []
        for b in range(p.shape[0]):
            tgt = seg_target[b, r].flatten()
            prd = p[b, r].flatten()
            if tgt.sum() == 0:
                continue
            inter = (prd * tgt).sum().item()
            denom = prd.sum().item() + tgt.sum().item()
            if denom > 0:
                scores.append(2.0 * inter / denom)
        out[name] = float(np.mean(scores)) if scores else 0.0
    return out


def train_one_epoch(model, loader, criterion, optimizer, device,
                    tumorcp_prob=0.5):
    model.train()
    running = {"total": 0.0, "cls": 0.0, "seg": 0.0}
    n_samples = 0
    correct = 0
    dice_acc = {"WT": [], "TC": [], "ET": []}
    for batch_idx, (imgs, labels, masks) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # TumorCP — 배치 내부 paste
        imgs, masks = tumor_copy_paste(imgs, masks, p=tumorcp_prob)

        cls_out, seg_main, aux_list = model(imgs, return_aux=True)
        total_loss, l_cls, l_seg = criterion(
            cls_out, seg_main, aux_list, labels, masks,
            model.log_var_cls, model.log_var_seg,
        )
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
            d = _dice_per_region(seg_main, masks)
            for k in dice_acc:
                dice_acc[k].append(d[k])

        if (batch_idx + 1) % 500 == 0:
            print(f"      batch {batch_idx + 1}/{len(loader)} | "
                  f"loss={running['total']/n_samples:.4f} "
                  f"acc={correct/n_samples*100:.1f}%")

    avg = {k: v / n_samples for k, v in running.items()}
    avg["acc"] = correct / n_samples * 100
    avg.update({f"dice_{k}": float(np.mean(v)) if v else 0.0
                for k, v in dice_acc.items()})
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running = {"total": 0.0, "cls": 0.0, "seg": 0.0}
    n_samples = 0
    correct = 0
    dice_acc = {"WT": [], "TC": [], "ET": []}
    for imgs, labels, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cls_out, seg_main, aux_list = model(imgs, return_aux=True)
        total_loss, l_cls, l_seg = criterion(
            cls_out, seg_main, aux_list, labels, masks,
            model.log_var_cls, model.log_var_seg,
        )
        bsz = imgs.size(0)
        n_samples += bsz
        running["total"] += total_loss.item() * bsz
        running["cls"] += l_cls.item() * bsz
        running["seg"] += l_seg.item() * bsz
        preds = (torch.sigmoid(cls_out.squeeze(1)) >= 0.5).float()
        correct += (preds == labels).sum().item()
        d = _dice_per_region(seg_main, masks)
        for k in dice_acc:
            dice_acc[k].append(d[k])
    avg = {k: v / n_samples for k, v in running.items()}
    avg["acc"] = correct / n_samples * 100
    avg.update({f"dice_{k}": float(np.mean(v)) if v else 0.0
                for k, v in dice_acc.items()})
    return avg


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "log_var_cls": float(model.log_var_cls.detach().cpu()),
        "log_var_seg": float(model.log_var_seg.detach().cpu()),
        "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                       else v) for k, v in CONFIG.items()},
    }, filepath)


def save_latest_checkpoint(filepath, *, phase, epoch, model, optimizer,
                           scheduler, history, best_val_loss,
                           early_stopping, swa_model=None,
                           swa_scheduler=None):
    """
    매 epoch 끝에서 호출되는 'resume용' 체크포인트.
    학습 중간에 프로세스가 죽어도 이 파일만 살아있으면
    동일 phase / 동일 epoch 다음 차례부터 재개할 수 있다.

    원자적 저장: <filepath>.tmp 로 저장 후 rename → 쓰는 도중 죽어도
    기존 체크포인트가 망가지지 않는다.
    """
    payload = {
        "phase": phase,                       # 1 or 2
        "epoch": epoch,                       # 이번 phase에서 완료된 epoch 번호 (1-based)
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict":
            scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "best_val_loss": best_val_loss,
        "early_stopping": early_stopping.state_dict(),
        "swa_model_state_dict":
            swa_model.state_dict() if swa_model is not None else None,
        "swa_scheduler_state_dict":
            swa_scheduler.state_dict() if swa_scheduler is not None else None,
        "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                       else v) for k, v in CONFIG.items()},
    }
    filepath = Path(filepath)
    tmp = filepath.with_suffix(filepath.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(filepath)


def load_latest_checkpoint(filepath, device):
    """latest 체크포인트가 있으면 dict 반환, 없으면 None."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    try:
        return torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        # 구버전 torch (weights_only 인자 없음)
        return torch.load(filepath, map_location=device)


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, n_epochs, phase_name,
                early_stopping, history, swa_model=None,
                swa_scheduler=None, swa_start=None,
                start_epoch=1, best_val_loss_init=float("inf"),
                phase_id=None, latest_path=None):
    """
    start_epoch        : 이어서 시작할 1-based epoch 번호 (resume 용)
    best_val_loss_init : resume 시 이전까지의 best val loss
    phase_id           : 1 또는 2 (latest 체크포인트에 기록)
    latest_path        : 매 epoch 끝에서 저장할 latest 체크포인트 경로
    """
    best_val_loss = best_val_loss_init
    best_metrics = None
    if start_epoch > n_epochs:
        print(f"  [{phase_name}] 이미 완료된 phase — skip.")
        return best_val_loss, best_metrics
    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion,
                                  optimizer, device,
                                  tumorcp_prob=CONFIG["tumorcp_prob"])
        val_m = validate(model, val_loader, criterion, device)

        if swa_model is not None and swa_start is not None and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr_now = swa_scheduler.get_last_lr()[0]
            swa_active = True
        else:
            if scheduler is not None:
                scheduler.step()
                lr_now = scheduler.get_last_lr()[0]
            else:
                lr_now = optimizer.param_groups[0]["lr"]
            swa_active = False

        dt = time.time() - t0
        print(
            f"  [{phase_name}] Epoch {epoch}/{n_epochs} ({dt:.1f}s) "
            f"{'(SWA)' if swa_active else ''}\n"
            f"    Train | loss={train_m['total']:.4f} cls={train_m['cls']:.4f} "
            f"seg={train_m['seg']:.4f} acc={train_m['acc']:.2f}% "
            f"WT={train_m['dice_WT']:.3f} TC={train_m['dice_TC']:.3f} "
            f"ET={train_m['dice_ET']:.3f}\n"
            f"    Val   | loss={val_m['total']:.4f} cls={val_m['cls']:.4f} "
            f"seg={val_m['seg']:.4f} acc={val_m['acc']:.2f}% "
            f"WT={val_m['dice_WT']:.3f} TC={val_m['dice_TC']:.3f} "
            f"ET={val_m['dice_ET']:.3f} LR={lr_now:.2e}\n"
            f"    log_var_cls={model.log_var_cls.item():.3f} "
            f"log_var_seg={model.log_var_seg.item():.3f}"
        )

        for k in ("total", "cls", "seg", "acc", "dice_WT", "dice_TC", "dice_ET"):
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])
        history["lr"].append(lr_now)
        history["log_var_cls"].append(float(model.log_var_cls.item()))
        history["log_var_seg"].append(float(model.log_var_seg.item()))

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            best_metrics = val_m
            save_checkpoint(model, optimizer, epoch, val_m,
                            CKPT_DIR / "sota_best.pth")
            print(f"    ★ best 저장 (val_loss={val_m['total']:.4f}, "
                  f"WT={val_m['dice_WT']:.3f})")

        stop = early_stopping(val_m["total"])

        # ── resume용 latest 체크포인트 (매 epoch) ──
        if latest_path is not None and phase_id is not None:
            save_latest_checkpoint(
                latest_path,
                phase=phase_id, epoch=epoch,
                model=model, optimizer=optimizer, scheduler=scheduler,
                history=history, best_val_loss=best_val_loss,
                early_stopping=early_stopping,
                swa_model=swa_model, swa_scheduler=swa_scheduler,
            )

        if stop:
            print(f"    ⚠ Early stop (patience={early_stopping.patience})")
            break
    return best_val_loss, best_metrics


def plot_curves(history):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_total"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    axes[0, 0].plot(epochs, history["train_total"], "b-o", label="train", ms=3)
    axes[0, 0].plot(epochs, history["val_total"], "r-o", label="val", ms=3)
    axes[0, 0].set_title("Total Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, history["train_cls"], "b-o", label="train cls", ms=3)
    axes[0, 1].plot(epochs, history["val_cls"], "r-o", label="val cls", ms=3)
    axes[0, 1].set_title("Cls Loss"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(epochs, history["train_seg"], "b-o", label="train seg", ms=3)
    axes[0, 2].plot(epochs, history["val_seg"], "r-o", label="val seg", ms=3)
    axes[0, 2].set_title("Seg Loss"); axes[0, 2].legend(); axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(epochs, history["train_acc"], "b-o", label="train acc", ms=3)
    axes[1, 0].plot(epochs, history["val_acc"], "r-o", label="val acc", ms=3)
    axes[1, 0].set_title("Cls Accuracy"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    for r, name in enumerate(["WT", "TC", "ET"]):
        axes[1, 1].plot(epochs, history[f"val_dice_{name}"], "-o",
                        label=f"val {name}", ms=3)
    axes[1, 1].set_title("Val Dice (WT/TC/ET)"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(epochs, history["log_var_cls"], "-o", label="log_var_cls", ms=3)
    axes[1, 2].plot(epochs, history["log_var_seg"], "-o", label="log_var_seg", ms=3)
    axes[1, 2].set_title("Uncertainty Weighting params")
    axes[1, 2].legend(); axes[1, 2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "sota_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("  Step 30 (Day 7 / sota): SOTA MM-MTL Training")
    print("=" * 60)
    print(f"  device: {CONFIG['device']}")
    print(f"  channel_mode: {CONFIG['channel_mode']}")
    if CONFIG["device"] == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 결과물 폴더 자동 생성 — 본 코드를 실행하면 outputs/{checkpoints,figures,logs}/sota/ 에 결과가 떨어진다.
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoader ──────────────────────────────────────────
    print(f"\n  building dataloaders ...")
    train_l, val_l, _, pos_weight = create_sota_dataloaders(
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
    model = create_sota_model(pretrained=True, in_ch=3, seg_classes=3).to(device)
    get_model_summary(model)

    # ── Loss ────────────────────────────────────────────────
    criterion = SOTAMultiTaskLoss(
        pos_weight=pos_weight.to(device),
        w_tv=CONFIG["w_tv"], w_bce=CONFIG["w_bce"], w_b=CONFIG["w_b"],
    )

    keys_metric = ["total", "cls", "seg", "acc", "dice_WT", "dice_TC", "dice_ET"]
    history = {f"train_{k}": [] for k in keys_metric}
    history.update({f"val_{k}": [] for k in keys_metric})
    history.update({"lr": [], "log_var_cls": [], "log_var_seg": []})

    # ── resume 체크: latest 체크포인트가 있으면 자동으로 이어서 학습 ──
    ckpt = load_latest_checkpoint(LATEST_CKPT, device)
    resume_phase = 1
    resume_epoch = 1                  # 다음에 실행할 epoch (1-based)
    resume_best_p1 = float("inf")
    resume_best_p2 = float("inf")
    resume_payload = None
    if ckpt is not None:
        print(f"\n  ▶ resume: '{LATEST_CKPT.name}' 발견 → "
              f"phase {ckpt['phase']}, epoch {ckpt['epoch']} 완료 지점에서 재개")
        # 모델 가중치/uncertainty params 복원
        model.load_state_dict(ckpt["model_state_dict"])
        # history 복원
        for k in history:
            if k in ckpt["history"]:
                history[k] = list(ckpt["history"][k])
        resume_phase = int(ckpt["phase"])
        finished_epoch = int(ckpt["epoch"])
        # 해당 phase의 총 epoch 수
        total_in_phase = (CONFIG["phase1_epochs"] if resume_phase == 1
                          else CONFIG["phase2_epochs"])
        if finished_epoch >= total_in_phase:
            # 이 phase는 끝났음 → 다음 phase 1 epoch 부터
            if resume_phase == 1:
                resume_phase = 2
                resume_epoch = 1
            else:
                resume_phase = 3   # 둘 다 끝 → 학습 본체 skip
                resume_epoch = 1
        else:
            resume_epoch = finished_epoch + 1
        if resume_phase == 2:
            resume_best_p2 = float(ckpt.get("best_val_loss", float("inf")))
        else:
            resume_best_p1 = float(ckpt.get("best_val_loss", float("inf")))
        resume_payload = ckpt
    else:
        print(f"\n  (resume 체크포인트 없음 → 처음부터 학습)")

    start_all = time.time()

    # ── Phase 1 ─────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  Phase 1: encoder freeze\n{'=' * 60}")
    freeze_encoder(model)
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=CONFIG["phase1_lr"], weight_decay=CONFIG["weight_decay"])
    sch1 = CosineAnnealingLR(opt1, T_max=CONFIG["phase1_epochs"])
    es1 = EarlyStopping(patience=999)
    if resume_phase == 1 and resume_payload is not None:
        opt1.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("scheduler_state_dict") is not None:
            sch1.load_state_dict(resume_payload["scheduler_state_dict"])
        es1.load_state_dict(resume_payload["early_stopping"])
    if resume_phase <= 1:
        train_phase(model, train_l, val_l, criterion, opt1, sch1, device,
                    CONFIG["phase1_epochs"], "Phase1", es1, history,
                    start_epoch=(resume_epoch if resume_phase == 1 else 1),
                    best_val_loss_init=resume_best_p1,
                    phase_id=1, latest_path=LATEST_CKPT)
    else:
        print(f"  Phase 1: 이미 완료된 상태 — skip.")

    # ── Phase 2 + SWA ──────────────────────────────────────
    print(f"\n{'=' * 60}\n  Phase 2: full fine-tune + SWA\n{'=' * 60}")
    unfreeze_encoder(model)
    opt2 = AdamW(model.parameters(),
                 lr=CONFIG["phase2_lr"], weight_decay=CONFIG["weight_decay"])
    sch2 = CosineAnnealingLR(opt2, T_max=CONFIG["phase2_epochs"])
    es2 = EarlyStopping(patience=CONFIG["early_stopping_patience"])

    swa_model = AveragedModel(model)
    swa_sch = SWALR(opt2, swa_lr=CONFIG["swa_lr"])

    if resume_phase == 2 and resume_payload is not None:
        opt2.load_state_dict(resume_payload["optimizer_state_dict"])
        if resume_payload.get("scheduler_state_dict") is not None:
            sch2.load_state_dict(resume_payload["scheduler_state_dict"])
        es2.load_state_dict(resume_payload["early_stopping"])
        if resume_payload.get("swa_model_state_dict") is not None:
            try:
                swa_model.load_state_dict(resume_payload["swa_model_state_dict"])
            except Exception as e:
                print(f"  [WARN] SWA model state 로드 실패 (무시): {e}")
        if resume_payload.get("swa_scheduler_state_dict") is not None:
            try:
                swa_sch.load_state_dict(resume_payload["swa_scheduler_state_dict"])
            except Exception as e:
                print(f"  [WARN] SWA scheduler state 로드 실패 (무시): {e}")

    if resume_phase <= 2:
        train_phase(model, train_l, val_l, criterion, opt2, sch2, device,
                    CONFIG["phase2_epochs"], "Phase2", es2, history,
                    swa_model=swa_model, swa_scheduler=swa_sch,
                    swa_start=CONFIG["swa_start_epoch"],
                    start_epoch=(resume_epoch if resume_phase == 2 else 1),
                    best_val_loss_init=resume_best_p2,
                    phase_id=2, latest_path=LATEST_CKPT)
    else:
        print(f"  Phase 2: 이미 완료된 상태 — skip.")

    # SWA BN 업데이트
    print(f"\n  Updating BN statistics for SWA model ...")
    try:
        update_bn(train_l, swa_model, device=device)
        torch.save({
            "model_state_dict": swa_model.module.state_dict(),
            "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                           else v) for k, v in CONFIG.items()},
        }, CKPT_DIR / "sota_swa.pth")
        print(f"  SWA checkpoint: {CKPT_DIR / 'sota_swa.pth'}")
    except Exception as e:
        print(f"  [WARN] SWA BN update 실패: {e}")

    save_checkpoint(model, opt2, len(history["train_total"]),
                    {k: history[f"val_{k}"][-1] for k in keys_metric},
                    CKPT_DIR / "sota_last.pth")

    with open(LOG_DIR / "sota_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"  history: {LOG_DIR / 'sota_history.json'}")

    plot_curves(history)
    print(f"  curves: {FIG_DIR / 'sota_training_curves.png'}")

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
    with open(LOG_DIR / "sota_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 학습이 정상적으로 끝났으니 resume용 latest 체크포인트는 정리한다.
    # (남겨두면 다음 실행에서 '학습 끝남' 상태로 인식되어 SWA가 빈 모델로
    #  재시도되는 등의 부작용이 있을 수 있음. sota_best.pth / sota_last.pth /
    #  sota_swa.pth 는 그대로 보존.)
    try:
        if LATEST_CKPT.exists():
            LATEST_CKPT.unlink()
            print(f"  resume 체크포인트 정리: {LATEST_CKPT.name} 삭제")
    except Exception as e:
        print(f"  [WARN] latest 체크포인트 삭제 실패 (무시): {e}")


if __name__ == "__main__":
    main()
