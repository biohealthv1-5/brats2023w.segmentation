# -*- coding: utf-8 -*-
"""
Step 27 (Day 6 / mmmt — Phase 2 (C)): 채널 Ablation 학습
=========================================================
260523v1updatemmmt.md §④ Phase 2 (C-1, C-2) 구현.

목적
----
"멀티모달의 *어느 채널이* 성능 향상에 기여했는가?" 를 분리하기 위해
*입력 채널만 바꾸고* 나머지 (모델/loss/스케줄/augment) 는 step24 와
**완전히 동일하게** 학습한다.

  - C-1: channel_mode='t1ce_only'    [T1ce, T1ce, T1ce]
  - C-2: channel_mode='flair_only'   [FLAIR, FLAIR, FLAIR]

산출물 폴더는 *step24 결과를 덮어쓰지 않도록* 채널 이름을 suffix 로 붙인다.

  outputs/checkpoints/mmmt_ablation/<mode>/{best,last}.pth
  outputs/logs/mmmt_ablation/<mode>/{history,summary}.json
  outputs/figures/mmmt_ablation/<mode>/training_curves.png

사용법
------
    # 두 모드 순차 학습 (RTX 4070 Laptop 기준 약 13시간)
    python code/mmmt/step27_ablation_train.py --mode t1ce_only
    python code/mmmt/step27_ablation_train.py --mode flair_only
    # 한 번에 둘 다 (순차 실행)
    python code/mmmt/step27_ablation_train.py --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step21_mmmt_dataset import create_mm_dataloaders                  # noqa: E402
from step22_mmmt_model import (                                        # noqa: E402
    create_mm_model, freeze_encoder, unfreeze_encoder, get_model_summary,
)
from step23_mmmt_losses import MMMTLoss                                # noqa: E402
# step24 의 루프 함수를 그대로 import 해 통제 변수 유지
from step24_mmmt_train import (                                        # noqa: E402
    EarlyStopping, train_one_epoch, validate, plot_curves,
)

# ─── 경로 ─────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_ROOT = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt_ablation"
LOG_ROOT = PROJECT_DIR / "outputs" / "logs" / "mmmt_ablation"
FIG_ROOT = PROJECT_DIR / "outputs" / "figures" / "mmmt_ablation"

VALID_MODES = ("t1ce_only", "flair_only")

# ─── step24 와 동일 하이퍼파라미터 (오직 channel_mode 만 다름) ─
BASE_CONFIG = {
    "phase1_epochs": 3,
    "phase1_lr": 1e-3,
    "phase2_epochs": 15,
    "phase2_lr": 1e-4,
    "batch_size": 16,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "early_stopping_patience": 5,
    "use_weighted_sampler": False,
    "beta_seg": 0.5,
    "tversky_alpha": 0.7, "tversky_beta": 0.3,
}


def _save_checkpoint(model, optimizer, epoch, metrics, filepath, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                       else v) for k, v in config.items()},
    }, filepath)


def _train_phase(model, train_l, val_l, criterion, optimizer, scheduler,
                 device, n_epochs, phase_name, early_stopping, history,
                 ckpt_dir: Path, config):
    best_val_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_l, criterion, optimizer, device)
        val_m = validate(model, val_l, criterion, device)

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

        if val_m["total"] < best_val_loss:
            best_val_loss = val_m["total"]
            _save_checkpoint(model, optimizer, epoch, val_m,
                             ckpt_dir / "best.pth", config)
            print(f"    ★ best 저장 (val_loss={val_m['total']:.4f}, "
                  f"WT={val_m['dice']:.3f})")

        if early_stopping(val_m["total"]):
            print(f"    ⚠ Early stop (patience={early_stopping.patience})")
            break
    return best_val_loss


def run_one_mode(mode: str):
    """단일 channel_mode 에 대해 step24 와 동일 루틴 학습."""
    assert mode in VALID_MODES, f"mode 는 {VALID_MODES} 중 하나여야 함"
    config = dict(BASE_CONFIG)
    config["channel_mode"] = mode
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = CKPT_ROOT / mode
    log_dir = LOG_ROOT / mode
    fig_dir = FIG_ROOT / mode
    for d in (ckpt_dir, log_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Step 27 (Phase 2 C): Ablation 학습  mode = '{mode}'")
    print("=" * 70)
    print(f"  device       : {config['device']}")
    print(f"  channel_mode : {config['channel_mode']}")
    print(f"  ckpt → {ckpt_dir}")
    print(f"  log  → {log_dir}")
    print(f"  fig  → {fig_dir}")

    # ── DataLoader ────────────────────────────────────────────
    print(f"\n  building dataloaders ...")
    train_l, val_l, _, pos_weight = create_mm_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        use_weighted_sampler=config["use_weighted_sampler"],
        channel_mode=config["channel_mode"],
    )
    print(f"  Train: {len(train_l.dataset):,}  ({len(train_l)} batches)")
    print(f"  Val:   {len(val_l.dataset):,}")
    print(f"  pos_weight: {pos_weight.item():.4f}")

    # ── Model (step24 와 동일) ────────────────────────────────
    device = torch.device(config["device"])
    print(f"\n  building model ...")
    model = create_mm_model(pretrained=True, in_ch=3, seg_classes=1).to(device)
    get_model_summary(model)

    # ── Loss ──────────────────────────────────────────────────
    criterion = MMMTLoss(
        pos_weight=pos_weight.to(device),
        beta_seg=config["beta_seg"],
        alpha_tv=config["tversky_alpha"], beta_tv=config["tversky_beta"],
    )

    # ── history dict ──────────────────────────────────────────
    keys_metric = ["total", "cls", "seg", "acc", "dice"]
    history = {f"train_{k}": [] for k in keys_metric}
    history.update({f"val_{k}": [] for k in keys_metric})
    history["lr"] = []

    start_all = time.time()

    # Phase 1: encoder freeze
    print(f"\n{'=' * 60}\n  Phase 1: encoder freeze\n{'=' * 60}")
    freeze_encoder(model)
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=config["phase1_lr"], weight_decay=config["weight_decay"])
    sch1 = CosineAnnealingLR(opt1, T_max=config["phase1_epochs"])
    es1 = EarlyStopping(patience=999)
    _train_phase(model, train_l, val_l, criterion, opt1, sch1, device,
                 config["phase1_epochs"], "Phase1", es1, history,
                 ckpt_dir, config)

    # Phase 2: full fine-tune
    print(f"\n{'=' * 60}\n  Phase 2: full fine-tune\n{'=' * 60}")
    unfreeze_encoder(model)
    opt2 = AdamW(model.parameters(),
                 lr=config["phase2_lr"], weight_decay=config["weight_decay"])
    sch2 = CosineAnnealingLR(opt2, T_max=config["phase2_epochs"])
    es2 = EarlyStopping(patience=config["early_stopping_patience"])
    _train_phase(model, train_l, val_l, criterion, opt2, sch2, device,
                 config["phase2_epochs"], "Phase2", es2, history,
                 ckpt_dir, config)

    # last 저장
    _save_checkpoint(model, opt2, len(history["train_total"]),
                     {k: history[f"val_{k}"][-1] for k in keys_metric},
                     ckpt_dir / "last.pth", config)

    with open(log_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n  history → {log_dir / 'history.json'}")

    # 학습 곡선 (step24 의 plot_curves 가 FIG_DIR 전역상수를 쓰므로
    # 여기서는 직접 그려 mmmt_ablation/<mode> 로 저장)
    _plot_curves_local(history, fig_dir / "training_curves.png")
    print(f"  curves  → {fig_dir / 'training_curves.png'}")

    total_time = (time.time() - start_all) / 60
    print(f"\n{'=' * 60}")
    print(f"  mode={mode} 완료. 총 학습 시간: {total_time:.1f}분")
    print(f"{'=' * 60}")

    summary = {
        "mode": mode,
        "total_minutes": round(total_time, 1),
        "total_epochs": len(history["train_total"]),
        "final_val": {k: history[f"val_{k}"][-1] for k in keys_metric},
        "config": {k: (str(v) if not isinstance(v, (int, float, str, bool))
                       else v) for k, v in config.items()},
    }
    with open(log_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def _plot_curves_local(history: dict, save: Path):
    """step24.plot_curves 의 로컬 버전 (저장 경로 지정 가능)."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    save.parent.mkdir(parents=True, exist_ok=True)

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

    axes[1, 1].plot(epochs, history["train_dice"], "b-o", label="train WT", ms=3)
    axes[1, 1].plot(epochs, history["val_dice"], "r-o", label="val WT", ms=3)
    axes[1, 1].set_title("WT Dice"); axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MM-MTL 채널 ablation 학습")
    parser.add_argument("--mode", type=str, default="both",
                        choices=("t1ce_only", "flair_only", "both"),
                        help="학습할 channel_mode (기본 both = 순차 실행)")
    args = parser.parse_args()

    if args.mode == "both":
        for m in VALID_MODES:
            print(f"\n\n>>>>>>>>>> START mode = {m} <<<<<<<<<<\n")
            run_one_mode(m)
    else:
        run_one_mode(args.mode)


if __name__ == "__main__":
    main()
