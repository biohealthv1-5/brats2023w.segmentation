# -*- coding: utf-8 -*-
"""
Step 25 (Day 6 / mmmt): Multi-Modal Multi-Task Evaluation
==========================================================
- Test 셋 평가 (WT Dice + IoU + Cls F1/AUROC/AUPRC)
- Confusion matrix / ROC / PR + Day 5 MTL 단일모달 결과와 4-way 비교

본 단계는 Day 6 *순수 멀티모달 효과*만 평가합니다. TTA / Temperature
Scaling / Conformal Prediction 등은 Day 7 `sota/` 패키지에서 진행됩니다.

사용법:
    python code/mmmt/step25_mmmt_evaluate.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score, classification_report,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step21_mmmt_dataset import create_mm_dataloaders
from step22_mmmt_model import create_mm_model

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt"
FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "mmmt"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt"
MTL_PREV_LOG = PROJECT_DIR / "outputs" / "logs" / "multitask"   # Day 5 결과 (4-way 비교용)


# ────────────────────────────────────────────────────────────
#  예측 수집
# ────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    all_seg_gt, all_seg_pred = [], []
    for imgs, labels, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        cls_logit, seg_out = model(imgs)
        cls_logit = cls_logit.squeeze(1)
        cls_probs = torch.sigmoid(cls_logit)
        seg_probs = torch.sigmoid(seg_out)
        all_labels.append(labels.numpy())
        all_probs.append(cls_probs.cpu().numpy())
        all_seg_gt.append(masks.numpy())
        all_seg_pred.append(seg_probs.cpu().numpy())
    return {
        "labels": np.concatenate(all_labels),
        "probs": np.concatenate(all_probs),
        "seg_gt": np.concatenate(all_seg_gt),
        "seg_pred": np.concatenate(all_seg_pred),
    }


def compute_seg_metrics(seg_gt: np.ndarray, seg_pred: np.ndarray,
                        threshold: float = 0.5):
    """(N, 1, H, W) → WT Dice/IoU (양성 슬라이스만)"""
    dices, ious = [], []
    for i in range(seg_gt.shape[0]):
        g = (seg_gt[i, 0].flatten() > 0.5).astype(np.float32)
        p = (seg_pred[i, 0].flatten() > threshold).astype(np.float32)
        if g.sum() == 0:
            continue
        inter = (g * p).sum()
        dices.append(2.0 * inter / max(g.sum() + p.sum(), 1e-8))
        ious.append(inter / max(g.sum() + p.sum() - inter, 1e-8))
    return {
        "n_positive_slices": len(dices),
        "dice_mean": float(np.mean(dices)) if dices else 0.0,
        "dice_median": float(np.median(dices)) if dices else 0.0,
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "iou_median": float(np.median(ious)) if ious else 0.0,
    }


# ────────────────────────────────────────────────────────────
#  시각화
# ────────────────────────────────────────────────────────────
def plot_diagnostics(labels, probs, seg_gt, seg_pred, out_dir: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    fpr, tpr, _ = roc_curve(labels, probs)
    pr_prec, pr_rec, _ = precision_recall_curve(labels, probs)
    auroc = auc(fpr, tpr)
    auprc = average_precision_score(labels, probs)
    f1 = f1_score(labels, preds)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                ax=axes[0, 0])
    axes[0, 0].set_title(f"Confusion Matrix (F1={f1:.4f})")
    axes[0, 0].set_xlabel("예측"); axes[0, 0].set_ylabel("실제")

    axes[0, 1].plot(fpr, tpr, lw=2, label=f"AUROC={auroc:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_title("ROC"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(pr_rec, pr_prec, lw=2, label=f"AUPRC={auprc:.4f}")
    axes[1, 0].set_title("PR"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlabel("Recall"); axes[1, 0].set_ylabel("Precision")

    # WT Dice 박스플롯
    dices = []
    for i in range(seg_gt.shape[0]):
        g = (seg_gt[i, 0].flatten() > 0.5).astype(np.float32)
        p = (seg_pred[i, 0].flatten() > 0.5).astype(np.float32)
        if g.sum() == 0:
            continue
        inter = (g * p).sum()
        dices.append(2.0 * inter / max(g.sum() + p.sum(), 1e-8))
    axes[1, 1].boxplot([dices], tick_labels=["WT"])
    axes[1, 1].set_title("WT Dice (test, 양성 슬라이스만)")
    axes[1, 1].set_ylabel("Dice"); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "mmmt_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 25 (Day 6 / mmmt): MM-MTL Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ─────────────────────────────────────────
    _, _, test_l, _ = create_mm_dataloaders(
        batch_size=16, num_workers=0,
        use_weighted_sampler=False, channel_mode="t1ce_flair_diff",
    )

    # ── Load model ──────────────────────────────────────────
    ckpt_path = CKPT_DIR / "mmmt_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint 없음: {ckpt_path}")
        return
    print(f"  loading {ckpt_path}")
    model = create_mm_model(pretrained=False, in_ch=3, seg_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Test 예측 ───────────────────────────────────────────
    print(f"\n  [test] collecting predictions ...")
    pack = collect_predictions(model, test_l, device)
    labels = pack["labels"]
    probs = pack["probs"]
    seg_gt = pack["seg_gt"]
    seg_pred = pack["seg_pred"]

    # ── 분류 지표 ───────────────────────────────────────────
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    cls_metrics = {
        "threshold": 0.5,
        "f1": float(f1_score(labels, preds)),
        "auroc": float(auc(*roc_curve(labels, probs)[:2])),
        "auprc": float(average_precision_score(labels, probs)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    print(f"\n  [Cls metrics]")
    print(f"    F1={cls_metrics['f1']:.4f}  AUROC={cls_metrics['auroc']:.4f}  "
          f"AUPRC={cls_metrics['auprc']:.4f}")
    print(f"    TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ── Segmentation 지표 ──────────────────────────────────
    seg_metrics = compute_seg_metrics(seg_gt, seg_pred, threshold=0.5)
    print(f"\n  [Seg metrics — WT]")
    print(f"    Dice={seg_metrics['dice_mean']:.4f}  IoU={seg_metrics['iou_mean']:.4f}  "
          f"n_pos={seg_metrics['n_positive_slices']}")

    # ── 결과 저장 ───────────────────────────────────────────
    plot_diagnostics(labels, probs, seg_gt, seg_pred, FIG_DIR)
    final = {
        "n_test": int(len(labels)),
        "cls": cls_metrics,
        "seg_wt": seg_metrics,
    }
    with open(LOG_DIR / "mmmt_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n  metrics saved: {LOG_DIR / 'mmmt_test_metrics.json'}")
    print(f"  figures:       {FIG_DIR}")

    # ── 4-way 비교 (이전 MTL 단일모달 결과) ────────────────
    prev = MTL_PREV_LOG / "mt_test_metrics.json"
    if prev.exists():
        try:
            with open(prev, "r", encoding="utf-8") as f:
                prev_m = json.load(f)
            print(f"\n  [4-way 비교 — Day5 MTL(FLAIR) vs Day6 MM-MTL(T1ce+FLAIR)]")
            print(f"    Day5 MTL : {prev_m}")
            print(f"    Day6 MMMT: {cls_metrics}  (+seg: {seg_metrics['dice_mean']:.4f})")
        except Exception as e:
            print(f"  [WARN] 이전 결과 비교 실패: {e}")

    print("=" * 60)


if __name__ == "__main__":
    main()
