# -*- coding: utf-8 -*-
"""
Step 19: Multi-Task 모델 평가 + 3-way 비교
=============================================
테스트셋에서 Multi-Task 모델을 평가하고,
Whole-Slice / Patch / Multi-Task 3-way 비교를 수행합니다.

사용법:
    python code/multitask/step19_multitask_evaluate.py
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step16_multitask_dataset import create_multitask_dataloaders
from step17_multitask_model import create_multitask_model

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "checkpoints" / "multitask"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures" / "multitask"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "multitask"
# 기존 결과 참조 (읽기만)
WHOLE_LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "whole"   # 기존 Whole-Slice 평가 결과
PATCH_LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "patch"   # 기존 Patch 평가 결과


@torch.no_grad()
def get_predictions(model, loader, device):
    """분류 예측 + 세분화 예측 수집"""
    model.eval()
    all_labels, all_probs, all_masks_gt, all_masks_pred = [], [], [], []

    for images, labels, masks in loader:
        images = images.to(device, non_blocking=True)
        cls_out, seg_out = model(images)

        cls_probs = torch.sigmoid(cls_out.squeeze(1)).cpu().numpy()
        seg_probs = torch.sigmoid(seg_out).cpu().numpy()  # (B,1,H,W)

        all_labels.extend(labels.numpy())
        all_probs.extend(cls_probs)
        all_masks_gt.extend(masks.numpy())        # (B,1,H,W)
        all_masks_pred.extend(seg_probs)

    return (np.array(all_labels), np.array(all_probs),
            np.array(all_masks_gt), np.array(all_masks_pred))


def compute_seg_metrics(masks_gt, masks_pred, threshold=0.5):
    """세분화 지표 계산 (Dice, IoU) — 종양 있는 슬라이스만"""
    dice_scores, iou_scores = [], []
    for i in range(len(masks_gt)):
        gt = (masks_gt[i].flatten() > 0.5).astype(float)
        pred = (masks_pred[i].flatten() > threshold).astype(float)
        if gt.sum() == 0:
            continue  # 종양 없는 슬라이스 제외
        intersection = (gt * pred).sum()
        union_dice = gt.sum() + pred.sum()
        union_iou = gt.sum() + pred.sum() - intersection
        dice = 2.0 * intersection / max(union_dice, 1e-8)
        iou = intersection / max(union_iou, 1e-8)
        dice_scores.append(dice)
        iou_scores.append(iou)
    return np.array(dice_scores), np.array(iou_scores)


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title, vmax in [
        (axes[0], cm, "d", "Confusion Matrix (절대값)", None),
        (axes[1], cm.astype("float") / cm.sum(axis=1)[:, np.newaxis],
         ".3f", "Confusion Matrix (정규화)", 1),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                    ax=ax, annot_kws={"size": 16}, linewidths=1, linecolor="white",
                    vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("예측"); ax.set_ylabel("실제")
    plt.suptitle("Multi-Task Classification", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return cm


def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0,1],[0,1], color="gray", lw=1.5, ls="--")
    ax.scatter(fpr[best_idx], tpr[best_idx], c="#2ecc71", s=150, zorder=5,
               edgecolors="black", lw=2, label=f"최적 Thr={best_thr:.3f}")
    ax.set_title(f"Multi-Task ROC (AUC={roc_auc:.4f})", fontsize=15, fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    ax.set_aspect("equal"); ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    return roc_auc, best_thr


def plot_segmentation_samples(images_list, masks_gt_list, masks_pred_list,
                              labels_list, probs_list, save_path, n=8):
    """예측 마스크 vs GT 마스크 시각화"""
    # 종양 있는 TP 샘플 선택
    tp_indices = [i for i in range(len(labels_list))
                  if labels_list[i] == 1 and (probs_list[i] >= 0.5)]
    if len(tp_indices) == 0:
        return
    np.random.seed(42)
    selected = np.random.choice(tp_indices, min(n, len(tp_indices)), replace=False)

    fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        # 원본 이미지 (정규화 역변환 생략, 첫 채널만)
        img = images_list[idx]
        gt_mask = masks_gt_list[idx].squeeze()
        pred_mask = (masks_pred_list[idx].squeeze() > 0.5).astype(float)

        axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].set_title("MRI")
        axes[row, 1].imshow(gt_mask, cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].set_title("GT Mask")
        axes[row, 2].imshow(pred_mask, cmap="Reds", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Pred Mask (p={probs_list[idx]:.3f})")
        # 오버레이
        axes[row, 3].imshow(img, cmap="gray")
        axes[row, 3].imshow(gt_mask, cmap="Greens", alpha=0.3)
        axes[row, 3].imshow(pred_mask, cmap="Reds", alpha=0.3)
        axes[row, 3].set_title("Overlay (G=GT, R=Pred)")

        for c in range(4):
            axes[row, c].axis("off")

    plt.suptitle("Multi-Task Segmentation 결과", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def plot_3way_comparison(mt_results, save_path):
    """Whole-Slice vs Patch vs Multi-Task 3-way 비교 차트"""
    # 기존 결과 로드
    ws_path = WHOLE_LOG_DIR / "evaluation_results.json"
    patch_path = PATCH_LOG_DIR / "step12_patch_eval.json"

    ws, patch = {}, {}
    if ws_path.exists():
        with open(ws_path) as f: ws = json.load(f)
    if patch_path.exists():
        with open(patch_path) as f: patch = json.load(f)

    # 비교 데이터 구성
    models_names = []
    metrics_data = {"Accuracy": [], "Precision": [], "Recall": [],
                    "F1-Score": [], "AUC-ROC": []}

    if ws:
        models_names.append("Whole-Slice\n(ResNet-18)")
        metrics_data["Accuracy"].append(ws.get("accuracy", 0))
        metrics_data["Precision"].append(ws.get("precision", 0))
        metrics_data["Recall"].append(ws.get("recall_sensitivity", 0))
        metrics_data["F1-Score"].append(ws.get("f1_score", 0))
        metrics_data["AUC-ROC"].append(ws.get("auc_roc", 0) * 100)

    if patch:
        models_names.append("Patch-Based\n(PatchCNN)")
        slice_best = patch.get("slice_results", {}).get("count", {})
        metrics_data["Accuracy"].append(slice_best.get("accuracy", 0))
        metrics_data["Precision"].append(slice_best.get("precision", 0))
        metrics_data["Recall"].append(slice_best.get("recall", 0))
        metrics_data["F1-Score"].append(slice_best.get("f1", 0))
        metrics_data["AUC-ROC"].append(slice_best.get("auc", 0) * 100)

    models_names.append("Multi-Task\n(Ours)")
    metrics_data["Accuracy"].append(mt_results["accuracy"])
    metrics_data["Precision"].append(mt_results["precision"])
    metrics_data["Recall"].append(mt_results["recall_sensitivity"])
    metrics_data["F1-Score"].append(mt_results["f1_score"])
    metrics_data["AUC-ROC"].append(mt_results["auc_roc"] * 100)

    # 차트
    n_metrics = len(metrics_data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    colors = ["#3498db", "#e67e22", "#2ecc71"][:len(models_names)]

    for i, (metric, values) in enumerate(metrics_data.items()):
        bars = axes[i].bar(models_names, values, color=colors, edgecolor="white", lw=1.5)
        axes[i].set_title(metric, fontsize=13, fontweight="bold")
        axes[i].set_ylim(60, 100)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("3-Way 모델 비교", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ 3-way 비교 차트 저장: {save_path.name}")


def main():
    print("=" * 60)
    print("  Step 19: Multi-Task 모델 평가")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    model_path = CHECKPOINT_DIR / "mt_best_model.pth"
    if not model_path.exists():
        print(f"\n[ERROR] 체크포인트 없음: {model_path}")
        return

    print(f"\n  모델 로드: {model_path.name}")
    model = create_multitask_model(pretrained=False)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Epoch: {ckpt['epoch']}, Val Acc: {ckpt['val_acc']:.2f}%, "
          f"Val Dice: {ckpt['val_dice']:.4f}")

    # DataLoader
    _, _, test_loader, _ = create_multitask_dataloaders(
        batch_size=32, num_workers=4
    )
    print(f"  Test samples: {len(test_loader.dataset):,}")

    # 예측 수집
    print(f"\n{'─' * 60}")
    print("  예측 수집 중...")
    labels, probs, masks_gt, masks_pred = get_predictions(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)
    print(f"  수집 완료: {len(labels):,} samples")

    # ─── 분류 지표 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Classification Report:")
    target_names = ["Negative", "Positive"]
    report = classification_report(labels, preds, target_names=target_names)
    print(report)

    report_dict = classification_report(labels, preds, target_names=target_names,
                                        output_dict=True)
    with open(LOG_DIR / "mt_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # ─── 시각화 ──────────────────────────────────────────────
    cm = plot_confusion_matrix(labels, preds, FIGURES_DIR / "mt_confusion_matrix.png")
    print(f"  ✓ CM 저장")

    roc_auc, best_thr = plot_roc_curve(labels, probs, FIGURES_DIR / "mt_roc_curve.png")
    print(f"  ✓ ROC 저장 (AUC={roc_auc:.4f})")

    # ─── 세분화 지표 ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  세분화 지표 계산 중...")
    dice_scores, iou_scores = compute_seg_metrics(masks_gt, masks_pred)
    mean_dice = dice_scores.mean() if len(dice_scores) > 0 else 0
    mean_iou = iou_scores.mean() if len(iou_scores) > 0 else 0
    print(f"  종양 슬라이스 수: {len(dice_scores):,}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    print(f"  Mean IoU:  {mean_iou:.4f}")

    # Dice 분포 히스토그램
    if len(dice_scores) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(dice_scores, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
        axes[0].axvline(mean_dice, color="red", ls="--", lw=2, label=f"Mean={mean_dice:.4f}")
        axes[0].set_title("Dice Score 분포", fontsize=13, fontweight="bold")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].hist(iou_scores, bins=50, color="#2ecc71", edgecolor="white", alpha=0.8)
        axes[1].axvline(mean_iou, color="red", ls="--", lw=2, label=f"Mean={mean_iou:.4f}")
        axes[1].set_title("IoU 분포", fontsize=13, fontweight="bold")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.suptitle("Multi-Task Segmentation 성능 분포", fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "mt_seg_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Dice/IoU 분포 저장")

    # ─── 세분화 시각화 샘플 ──────────────────────────────────
    # 원본 이미지 수집 (정규화 전 grayscale)
    print("  세분화 샘플 시각화 중...")
    import cv2, pandas as pd
    splits_df = pd.read_csv(PROJECT_DIR / "processed" / "splits.csv")
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    slice_dir = PROJECT_DIR / "processed" / "slices"

    raw_images = []
    for i in range(min(len(test_df), len(labels))):
        fname = test_df.iloc[i]["filename"]
        arr = np.fromfile(str(slice_dir / fname), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
        raw_images.append(img)

    if len(raw_images) > 0:
        plot_segmentation_samples(
            raw_images, masks_gt, masks_pred, labels, probs,
            FIGURES_DIR / "mt_segmentation_samples.png", n=8
        )
        print(f"  ✓ 세분화 샘플 시각화 저장")

    # ─── 결과 저장 ───────────────────────────────────────────
    tn, fp, fn, tp = cm.ravel()
    precision_val = tp / max(tp + fp, 1) * 100
    recall_val = tp / max(tp + fn, 1) * 100
    specificity = tn / max(tn + fp, 1) * 100
    f1 = 2 * (precision_val * recall_val) / max(precision_val + recall_val, 1)
    accuracy = (preds == labels).mean() * 100

    results = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision_val, 2),
        "recall_sensitivity": round(recall_val, 2),
        "specificity": round(specificity, 2),
        "f1_score": round(f1, 2),
        "auc_roc": round(roc_auc, 4),
        "optimal_threshold": round(float(best_thr), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "segmentation": {
            "mean_dice": round(float(mean_dice), 4),
            "mean_iou": round(float(mean_iou), 4),
            "num_tumor_slices": int(len(dice_scores)),
        },
        "total_test_samples": int(len(labels)),
    }

    with open(LOG_DIR / "mt_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ─── 3-way 비교 ─────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  3-way 비교 차트 생성 중...")
    plot_3way_comparison(results, FIGURES_DIR / "mt_3way_comparison.png")

    # ─── 최종 출력 ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Multi-Task 평가 결과 (Test Set)")
    print(f"{'═' * 60}")
    print(f"  [Classification]")
    print(f"    Accuracy:    {accuracy:.2f}%")
    print(f"    Precision:   {precision_val:.2f}%")
    print(f"    Recall:      {recall_val:.2f}%")
    print(f"    Specificity: {specificity:.2f}%")
    print(f"    F1-Score:    {f1:.2f}%")
    print(f"    AUC-ROC:     {roc_auc:.4f}")
    print(f"  [Segmentation]")
    print(f"    Mean Dice:   {mean_dice:.4f}")
    print(f"    Mean IoU:    {mean_iou:.4f}")
    print(f"{'─' * 60}")
    print(f"    TN: {tn:,}  FP: {fp:,}")
    print(f"    FN: {fn:,}  TP: {tp:,}")
    print(f"{'═' * 60}")
    print(f"  결과 저장: {LOG_DIR / 'mt_evaluation_results.json'}")


if __name__ == "__main__":
    main()
