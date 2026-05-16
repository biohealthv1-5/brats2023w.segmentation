# -*- coding: utf-8 -*-
"""
Step 6: 모델 평가
==================
테스트셋에서 최적 모델의 성능을 평가하고 시각화합니다.
  - Confusion Matrix
  - ROC Curve & AUC
  - Classification Report
  - 예측 확률 분포

사용법:
    python code/step6_evaluate.py
"""

import os
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
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 한글 폰트 설정
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# 프로젝트 모듈 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_dataset import create_dataloaders
from step4_model import create_model

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
LOG_DIR = PROJECT_DIR / "outputs" / "logs"


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    모델의 예측 결과를 수집합니다.

    Returns:
        all_labels: 실제 라벨 (numpy array)
        all_probs: 예측 확률 (numpy array)
        all_preds: 예측 클래스 (numpy array)
    """
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    return all_labels, all_probs, all_preds


def plot_confusion_matrix(labels, preds, save_path):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 절대값 Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axes[0],
        annot_kws={"size": 16},
        linewidths=1,
        linecolor="white",
    )
    axes[0].set_title("Confusion Matrix (절대값)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("예측 (Predicted)", fontsize=12)
    axes[0].set_ylabel("실제 (Actual)", fontsize=12)

    # 정규화 Confusion Matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=axes[1],
        annot_kws={"size": 16},
        linewidths=1,
        linecolor="white",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Confusion Matrix (정규화)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("예측 (Predicted)", fontsize=12)
    axes[1].set_ylabel("실제 (Actual)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Confusion Matrix 저장: {save_path.name}")

    return cm


def plot_roc_curve(labels, probs, save_path):
    """ROC Curve 시각화"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    # ROC Curve
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--",
            label="Random (AUC = 0.5)")

    # 최적 threshold 찾기 (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    ax.scatter(fpr[best_idx], tpr[best_idx], color="#2ecc71", s=150,
               zorder=5, edgecolors="black", linewidth=2,
               label=f"최적 Threshold = {best_threshold:.3f}")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.4f})", fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ ROC Curve 저장: {save_path.name}")

    return roc_auc, best_threshold


def plot_precision_recall_curve(labels, probs, save_path):
    """Precision-Recall Curve 시각화"""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap_score = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(recall, precision, color="#3498db", lw=2.5,
            label=f"PR Curve (AP = {ap_score:.4f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_title(f"Precision-Recall Curve (AP = {ap_score:.4f})",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ PR Curve 저장: {save_path.name}")

    return ap_score


def plot_probability_distribution(labels, probs, save_path):
    """예측 확률 분포 시각화"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Negative 클래스
    ax.hist(
        probs[labels == 0],
        bins=50,
        alpha=0.6,
        color="#2ecc71",
        label=f"Negative (n={int((labels==0).sum()):,})",
        edgecolor="white",
        density=True,
    )
    # Positive 클래스
    ax.hist(
        probs[labels == 1],
        bins=50,
        alpha=0.6,
        color="#e74c3c",
        label=f"Positive (n={int((labels==1).sum()):,})",
        edgecolor="white",
        density=True,
    )

    ax.axvline(x=0.5, color="black", linestyle="--", lw=2,
               label="Threshold (0.5)")

    ax.set_title("예측 확률 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("예측 확률 (Sigmoid Output)", fontsize=12)
    ax.set_ylabel("밀도 (Density)", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 확률 분포 저장: {save_path.name}")


def main():
    print("=" * 60)
    print("  Step 6: 모델 평가")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # 디렉토리 생성
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 체크포인트 확인
    best_model_path = CHECKPOINT_DIR / "best_model.pth"
    if not best_model_path.exists():
        print(f"\n[ERROR] 체크포인트가 없습니다: {best_model_path}")
        print("  step5_train.py를 먼저 실행하세요.")
        return

    # ─── 모델 로드 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  최적 모델 로드 중...")
    model = create_model(pretrained=False)
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  체크포인트 Epoch: {checkpoint['epoch']}")
    print(f"  체크포인트 Val Loss: {checkpoint['val_loss']:.4f}")
    print(f"  체크포인트 Val Acc: {checkpoint['val_acc']:.2f}%")

    # ─── DataLoader ──────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Test DataLoader 생성 중...")
    _, _, test_loader, _ = create_dataloaders(
        batch_size=64, num_workers=4,
    )
    print(f"  Test samples: {len(test_loader.dataset):,}")

    # ─── 예측 수집 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  예측 수집 중...")
    labels, probs, preds = get_predictions(model, test_loader, device)
    print(f"  수집 완료: {len(labels):,} samples")

    # ─── Classification Report ───────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Classification Report:")
    print(f"{'─' * 60}")
    target_names = ["Negative (종양 없음)", "Positive (종양 존재)"]
    report = classification_report(labels, preds, target_names=target_names)
    print(report)

    # 리포트를 파일로 저장
    report_dict = classification_report(labels, preds, target_names=target_names, output_dict=True)
    with open(LOG_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # ─── 시각화 ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  시각화 생성 중...")
    print(f"{'─' * 60}")

    # 1. Confusion Matrix
    cm = plot_confusion_matrix(
        labels, preds, FIGURES_DIR / "07_confusion_matrix.png"
    )

    # 2. ROC Curve
    roc_auc, best_threshold = plot_roc_curve(
        labels, probs, FIGURES_DIR / "08_roc_curve.png"
    )

    # 3. Precision-Recall Curve
    ap_score = plot_precision_recall_curve(
        labels, probs, FIGURES_DIR / "09_pr_curve.png"
    )

    # 4. 예측 확률 분포
    plot_probability_distribution(
        labels, probs, FIGURES_DIR / "10_probability_distribution.png"
    )

    # ─── 최종 결과 요약 ──────────────────────────────────────
    accuracy = (preds == labels).mean() * 100
    tn, fp, fn, tp = cm.ravel()
    precision_val = tp / max(tp + fp, 1) * 100
    recall_val = tp / max(tp + fn, 1) * 100
    specificity = tn / max(tn + fp, 1) * 100
    f1 = 2 * (precision_val * recall_val) / max(precision_val + recall_val, 1)

    results = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision_val, 2),
        "recall_sensitivity": round(recall_val, 2),
        "specificity": round(specificity, 2),
        "f1_score": round(f1, 2),
        "auc_roc": round(roc_auc, 4),
        "ap_score": round(ap_score, 4),
        "optimal_threshold": round(best_threshold, 4),
        "confusion_matrix": {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
        },
        "total_test_samples": int(len(labels)),
    }

    # 결과 저장
    with open(LOG_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'═' * 60}")
    print("  최종 평가 결과 (Test Set)")
    print(f"{'═' * 60}")
    print(f"  Accuracy:    {accuracy:.2f}%")
    print(f"  Precision:   {precision_val:.2f}%")
    print(f"  Recall:      {recall_val:.2f}%")
    print(f"  Specificity: {specificity:.2f}%")
    print(f"  F1-Score:    {f1:.2f}%")
    print(f"  AUC-ROC:     {roc_auc:.4f}")
    print(f"  AP Score:    {ap_score:.4f}")
    print(f"{'─' * 60}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {tn:,}  FP: {fp:,}")
    print(f"    FN: {fn:,}  TP: {tp:,}")
    print(f"{'═' * 60}")
    print(f"\n  결과 저장: {LOG_DIR / 'evaluation_results.json'}")
    print(f"  시각화 저장: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
