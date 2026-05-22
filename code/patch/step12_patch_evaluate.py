# -*- coding: utf-8 -*-
"""
Step 12: 패치 모델 평가
========================
패치 단위 성능 + 슬라이스 단위 집계 성능을 모두 평가하고,
기존 Whole-Slice 모델과 직접 비교합니다.

Level 1: 패치 단위 — Accuracy, F1, AUC-ROC, Confusion Matrix
Level 2: 슬라이스 단위 — max/mean/count 집계 → 슬라이스 분류
Level 3: Whole-Slice vs Patch 비교 표

사용법:
    python code/patch/step12_patch_evaluate.py
"""

import sys, json
import numpy as np
import pandas as pd
import torch
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
from step9_patch_dataset import create_patch_dataloaders, PATCH_CSV
from step10_patch_model import PatchCNN

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "patch" / "patch_best_model.pth"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures" / "patch" / "patch_eval"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "patch"
WHOLE_LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "whole"   # 카테고리별 분리: whole 결과는 별도 폴더
PREV_EVAL = WHOLE_LOG_DIR / "evaluation_results.json"   # whole-slice 결과 (step6_evaluate.py 산출물)


# ═══════════════════════════════════════════════════════════════
#  패치 단위 예측 수집
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def get_patch_predictions(model, loader, device):
    """모든 패치에 대한 예측 수집"""
    model.eval()
    all_labels, all_probs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    return np.array(all_labels), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════
#  슬라이스 단위 집계
# ═══════════════════════════════════════════════════════════════
def aggregate_to_slice(test_df, patch_probs, method="max", threshold=0.5, count_n=2):
    """
    패치 예측을 슬라이스 단위로 집계합니다.

    method:
        "max"   — 패치 중 최대 확률
        "mean"  — 패치 평균 확률
        "count" — 양성 패치 ≥ count_n개
    """
    df = test_df.copy()
    df["prob"] = patch_probs

    grouped = df.groupby("slice_filename")

    slice_results = []
    for fname, g in grouped:
        gt_label = g["slice_label"].iloc[0]  # 슬라이스 GT

        if method == "max":
            score = g["prob"].max()
            pred = 1 if score >= threshold else 0
        elif method == "mean":
            score = g["prob"].mean()
            pred = 1 if score >= threshold else 0
        elif method == "count":
            pos_patches = (g["prob"] >= threshold).sum()
            score = pos_patches / len(g)
            pred = 1 if pos_patches >= count_n else 0
        else:
            raise ValueError(f"Unknown method: {method}")

        slice_results.append({
            "filename": fname,
            "gt_label": gt_label,
            "score": score,
            "pred": pred,
        })

    sdf = pd.DataFrame(slice_results)
    return sdf["gt_label"].values, sdf["score"].values, sdf["pred"].values


# ═══════════════════════════════════════════════════════════════
#  시각화 함수들
# ═══════════════════════════════════════════════════════════════
def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                ax=ax, annot_kws={"size": 16}, linewidths=1, linecolor="white")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("예측", fontsize=12)
    ax.set_ylabel("실제", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path.name}")
    return cm


def plot_roc(labels, probs, title, save_path):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    j = np.argmax(tpr - fpr)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "gray", lw=1.5, ls="--")
    ax.scatter(fpr[j], tpr[j], color="#2ecc71", s=150, zorder=5,
               edgecolors="black", lw=2, label=f"Best θ={thresholds[j]:.3f}")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("FPR", fontsize=12)
    ax.set_ylabel("TPR", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path.name}")
    return roc_auc, thresholds[j]


def plot_comparison_bar(ws_results, patch_results, save_path):
    """Whole-Slice vs Patch 비교 바 차트"""
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC-ROC"]
    ws_vals = [
        ws_results["accuracy"], ws_results["precision"],
        ws_results["recall_sensitivity"], ws_results["specificity"],
        ws_results["f1_score"], ws_results["auc_roc"] * 100,
    ]
    pa_vals = [
        patch_results["accuracy"], patch_results["precision"],
        patch_results["recall"], patch_results["specificity"],
        patch_results["f1"], patch_results["auc_roc"] * 100,
    ]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w/2, ws_vals, w, label="Whole-Slice", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, pa_vals, w, label="Patch-Based (max)", color="#e74c3c", alpha=0.85)

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("점수 (%)", fontsize=12)
    ax.set_title("Whole-Slice vs Patch-Based 성능 비교 (슬라이스 단위)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_fn_fp_comparison(ws_results, patch_cm, save_path):
    """FN/FP 개수 비교 바 차트"""
    tn, fp, fn, tp = patch_cm.ravel()
    categories = ["FN (놓친 종양)", "FP (오탐)"]
    ws_vals = [ws_results["confusion_matrix"]["FN"], ws_results["confusion_matrix"]["FP"]]
    pa_vals = [fn, fp]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, ws_vals, w, label="Whole-Slice", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w/2, pa_vals, w, label="Patch-Based", color="#e74c3c", alpha=0.85)

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10,
                    f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=13)
    ax.set_ylabel("개수", fontsize=12)
    ax.set_title("FN / FP 비교: Whole-Slice vs Patch-Based",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path.name}")


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Step 12: 패치 모델 평가")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 모델 로드 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  패치 모델 로드 중...")
    if not CHECKPOINT.exists():
        print(f"  [ERROR] 체크포인트 없음: {CHECKPOINT}")
        return

    model = PatchCNN().to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✓ 로드 완료")

    # ─── DataLoader (테스트셋 전체) ──────────────────────────
    print(f"\n{'─' * 60}")
    print("  Test DataLoader 생성 중...")
    _, _, test_loader, _ = create_patch_dataloaders(
        batch_size=256, num_workers=0, neg_subsample_ratio=None,
    )
    print(f"  Test 패치: {len(test_loader.dataset):,}")

    # ─── 패치 CSV 로드 (슬라이스 집계용) ─────────────────────
    df_all = pd.read_csv(PATCH_CSV)
    test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════
    # Level 1: 패치 단위 평가
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Level 1: 패치 단위 평가")
    print(f"{'═' * 60}")

    p_labels, p_probs = get_patch_predictions(model, test_loader, device)
    p_preds = (p_probs >= 0.5).astype(int)

    print(f"  패치 수집 완료: {len(p_labels):,}")

    # Confusion Matrix
    p_cm = plot_confusion_matrix(
        p_labels, p_preds, "패치 단위 Confusion Matrix",
        FIGURES_DIR / "12_patch_confusion_matrix.png")
    tn, fp, fn, tp = p_cm.ravel()

    # Classification Report
    print(f"\n  패치 Classification Report:")
    print(classification_report(p_labels, p_preds,
          target_names=["Negative", "Positive"]))

    # ROC
    p_auc, p_best_thr = plot_roc(
        p_labels, p_probs, "패치 단위 ROC Curve",
        FIGURES_DIR / "12_patch_roc.png")

    patch_acc = (p_preds == p_labels).mean() * 100
    patch_prec = tp / max(tp + fp, 1) * 100
    patch_rec = tp / max(tp + fn, 1) * 100
    patch_spec = tn / max(tn + fp, 1) * 100
    patch_f1 = 2 * patch_prec * patch_rec / max(patch_prec + patch_rec, 1)

    print(f"  Patch Accuracy:  {patch_acc:.2f}%")
    print(f"  Patch AUC-ROC:   {p_auc:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Level 2: 슬라이스 단위 집계
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Level 2: 슬라이스 단위 집계 평가")
    print(f"{'═' * 60}")

    slice_results = {}
    for method in ["max", "mean", "count"]:
        s_labels, s_scores, s_preds = aggregate_to_slice(
            test_df, p_probs, method=method, threshold=0.5, count_n=2)

        s_cm = confusion_matrix(s_labels, s_preds)
        s_tn, s_fp, s_fn, s_tp = s_cm.ravel()
        s_acc = (s_preds == s_labels).mean() * 100
        s_prec = s_tp / max(s_tp + s_fp, 1) * 100
        s_rec = s_tp / max(s_tp + s_fn, 1) * 100
        s_spec = s_tn / max(s_tn + s_fp, 1) * 100
        s_f1 = 2 * s_prec * s_rec / max(s_prec + s_rec, 1)
        try:
            s_auc = auc(*roc_curve(s_labels, s_scores)[:2])
        except Exception:
            s_auc = 0.0

        slice_results[method] = {
            "accuracy": round(s_acc, 2),
            "precision": round(s_prec, 2),
            "recall": round(s_rec, 2),
            "specificity": round(s_spec, 2),
            "f1": round(s_f1, 2),
            "auc_roc": round(s_auc, 4),
            "cm": {"TN": int(s_tn), "FP": int(s_fp), "FN": int(s_fn), "TP": int(s_tp)},
        }

        print(f"\n  [{method.upper()}] Acc={s_acc:.2f}% F1={s_f1:.2f}% "
              f"AUC={s_auc:.4f} | FN={s_fn} FP={s_fp}")

        # 최적 집계 방법에 대해 시각화
        plot_confusion_matrix(
            s_labels, s_preds, f"슬라이스 단위 CM ({method})",
            FIGURES_DIR / f"12_slice_{method}_cm.png")

    # Best method (max가 보통 최적)
    best_method = max(slice_results, key=lambda m: slice_results[m]["f1"])
    best = slice_results[best_method]
    print(f"\n  ★ 최적 집계: {best_method.upper()} (F1={best['f1']:.2f}%)")

    # Best method ROC
    s_labels, s_scores, s_preds = aggregate_to_slice(
        test_df, p_probs, method=best_method)
    s_auc, s_thr = plot_roc(
        s_labels, s_scores, f"슬라이스 단위 ROC ({best_method})",
        FIGURES_DIR / "12_slice_best_roc.png")

    # ═══════════════════════════════════════════════════════════
    # Level 3: Whole-Slice vs Patch 비교
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  Level 3: Whole-Slice vs Patch-Based 비교")
    print(f"{'═' * 60}")

    if PREV_EVAL.exists():
        with open(PREV_EVAL, "r", encoding="utf-8") as f:
            ws = json.load(f)

        print(f"\n  {'지표':<20s} {'Whole-Slice':>12s} {'Patch('+best_method+')':>14s} {'차이':>8s}")
        print(f"  {'─'*56}")

        comparisons = [
            ("Accuracy", ws["accuracy"], best["accuracy"]),
            ("Precision", ws["precision"], best["precision"]),
            ("Recall", ws["recall_sensitivity"], best["recall"]),
            ("Specificity", ws["specificity"], best["specificity"]),
            ("F1-Score", ws["f1_score"], best["f1"]),
            ("AUC-ROC", ws["auc_roc"]*100, best["auc_roc"]*100),
        ]
        for name, w_val, p_val in comparisons:
            diff = p_val - w_val
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {name:<20s} {w_val:>11.2f}% {p_val:>13.2f}% {arrow}{abs(diff):>6.2f}%")

        ws_fn = ws["confusion_matrix"]["FN"]
        ws_fp = ws["confusion_matrix"]["FP"]
        pa_fn = best["cm"]["FN"]
        pa_fp = best["cm"]["FP"]
        print(f"\n  {'FN (놓친 종양)':<20s} {ws_fn:>11,d}  {pa_fn:>13,d}  {'↓' if pa_fn < ws_fn else '↑'}{abs(pa_fn-ws_fn):>5d}")
        print(f"  {'FP (오탐)':<20s} {ws_fp:>11,d}  {pa_fp:>13,d}  {'↓' if pa_fp < ws_fp else '↑'}{abs(pa_fp-ws_fp):>5d}")

        # 비교 차트
        plot_comparison_bar(ws, best, FIGURES_DIR / "12_comparison_bar.png")

        # FN/FP 비교 차트
        best_cm = confusion_matrix(s_labels, s_preds)
        plot_fn_fp_comparison(ws, best_cm, FIGURES_DIR / "12_fn_fp_comparison.png")
    else:
        print("  [SKIP] whole-slice 평가 결과 없음 — 비교 생략")

    # ─── 결과 저장 ───────────────────────────────────────────
    final = {
        "patch_level": {
            "accuracy": round(patch_acc, 2),
            "precision": round(patch_prec, 2),
            "recall": round(patch_rec, 2),
            "specificity": round(patch_spec, 2),
            "f1": round(patch_f1, 2),
            "auc_roc": round(p_auc, 4),
            "total_patches": int(len(p_labels)),
            "cm": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        },
        "slice_level": slice_results,
        "best_aggregation": best_method,
    }
    save_path = LOG_DIR / "step12_patch_eval.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'─' * 60}")
    print(f"  결과 저장: {save_path}")
    print(f"  시각화 저장: {FIGURES_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
