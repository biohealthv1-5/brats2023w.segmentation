# -*- coding: utf-8 -*-
"""
Step 25b (Day 6 / mmmt — Phase 1 (A)): Threshold 동등 보정 재평가
=====================================================================
260523v1updatemmmt.md §④ Phase 1 (A) 구현.

목적
----
Day 5 MTL 의 ROC-Youden 최적 임계값(= 0.3847)은 그 모델의 "운용 지점"이고,
260522mmmt.md §3.1·§5.2 에서 보고된 Day 5 F1=93.91 / FP=305 / FN=1136 은
*그 threshold* 에서 산출된 값이다.

Day 6 MM-MTL 의 기본 평가(step25)는 threshold=0.5 에서 수행되었으므로
"멀티모달의 *순수* 효과"를 분리하려면 **두 모델을 같은 운용 지점에서**
비교해야 한다. 본 스크립트는 mmmt_best.pth 를 그대로 추론하고
여러 threshold 에서 cls 메트릭을 재계산한다.

산출물
------
- outputs/logs/mmmt/mmmt_test_metrics_thr_sweep.json
    threshold 별 confusion matrix · F1 · precision · recall · accuracy
- outputs/figures/mmmt/threshold_sweep.png
    threshold vs (F1, FP, FN) 곡선
- 콘솔에 4-way 표 (Day5@0.3847 vs Day6@0.5 vs Day6@0.3847) 출력

사용법
------
    python code/mmmt/step25b_mmmt_evaluate_thr.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step21_mmmt_dataset import create_mm_dataloaders
from step22_mmmt_model import create_mm_model
from step25_mmmt_evaluate import collect_predictions, compute_seg_metrics

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt"
FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "mmmt"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt"
MTL_LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "multitask"

# Day 5 MTL ROC-Youden optimal threshold (mt_evaluation_results.json 기록치)
DAY5_OPTIMAL_THR = 0.3847

# 비교용 threshold 그리드
THR_GRID = [0.30, 0.35, DAY5_OPTIMAL_THR, 0.40, 0.45, 0.50]


# ─────────────────────────────────────────────────────────────
#  메트릭 계산
# ─────────────────────────────────────────────────────────────
def metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, thr: float):
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (preds == labels).mean()
    return {
        "threshold": round(float(thr), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "accuracy":    round(float(accuracy * 100), 2),
        "precision":   round(float(precision * 100), 2),
        "recall":      round(float(recall * 100), 2),
        "specificity": round(float(specificity * 100), 2),
        "f1":          round(float(f1 * 100), 2),
    }


def plot_threshold_sweep(labels, probs, save_path: Path,
                         highlight_thr: float = DAY5_OPTIMAL_THR):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    thr_axis = np.linspace(0.05, 0.95, 91)
    f1s, fps, fns, recalls = [], [], [], []
    for t in thr_axis:
        m = metrics_at_threshold(labels, probs, float(t))
        f1s.append(m["f1"])
        fps.append(m["fp"])
        fns.append(m["fn"])
        recalls.append(m["recall"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax in axes:
        ax.axvline(highlight_thr, color="#e74c3c", ls="--", lw=1.5,
                   label=f"Day5 운용지점 = {highlight_thr:.4f}")
        ax.axvline(0.5, color="#7f8c8d", ls=":", lw=1.5, label="기본 0.5")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Threshold")

    axes[0].plot(thr_axis, f1s, "-o", ms=2, color="#2ecc71")
    axes[0].set_ylabel("F1 (%)"); axes[0].set_title("F1 vs Threshold")
    axes[0].legend()

    axes[1].plot(thr_axis, fps, "-o", ms=2, color="#e67e22", label="FP")
    axes[1].plot(thr_axis, fns, "-s", ms=2, color="#3498db", label="FN")
    axes[1].set_ylabel("Count")
    axes[1].set_title("FP / FN vs Threshold")
    axes[1].legend()

    axes[2].plot(thr_axis, recalls, "-o", ms=2, color="#9b59b6")
    axes[2].set_ylabel("Recall (%)")
    axes[2].set_title("Recall(Sensitivity) vs Threshold")
    axes[2].legend()

    plt.suptitle("MM-MTL Test — Threshold Sweep",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_day5_metrics():
    """Day5 mt_evaluation_results.json 에서 비교용 행을 가져온다."""
    p = MTL_LOG_DIR / "mt_evaluation_results.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    cm = d.get("confusion_matrix", {})
    return {
        "model": "Day5 MTL (FLAIR-only)",
        "threshold": d.get("optimal_threshold"),
        "tp": cm.get("TP"), "tn": cm.get("TN"),
        "fp": cm.get("FP"), "fn": cm.get("FN"),
        "accuracy":  d.get("accuracy"),
        "precision": d.get("precision"),
        "recall":    d.get("recall_sensitivity"),
        "specificity": d.get("specificity"),
        "f1":        d.get("f1_score"),
        "auc_roc":   d.get("auc_roc"),
    }


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 25b: MM-MTL Threshold 동등 보정 (Day 5 운용지점 = 0.3847)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    ckpt_path = CKPT_DIR / "mmmt_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint 없음: {ckpt_path}")
        return

    # ── DataLoader (test split, T1ce+FLAIR+|diff|) ─────────────
    _, _, test_l, _ = create_mm_dataloaders(
        batch_size=16, num_workers=0,
        use_weighted_sampler=False, channel_mode="t1ce_flair_diff",
    )
    print(f"  test samples: {len(test_l.dataset):,}")

    # ── Load model ────────────────────────────────────────────
    print(f"  loading {ckpt_path.name}")
    model = create_mm_model(pretrained=False, in_ch=3, seg_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 예측 수집 (한 번만) ───────────────────────────────────
    print("  collecting predictions on test ...")
    pack = collect_predictions(model, test_l, device)
    labels = pack["labels"].astype(int)
    probs = pack["probs"].astype(float)
    seg_gt = pack["seg_gt"]
    seg_pred = pack["seg_pred"]

    # ── threshold 무관 지표 (AUROC/AUPRC) ─────────────────────
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auroc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(labels, probs))
    # ROC-Youden optimal (Day6 자체 기준)
    j_idx = int(np.argmax(tpr - fpr))
    day6_optimal_thr = float(thresholds[j_idx])

    print(f"\n  [Threshold-invariant]")
    print(f"    AUROC = {auroc:.4f}")
    print(f"    AUPRC = {auprc:.4f}")
    print(f"    Day6 자체 ROC-Youden optimal = {day6_optimal_thr:.4f}")

    # ── threshold sweep ──────────────────────────────────────
    sweep = [metrics_at_threshold(labels, probs, t) for t in THR_GRID]

    # ── seg 메트릭 (threshold 무관, seg_thr=0.5 고정) ─────────
    seg_metrics = compute_seg_metrics(seg_gt, seg_pred, threshold=0.5)

    # ── 출력 표 ──────────────────────────────────────────────
    print(f"\n  [Threshold sweep — Day6 MM-MTL]")
    print(f"  {'thr':>6} | {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} | "
          f"{'P%':>6} {'R%':>6} {'F1%':>6} {'Acc%':>6}")
    print("  " + "-" * 64)
    for m in sweep:
        print(f"  {m['threshold']:>6.4f} | "
              f"{m['tp']:>5} {m['tn']:>5} {m['fp']:>5} {m['fn']:>5} | "
              f"{m['precision']:>6.2f} {m['recall']:>6.2f} "
              f"{m['f1']:>6.2f} {m['accuracy']:>6.2f}")

    # ── Day5 비교 ───────────────────────────────────────────
    day5 = load_day5_metrics()
    if day5 is not None:
        print(f"\n  [Day5 ↔ Day6 동등 운용지점 비교 (thr={DAY5_OPTIMAL_THR})]")
        print(f"    Day5 (FLAIR-only)   : "
              f"F1={day5['f1']:>6.2f}  FP={day5['fp']:>5}  FN={day5['fn']:>5}  "
              f"R={day5['recall']:>6.2f}  P={day5['precision']:>6.2f}")
        # Day6 의 동일 threshold 행
        d6_fair = next(m for m in sweep if abs(m["threshold"] - DAY5_OPTIMAL_THR) < 1e-4)
        print(f"    Day6 (T1ce+FLAIR+d) : "
              f"F1={d6_fair['f1']:>6.2f}  FP={d6_fair['fp']:>5}  FN={d6_fair['fn']:>5}  "
              f"R={d6_fair['recall']:>6.2f}  P={d6_fair['precision']:>6.2f}")
        d6_default = next(m for m in sweep if abs(m["threshold"] - 0.5) < 1e-4)
        print(f"    Day6 @ thr=0.5      : "
              f"F1={d6_default['f1']:>6.2f}  FP={d6_default['fp']:>5}  FN={d6_default['fn']:>5}  "
              f"R={d6_default['recall']:>6.2f}  P={d6_default['precision']:>6.2f}")
        delta_fp = (d6_fair["fp"] or 0) - (day5["fp"] or 0)
        delta_fn = (d6_fair["fn"] or 0) - (day5["fn"] or 0)
        print(f"    Δ(Day6@fair − Day5)  : ΔFP = {delta_fp:+d}  ΔFN = {delta_fn:+d}")
    else:
        print(f"\n  [WARN] Day5 결과(mt_evaluation_results.json) 없음 — 비교 생략")

    # ── 그림 저장 ────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_threshold_sweep(labels, probs,
                         FIG_DIR / "threshold_sweep.png",
                         highlight_thr=DAY5_OPTIMAL_THR)
    print(f"\n  threshold sweep figure → {FIG_DIR / 'threshold_sweep.png'}")

    # ── JSON 저장 ───────────────────────────────────────────
    result = {
        "n_test": int(len(labels)),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "day5_optimal_thr": DAY5_OPTIMAL_THR,
        "day6_optimal_thr": round(day6_optimal_thr, 4),
        "threshold_sweep": sweep,
        "seg_wt": seg_metrics,
        "day5_reference": day5,
    }
    out = LOG_DIR / "mmmt_test_metrics_thr_sweep.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  metrics            → {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
