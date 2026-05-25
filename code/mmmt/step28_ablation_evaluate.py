# -*- coding: utf-8 -*-
"""
Step 28 (Day 6 / mmmt — Phase 2 (C) 평가): 채널 Ablation 통합 표
================================================================
260523v1updatemmmt.md §④ Phase 2 (C) 평가 단계.

step27 로 학습된 두 ablation 체크포인트(`t1ce_only`, `flair_only`)와
기존 step24 의 멀티모달(`t1ce_flair_diff`) 체크포인트를 동일 test split
에서 평가하고, *threshold 동등 보정*(step25b 와 같은 0.3847)도 함께
적용한다.

산출물
------
- outputs/logs/mmmt_ablation/ablation_table.json
    3-row × (TP/TN/FP/FN, F1, Precision, Recall, AUROC, AUPRC, WT Dice)
    × {thr=0.3847, thr=0.5}
- outputs/figures/mmmt_ablation/ablation_table.png
    비교 막대 차트 (F1, Recall, FP, WT Dice 각 1패널)

사용법
------
    python code/mmmt/step28_ablation_evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, average_precision_score, f1_score,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step21_mmmt_dataset import create_mm_dataloaders                    # noqa: E402
from step22_mmmt_model import create_mm_model                            # noqa: E402
# NOTE: step25.collect_predictions 는 전체 test split (≈25k slices)의
#  seg_gt/seg_pred 를 float32 (~4.7GiB×2) 로 메모리에 들고 있다가
#  np.concatenate 로 합쳐 OOM 을 유발한다. step28 에서는 seg 지표만
#  필요하므로 batch-incremental 로 계산하는 _collect_predictions_lowmem
#  을 별도로 정의해 사용한다. (compute_seg_metrics 는 그대로 import)
from step25_mmmt_evaluate import compute_seg_metrics  # noqa: E402  (현재는 사용 안 함, 호환성 보존)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MMMT_CKPT = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt" / "mmmt_best.pth"
ABL_CKPT_ROOT = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt_ablation"
ABL_LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt_ablation"
ABL_FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "mmmt_ablation"

DAY5_THR = 0.3847
DAY6_DEFAULT_THR = 0.5

# (display name, channel_mode, ckpt path)
RUNS = [
    ("Day6 baseline [T1ce,FLAIR,|diff|]", "t1ce_flair_diff", MMMT_CKPT),
    ("Ablation C-1: T1ce-only",          "t1ce_only",        ABL_CKPT_ROOT / "t1ce_only"  / "best.pth"),
    ("Ablation C-2: FLAIR-only",         "flair_only",       ABL_CKPT_ROOT / "flair_only" / "best.pth"),
]


@torch.no_grad()
def _collect_predictions_lowmem(model, loader, device, seg_threshold: float = 0.5):
    """메모리-절약형 예측 수집.

    step25 의 collect_predictions 는 (N≈25115, 1, 224, 224) float32 텐서
    두 개를 한꺼번에 만들면서(≈ 4.69 GiB ×2, concatenate 순간 추가 사본까지
    포함하면 10~15 GiB 가 필요해) MemoryError 가 발생한다.

    여기서는
      - 분류용 labels / probs 는 배치마다 numpy 로만 떼서 누적
      - 분할(WT seg) 지표는 배치마다 즉시 Dice / IoU 를 계산해 스칼라만 저장
    함으로써 메모리 사용량을 GB 단위에서 MB 단위로 떨어뜨린다.

    반환값은 step25.collect_predictions 와 다르며, _evaluate_one 가
    필요로 하는 정보만 담는다.
    """
    model.eval()
    all_labels, all_probs = [], []
    dices, ious = [], []

    for imgs, labels, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        cls_logit, seg_out = model(imgs)
        cls_logit = cls_logit.squeeze(1)
        cls_probs = torch.sigmoid(cls_logit).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(cls_probs)

        # ── 배치별 seg 지표 즉시 계산 (전체 텐서를 누적하지 않음) ──
        # bool 로 즉시 이진화해 메모리 풋프린트를 더 줄인다.
        seg_pred_bin = (torch.sigmoid(seg_out) > seg_threshold).cpu().numpy()
        seg_gt_bin = (masks.numpy() > 0.5)
        # (B, 1, H, W) → 샘플별
        for i in range(seg_gt_bin.shape[0]):
            g = seg_gt_bin[i, 0]
            if not g.any():
                continue  # 음성 슬라이스는 step25 와 동일하게 스킵
            p = seg_pred_bin[i, 0]
            g_sum = float(g.sum())
            p_sum = float(p.sum())
            inter = float(np.logical_and(g, p).sum())
            dices.append(2.0 * inter / max(g_sum + p_sum, 1e-8))
            ious.append(inter / max(g_sum + p_sum - inter, 1e-8))

        # GPU 메모리도 매 배치 정리
        del imgs, cls_logit, seg_out, cls_probs, seg_pred_bin
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seg = {
        "n_positive_slices": len(dices),
        "dice_mean":   float(np.mean(dices))   if dices else 0.0,
        "dice_median": float(np.median(dices)) if dices else 0.0,
        "iou_mean":    float(np.mean(ious))    if ious  else 0.0,
        "iou_median":  float(np.median(ious))  if ious  else 0.0,
    }
    return {
        "labels": np.concatenate(all_labels),
        "probs":  np.concatenate(all_probs),
        "seg":    seg,
    }


def _metrics_at_thr(labels, probs, thr):
    preds = (probs >= thr).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (preds == labels).mean()
    return dict(
        threshold=round(float(thr), 4),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        precision=round(float(precision * 100), 2),
        recall=round(float(recall * 100), 2),
        f1=round(float(f1 * 100), 2),
        accuracy=round(float(accuracy * 100), 2),
    )


def _evaluate_one(name, mode, ckpt_path, device):
    print(f"\n  ── {name}  (mode={mode})")
    if not Path(ckpt_path).exists():
        print(f"    [WARN] ckpt 없음: {ckpt_path} — 건너뜀")
        return None

    _, _, test_l, _ = create_mm_dataloaders(
        batch_size=16, num_workers=0,
        use_weighted_sampler=False, channel_mode=mode,
    )
    model = create_mm_model(pretrained=False, in_ch=3, seg_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 메모리-절약 예측 수집 (seg 지표는 배치마다 즉시 계산) ──
    pack = _collect_predictions_lowmem(model, test_l, device, seg_threshold=0.5)
    labels = pack["labels"].astype(int)
    probs = pack["probs"].astype(float)
    seg = pack["seg"]

    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(labels, probs))

    row = {
        "name": name,
        "mode": mode,
        "ckpt": str(ckpt_path),
        "n_test": int(len(labels)),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "@0.5":    _metrics_at_thr(labels, probs, DAY6_DEFAULT_THR),
        "@0.3847": _metrics_at_thr(labels, probs, DAY5_THR),
        "seg_wt": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in seg.items()},
    }

    print(f"    AUROC={row['auroc']:.4f}  AUPRC={row['auprc']:.4f}  "
          f"WT Dice={seg['dice_mean']:.4f}")
    print(f"    @0.5    : F1={row['@0.5']['f1']:.2f}  "
          f"R={row['@0.5']['recall']:.2f}  "
          f"FP={row['@0.5']['fp']}  FN={row['@0.5']['fn']}")
    print(f"    @0.3847 : F1={row['@0.3847']['f1']:.2f}  "
          f"R={row['@0.3847']['recall']:.2f}  "
          f"FP={row['@0.3847']['fp']}  FN={row['@0.3847']['fn']}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _plot_table(rows, save: Path):
    """막대 차트 4패널 — F1@fair, Recall@fair, FP@fair, WT Dice."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    valid = [r for r in rows if r is not None]
    if not valid:
        print("  [WARN] 시각화 대상 없음")
        return
    names = [r["name"] for r in valid]
    f1s = [r["@0.3847"]["f1"] for r in valid]
    recs = [r["@0.3847"]["recall"] for r in valid]
    fps = [r["@0.3847"]["fp"] for r in valid]
    dices = [r["seg_wt"]["dice_mean"] * 100 for r in valid]

    colors = ["#2ecc71", "#e67e22", "#3498db"][:len(valid)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        ("F1 @0.3847 (%)", f1s),
        ("Recall @0.3847 (%)", recs),
        ("FP @0.3847", fps),
        ("WT Dice ×100", dices),
    ]
    for ax, (title, vals) in zip(axes, panels):
        bars = ax.bar(names, vals, color=colors, edgecolor="white", lw=1.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{v:.1f}" if isinstance(v, float) else f"{v}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        # x-tick 회전
        ax.tick_params(axis="x", labelrotation=12)

    plt.suptitle("채널 Ablation 비교 (test, threshold=0.3847 공정 비교 기준)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 70)
    print("  Step 28 (Phase 2 C 평가): 채널 Ablation 통합 표")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    ABL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ABL_FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, mode, ckpt in RUNS:
        rows.append(_evaluate_one(name, mode, ckpt, device))

    # ── 콘솔 표 ───────────────────────────────────────────────
    print(f"\n  [Ablation Table — test split]")
    header = (f"  {'model':<40s} {'thr':>7} {'F1':>6} {'P%':>6} {'R%':>6} "
              f"{'FP':>6} {'FN':>6} {'AUROC':>6} {'Dice':>6}")
    print(header); print("  " + "─" * (len(header) - 2))
    for r in rows:
        if r is None:
            continue
        for tag in ("@0.5", "@0.3847"):
            m = r[tag]
            print(f"  {r['name']:<40s} {m['threshold']:>7.4f} "
                  f"{m['f1']:>6.2f} {m['precision']:>6.2f} "
                  f"{m['recall']:>6.2f} {m['fp']:>6d} {m['fn']:>6d} "
                  f"{r['auroc']:>6.4f} {r['seg_wt']['dice_mean']:>6.4f}")

    # ── 저장 ──────────────────────────────────────────────────
    out_json = ABL_LOG_DIR / "ablation_table.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "day5_optimal_thr": DAY5_THR,
            "day6_default_thr": DAY6_DEFAULT_THR,
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  table JSON  → {out_json}")

    _plot_table(rows, ABL_FIG_DIR / "ablation_table.png")
    print(f"  figure      → {ABL_FIG_DIR / 'ablation_table.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
