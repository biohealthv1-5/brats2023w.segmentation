# -*- coding: utf-8 -*-
"""
Step 26 (Day 6 / mmmt — Phase 1 (B)): Day5 ↔ Day6 FN 회복 차분 분석
===================================================================
260523v1updatemmmt.md §④ Phase 1 (B) 구현.

목적
----
Day 5 MTL(FLAIR-only) 과 Day 6 MM-MTL(T1ce + FLAIR + |diff|) 의
**테스트 슬라이스별 예측을 1:1로 비교**해

    1) recovered    = Day5 FN ∩ Day6 TP  (멀티모달이 새로 잡아낸 슬라이스)
    2) still_missed = Day5 FN ∩ Day6 FN  (멀티모달도 못 잡은 슬라이스)
    3) regressed    = Day5 TP ∩ Day6 FN  (Day5 가 잡았는데 Day6 가 놓친 슬라이스)
    4) common_tp    = Day5 TP ∩ Day6 TP
    5) new_fp       = Day5 TN ∩ Day6 FP  (멀티모달이 새로 만든 위양성)
    6) common_tn / common_fp ...

각 그룹에 대해 *슬라이스 z-위치*, *종양 픽셀 수*, *Day5/Day6 확률*
분포를 비교한다. step25b 의 threshold-동등 보정(0.3847)을 함께 적용해
"공정 비교"와 "기본 운용지점(0.5)" 두 시나리오 모두에서 차분한다.

산출물
------
- outputs/logs/mmmt/fn_diff_per_slice.csv
    filename, patient_id, slice_idx, label, tumor_pixels,
    day5_prob, day6_prob,
    group_fair  (threshold=0.3847 양쪽 동일)
    group_split (Day5=0.3847, Day6=0.5  ← 보고서의 원본 비교)
- outputs/logs/mmmt/fn_diff_summary.json
    그룹별 카운트 + 메트릭 차분 표
- outputs/figures/mmmt/fn_diff_zdist.png
- outputs/figures/mmmt/fn_diff_tumor_size.png
- outputs/figures/mmmt/fn_diff_prob_scatter.png

사용법
------
    python code/mmmt/step26_fn_diff_analysis.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 동일 폴더 import (이미 다른 step 들이 쓰는 패턴)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multitask"))

# Day 6 (mmmt) 자원
from step21_mmmt_dataset import create_mm_dataloaders          # noqa: E402
from step22_mmmt_model import create_mm_model                  # noqa: E402

# Day 5 (multitask) 자원
from step16_multitask_dataset import create_multitask_dataloaders  # noqa: E402
from step17_multitask_model import create_multitask_model         # noqa: E402


# ─── 경로 ─────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MT_CKPT = PROJECT_DIR / "outputs" / "checkpoints" / "multitask" / "mt_best_model.pth"
MM_CKPT = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt" / "mmmt_best.pth"
SPLITS_CSV = PROJECT_DIR / "processed" / "splits.csv"
SEG_DIR = PROJECT_DIR / "processed" / "seg_masks"

LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt"
FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "mmmt"

# Day5 ROC-Youden optimal threshold (mt_evaluation_results.json 기록치)
DAY5_THR = 0.3847
DAY6_DEFAULT_THR = 0.5


# ─── 예측 수집 ────────────────────────────────────────────────
@torch.no_grad()
def _collect_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """loader 순서대로 (probs, labels) 수집. shuffle=False 가정."""
    model.eval()
    probs_list, labels_list = [], []
    for batch in loader:
        # 두 dataset 모두 (img, label, mask) 반환
        imgs, labels, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        cls_out, _ = model(imgs)
        p = torch.sigmoid(cls_out.squeeze(1)).cpu().numpy()
        probs_list.append(p)
        labels_list.append(labels.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


# ─── 그룹 분류 ────────────────────────────────────────────────
def _classify_groups(labels: np.ndarray,
                     day5_pred: np.ndarray,
                     day6_pred: np.ndarray) -> np.ndarray:
    """
    label / day5_pred / day6_pred (binary 0/1) → group str array.

    그룹 정의:
      label=1 (양성 슬라이스):
        - recovered    : day5=0, day6=1   (Day5 FN, Day6 TP)
        - still_missed : day5=0, day6=0   (Day5 FN, Day6 FN)
        - regressed    : day5=1, day6=0   (Day5 TP, Day6 FN)
        - common_tp    : day5=1, day6=1
      label=0 (음성 슬라이스):
        - common_tn    : day5=0, day6=0
        - new_fp       : day5=0, day6=1   (Day6 가 새로 만든 위양성)
        - cleaned_fp   : day5=1, day6=0   (Day5 위양성을 Day6 가 제거)
        - common_fp    : day5=1, day6=1
    """
    g = np.empty(labels.shape, dtype=object)
    pos = labels == 1
    neg = ~pos
    # positives
    g[pos & (day5_pred == 0) & (day6_pred == 1)] = "recovered"
    g[pos & (day5_pred == 0) & (day6_pred == 0)] = "still_missed"
    g[pos & (day5_pred == 1) & (day6_pred == 0)] = "regressed"
    g[pos & (day5_pred == 1) & (day6_pred == 1)] = "common_tp"
    # negatives
    g[neg & (day5_pred == 0) & (day6_pred == 0)] = "common_tn"
    g[neg & (day5_pred == 0) & (day6_pred == 1)] = "new_fp"
    g[neg & (day5_pred == 1) & (day6_pred == 0)] = "cleaned_fp"
    g[neg & (day5_pred == 1) & (day6_pred == 1)] = "common_fp"
    return g


# ─── 종양 픽셀 수 ──────────────────────────────────────────────
def _tumor_pixels(filenames: List[str]) -> np.ndarray:
    """seg_masks/<filename> 에서 (mask>127) 픽셀 수. 없는 파일은 0."""
    import cv2
    out = np.zeros(len(filenames), dtype=np.int32)
    for i, fn in enumerate(filenames):
        p = SEG_DIR / fn
        if not p.exists():
            continue
        arr = np.fromfile(str(p), dtype=np.uint8)
        m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[i] = int((m > 127).sum())
    return out


# ─── 메트릭 ────────────────────────────────────────────────────
def _binary_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict:
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn,
        precision=round(float(precision * 100), 2),
        recall=round(float(recall * 100), 2),
        f1=round(float(f1 * 100), 2),
        accuracy=round(float((preds == labels).mean() * 100), 2),
    )


# ─── 시각화 ────────────────────────────────────────────────────
def _plot_zdist(df: pd.DataFrame, save: Path,
                group_col: str = "group_fair"):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    focus = ["recovered", "still_missed", "regressed", "common_tp"]
    colors = {"recovered": "#2ecc71", "still_missed": "#e74c3c",
              "regressed": "#9b59b6", "common_tp": "#3498db"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) z 분포 히스토그램
    bins = np.arange(0, df["slice_idx"].max() + 5, 4)
    for grp in focus:
        sub = df[df[group_col] == grp]
        if len(sub) == 0:
            continue
        axes[0].hist(sub["slice_idx"], bins=bins, alpha=0.45,
                     color=colors[grp], label=f"{grp} (n={len(sub)})")
    axes[0].set_xlabel("Brain z-index (axial)")
    axes[0].set_ylabel("# slices")
    axes[0].set_title(f"슬라이스 z-위치 분포 ({group_col})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (2) z-bin 별 recovered 비율 (양성 슬라이스만)
    pos_mask = df["label"] == 1
    bin_edges = np.arange(0, df["slice_idx"].max() + 9, 8)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    rec_ratio = []
    counts = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        sub = df[pos_mask & (df["slice_idx"] >= lo) & (df["slice_idx"] < hi)]
        n = len(sub)
        rec = int((sub[group_col] == "recovered").sum())
        rec_ratio.append(rec / max(n, 1))
        counts.append(n)
    axes[1].bar(centers, rec_ratio, width=7.0, color="#2ecc71", alpha=0.8,
                edgecolor="white")
    for c, r, n in zip(centers, rec_ratio, counts):
        if n > 0:
            axes[1].text(c, r + 0.005, f"{n}", ha="center", fontsize=7,
                         color="#555")
    axes[1].set_xlabel("Brain z-index (axial)")
    axes[1].set_ylabel("recovered ratio (Day5 FN → Day6 TP)")
    axes[1].set_title(f"z-bin 별 회복률 ({group_col})")
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle(f"Day5↔Day6 FN 차분 — z 분포 [{group_col}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_tumor_size(df: pd.DataFrame, save: Path,
                     group_col: str = "group_fair"):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    focus = ["recovered", "still_missed", "regressed", "common_tp"]
    colors = {"recovered": "#2ecc71", "still_missed": "#e74c3c",
              "regressed": "#9b59b6", "common_tp": "#3498db"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # log-scaled hist (양성 슬라이스 중에서만 의미 있음 — tumor_pixels>0)
    for grp in focus:
        sub = df[(df[group_col] == grp) & (df["tumor_pixels"] > 0)]
        if len(sub) == 0:
            continue
        axes[0].hist(np.log10(sub["tumor_pixels"].values + 1),
                     bins=40, alpha=0.45, color=colors[grp],
                     label=f"{grp} (n={len(sub)}, med={int(sub['tumor_pixels'].median())})")
    axes[0].set_xlabel("log10(tumor pixels + 1)")
    axes[0].set_ylabel("# slices")
    axes[0].set_title(f"종양 크기 분포 ({group_col})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # boxplot
    data, names, cs = [], [], []
    for grp in focus:
        sub = df[(df[group_col] == grp) & (df["tumor_pixels"] > 0)]
        if len(sub) > 0:
            data.append(sub["tumor_pixels"].values)
            names.append(f"{grp}\nn={len(sub)}")
            cs.append(colors[grp])
    if data:
        bp = axes[1].boxplot(data, tick_labels=names, patch_artist=True,
                             showfliers=False)
        for patch, c in zip(bp["boxes"], cs):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("tumor pixels (log)")
        axes[1].set_title("그룹별 종양 크기 박스플롯")
        axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle(f"Day5↔Day6 FN 차분 — 종양 크기 [{group_col}]",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_prob_scatter(df: pd.DataFrame, save: Path):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 양성 슬라이스
    axes[0].scatter(pos["day5_prob"], pos["day6_prob"], s=3, alpha=0.25,
                    color="#e74c3c")
    axes[0].axvline(DAY5_THR, color="#34495e", ls="--", lw=1,
                    label=f"Day5 thr={DAY5_THR}")
    axes[0].axhline(DAY5_THR, color="#2ecc71", ls="--", lw=1,
                    label=f"Day6 thr(fair)={DAY5_THR}")
    axes[0].axhline(DAY6_DEFAULT_THR, color="#7f8c8d", ls=":", lw=1,
                    label=f"Day6 thr(default)={DAY6_DEFAULT_THR}")
    axes[0].plot([0, 1], [0, 1], color="#bdc3c7", lw=0.8)
    axes[0].set_xlabel("Day5 prob (FLAIR-only)")
    axes[0].set_ylabel("Day6 prob (T1ce+FLAIR+|diff|)")
    axes[0].set_title(f"양성 슬라이스 (n={len(pos):,})")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(-0.02, 1.02); axes[0].set_ylim(-0.02, 1.02)

    # 음성 슬라이스
    axes[1].scatter(neg["day5_prob"], neg["day6_prob"], s=3, alpha=0.25,
                    color="#3498db")
    axes[1].axvline(DAY5_THR, color="#34495e", ls="--", lw=1)
    axes[1].axhline(DAY5_THR, color="#2ecc71", ls="--", lw=1)
    axes[1].axhline(DAY6_DEFAULT_THR, color="#7f8c8d", ls=":", lw=1)
    axes[1].plot([0, 1], [0, 1], color="#bdc3c7", lw=0.8)
    axes[1].set_xlabel("Day5 prob (FLAIR-only)")
    axes[1].set_ylabel("Day6 prob (T1ce+FLAIR+|diff|)")
    axes[1].set_title(f"음성 슬라이스 (n={len(neg):,})")
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(-0.02, 1.02); axes[1].set_ylim(-0.02, 1.02)

    plt.suptitle("Day5↔Day6 확률 산점도 (test split)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()


# ─── 메인 ──────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Step 26 (Phase 1 B): Day5 ↔ Day6 FN 회복 차분 분석")
    print("=" * 70)

    if not MT_CKPT.exists():
        print(f"[ERROR] Day5 ckpt 없음: {MT_CKPT}"); return
    if not MM_CKPT.exists():
        print(f"[ERROR] Day6 ckpt 없음: {MM_CKPT}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ── 테스트 split 메타 (순서 고정) ─────────────────────────
    splits_df = pd.read_csv(SPLITS_CSV)
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    n_test = len(test_df)
    print(f"  test samples: {n_test:,}")

    # ── Day5 (multitask, FLAIR-only) 예측 ─────────────────────
    print("\n  [Day5] loading mt_best_model.pth + FLAIR-only dataloader ...")
    _, _, mt_test_l, _ = create_multitask_dataloaders(
        batch_size=32, num_workers=0,
    )
    mt_model = create_multitask_model(pretrained=False).to(device)
    mt_ckpt = torch.load(MT_CKPT, map_location=device, weights_only=True)
    mt_model.load_state_dict(mt_ckpt["model_state_dict"])
    day5_probs, day5_labels = _collect_probs(mt_model, mt_test_l, device)
    print(f"    Day5 probs collected: {day5_probs.shape}, "
          f"label balance pos={int(day5_labels.sum()):,}/{len(day5_labels):,}")
    del mt_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Day6 (mmmt, T1ce+FLAIR+|diff|) 예측 ──────────────────
    print("\n  [Day6] loading mmmt_best.pth + T1ce+FLAIR+|diff| dataloader ...")
    _, _, mm_test_l, _ = create_mm_dataloaders(
        batch_size=16, num_workers=0,
        use_weighted_sampler=False, channel_mode="t1ce_flair_diff",
    )
    mm_model = create_mm_model(pretrained=False, in_ch=3, seg_classes=1).to(device)
    mm_ckpt = torch.load(MM_CKPT, map_location=device)
    mm_model.load_state_dict(mm_ckpt["model_state_dict"])
    day6_probs, day6_labels = _collect_probs(mm_model, mm_test_l, device)
    print(f"    Day6 probs collected: {day6_probs.shape}, "
          f"label balance pos={int(day6_labels.sum()):,}/{len(day6_labels):,}")
    del mm_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── 정합성 체크 ───────────────────────────────────────────
    assert len(day5_labels) == len(day6_labels) == n_test, (
        f"길이 불일치: Day5={len(day5_labels)} Day6={len(day6_labels)} "
        f"splits={n_test}"
    )
    if not np.array_equal(day5_labels, day6_labels):
        n_diff = int((day5_labels != day6_labels).sum())
        raise AssertionError(
            f"라벨 순서 불일치({n_diff}개): 두 dataset 의 test_df 정렬이 "
            f"다릅니다. 즉시 중단."
        )
    if not np.array_equal(day5_labels.astype(int),
                          test_df["label"].values.astype(int)):
        raise AssertionError(
            "splits.csv test 라벨과 dataloader 라벨이 다릅니다. "
            "shuffle 또는 sampler 가 활성화되었는지 확인."
        )
    labels = day5_labels.astype(int)

    # ── 종양 픽셀 수 ──────────────────────────────────────────
    print("\n  computing tumor pixel counts from seg_masks ...")
    tumor_pix = _tumor_pixels(test_df["filename"].tolist())
    print(f"    median(positive) = "
          f"{int(np.median(tumor_pix[tumor_pix > 0])) if (tumor_pix > 0).any() else 0:,} px")

    # ── 그룹 분류 (두 시나리오) ───────────────────────────────
    # (1) fair  : Day5=0.3847, Day6=0.3847  (동등 운용지점)
    # (2) split : Day5=0.3847, Day6=0.5      (원본 보고치 비교)
    day5_pred_fair = (day5_probs >= DAY5_THR).astype(int)
    day6_pred_fair = (day6_probs >= DAY5_THR).astype(int)
    day6_pred_def = (day6_probs >= DAY6_DEFAULT_THR).astype(int)

    g_fair = _classify_groups(labels, day5_pred_fair, day6_pred_fair)
    g_split = _classify_groups(labels, day5_pred_fair, day6_pred_def)

    # ── CSV 저장 ──────────────────────────────────────────────
    out_df = pd.DataFrame({
        "filename":     test_df["filename"].values,
        "patient_id":   test_df["patient_id"].values,
        "slice_idx":    test_df["slice_idx"].values.astype(int),
        "label":        labels,
        "tumor_pixels": tumor_pix,
        "day5_prob":    np.round(day5_probs, 6),
        "day6_prob":    np.round(day6_probs, 6),
        "day5_pred":    day5_pred_fair,
        "day6_pred_fair":    day6_pred_fair,
        "day6_pred_default": day6_pred_def,
        "group_fair":   g_fair,
        "group_split":  g_split,
    })
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / "fn_diff_per_slice.csv"
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  per-slice CSV → {csv_path}")

    # ── 그룹 카운트 / 메트릭 차분 ─────────────────────────────
    def _grp_counts(arr):
        u, c = np.unique(arr, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}

    summary = {
        "n_test": int(n_test),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "thresholds": {
            "day5":            DAY5_THR,
            "day6_fair":       DAY5_THR,
            "day6_default":    DAY6_DEFAULT_THR,
        },
        "metrics": {
            "day5":            _binary_metrics(labels, day5_pred_fair),
            "day6_fair@0.3847":  _binary_metrics(labels, day6_pred_fair),
            "day6_default@0.5":  _binary_metrics(labels, day6_pred_def),
        },
        "group_counts_fair":  _grp_counts(g_fair),
        "group_counts_split": _grp_counts(g_split),
    }

    # 그룹별 종양 픽셀 / z 통계 (양성 슬라이스 중심)
    def _grp_stats(group_col):
        out = {}
        for grp, sub in out_df.groupby(group_col):
            pos_sub = sub[sub["label"] == 1]
            out[grp] = {
                "n": int(len(sub)),
                "n_positive": int(len(pos_sub)),
                "tumor_px_median": int(pos_sub["tumor_pixels"].median())
                                   if len(pos_sub) else 0,
                "tumor_px_mean":   round(float(pos_sub["tumor_pixels"].mean()), 1)
                                   if len(pos_sub) else 0.0,
                "z_median":        int(sub["slice_idx"].median()) if len(sub) else 0,
                "day5_prob_mean":  round(float(sub["day5_prob"].mean()), 4) if len(sub) else 0.0,
                "day6_prob_mean":  round(float(sub["day6_prob"].mean()), 4) if len(sub) else 0.0,
            }
        return out

    summary["group_stats_fair"]  = _grp_stats("group_fair")
    summary["group_stats_split"] = _grp_stats("group_split")

    json_path = LOG_DIR / "fn_diff_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  summary JSON  → {json_path}")

    # ── 콘솔 표 ───────────────────────────────────────────────
    print("\n  [그룹 카운트 — fair (양쪽 thr=0.3847)]")
    for grp in ("recovered", "still_missed", "regressed", "common_tp",
                "new_fp", "cleaned_fp", "common_fp", "common_tn"):
        n = summary["group_counts_fair"].get(grp, 0)
        print(f"    {grp:>14s}  : {n:>7,}")

    print("\n  [그룹 카운트 — split (Day5=0.3847, Day6=0.5)]")
    for grp in ("recovered", "still_missed", "regressed", "common_tp",
                "new_fp", "cleaned_fp", "common_fp", "common_tn"):
        n = summary["group_counts_split"].get(grp, 0)
        print(f"    {grp:>14s}  : {n:>7,}")

    m = summary["metrics"]
    print("\n  [Metric 차분]")
    print(f"    {'scheme':<22s} {'F1':>6} {'Recall':>7} {'Prec':>7} "
          f"{'FP':>6} {'FN':>6}")
    for tag in ("day5", "day6_fair@0.3847", "day6_default@0.5"):
        x = m[tag]
        print(f"    {tag:<22s} {x['f1']:>6.2f} {x['recall']:>7.2f} "
              f"{x['precision']:>7.2f} {x['fp']:>6d} {x['fn']:>6d}")

    # ── 그림 저장 ─────────────────────────────────────────────
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_zdist(out_df, FIG_DIR / "fn_diff_zdist.png", group_col="group_fair")
    _plot_tumor_size(out_df, FIG_DIR / "fn_diff_tumor_size.png",
                     group_col="group_fair")
    _plot_prob_scatter(out_df, FIG_DIR / "fn_diff_prob_scatter.png")
    print(f"\n  figures       → {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
