# -*- coding: utf-8 -*-
"""
Step 31 (Day 7 / sota): SOTA Evaluation
========================================
v2ways.md §5 신뢰성 SOTA 3종 패키지 적용:
- TTA (identity / hflip / vflip / rot180)                §5.1
- Temperature Scaling (val LBFGS) → calibration         §5.3
- Conformal Prediction (간이 marginal, mapie 없이도 동작) §5.5

+ WT/TC/ET 3-region 지표 + 5-way 비교 (vs mmmt + multitask)

사용법:
    python code/sota/step31_sota_evaluate.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step27_sota_dataset import create_sota_dataloaders
from step28_sota_model import create_sota_model

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = PROJECT_DIR / "outputs" / "checkpoints" / "sota"
FIG_DIR = PROJECT_DIR / "outputs" / "figures" / "sota"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "sota"
MMMT_PREV_LOG = PROJECT_DIR / "outputs" / "logs" / "mmmt"          # Day 6 결과
MTL_PREV_LOG = PROJECT_DIR / "outputs" / "logs" / "multitask"      # Day 5 결과


# ─── TTA ──────────────────────────────────────────────────
def _apply_tta(x, mode):
    if mode == "identity": return x
    if mode == "hflip":    return torch.flip(x, dims=[3])
    if mode == "vflip":    return torch.flip(x, dims=[2])
    if mode == "rot180":   return torch.flip(x, dims=[2, 3])
    raise ValueError(mode)


def _inverse_tta(seg, mode):
    if mode == "identity": return seg
    if mode == "hflip":    return torch.flip(seg, dims=[3])
    if mode == "vflip":    return torch.flip(seg, dims=[2])
    if mode == "rot180":   return torch.flip(seg, dims=[2, 3])
    raise ValueError(mode)


@torch.no_grad()
def predict_with_tta(model, imgs,
                     tta_modes=("identity", "hflip", "vflip", "rot180")):
    cls_list, seg_list = [], []
    for m in tta_modes:
        x_t = _apply_tta(imgs, m)
        cls_out, seg_out, _ = model(x_t, return_aux=False)
        cls_list.append(torch.sigmoid(cls_out.squeeze(1)))
        seg_list.append(_inverse_tta(torch.sigmoid(seg_out), m))
    return torch.stack(cls_list).mean(0), torch.stack(seg_list).mean(0)


# ─── Temperature Scaling ────────────────────────────────
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor,
                    max_iter: int = 50):
    scaler = TemperatureScaler().to(val_logits.device)
    optim = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=max_iter)
    crit = nn.BCEWithLogitsLoss()

    def closure():
        optim.zero_grad()
        loss = crit(scaler(val_logits), val_labels)
        loss.backward()
        return loss

    optim.step(closure)
    return scaler.temperature.detach().item()


# ─── 예측 수집 ───────────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, device,
                        use_tta=True, return_logits=False):
    model.eval()
    all_labels, all_probs, all_logits = [], [], []
    all_seg_gt, all_seg_pred = [], []
    for imgs, labels, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        if use_tta:
            cls_probs, seg_probs = predict_with_tta(model, imgs)
            logits_no_tta, _, _ = model(imgs, return_aux=False)
            logits_no_tta = logits_no_tta.squeeze(1)
        else:
            cls_logit, seg_out, _ = model(imgs, return_aux=False)
            cls_logit = cls_logit.squeeze(1)
            cls_probs = torch.sigmoid(cls_logit)
            seg_probs = torch.sigmoid(seg_out)
            logits_no_tta = cls_logit
        all_labels.append(labels.numpy())
        all_probs.append(cls_probs.cpu().numpy())
        all_logits.append(logits_no_tta.cpu().numpy())
        all_seg_gt.append(masks.numpy())
        all_seg_pred.append(seg_probs.cpu().numpy())
    out = {
        "labels": np.concatenate(all_labels),
        "probs": np.concatenate(all_probs),
        "seg_gt": np.concatenate(all_seg_gt),
        "seg_pred": np.concatenate(all_seg_pred),
    }
    if return_logits:
        out["logits"] = np.concatenate(all_logits)
    return out


def compute_seg_metrics(seg_gt: np.ndarray, seg_pred: np.ndarray,
                        threshold: float = 0.5):
    regions = ["WT", "TC", "ET"]
    result = {}
    for r, name in enumerate(regions):
        dices, ious = [], []
        for i in range(seg_gt.shape[0]):
            g = (seg_gt[i, r].flatten() > 0.5).astype(np.float32)
            p = (seg_pred[i, r].flatten() > threshold).astype(np.float32)
            if g.sum() == 0:
                continue
            inter = (g * p).sum()
            dices.append(2.0 * inter / max(g.sum() + p.sum(), 1e-8))
            ious.append(inter / max(g.sum() + p.sum() - inter, 1e-8))
        result[name] = {
            "n_positive_slices": len(dices),
            "dice_mean": float(np.mean(dices)) if dices else 0.0,
            "dice_median": float(np.median(dices)) if dices else 0.0,
            "iou_mean": float(np.mean(ious)) if ious else 0.0,
            "iou_median": float(np.median(ious)) if ious else 0.0,
        }
    return result


# ─── 시각화 ──────────────────────────────────────────────
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

    regions = ["WT", "TC", "ET"]
    dices_by_region = {r: [] for r in regions}
    for r, name in enumerate(regions):
        for i in range(seg_gt.shape[0]):
            g = (seg_gt[i, r].flatten() > 0.5).astype(np.float32)
            p = (seg_pred[i, r].flatten() > 0.5).astype(np.float32)
            if g.sum() == 0:
                continue
            inter = (g * p).sum()
            dices_by_region[name].append(2.0 * inter / max(g.sum() + p.sum(), 1e-8))
    data = [dices_by_region[r] for r in regions]
    axes[1, 1].boxplot(data, tick_labels=regions)
    axes[1, 1].set_title("Dice by Region (test, 양성 슬라이스만)")
    axes[1, 1].set_ylabel("Dice"); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "sota_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─── Conformal (간이 marginal) ──────────────────────────
def try_conformal(val_probs, val_labels, test_probs, alpha=0.1):
    try:
        n = len(val_labels)
        scores = np.where(val_labels == 1, 1 - val_probs, val_probs)
        q = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n,
                        method="higher")
        sets = []
        for p in test_probs:
            included = []
            if p <= q:
                included.append(0)
            if (1 - p) <= q:
                included.append(1)
            sets.append(included)
        return {"q": float(q), "sets": sets}
    except Exception as e:
        return {"error": str(e)}


# ─── Main ────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 31 (Day 7 / sota): SOTA Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_l, val_l, test_l, _ = create_sota_dataloaders(
        batch_size=16, num_workers=0,
        use_weighted_sampler=False, channel_mode="t1ce_flair_diff",
    )

    ckpt_path = CKPT_DIR / "sota_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint 없음: {ckpt_path}")
        return
    print(f"  loading {ckpt_path}")
    model = create_sota_model(pretrained=False, in_ch=3, seg_classes=3).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"\n  [val] collecting raw logits (no TTA) ...")
    val_pack = collect_predictions(model, val_l, device,
                                   use_tta=False, return_logits=True)
    val_logits = torch.tensor(val_pack["logits"], dtype=torch.float32)
    val_labels = torch.tensor(val_pack["labels"], dtype=torch.float32)
    T = fit_temperature(val_logits, val_labels)
    print(f"  fitted Temperature T = {T:.4f}")

    print(f"\n  [test] collecting predictions with TTA ...")
    test_pack = collect_predictions(model, test_l, device,
                                    use_tta=True, return_logits=True)
    labels = test_pack["labels"]
    raw_logits = test_pack["logits"]
    seg_gt = test_pack["seg_gt"]
    seg_pred = test_pack["seg_pred"]

    probs_tta = test_pack["probs"]
    probs_tscaled = 1.0 / (1.0 + np.exp(-(raw_logits / T)))

    metrics = {}
    for tag, p in [("tta", probs_tta), ("temp_scaled", probs_tscaled)]:
        preds = (p >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        metrics[tag] = {
            "threshold": 0.5,
            "f1": float(f1_score(labels, preds)),
            "auroc": float(auc(*roc_curve(labels, p)[:2])),
            "auprc": float(average_precision_score(labels, p)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    seg_metrics = compute_seg_metrics(seg_gt, seg_pred, threshold=0.5)
    print("\n  [Seg metrics]")
    for r, m in seg_metrics.items():
        print(f"    {r}: dice={m['dice_mean']:.4f}  iou={m['iou_mean']:.4f}  "
              f"n_pos={m['n_positive_slices']}")

    try:
        val_probs_for_cp = torch.sigmoid(val_logits / T).numpy()
        cp = try_conformal(val_probs_for_cp, val_pack["labels"], probs_tscaled,
                           alpha=0.1)
        if "q" in cp:
            covered = 0
            for true_y, included in zip(labels, cp["sets"]):
                if int(true_y) in included:
                    covered += 1
            cp["coverage"] = covered / len(labels)
            cp["sets"] = cp["sets"][:50]
    except Exception as e:
        cp = {"error": str(e)}

    plot_diagnostics(labels, probs_tta, seg_gt, seg_pred, FIG_DIR)

    final = {
        "n_test": int(len(labels)),
        "temperature": T,
        "cls": metrics,
        "seg": seg_metrics,
        "conformal": cp,
    }
    with open(LOG_DIR / "sota_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n  metrics saved: {LOG_DIR / 'sota_test_metrics.json'}")
    print(f"  figures:       {FIG_DIR}")

    # ── 다중 비교 (Day5 MTL / Day6 MMMT / Day7 SOTA) ────
    print(f"\n  [Multi-way 비교]")
    # Day 5 결과 파일명은 실제로 `mt_evaluation_results.json` (구버전 호환을 위해 두 가지 모두 시도)
    mtl_candidates = [MTL_PREV_LOG / "mt_test_metrics.json",
                      MTL_PREV_LOG / "mt_evaluation_results.json"]
    mtl_path = next((p for p in mtl_candidates if p.exists()), mtl_candidates[0])
    for tag, path in [("Day5 MTL (multitask)", mtl_path),
                      ("Day6 MMMT (mmmt)", MMMT_PREV_LOG / "mmmt_test_metrics.json")]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                print(f"    {tag}: {prev}")
            except Exception as e:
                print(f"  [WARN] {tag} 비교 실패: {e}")
    print(f"    Day7 SOTA   : {metrics['tta']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
