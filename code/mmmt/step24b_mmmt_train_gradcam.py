# -*- coding: utf-8 -*-
"""
Step 24b (Day 6 / mmmt): MM-MTL 모델 — Train Split Grad-CAM 시각화
=====================================================================
학습된 MMMTBrainNet (step24 산출물) 의 *학습셋(train)* 에서
3채널 입력 [T1ce, FLAIR, |T1ce-FLAIR|] 에 대한 Grad-CAM 을 시각화합니다.

각 슬라이스마다 6열 그리드를 출력:
  T1ce | FLAIR | |T1ce-FLAIR| | Seg GT | Grad-CAM | Seg Pred

질문:
  1) (TP) 멀티모달 입력 어디를 보고 양성으로 분류하는가?
  2) (FN) 학습셋에서도 멀티모달이 놓치는 패턴은?
  3) (FP) 학습셋에서도 오탐하는 케이스?

Grad-CAM 타겟 레이어: model.enc5 (= ResNet-18 layer4)
분류 헤드 cls_out 의 logit 에 대해 역전파.

사용법:
    python code/mmmt/step24b_mmmt_train_gradcam.py
"""

import sys
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step21_mmmt_dataset import (
    IMAGENET_MEAN, IMAGENET_STD,
    FLAIR_DIR, T1CE_DIR, WT_DIR, SPLITS_CSV,
    _imread_gray,
)
from step22_mmmt_model import create_mm_model

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "mmmt" / "mmmt_best.pth"
GRADCAM_DIR = PROJECT_DIR / "outputs" / "figures" / "mmmt" / "gradcam_train"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "mmmt"

# ─── 설정 ────────────────────────────────────────────────────
IMAGE_SIZE = 224
NUM_SAMPLES = 8
SUBSAMPLE_IOU = 400
THRESHOLD = 0.5
CAM_THRESHOLD = 0.5
CHANNEL_MODE = "t1ce_flair_diff"   # step24 학습 시 사용한 모드와 동일해야 함
RNG_SEED = 42

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  Grad-CAM 엔진 (MM-MTL용)
# ═══════════════════════════════════════════════════════════════
class MMMTGradCAM:
    """
    MMMTBrainNet 의 분류 헤드(cls_out) 를 타겟으로
    enc5(= ResNet-18 layer4) 의 Grad-CAM 계산.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, device):
        """
        x : (3, H, W) tensor (정규화된 상태)
        return : (cam_224x224, prob, seg_pred_224x224)
        """
        with torch.enable_grad():
            x = x.unsqueeze(0).to(device).requires_grad_(True)
            self.model.zero_grad()
            cls_out, seg_out = self.model(x)
            logit = cls_out.squeeze()
            logit.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        prob = torch.sigmoid(logit).item()
        seg_prob = torch.sigmoid(seg_out).squeeze().detach().cpu().numpy()
        return cam, prob, seg_prob


# ═══════════════════════════════════════════════════════════════
#  데이터 유틸 — 3채널 멀티모달 입력
# ═══════════════════════════════════════════════════════════════
def _compose_3ch(t1ce: np.ndarray, flair: np.ndarray, mode: str = CHANNEL_MODE):
    t1ce_f = t1ce.astype(np.float32) / 255.0
    flair_f = flair.astype(np.float32) / 255.0
    if mode == "t1ce_flair_diff":
        c3 = np.abs(t1ce_f - flair_f)
        return np.stack([t1ce_f, flair_f, c3], axis=-1), c3
    elif mode == "t1ce_flair_ratio":
        c3 = np.clip(t1ce_f / (flair_f + 1e-3), 0.0, 1.0)
        return np.stack([t1ce_f, flair_f, c3], axis=-1), c3
    elif mode == "t1ce_flair_mul":
        c3 = t1ce_f * flair_f
        return np.stack([c3, t1ce_f, flair_f], axis=-1), c3
    else:
        raise ValueError(f"unknown channel_mode {mode}")


def load_mm_inputs(filename: str):
    """
    return:
        tensor   : (3,H,W)  정규화된 입력
        t1ce_u8  : (H,W)    원본 0-255
        flair_u8 : (H,W)    원본 0-255
        c3_f01   : (H,W)    3번째 채널 (0~1)
        mask_gt  : (H,W)    WT GT mask 0/255
    """
    t1ce = _imread_gray(T1CE_DIR / filename)
    flair = _imread_gray(FLAIR_DIR / filename)
    if t1ce is None: t1ce = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if flair is None: flair = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    img_3ch, c3_f01 = _compose_3ch(t1ce, flair, CHANNEL_MODE)        # (H,W,3) [0,1]
    img_norm = (img_3ch - _MEAN) / _STD                              # ImageNet norm
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1).astype(np.float32))

    mask = _imread_gray(WT_DIR / filename)
    if mask is None:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask_gt = (mask > 127).astype(np.uint8) * 255

    return tensor, t1ce, flair, c3_f01, mask_gt


def compute_iou(a_heat: np.ndarray, mask_bin: np.ndarray, thresh: float = CAM_THRESHOLD):
    a = (a_heat >= thresh).astype(np.uint8)
    b = (mask_bin > 0).astype(np.uint8)
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / max(union, 1)


# ═══════════════════════════════════════════════════════════════
#  시각화
# ═══════════════════════════════════════════════════════════════
def visualize_samples(samples, category, save_dir, gradcam_engine, device):
    """
    카테고리(TP/FN/FP)별 7열 그리드:
      T1ce | FLAIR | |T1ce-FLAIR| | Seg GT | Grad-CAM | Seg Pred | MRI+CAM
    """
    n = len(samples)
    if n == 0:
        return []

    n_cols = 7
    fig, axes = plt.subplots(n, n_cols, figsize=(3.0 * n_cols, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["T1ce", "FLAIR", "|T1ce-FLAIR|", "Seg GT",
              "Grad-CAM (cls)", "Seg Pred", "FLAIR + CAM"]
    rows_info = []

    for i, row in enumerate(samples):
        fname = row["filename"]
        label = int(row["label"])
        tensor, t1ce_u8, flair_u8, c3, mask_gt = load_mm_inputs(fname)
        cam, prob, seg_pred = gradcam_engine(tensor, device)
        seg_bin = (seg_pred >= THRESHOLD).astype(np.uint8) * 255
        iou_cam_gt = compute_iou(cam, mask_gt)
        iou_seg_gt = compute_iou((seg_pred >= THRESHOLD).astype(np.float32), mask_gt)
        rows_info.append({
            "filename": fname, "label": label, "prob": float(prob),
            "iou_cam_gt": float(iou_cam_gt),
            "iou_seg_gt": float(iou_seg_gt),
        })

        cam_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
        flair3 = cv2.cvtColor(flair_u8, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(flair3, 0.55, cam_color, 0.45, 0)

        axes[i, 0].imshow(t1ce_u8, cmap="gray")
        axes[i, 0].set_title(f"{titles[0]}\nlabel={label}, prob={prob:.3f}",
                             fontsize=9)
        axes[i, 1].imshow(flair_u8, cmap="gray")
        axes[i, 1].set_title(titles[1], fontsize=9)
        axes[i, 2].imshow(c3, cmap="magma", vmin=0, vmax=1)
        axes[i, 2].set_title(titles[2], fontsize=9)
        axes[i, 3].imshow(mask_gt, cmap="Reds", vmin=0, vmax=255)
        axes[i, 3].set_title(titles[3], fontsize=9)
        axes[i, 4].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[i, 4].set_title(f"{titles[4]}\nIoU(CAM,GT)={iou_cam_gt:.3f}",
                             fontsize=9)
        axes[i, 5].imshow(seg_bin, cmap="Reds", vmin=0, vmax=255)
        axes[i, 5].set_title(f"{titles[5]}\nIoU(Seg,GT)={iou_seg_gt:.3f}",
                             fontsize=9)
        axes[i, 6].imshow(overlay)
        axes[i, 6].set_title(titles[6], fontsize=9)

        axes[i, 0].set_ylabel(fname.replace(".png", ""),
                              fontsize=7, rotation=0, labelpad=120, va="center")
        for j in range(n_cols):
            axes[i, j].axis("off")

    fig.suptitle(f"[MM-MTL | TRAIN] Grad-CAM: {category} "
                 f"(channels = {CHANNEL_MODE}, n={n})",
                 fontsize=16, fontweight="bold", y=1.005)
    plt.tight_layout()
    save_path = save_dir / f"mmmt_train_gradcam_{category.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {category} 시각화 저장: {save_path.name}")
    return rows_info


def plot_iou_distribution(iou_rows, save_dir, fname, title):
    if not iou_rows:
        return
    vals = [x["iou_cam_gt"] for x in iou_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=25, color="#16a085", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(vals), color="#e74c3c", ls="--", lw=2,
               label=f"평균 IoU = {np.mean(vals):.3f}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("IoU (Grad-CAM ↔ WT GT)", fontsize=11)
    ax.set_ylabel("빈도", fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ IoU 분포 저장: {fname}")


def plot_cam_vs_seg(iou_rows, save_dir):
    if not iou_rows:
        return
    cam_ious = np.array([x["iou_cam_gt"] for x in iou_rows])
    seg_ious = np.array([x["iou_seg_gt"] for x in iou_rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(seg_ious, cam_ious, c="#16a085", s=18, alpha=0.65, edgecolors="white")
    lim = max(cam_ious.max(), seg_ious.max(), 0.1)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel("IoU (Seg head 예측 ↔ WT GT)", fontsize=11)
    ax.set_ylabel("IoU (Grad-CAM ↔ WT GT)", fontsize=11)
    ax.set_title("MM-MTL Train TP: Seg head vs Grad-CAM 위치 정확도",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "mmmt_train_cam_vs_seg_iou.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ CAM vs Seg IoU 산점도 저장")


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Step 24b: MM-MTL Grad-CAM @ TRAIN split")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Channel mode: {CHANNEL_MODE}")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        print(f"  [ERROR] 체크포인트 없음: {CHECKPOINT}")
        print("  먼저 step24_mmmt_train.py 를 실행하여 mmmt_best.pth 를 생성하세요.")
        return
    if not T1CE_DIR.exists() or len(list(T1CE_DIR.glob('*.png'))) == 0:
        print(f"  [ERROR] T1ce 슬라이스 없음: {T1CE_DIR}")
        print("  먼저 step20_mmmt_preprocess.py 를 실행하세요.")
        return

    # ─── 모델 로드 ───────────────────────────────────────────
    model = create_mm_model(pretrained=False, in_ch=3, seg_classes=1)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    metrics = ckpt.get("metrics", {})
    print(f"  ✓ MM-MTL 모델 로드 (Epoch {ckpt['epoch']}, "
          f"Val Acc={metrics.get('acc', 0):.2f}%, "
          f"Val WT Dice={metrics.get('dice', 0):.4f})")

    gradcam = MMMTGradCAM(model, model.enc5)
    print(f"  ✓ Grad-CAM target = model.enc5 (ResNet-18 layer4)")

    # ─── 학습셋 로드 ─────────────────────────────────────────
    df = pd.read_csv(SPLITS_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"  Train 샘플 수: {len(train_df):,}")

    # ─── 학습셋 분류 예측 수집 ────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  학습셋 분류 예측 수집 중...")
    probs = np.zeros(len(train_df), dtype=np.float32)
    for i in range(len(train_df)):
        if i % 1000 == 0:
            print(f"    {i:,}/{len(train_df):,} ({i/len(train_df)*100:.1f}%)")
        fname = train_df.iloc[i]["filename"]
        tensor, _, _, _, _ = load_mm_inputs(fname)
        with torch.no_grad():
            cls_out, _ = model(tensor.unsqueeze(0).to(device))
            probs[i] = torch.sigmoid(cls_out.squeeze()).item()
    train_df["prob"] = probs
    train_df["pred"] = (train_df["prob"] >= THRESHOLD).astype(int)

    tp_df = train_df[(train_df["label"] == 1) & (train_df["pred"] == 1)]
    fn_df = train_df[(train_df["label"] == 1) & (train_df["pred"] == 0)]
    fp_df = train_df[(train_df["label"] == 0) & (train_df["pred"] == 1)]
    tn_df = train_df[(train_df["label"] == 0) & (train_df["pred"] == 0)]
    print(f"\n  [Train confusion]")
    print(f"    TP: {len(tp_df):,}  FN: {len(fn_df):,}")
    print(f"    FP: {len(fp_df):,}  TN: {len(tn_df):,}")

    # ═════════════════════════════════════════════════════════
    # Q1: TP
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  [Q1] TP : 멀티모달 입력 어디를 보고 양성으로 분류?")
    print(f"{'═' * 60}")
    tp_sorted = tp_df.sort_values("prob", ascending=False)
    tp_high = tp_sorted.head(NUM_SAMPLES // 2)
    tp_low = tp_sorted.tail(NUM_SAMPLES // 2)
    tp_samples = pd.concat([tp_high, tp_low]).to_dict("records")
    visualize_samples(tp_samples, "TP", GRADCAM_DIR, gradcam, device)

    print(f"  서브샘플 IoU 계산 중 (최대 {SUBSAMPLE_IOU}개)...")
    tp_sub = tp_df.sample(n=min(SUBSAMPLE_IOU, len(tp_df)), random_state=RNG_SEED)
    all_tp_rows = []
    for _, row in tp_sub.iterrows():
        tensor, _, _, _, mask_gt = load_mm_inputs(row["filename"])
        cam, prob, seg_pred = gradcam(tensor, device)
        all_tp_rows.append({
            "filename": row["filename"], "prob": float(prob),
            "iou_cam_gt": float(compute_iou(cam, mask_gt)),
            "iou_seg_gt": float(compute_iou(
                (seg_pred >= THRESHOLD).astype(np.float32), mask_gt)),
        })
    plot_iou_distribution(all_tp_rows, GRADCAM_DIR,
                          "mmmt_train_tp_iou_distribution.png",
                          "MM-MTL Train TP — Grad-CAM ↔ WT GT IoU 분포")
    plot_cam_vs_seg(all_tp_rows, GRADCAM_DIR)

    mean_cam_iou = float(np.mean([r["iou_cam_gt"] for r in all_tp_rows]))
    mean_seg_iou = float(np.mean([r["iou_seg_gt"] for r in all_tp_rows]))
    print(f"  ★ Train TP 평균 IoU (Grad-CAM ↔ WT GT): {mean_cam_iou:.4f}")
    print(f"  ★ Train TP 평균 IoU (Seg head ↔ WT GT): {mean_seg_iou:.4f}")

    # ═════════════════════════════════════════════════════════
    # Q2: FN
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  [Q2] FN ({len(fn_df):,}개) : 학습셋에서도 놓치는가?")
    print(f"{'═' * 60}")
    if len(fn_df) > 0:
        fn_sorted = fn_df.sort_values("prob", ascending=False)
        fn_border = fn_sorted.head(NUM_SAMPLES // 2)
        fn_far = fn_sorted.tail(NUM_SAMPLES // 2)
        fn_samples = pd.concat([fn_border, fn_far]).to_dict("records")
        visualize_samples(fn_samples, "FN", GRADCAM_DIR, gradcam, device)
    else:
        print("  학습셋 FN 없음.")

    # ═════════════════════════════════════════════════════════
    # Q3: FP
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  [Q3] FP ({len(fp_df):,}개) : 학습셋에서도 오탐?")
    print(f"{'═' * 60}")
    if len(fp_df) > 0:
        fp_sorted = fp_df.sort_values("prob", ascending=False)
        fp_high = fp_sorted.head(NUM_SAMPLES // 2)
        fp_low = fp_sorted.tail(NUM_SAMPLES // 2)
        fp_samples = pd.concat([fp_high, fp_low]).to_dict("records")
        visualize_samples(fp_samples, "FP", GRADCAM_DIR, gradcam, device)
    else:
        print("  학습셋 FP 없음.")

    # ─── 결과 저장 ───────────────────────────────────────────
    summary = {
        "split": "train",
        "channel_mode": CHANNEL_MODE,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_val_acc": float(metrics.get("acc", 0)),
        "checkpoint_val_dice": float(metrics.get("dice", 0)),
        "n_train_total": int(len(train_df)),
        "TP": int(len(tp_df)), "FN": int(len(fn_df)),
        "FP": int(len(fp_df)), "TN": int(len(tn_df)),
        "train_accuracy_pct": round(float((train_df["pred"] == train_df["label"]).mean()) * 100, 4),
        "iou_subsample_n": int(len(all_tp_rows)),
        "mean_iou_CAM_GT_trainTP": round(mean_cam_iou, 4),
        "mean_iou_SEG_GT_trainTP": round(mean_seg_iou, 4),
        "FN_prob_mean": (round(float(fn_df["prob"].mean()), 4) if len(fn_df) else None),
        "FP_prob_mean": (round(float(fp_df["prob"].mean()), 4) if len(fp_df) else None),
    }
    out_json = GRADCAM_DIR / "mmmt_train_gradcam_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    out_csv = GRADCAM_DIR / "mmmt_train_gradcam_subsample.csv"
    pd.DataFrame(all_tp_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'═' * 60}")
    print("  MM-MTL Train Grad-CAM 분석 요약")
    print(f"{'═' * 60}")
    print(f"  TP/FN/FP/TN = {len(tp_df):,}/{len(fn_df):,}/{len(fp_df):,}/{len(tn_df):,}")
    print(f"  Train acc = {summary['train_accuracy_pct']:.2f}%")
    print(f"  Train TP 평균 IoU CAM↔WT GT : {mean_cam_iou:.4f}")
    print(f"  Train TP 평균 IoU Seg↔WT GT : {mean_seg_iou:.4f}")
    print(f"  결과: {out_json}")
    print(f"  서브샘플 CSV: {out_csv}")
    print(f"  시각화: {GRADCAM_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
