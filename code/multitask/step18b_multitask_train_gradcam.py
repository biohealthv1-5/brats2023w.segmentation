# -*- coding: utf-8 -*-
"""
Step 18b: Multi-Task 모델 — Train Split Grad-CAM 시각화
=========================================================
학습된 MultiTaskBrainNet (step18 산출물) 의 *학습셋(train)* 에서
Grad-CAM 히트맵을 만들어:
  - 분류 헤드(cls_out) 가 보는 영역
  - 세분화 GT mask
  - 세분화 예측 mask (seg_out)
세 가지를 동시에 비교합니다.

질문:
  1) (TP)  분류기가 본 영역이 실제 종양과 일치하는가?
  2) (FN)  학습셋에서도 놓치는 케이스가 있는가? (under-fit / 작은 종양)
  3) (FP)  학습셋에서도 오탐하는 케이스가 있는가? (혼란 패턴)

Grad-CAM 타겟 레이어: model.enc5  (= ResNet-18 layer4)

사용법:
    python code/multitask/step18b_multitask_train_gradcam.py
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
from PIL import Image
from torchvision import transforms

# Windows 콘솔 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step16_multitask_dataset import (
    IMAGENET_MEAN, IMAGENET_STD, SLICE_DIR, MASK_DIR, SPLITS_CSV,
)
from step17_multitask_model import create_multitask_model

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "multitask" / "mt_best_model.pth"
GRADCAM_DIR = PROJECT_DIR / "outputs" / "figures" / "multitask" / "gradcam_train"
LOG_DIR = PROJECT_DIR / "outputs" / "logs" / "multitask"

# ─── 설정 ────────────────────────────────────────────────────
IMAGE_SIZE = 224
NUM_SAMPLES = 8           # 카테고리(TP/FN/FP)별 시각화 샘플 수
SUBSAMPLE_IOU = 400       # IoU 통계를 위한 서브샘플 크기 (학습셋은 크니까 제한)
THRESHOLD = 0.5
CAM_THRESHOLD = 0.5       # IoU 계산용 CAM 이진화 임계값
RNG_SEED = 42


# ═══════════════════════════════════════════════════════════════
#  Grad-CAM 엔진 (Multi-Task용)
# ═══════════════════════════════════════════════════════════════
class MultiTaskGradCAM:
    """
    MultiTaskBrainNet 의 분류 헤드(cls_out) 를 타겟으로
    enc5(= ResNet-18 layer4) 의 Grad-CAM 을 계산.
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
            cls_out, seg_out = self.model(x)        # cls_out: (1,1)  seg_out: (1,1,H,W)
            logit = cls_out.squeeze()               # scalar
            logit.backward()

        # ─ CAM ─
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)              # (1,C,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        prob = torch.sigmoid(logit).item()
        seg_prob = torch.sigmoid(seg_out).squeeze().detach().cpu().numpy()   # (H,W)

        return cam, prob, seg_prob


# ═══════════════════════════════════════════════════════════════
#  데이터 유틸
# ═══════════════════════════════════════════════════════════════
_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_slice_tensor(filename: str):
    """슬라이스 PNG → (정규화 텐서, 원본 grayscale, 종양 마스크 GT)"""
    img_path = SLICE_DIR / filename
    arr = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    mask_path = MASK_DIR / filename
    if mask_path.exists():
        m_arr = np.fromfile(str(mask_path), dtype=np.uint8)
        mask = cv2.imdecode(m_arr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    else:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask_bin = (mask > 127).astype(np.uint8) * 255

    img_3ch = np.stack([img, img, img], axis=-1)
    pil = Image.fromarray(img_3ch)
    tensor = _NORM(pil)
    return tensor, img, mask_bin


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
    카테고리(TP/FN/FP)별 5열 그리드:
      MRI | Seg GT | Grad-CAM | Seg-Pred | 오버레이(MRI+CAM)
    """
    n = len(samples)
    if n == 0:
        return []

    fig, axes = plt.subplots(n, 5, figsize=(20, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MRI", "Seg GT", "Grad-CAM (cls)", "Seg Pred", "MRI + CAM"]
    rows_info = []

    for i, row in enumerate(samples):
        fname = row["filename"]
        label = int(row["label"])
        tensor, gray, mask_gt = load_slice_tensor(fname)
        cam, prob, seg_pred = gradcam_engine(tensor, device)
        seg_pred_bin = (seg_pred >= THRESHOLD).astype(np.uint8) * 255
        iou_cam_gt = compute_iou(cam, mask_gt)
        iou_seg_gt = compute_iou((seg_pred >= THRESHOLD).astype(np.float32), mask_gt)
        rows_info.append({
            "filename": fname, "label": label, "prob": prob,
            "iou_cam_gt": iou_cam_gt, "iou_seg_gt": iou_seg_gt,
        })

        # CAM 컬러맵
        cam_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(gray3, 0.55, cam_color, 0.45, 0)

        axes[i, 0].imshow(gray, cmap="gray")
        axes[i, 0].set_title(f"{col_titles[0]}\nlabel={label}, prob={prob:.3f}",
                             fontsize=10)
        axes[i, 1].imshow(mask_gt, cmap="Reds", vmin=0, vmax=255)
        axes[i, 1].set_title(col_titles[1], fontsize=10)
        axes[i, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title(f"{col_titles[2]}\nIoU(CAM,GT)={iou_cam_gt:.3f}",
                             fontsize=10)
        axes[i, 3].imshow(seg_pred_bin, cmap="Reds", vmin=0, vmax=255)
        axes[i, 3].set_title(f"{col_titles[3]}\nIoU(Seg,GT)={iou_seg_gt:.3f}",
                             fontsize=10)
        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title(col_titles[4], fontsize=10)

        axes[i, 0].set_ylabel(fname.replace(".png", ""),
                              fontsize=7, rotation=0, labelpad=120, va="center")
        for j in range(5):
            axes[i, j].axis("off")

    fig.suptitle(f"[Multi-Task | TRAIN] Grad-CAM: {category} (n={n})",
                 fontsize=16, fontweight="bold", y=1.005)
    plt.tight_layout()
    save_path = save_dir / f"mt_train_gradcam_{category.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {category} 시각화 저장: {save_path.name}")
    return rows_info


def plot_iou_distribution(iou_rows, save_dir, fname, title):
    """CAM-GT IoU 분포 히스토그램"""
    if not iou_rows:
        return
    vals = [x["iou_cam_gt"] for x in iou_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=25, color="#3498db", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(vals), color="#e74c3c", ls="--", lw=2,
               label=f"평균 IoU = {np.mean(vals):.3f}")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("IoU (Grad-CAM ↔ Seg GT)", fontsize=11)
    ax.set_ylabel("빈도", fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ IoU 분포 저장: {fname}")


def plot_cam_vs_seg(iou_rows, save_dir):
    """Grad-CAM IoU vs Seg-Pred IoU 비교 산점도"""
    if not iou_rows:
        return
    cam_ious = np.array([x["iou_cam_gt"] for x in iou_rows])
    seg_ious = np.array([x["iou_seg_gt"] for x in iou_rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(seg_ious, cam_ious, c="#9b59b6", s=18, alpha=0.65, edgecolors="white")
    lim = max(cam_ious.max(), seg_ious.max(), 0.1)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel("IoU (Seg head 예측 ↔ GT)", fontsize=11)
    ax.set_ylabel("IoU (Grad-CAM ↔ GT)", fontsize=11)
    ax.set_title("학습셋 TP: Seg head 위치 정확도 vs Grad-CAM 위치 정확도",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "mt_train_cam_vs_seg_iou.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ CAM vs Seg IoU 산점도 저장")


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Step 18b: Multi-Task Grad-CAM @ TRAIN split")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 모델 로드 ───────────────────────────────────────────
    if not CHECKPOINT.exists():
        print(f"  [ERROR] 체크포인트 없음: {CHECKPOINT}")
        print("  먼저 step18_multitask_train.py 를 실행하여 mt_best_model.pth 를 생성하세요.")
        return

    model = create_multitask_model(pretrained=False)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  ✓ Multi-Task 모델 로드 (Epoch {ckpt['epoch']}, "
          f"Val Acc={ckpt['val_acc']:.2f}%, Val Dice={ckpt['val_dice']:.4f})")

    gradcam = MultiTaskGradCAM(model, model.enc5)
    print(f"  ✓ Grad-CAM target = model.enc5 (ResNet-18 layer4)")

    # ─── 학습셋 로드 ─────────────────────────────────────────
    df = pd.read_csv(SPLITS_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"  Train 샘플 수: {len(train_df):,}")

    # ─── 학습셋 예측 수집 (분류 확률) ────────────────────────
    print(f"\n{'─' * 60}")
    print("  학습셋 분류 예측 수집 중...")
    probs = np.zeros(len(train_df), dtype=np.float32)
    for i in range(len(train_df)):
        if i % 1000 == 0:
            print(f"    {i:,}/{len(train_df):,} ({i/len(train_df)*100:.1f}%)")
        fname = train_df.iloc[i]["filename"]
        tensor, _, _ = load_slice_tensor(fname)
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
    # 질문 1: TP — Grad-CAM ↔ Seg GT
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  [Q1] TP : Grad-CAM 이 실제 종양 영역과 일치하는가?")
    print(f"{'═' * 60}")
    tp_sorted = tp_df.sort_values("prob", ascending=False)
    tp_high = tp_sorted.head(NUM_SAMPLES // 2)
    tp_low = tp_sorted.tail(NUM_SAMPLES // 2)
    tp_samples = pd.concat([tp_high, tp_low]).to_dict("records")
    tp_vis = visualize_samples(tp_samples, "TP", GRADCAM_DIR, gradcam, device)

    # 서브샘플로 전체 IoU 통계
    print(f"  서브샘플 IoU 계산 중 (최대 {SUBSAMPLE_IOU}개)...")
    tp_sub = tp_df.sample(n=min(SUBSAMPLE_IOU, len(tp_df)), random_state=RNG_SEED)
    all_tp_rows = []
    for _, row in tp_sub.iterrows():
        tensor, _, mask_gt = load_slice_tensor(row["filename"])
        cam, prob, seg_pred = gradcam(tensor, device)
        all_tp_rows.append({
            "filename": row["filename"], "prob": float(prob),
            "iou_cam_gt": float(compute_iou(cam, mask_gt)),
            "iou_seg_gt": float(compute_iou(
                (seg_pred >= THRESHOLD).astype(np.float32), mask_gt)),
        })
    plot_iou_distribution(all_tp_rows, GRADCAM_DIR,
                          "mt_train_tp_iou_distribution.png",
                          "Train TP — Grad-CAM ↔ Seg GT IoU 분포")
    plot_cam_vs_seg(all_tp_rows, GRADCAM_DIR)

    mean_cam_iou = float(np.mean([r["iou_cam_gt"] for r in all_tp_rows]))
    mean_seg_iou = float(np.mean([r["iou_seg_gt"] for r in all_tp_rows]))
    print(f"  ★ Train TP 평균 IoU (Grad-CAM ↔ GT): {mean_cam_iou:.4f}")
    print(f"  ★ Train TP 평균 IoU (Seg head ↔ GT): {mean_seg_iou:.4f}")

    # ═════════════════════════════════════════════════════════
    # 질문 2: FN — 학습셋에서도 놓치는 케이스?
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  [Q2] FN ({len(fn_df):,}개) : 학습셋에서도 놓치는가?")
    print(f"{'═' * 60}")
    if len(fn_df) > 0:
        fn_sorted = fn_df.sort_values("prob", ascending=False)
        fn_border = fn_sorted.head(NUM_SAMPLES // 2)
        fn_far = fn_sorted.tail(NUM_SAMPLES // 2)
        fn_samples = pd.concat([fn_border, fn_far]).to_dict("records")
        fn_vis = visualize_samples(fn_samples, "FN", GRADCAM_DIR, gradcam, device)
    else:
        fn_vis = []
        print("  학습셋 FN 없음 — 분류기가 학습셋 양성 슬라이스를 모두 맞춤.")

    # ═════════════════════════════════════════════════════════
    # 질문 3: FP — 학습셋에서도 오탐?
    # ═════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print(f"  [Q3] FP ({len(fp_df):,}개) : 학습셋에서도 무엇을 착각하는가?")
    print(f"{'═' * 60}")
    if len(fp_df) > 0:
        fp_sorted = fp_df.sort_values("prob", ascending=False)
        fp_high = fp_sorted.head(NUM_SAMPLES // 2)
        fp_low = fp_sorted.tail(NUM_SAMPLES // 2)
        fp_samples = pd.concat([fp_high, fp_low]).to_dict("records")
        fp_vis = visualize_samples(fp_samples, "FP", GRADCAM_DIR, gradcam, device)
    else:
        fp_vis = []
        print("  학습셋 FP 없음 — 분류기가 학습셋 음성 슬라이스를 모두 맞춤.")

    # ─── 결과 저장 ───────────────────────────────────────────
    summary = {
        "split": "train",
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_val_acc": float(ckpt["val_acc"]),
        "checkpoint_val_dice": float(ckpt["val_dice"]),
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
    out_json = GRADCAM_DIR / "mt_train_gradcam_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 서브샘플 행도 저장 (분석용)
    out_csv = GRADCAM_DIR / "mt_train_gradcam_subsample.csv"
    pd.DataFrame(all_tp_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'═' * 60}")
    print("  Train Grad-CAM 분석 요약 (Multi-Task)")
    print(f"{'═' * 60}")
    print(f"  TP/FN/FP/TN = {len(tp_df):,}/{len(fn_df):,}/{len(fp_df):,}/{len(tn_df):,}")
    print(f"  Train acc = {summary['train_accuracy_pct']:.2f}%")
    print(f"  Train TP 평균 IoU CAM↔GT  : {mean_cam_iou:.4f}")
    print(f"  Train TP 평균 IoU Seg↔GT  : {mean_seg_iou:.4f}")
    print(f"  결과: {out_json}")
    print(f"  서브샘플 CSV: {out_csv}")
    print(f"  시각화: {GRADCAM_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
