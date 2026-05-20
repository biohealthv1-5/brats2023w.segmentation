# -*- coding: utf-8 -*-
"""
Step 13: 패치 Grad-CAM 분석
============================
패치 분류 모델의 Grad-CAM 히트맵을 생성하고,
패치별 히트맵을 원래 슬라이스 위치에 재조립하여
seg 마스크와의 IoU를 비교합니다.

기존 whole-slice Grad-CAM (IoU=0.145)과의 개선 여부를 확인합니다.

사용법:
    python code/step13_patch_gradcam.py
"""

import sys, json, cv2
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step10_patch_model import PatchCNN

# ─── 경로 ────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_DIR / "data" / "BraTS-GLI" / "training"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
SLICE_DIR = PROJECT_DIR / "processed" / "slices"
PATCH_CSV = PROJECT_DIR / "processed" / "patch_labels.csv"
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "patch_best_model.pth"
GRADCAM_DIR = PROJECT_DIR / "outputs" / "figures" / "patch_gradcam"

# ─── 설정 ────────────────────────────────────────────────────
PATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_SAMPLES = 6          # 카테고리별 시각화 샘플
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════
#  Grad-CAM for PatchCNN
# ═══════════════════════════════════════════════════════════════
class PatchGradCAM:
    """PatchCNN의 마지막 Conv 레이어 대상 Grad-CAM"""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target = model.get_gradcam_target_layer()
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, device):
        with torch.enable_grad():
            x = x.unsqueeze(0).to(device).requires_grad_(True)
            self.model.zero_grad()
            logit = self.model(x).squeeze()
            logit.backward()
        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (PATCH_SIZE, PATCH_SIZE))
        prob = torch.sigmoid(logit).item()
        return cam, prob


# ═══════════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════════
def load_slice_gray(filename):
    buf = np.fromfile(str(SLICE_DIR / filename), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img if img is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)


def load_seg_mask(patient_id, slice_idx):
    seg_path = DATA_DIR / patient_id / f"{patient_id}-seg.nii.gz"
    if not seg_path.exists():
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    s = seg[:, :, slice_idx]
    mask = (s > 0).astype(np.uint8) * 255
    return cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


def patch_to_tensor(gray_img, x, y):
    patch = gray_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
    p3 = np.stack([patch]*3, axis=-1)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm(Image.fromarray(p3))


def compute_iou(a, b, thresh=0.5):
    a_bin = (a >= thresh).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)
    inter = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()
    return inter / max(union, 1)


# ═══════════════════════════════════════════════════════════════
#  슬라이스 단위 히트맵 재조립
# ═══════════════════════════════════════════════════════════════
def reconstruct_slice_heatmap(slice_patches, gradcam_engine, gray_img, device):
    """
    한 슬라이스의 모든 패치에 대해 Grad-CAM을 수행하고
    224×224 히트맵으로 재조립합니다.
    """
    heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    count = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for _, p in slice_patches.iterrows():
        x, y = int(p["x_start"]), int(p["y_start"])
        tensor = patch_to_tensor(gray_img, x, y)
        cam, prob = gradcam_engine(tensor, device)
        heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += cam
        count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    count = np.maximum(count, 1)
    heatmap /= count
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


# ═══════════════════════════════════════════════════════════════
#  시각화
# ═══════════════════════════════════════════════════════════════
def visualize_slice_gradcam(samples_info, category, gradcam_engine, device, save_dir):
    """
    슬라이스 단위 재조립 히트맵 시각화
    열: 원본 | Seg 마스크 | Patch-CAM | 오버레이
    """
    n = len(samples_info)
    if n == 0:
        return []

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    ious = []
    for i, (fname, patches_df) in enumerate(samples_info):
        row0 = patches_df.iloc[0]
        patient_id = row0["patient_id"]
        slice_idx = int(row0["slice_idx"])

        gray = load_slice_gray(fname)
        seg = load_seg_mask(patient_id, slice_idx)
        heatmap = reconstruct_slice_heatmap(patches_df, gradcam_engine, gray, device)
        iou = compute_iou(heatmap, seg)
        ious.append(iou)

        cam_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(gray3, 0.6, cam_color, 0.4, 0)

        axes[i, 0].imshow(gray, cmap="gray")
        axes[i, 0].set_title("MRI 슬라이스", fontsize=10)

        axes[i, 1].imshow(seg, cmap="Reds", vmin=0, vmax=255)
        axes[i, 1].set_title("Seg 마스크 (GT)", fontsize=10)

        axes[i, 2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Patch Grad-CAM\nIoU={iou:.3f}", fontsize=10)

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("오버레이", fontsize=10)

        axes[i, 0].set_ylabel(fname.replace(".png", ""), fontsize=7,
                              rotation=0, labelpad=120, va="center")
        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle(f"Patch Grad-CAM: {category} (n={n})",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_dir / f"patch_gradcam_{category.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {category} 시각화 저장")
    return ious


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Step 13: 패치 Grad-CAM 분석")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 모델 ────────────────────────────────────────────────
    model = PatchCNN().to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✓ 모델 로드")
    gradcam = PatchGradCAM(model)

    # ─── 테스트 패치 로드 ────────────────────────────────────
    df = pd.read_csv(PATCH_CSV)
    test_df = df[df["split"] == "test"].copy()

    # 슬라이스별 패치 예측 (max 집계)
    print(f"\n{'─' * 60}")
    print("  슬라이스별 패치 예측 수집 중...")
    slice_groups = test_df.groupby("slice_filename")
    slice_info = []

    total = len(slice_groups)
    for i, (fname, grp) in enumerate(slice_groups):
        if i % 1000 == 0:
            print(f"    {i:,}/{total:,} ({i/total*100:.1f}%)")

        probs = []
        for _, p in grp.iterrows():
            gray = load_slice_gray(fname)
            t = patch_to_tensor(gray, int(p["x_start"]), int(p["y_start"]))
            with torch.no_grad():
                logit = model(t.unsqueeze(0).to(device)).squeeze()
                prob = torch.sigmoid(logit).item()
            probs.append(prob)

        grp = grp.copy()
        grp["prob"] = probs
        max_prob = max(probs)
        gt = grp["slice_label"].iloc[0]
        pred = 1 if max_prob >= 0.5 else 0

        slice_info.append({
            "fname": fname, "gt": gt, "pred": pred,
            "max_prob": max_prob, "patches_df": grp,
        })

    # ─── TP / FN / FP 분류 ───────────────────────────────────
    tp_slices = [s for s in slice_info if s["gt"] == 1 and s["pred"] == 1]
    fn_slices = [s for s in slice_info if s["gt"] == 1 and s["pred"] == 0]
    fp_slices = [s for s in slice_info if s["gt"] == 0 and s["pred"] == 1]

    print(f"\n  TP: {len(tp_slices):,}  FN: {len(fn_slices):,}  FP: {len(fp_slices):,}")

    # ─── TP 시각화 + IoU ─────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  [TP] Patch Grad-CAM vs Seg 마스크")
    print(f"{'═' * 60}")

    tp_sorted = sorted(tp_slices, key=lambda s: s["max_prob"], reverse=True)
    tp_samples = [(s["fname"], s["patches_df"]) for s in tp_sorted[:NUM_SAMPLES]]
    tp_ious = visualize_slice_gradcam(tp_samples, "TP", gradcam, device, GRADCAM_DIR)

    # 서브샘플 IoU 통계
    print("  전체 TP IoU 계산 (최대 200개)...")
    tp_sub = tp_sorted[:min(200, len(tp_sorted))]
    all_tp_ious = []
    for s in tp_sub:
        gray = load_slice_gray(s["fname"])
        seg = load_seg_mask(
            s["patches_df"].iloc[0]["patient_id"],
            int(s["patches_df"].iloc[0]["slice_idx"]))
        hm = reconstruct_slice_heatmap(s["patches_df"], gradcam, gray, device)
        all_tp_ious.append(compute_iou(hm, seg))

    mean_iou = np.mean(all_tp_ious) if all_tp_ious else 0
    print(f"  ★ 패치 TP 평균 IoU: {mean_iou:.4f}")

    # IoU 분포 히스토그램
    if all_tp_ious:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_tp_ious, bins=20, color="#e74c3c", edgecolor="white", alpha=0.8)
        ax.axvline(mean_iou, color="black", ls="--", lw=2,
                   label=f"평균 IoU={mean_iou:.3f}")
        ax.set_title("Patch TP: Grad-CAM ↔ Seg IoU 분포",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("IoU", fontsize=12)
        ax.set_ylabel("빈도", fontsize=12)
        ax.legend(fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(GRADCAM_DIR / "patch_iou_distribution.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ IoU 분포 저장")

    # ─── FN 시각화 ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  [FN] {len(fn_slices):,}개 — 왜 놓쳤는가?")
    print(f"{'═' * 60}")

    fn_sorted = sorted(fn_slices, key=lambda s: s["max_prob"], reverse=True)
    fn_samples = [(s["fname"], s["patches_df"]) for s in fn_sorted[:NUM_SAMPLES]]
    fn_ious = visualize_slice_gradcam(fn_samples, "FN", gradcam, device, GRADCAM_DIR)

    # ─── FP 시각화 ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  [FP] {len(fp_slices):,}개 — 무엇을 착각했는가?")
    print(f"{'═' * 60}")

    fp_sorted = sorted(fp_slices, key=lambda s: s["max_prob"], reverse=True)
    fp_samples = [(s["fname"], s["patches_df"]) for s in fp_sorted[:NUM_SAMPLES]]
    fp_ious = visualize_slice_gradcam(fp_samples, "FP", gradcam, device, GRADCAM_DIR)

    # ─── IoU 비교 (Whole-Slice vs Patch) ─────────────────────
    prev_gradcam = PROJECT_DIR / "outputs" / "figures" / "gradcam" / "gradcam_analysis.json"
    if prev_gradcam.exists():
        with open(prev_gradcam, "r", encoding="utf-8") as f:
            prev = json.load(f)
        prev_iou = prev.get("TP_평균_IoU", prev.get("TP_\ud3c9\uade0_IoU", 0))
        print(f"\n{'═' * 60}")
        print(f"  IoU 비교:")
        print(f"  Whole-Slice Grad-CAM 평균 IoU: {prev_iou:.4f}")
        print(f"  Patch Grad-CAM 평균 IoU:       {mean_iou:.4f}")
        diff = mean_iou - prev_iou
        print(f"  개선: {'↑' if diff > 0 else '↓'}{abs(diff):.4f} "
              f"({diff/max(prev_iou,0.001)*100:+.1f}%)")
        print(f"{'═' * 60}")

    # ─── 결과 저장 ───────────────────────────────────────────
    results = {
        "patch_tp_mean_iou": round(mean_iou, 4),
        "tp_slices": len(tp_slices),
        "fn_slices": len(fn_slices),
        "fp_slices": len(fp_slices),
        "samples_iou_computed": len(all_tp_ious),
    }
    with open(GRADCAM_DIR / "patch_gradcam_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  결과: {GRADCAM_DIR / 'patch_gradcam_analysis.json'}")
    print(f"  시각화: {GRADCAM_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
