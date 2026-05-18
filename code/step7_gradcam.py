# -*- coding: utf-8 -*-
"""
Step 7: Grad-CAM 분석 (3일차)
==============================
TP / FN / FP 케이스별 Grad-CAM 히트맵을 생성하고,
segmentation 마스크와 비교하여 모델의 판단 근거를 시각적으로 분석합니다.

질문1 (TP): seg 마스크와 히트맵이 일치하는가?
질문2 (FN 888개): 왜 놓쳤는가?
질문3 (FP 536개): 무엇을 착각했는가?

사용법:
    python code/step7_gradcam.py
"""

import os, sys, json, cv2
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

# Windows 콘솔 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_dataset import IMAGENET_MEAN, IMAGENET_STD
from step4_model import create_model

# ─── 경로 설정 ───────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (
    PROJECT_DIR / "data" / "BraTS-GLI" / "training"
    / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)
SLICE_DIR = PROJECT_DIR / "processed" / "slices"
SPLITS_CSV = PROJECT_DIR / "processed" / "splits.csv"
CHECKPOINT = PROJECT_DIR / "outputs" / "checkpoints" / "best_model.pth"
GRADCAM_DIR = PROJECT_DIR / "outputs" / "figures" / "gradcam"

# ─── 설정 ────────────────────────────────────────────────────
NUM_SAMPLES = 8          # 각 카테고리별 시각화 샘플 수
MODALITY = "t2f"
IMAGE_SIZE = 224
THRESHOLD = 0.5


# ═══════════════════════════════════════════════════════════════
#  Grad-CAM 엔진
# ═══════════════════════════════════════════════════════════════
class GradCAM:
    """ResNet-18 layer4 대상 Grad-CAM"""

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # 훅 등록
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def __call__(self, x, device):
        """입력 텐서 x → (heatmap_224x224, prob)"""
        # forward (grad 필요하므로 no_grad 해제)
        with torch.enable_grad():
            x = x.unsqueeze(0).to(device).requires_grad_(True)
            self.model.zero_grad()
            logit = self.model(x).squeeze()
            logit.backward()

        # GAP → 가중합
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # 0-1 정규화
        if cam.max() > 0:
            cam = cam / cam.max()

        # 224×224로 리사이즈
        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

        prob = torch.sigmoid(logit).item()
        return cam_resized, prob


# ═══════════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════════
def load_seg_mask(patient_id: str, slice_idx: int) -> np.ndarray:
    """원본 NIfTI seg 마스크에서 해당 슬라이스를 224×224로 추출"""
    seg_path = DATA_DIR / patient_id / f"{patient_id}-seg.nii.gz"
    if not seg_path.exists():
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    seg_slice = seg[:, :, slice_idx]
    mask = (seg_slice > 0).astype(np.uint8) * 255
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return mask


def load_slice_tensor(filename: str):
    """슬라이스 PNG → 정규화된 텐서"""
    img_path = SLICE_DIR / filename
    arr = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    img_3ch = np.stack([img, img, img], axis=-1)
    pil = Image.fromarray(img_3ch)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(pil), img


def compute_iou(cam: np.ndarray, mask: np.ndarray, cam_thresh: float = 0.5) -> float:
    """Grad-CAM 히트맵과 seg 마스크 간 IoU 계산"""
    cam_bin = (cam >= cam_thresh).astype(np.uint8)
    mask_bin = (mask > 0).astype(np.uint8)
    intersection = (cam_bin & mask_bin).sum()
    union = (cam_bin | mask_bin).sum()
    return intersection / max(union, 1)


# ═══════════════════════════════════════════════════════════════
#  시각화 함수
# ═══════════════════════════════════════════════════════════════
def visualize_samples(samples, category, save_dir, gradcam_engine, device):
    """
    카테고리(TP/FN/FP)별 샘플들을 4열 그리드로 시각화
    열: 원본 | Seg 마스크 | Grad-CAM | 오버레이
    """
    n = len(samples)
    if n == 0:
        return []

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["MRI 슬라이스", "Seg 마스크 (GT)", "Grad-CAM", "오버레이"]
    ious = []

    for i, row in enumerate(samples):
        filename = row["filename"]
        patient_id = row["patient_id"]
        slice_idx = int(row["slice_idx"])

        # 데이터 준비
        tensor, gray_img = load_slice_tensor(filename)
        seg_mask = load_seg_mask(patient_id, slice_idx)
        cam, prob = gradcam_engine(tensor, device)
        iou = compute_iou(cam, seg_mask)
        ious.append({"filename": filename, "prob": prob, "iou": iou})

        # 컬러맵 적용
        cam_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

        gray_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(gray_3ch, 0.6, cam_color, 0.4, 0)

        # seg 마스크 윤곽선 오버레이
        seg_contours = seg_mask.copy()

        # 그리기
        axes[i, 0].imshow(gray_img, cmap="gray")
        axes[i, 0].set_title(f"{titles[0]}\nprob={prob:.3f}", fontsize=10)

        axes[i, 1].imshow(seg_mask, cmap="Reds", vmin=0, vmax=255)
        axes[i, 1].set_title(titles[1], fontsize=10)

        axes[i, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title(f"{titles[2]}\nIoU={iou:.3f}", fontsize=10)

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(titles[3], fontsize=10)

        # 파일명 표시
        axes[i, 0].set_ylabel(filename.replace(".png", ""), fontsize=8, rotation=0,
                              labelpad=120, va="center")

        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle(f"Grad-CAM 분석: {category} (n={n})", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_path = save_dir / f"gradcam_{category.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {category} 시각화 저장: {save_path.name}")
    return ious


# ═══════════════════════════════════════════════════════════════
#  통계 요약 시각화
# ═══════════════════════════════════════════════════════════════
def plot_iou_distribution(tp_ious, save_dir):
    """TP 샘플들의 CAM-Seg IoU 분포 히스토그램"""
    if not tp_ious:
        return
    vals = [x["iou"] for x in tp_ious]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(vals), color="#e74c3c", linestyle="--", lw=2,
               label=f"평균 IoU={np.mean(vals):.3f}")
    ax.set_title("TP: Grad-CAM ↔ Seg 마스크 IoU 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("빈도", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "gradcam_iou_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ IoU 분포 저장: gradcam_iou_distribution.png")


def plot_prob_comparison(fn_probs, fp_probs, save_dir):
    """FN/FP 예측 확률 비교 히스토그램"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if fn_probs:
        axes[0].hist(fn_probs, bins=20, color="#e67e22", edgecolor="white", alpha=0.8)
        axes[0].axvline(0.5, color="red", linestyle="--", lw=2, label="Threshold=0.5")
        axes[0].set_title(f"FN 예측 확률 분포 (n={len(fn_probs)})", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("예측 확률", fontsize=11)
        axes[0].set_ylabel("빈도", fontsize=11)
        axes[0].legend()
        axes[0].spines[["top", "right"]].set_visible(False)

    if fp_probs:
        axes[1].hist(fp_probs, bins=20, color="#9b59b6", edgecolor="white", alpha=0.8)
        axes[1].axvline(0.5, color="red", linestyle="--", lw=2, label="Threshold=0.5")
        axes[1].set_title(f"FP 예측 확률 분포 (n={len(fp_probs)})", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("예측 확률", fontsize=11)
        axes[1].set_ylabel("빈도", fontsize=11)
        axes[1].legend()
        axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_dir / "gradcam_fn_fp_prob.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ FN/FP 확률 분포 저장: gradcam_fn_fp_prob.png")


def plot_fn_tumor_size(fn_df, save_dir):
    """FN 케이스의 종양 크기(seg 마스크 내 종양 픽셀 비율) 분석"""
    tumor_fracs = []
    for _, row in fn_df.iterrows():
        seg = load_seg_mask(row["patient_id"], int(row["slice_idx"]))
        frac = (seg > 0).sum() / (IMAGE_SIZE * IMAGE_SIZE)
        tumor_fracs.append(frac)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tumor_fracs, bins=25, color="#e74c3c", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(tumor_fracs), color="black", linestyle="--", lw=2,
               label=f"평균={np.mean(tumor_fracs):.4f}")
    ax.set_title(f"FN: 놓친 종양의 크기 분포 (n={len(tumor_fracs)})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("종양 픽셀 비율 (seg mask)", fontsize=11)
    ax.set_ylabel("빈도", fontsize=11)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir / "gradcam_fn_tumor_size.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ FN 종양 크기 분포 저장: gradcam_fn_tumor_size.png")
    return tumor_fracs


# ═══════════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Step 7: Grad-CAM 분석 (3일차)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 모델 로드 ───────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  모델 로드 중...")
    if not CHECKPOINT.exists():
        print(f"  [ERROR] 체크포인트 없음: {CHECKPOINT}")
        return

    model = create_model(pretrained=False)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  ✓ 모델 로드 완료 (Epoch {ckpt['epoch']})")

    # Grad-CAM 엔진 초기화 (layer4 타겟)
    gradcam = GradCAM(model, model.layer4)
    print("  ✓ Grad-CAM 엔진 초기화 (target: layer4)")

    # ─── 테스트셋 예측 수집 ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  테스트셋 예측 수집 중...")
    df = pd.read_csv(SPLITS_CSV)
    test_df = df[df["split"] == "test"].copy()
    print(f"  Test 샘플 수: {len(test_df):,}")

    # 각 샘플의 예측 확률 계산
    probs_list = []
    total = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows()):
        if i % 500 == 0:
            print(f"    진행: {i:,}/{total:,} ({i/total*100:.1f}%)")
        tensor, _ = load_slice_tensor(row["filename"])
        with torch.no_grad():
            logit = model(tensor.unsqueeze(0).to(device)).squeeze()
            prob = torch.sigmoid(logit).item()
        probs_list.append(prob)

    test_df["prob"] = probs_list
    test_df["pred"] = (test_df["prob"] >= THRESHOLD).astype(int)

    # ─── TP / FN / FP / TN 분류 ─────────────────────────────
    tp_df = test_df[(test_df["label"] == 1) & (test_df["pred"] == 1)]
    fn_df = test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)]
    fp_df = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)]
    tn_df = test_df[(test_df["label"] == 0) & (test_df["pred"] == 0)]

    print(f"\n  TP: {len(tp_df):,}  FN: {len(fn_df):,}")
    print(f"  FP: {len(fp_df):,}  TN: {len(tn_df):,}")

    # ─── 질문 1: TP Grad-CAM vs Seg 마스크 ──────────────────
    print(f"\n{'═' * 60}")
    print("  [질문1] TP: Seg 마스크와 히트맵 비교")
    print(f"{'═' * 60}")

    # 다양한 확률 범위에서 샘플 선택 (고확률 / 저확률 TP)
    tp_sorted = tp_df.sort_values("prob", ascending=False)
    tp_high = tp_sorted.head(NUM_SAMPLES // 2)
    tp_low = tp_sorted.tail(NUM_SAMPLES // 2)
    tp_samples = pd.concat([tp_high, tp_low]).to_dict("records")

    tp_ious = visualize_samples(tp_samples, "TP", GRADCAM_DIR, gradcam, device)
    mean_iou = np.mean([x["iou"] for x in tp_ious]) if tp_ious else 0
    print(f"  평균 IoU (CAM ↔ Seg): {mean_iou:.3f}")

    # 전체 TP IoU 분포 (서브샘플)
    print("  전체 TP IoU 계산 중 (최대 200개 서브샘플)...")
    tp_sub = tp_df.sample(n=min(200, len(tp_df)), random_state=42)
    all_tp_ious = []
    for _, row in tp_sub.iterrows():
        tensor, _ = load_slice_tensor(row["filename"])
        cam, prob = gradcam(tensor, device)
        seg = load_seg_mask(row["patient_id"], int(row["slice_idx"]))
        iou = compute_iou(cam, seg)
        all_tp_ious.append({"filename": row["filename"], "prob": prob, "iou": iou})
    plot_iou_distribution(all_tp_ious, GRADCAM_DIR)
    print(f"  전체 TP 평균 IoU: {np.mean([x['iou'] for x in all_tp_ious]):.3f}")

    # ─── 질문 2: FN 분석 ─────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  [질문2] FN ({len(fn_df):,}개): 왜 놓쳤는가?")
    print(f"{'═' * 60}")

    # 확률이 threshold에 가장 가까운 (경계선) 샘플 + 가장 먼 샘플
    fn_sorted = fn_df.sort_values("prob", ascending=False)
    fn_border = fn_sorted.head(NUM_SAMPLES // 2)  # threshold 근처
    fn_far = fn_sorted.tail(NUM_SAMPLES // 2)     # 매우 낮은 확률
    fn_samples = pd.concat([fn_border, fn_far]).to_dict("records")

    fn_ious = visualize_samples(fn_samples, "FN", GRADCAM_DIR, gradcam, device)

    # FN 종양 크기 분석 (서브샘플)
    fn_sub = fn_df.sample(n=min(300, len(fn_df)), random_state=42)
    tumor_fracs = plot_fn_tumor_size(fn_sub, GRADCAM_DIR)
    small_tumor = sum(1 for f in tumor_fracs if f < 0.01)
    print(f"  종양 픽셀 비율 < 1% 인 FN: {small_tumor}/{len(tumor_fracs)} "
          f"({small_tumor/max(len(tumor_fracs),1)*100:.1f}%)")

    # ─── 질문 3: FP 분석 ─────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  [질문3] FP ({len(fp_df):,}개): 무엇을 착각했는가?")
    print(f"{'═' * 60}")

    fp_sorted = fp_df.sort_values("prob", ascending=False)
    fp_high = fp_sorted.head(NUM_SAMPLES // 2)   # 가장 확신한 오탐
    fp_low = fp_sorted.tail(NUM_SAMPLES // 2)    # 경계선 오탐
    fp_samples = pd.concat([fp_high, fp_low]).to_dict("records")

    fp_ious = visualize_samples(fp_samples, "FP", GRADCAM_DIR, gradcam, device)

    # ─── FN/FP 확률 분포 비교 ────────────────────────────────
    plot_prob_comparison(fn_df["prob"].tolist(), fp_df["prob"].tolist(), GRADCAM_DIR)

    # ─── 결과 저장 ───────────────────────────────────────────
    results = {
        "총_테스트_샘플": len(test_df),
        "TP": len(tp_df), "FN": len(fn_df),
        "FP": len(fp_df), "TN": len(tn_df),
        "TP_평균_IoU": round(np.mean([x["iou"] for x in all_tp_ious]), 4),
        "FN_작은종양_비율": round(small_tumor / max(len(tumor_fracs), 1), 4),
        "FN_확률_평균": round(fn_df["prob"].mean(), 4),
        "FP_확률_평균": round(fp_df["prob"].mean(), 4),
    }
    results_path = GRADCAM_DIR / "gradcam_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ─── 최종 요약 ───────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Grad-CAM 분석 최종 요약")
    print(f"{'═' * 60}")
    print(f"  [질문1] TP 평균 IoU (CAM↔Seg): {results['TP_평균_IoU']:.4f}")
    print(f"  [질문2] FN 중 작은 종양 비율:  {results['FN_작은종양_비율']*100:.1f}%")
    print(f"           FN 평균 예측 확률:     {results['FN_확률_평균']:.4f}")
    print(f"  [질문3] FP 평균 예측 확률:     {results['FP_확률_평균']:.4f}")
    print(f"{'─' * 60}")
    print(f"  결과 저장: {results_path}")
    print(f"  시각화 저장: {GRADCAM_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
