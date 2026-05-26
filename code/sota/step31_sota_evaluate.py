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

import os
import sys
import gc
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import hashlib
import subprocess
import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score, roc_auc_score, brier_score_loss,
)
try:
    from scipy.ndimage import (
        label as cc_label,
        binary_closing,
        distance_transform_edt,
    )
    _HAS_SCIPY = True
except Exception:  # scipy 미설치 환경 대비
    cc_label = None
    binary_closing = None
    distance_transform_edt = None
    _HAS_SCIPY = False

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# CUDA 메모리 단편화 완화 옵션.
# 주의: `expandable_segments:True` 는 Windows 에서 공식 지원되지 않으므로
# (PyTorch CUDACachingAllocator 가 Linux 전용 mmap/UVA 경로를 사용) Windows 에서는
# 해당 옵션을 켜면 ~TensorImpl 시 Illegal Memory Access / Allocator 손상이 발생합니다.
# 따라서 OS 에 따라 분기해서 적용합니다.
# OOM 다발 환경 대응을 위해 max_split_size_mb 도 더 작게 (64) 잡는다.
if sys.platform == "win32":
    # Windows: expandable_segments 비활성. split-size 만 줄여 단편화를 완화.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "max_split_size_mb:64",
    )
else:
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64",
    )

# cudnn benchmark 는 입력 shape 가 매번 같을 때 빨라지지만, 평가 시 workspace 가
# 추가로 잡혀 OOM 의 원인이 될 수 있으므로 평가 스크립트에서는 끈다.
try:
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

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


def _autocast_ctx(device, enabled=True):
    """CUDA 일 때만 fp16 autocast, 그 외에는 nullcontext."""
    if enabled and device.type == "cuda":
        try:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        except (AttributeError, TypeError):
            # 구버전 PyTorch 대응
            return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def predict_with_tta(model, imgs,
                     tta_modes=("identity", "hflip", "vflip", "rot180"),
                     return_identity_logit=False,
                     use_amp=True):
    """
    TTA 예측. 메모리 절약을 위해 stack 대신 running-mean(누적합) 사용.
    return_identity_logit=True 면 identity 모드의 원본 cls logit 도 같이 반환
    (Temperature Scaling 용 raw logit 을 추가 forward 없이 얻기 위함).

    누적 버퍼(cls_logit_sum/seg_sum)는 첫 forward 직후 CPU 로 옮겨서, 다음 forward
    의 중간 활성과 동시에 GPU VRAM 에 살아있지 않도록 한다. (Windows 8GB VRAM 환경
    대응)

    [P4 — A-1 부분 채택, VRAM 동일] cls 측 누적은 *logit 기준* 으로 수행하고
    마지막 한 번만 sigmoid 한다 ("logit-mean"). seg 측은 단조 사상으로 큰 차이가
    없고 binary mask 산출 시점이 별도이므로 종전과 같이 sigmoid 후 평균을 유지.
    """
    cls_logit_sum = None  # CPU float32 (logit-mean 누적)
    seg_sum = None        # CPU float32 (sigmoid 후 평균 — seg)
    identity_logit = None # CPU float32
    n = float(len(tta_modes))
    device = imgs.device
    is_cuda = (device.type == "cuda")
    for m in tta_modes:
        x_t = _apply_tta(imgs, m)
        with _autocast_ctx(device, enabled=use_amp):
            cls_out, seg_out, _ = model(x_t, return_aux=False)
        # autocast 이후 fp32 로 승격해서 CPU 로 옮김 (cls 는 sigmoid 하지 않음)
        cls_logit_1d = cls_out.squeeze(1).float()
        cl_cpu = cls_logit_1d.detach().to("cpu", copy=True)
        if m == "identity":
            # raw identity logit (Temperature Scaling 용) 은 CPU 로 별도 보관
            identity_logit = cl_cpu.clone()
        seg_p_gpu = _inverse_tta(torch.sigmoid(seg_out.float()), m)
        seg_p = seg_p_gpu.detach().to("cpu", copy=True)
        # CPU 누적 — cls 는 logit, seg 는 prob
        cls_logit_sum = cl_cpu if cls_logit_sum is None else cls_logit_sum + cl_cpu
        seg_sum = seg_p if seg_sum is None else seg_sum + seg_p
        # GPU 측 중간 텐서 즉시 해제. seg_p_gpu 까지 명시적으로 del 해야
        # 다음 forward 의 활성과 GPU 에 동시에 살아있지 않음.
        del x_t, cls_out, seg_out, cls_logit_1d, cl_cpu, seg_p_gpu, seg_p
        # 비동기 커널이 끝난 뒤 free 가 일어나야 allocator 가 안전.
        # 매 TTA step 마다 synchronize 한 번으로 손상된 주소 재사용 방지.
        if is_cuda:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    # cls: logit-mean → 마지막에 단 한 번 sigmoid
    cls_mean_logit = cls_logit_sum / n
    cls_mean = torch.sigmoid(cls_mean_logit)
    seg_mean = seg_sum / n
    if return_identity_logit:
        return cls_mean, seg_mean, identity_logit
    return cls_mean, seg_mean


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
def _free_cuda(sync: bool = True):
    """CUDA 캐시/임시 텐서를 적극적으로 회수.

    sync=True 면 empty_cache 전에 synchronize 를 한 번 호출해
    아직 끝나지 않은 비동기 커널 때문에 allocator 가 잘못된 주소를
    재사용하는 것(=Illegal Memory Access / 알 수 없이 죽는 현상)을 방지한다.
    """
    if torch.cuda.is_available():
        try:
            if sync:
                # 미완 커널이 메모리 영역을 들고 있는 상태에서 free 하면
                # 다음 alloc 이 손상된 주소를 잡아 step31 이 조용히 죽는다.
                torch.cuda.synchronize()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# PyTorch 버전에 따라 OutOfMemoryError 가 없을 수 있어 안전 가드.
_CUDA_OOM_TYPES = (RuntimeError,)
if hasattr(torch.cuda, "OutOfMemoryError"):
    _CUDA_OOM_TYPES = (RuntimeError, torch.cuda.OutOfMemoryError)  # type: ignore[attr-defined]


def _forward_batch(model, imgs, use_tta, use_amp=True):
    """한 배치 forward. (cls_probs, seg_probs, identity_logit) 반환 — 모두 CPU 텐서."""
    device = imgs.device
    if use_tta:
        # predict_with_tta 가 이미 CPU 텐서를 반환
        cls_probs, seg_probs, id_logit = predict_with_tta(
            model, imgs, return_identity_logit=True, use_amp=use_amp,
        )
        return cls_probs, seg_probs, id_logit
    else:
        with _autocast_ctx(device, enabled=use_amp):
            cls_logit, seg_out, _ = model(imgs, return_aux=False)
        cls_logit = cls_logit.squeeze(1).float()
        # 곧바로 CPU 로 분리 → GPU 누수 방지
        cls_logit_cpu = cls_logit.detach().to("cpu", copy=True)
        seg_probs_gpu = torch.sigmoid(seg_out.float())
        seg_probs = seg_probs_gpu.detach().to("cpu", copy=True)
        cls_probs = torch.sigmoid(cls_logit_cpu)
        id_logit = cls_logit_cpu
        # GPU 텐서를 모두 명시적으로 떨궈서 다음 배치 alloc 시
        # 잔여 활성과 충돌하지 않게 한다.
        del cls_logit, seg_out, seg_probs_gpu
        if device.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        return cls_probs, seg_probs, id_logit


def _run_batch_safely(model, imgs, use_tta, device, use_amp=True):
    """
    CUDA OOM / 런타임 에러를 받아 (a) 캐시 비우고 재시도 → (b) 절반 분할 → (c) CPU 폴백.
    """
    try:
        with torch.inference_mode():
            return _forward_batch(model, imgs, use_tta, use_amp=use_amp)
    except _CUDA_OOM_TYPES as e:
        msg = str(e).lower()
        is_cuda_err = (
            ("cuda" in msg) or ("out of memory" in msg) or ("cublas" in msg)
            or ("accelerator" in msg)
            or (hasattr(torch.cuda, "OutOfMemoryError")
                and isinstance(e, torch.cuda.OutOfMemoryError))  # type: ignore[attr-defined]
        )
        if not is_cuda_err:
            raise
        print(f"  [WARN] CUDA 런타임 에러 감지 → 캐시 비우고 재시도: {e}")
        _free_cuda()

    # (1) 캐시 비우고 한 번 더 시도
    try:
        with torch.inference_mode():
            return _forward_batch(model, imgs, use_tta, use_amp=use_amp)
    except _CUDA_OOM_TYPES as e:
        print(f"  [WARN] 재시도 실패 → 배치 분할 시도: {e}")
        _free_cuda()

    # (2) 배치 절반씩 쪼개서 시도
    if imgs.size(0) > 1:
        mid = imgs.size(0) // 2
        outs1 = _run_batch_safely(model, imgs[:mid], use_tta, device, use_amp=use_amp)
        outs2 = _run_batch_safely(model, imgs[mid:], use_tta, device, use_amp=use_amp)
        return (
            torch.cat([outs1[0], outs2[0]], dim=0),
            torch.cat([outs1[1], outs2[1]], dim=0),
            torch.cat([outs1[2], outs2[2]], dim=0),
        )

    # (3) 단일 샘플도 안 되면 CPU 로 폴백
    print("  [WARN] CUDA 단일 샘플 forward 도 실패 → CPU 폴백")
    _free_cuda()
    model_cpu = model.to("cpu")
    try:
        with torch.inference_mode():
            # CPU 폴백 시에는 autocast off
            outs = _forward_batch(model_cpu, imgs.cpu(), use_tta, use_amp=False)
    finally:
        if device.type == "cuda":
            try:
                model.to(device)
            except Exception as e:
                print(f"  [WARN] 모델을 다시 GPU로 옮기지 못함 → 이후 CPU로 진행: {e}")
    return outs


def collect_predictions(model, loader, device,
                        use_tta=True, return_logits=False, use_amp=True):
    """
    배치별 예측 수집. 핵심 원칙:

      * GPU(VRAM, 8GB) 은 *오직* 모델 forward 만 담당.
      * cls_probs / seg_probs / id_logit / labels / masks 는 forward 직후
        곧바로 CPU(numpy) 로 넘긴다. (이미 _forward_batch / predict_with_tta
        에서 .to("cpu") 처리됨)
      * GPU 측 입력(imgs) 도 numpy 변환 직후 즉시 del + synchronize.
      * 시스템 RAM 사용량 절감을 위해 누적 시 fp16 numpy 사용 (필요 시 fp32
        로 다시 promote).
    """
    model.eval()
    all_labels, all_probs, all_logits = [], [], []
    all_seg_gt, all_seg_pred = [], []
    is_cuda = (device.type == "cuda")
    # Windows + 8GB VRAM 환경에서는 매 배치마다 캐시 비우기가 안전.
    free_every = 1 if (sys.platform == "win32" and is_cuda) else 5
    for batch_idx, (imgs, labels, masks) in enumerate(loader):
        imgs_gpu = imgs.to(device, non_blocking=True)
        cls_probs, seg_probs, id_logit = _run_batch_safely(
            model, imgs_gpu, use_tta, device, use_amp=use_amp,
        )
        # GPU 입력은 forward 가 끝났으니 지체 없이 해제.
        del imgs_gpu
        if is_cuda:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # cls_probs / seg_probs / id_logit 은 이미 CPU 텐서 → numpy 변환.
        # 시스템 RAM 사용량을 줄이기 위해 fp16 으로 누적한다.
        all_labels.append(np.asarray(labels.numpy(), dtype=np.float32))
        all_probs.append(cls_probs.numpy().astype(np.float16, copy=False))
        all_logits.append(id_logit.numpy().astype(np.float16, copy=False))
        all_seg_gt.append(masks.numpy().astype(np.float16, copy=False))
        all_seg_pred.append(seg_probs.to(torch.float16).numpy())

        # CPU 텐서 핸들도 즉시 del 해서 GC 가 빨리 회수하게 함.
        del cls_probs, seg_probs, id_logit, imgs, labels, masks

        if (batch_idx + 1) % free_every == 0:
            _free_cuda()
    _free_cuda()

    # 최종 합치기. 메트릭/시각화는 fp32 가 안전하므로 promote.
    labels_np = np.concatenate(all_labels).astype(np.float32)
    probs_np = np.concatenate(all_probs).astype(np.float32)
    seg_gt_np = np.concatenate(all_seg_gt).astype(np.float32)
    seg_pred_np = np.concatenate(all_seg_pred).astype(np.float32)

    # 누적 리스트 즉시 해제 (concatenate 직후 RAM 피크 완화).
    del all_labels, all_probs, all_seg_gt, all_seg_pred
    gc.collect()

    out = {
        "labels": labels_np,
        "probs": probs_np,
        "seg_gt": seg_gt_np,
        "seg_pred": seg_pred_np,
    }
    if return_logits:
        out["logits"] = np.concatenate(all_logits).astype(np.float32)
    del all_logits
    gc.collect()
    return out


def compute_seg_metrics(seg_gt: np.ndarray, seg_pred: np.ndarray,
                        threshold: float = 0.5):
    """region 별 dice/iou.

    seg_gt / seg_pred shape: (N, C, H, W). C 가 1~3 가운데 어떤 값이든 대응 —
    region 이름은 ["WT", "TC", "ET"] 첫 C 개만 사용. (P2 의 single-region
    호출에서도 안전하게 동작.)
    """
    regions = ["WT", "TC", "ET"]
    n_ch = int(seg_gt.shape[1])
    result = {}
    for r in range(min(n_ch, len(regions))):
        name = regions[r]
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


# ─── P2: Per-region threshold sweep (A-4) ────────────────
def sweep_threshold_per_region(val_seg_gt: np.ndarray, val_seg_pred: np.ndarray,
                               grid: np.ndarray = None):
    """val 에서 region 별 best threshold (dice 기준) 를 찾아 dict 로 반환.

    forward 추가 없이 raw npz 위에서만 동작 — CPU/numpy.
    """
    if grid is None:
        grid = np.linspace(0.30, 0.70, 21)
    best = {}
    for r, name in enumerate(["WT", "TC", "ET"]):
        best_t, best_d = 0.5, -1.0
        for t in grid:
            dices = []
            for i in range(val_seg_gt.shape[0]):
                g = (val_seg_gt[i, r] > 0.5).astype(np.float32)
                p = (val_seg_pred[i, r] > t).astype(np.float32)
                if g.sum() == 0:
                    continue
                inter = (g * p).sum()
                dices.append(2.0 * inter / max(g.sum() + p.sum(), 1e-8))
            d = float(np.mean(dices)) if dices else 0.0
            if d > best_d:
                best_t, best_d = float(t), d
        best[name] = {"threshold": best_t, "val_dice": best_d}
    return best


def sweep_threshold_cls(val_labels: np.ndarray, val_probs: np.ndarray,
                        grid: np.ndarray = None):
    """cls 도 val F1-optimal threshold 한 줄 산출 (A-4 부록)."""
    if grid is None:
        grid = np.linspace(0.30, 0.70, 41)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (val_probs >= t).astype(int)
        try:
            f1 = f1_score(val_labels, pred)
        except Exception:
            f1 = 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return {"threshold": best_t, "val_f1": best_f1}


def plot_threshold_sweep(val_seg_gt: np.ndarray, val_seg_pred: np.ndarray,
                         out_dir: Path,
                         grid: np.ndarray = None):
    """region 별 dice vs threshold 곡선 → PNG 3장."""
    import matplotlib.pyplot as plt
    if grid is None:
        grid = np.linspace(0.30, 0.70, 21)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, name in enumerate(["WT", "TC", "ET"]):
        curve = []
        for t in grid:
            dices = []
            for i in range(val_seg_gt.shape[0]):
                g = (val_seg_gt[i, r] > 0.5).astype(np.float32)
                p = (val_seg_pred[i, r] > t).astype(np.float32)
                if g.sum() == 0:
                    continue
                inter = (g * p).sum()
                dices.append(2.0 * inter / max(g.sum() + p.sum(), 1e-8))
            curve.append(float(np.mean(dices)) if dices else 0.0)
        curve = np.array(curve)
        best_idx = int(np.argmax(curve))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, curve, marker="o", lw=2)
        ax.axvline(grid[best_idx], color="r", ls="--",
                   label=f"best t={grid[best_idx]:.2f}, dice={curve[best_idx]:.4f}")
        ax.set_xlabel("threshold")
        ax.set_ylabel(f"val dice ({name})")
        ax.set_title(f"Per-region threshold sweep — {name}")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"threshold_sweep_{name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()


# ─── P3: Post-processing — CC filter + 3x3 closing (A-5) ──
def postproc_seg(seg_pred_bin: np.ndarray, min_size: int = 10,
                 do_close: bool = True, kernel: int = 3) -> np.ndarray:
    """seg_pred_bin: (N, R, H, W) 0/1 array. CC < min_size 제거 + 3x3 closing."""
    if not _HAS_SCIPY:
        print("  [WARN] scipy 미설치 → postproc 건너뜀. 원본 binary 반환.")
        return seg_pred_bin.copy()
    out = np.zeros_like(seg_pred_bin)
    struct = np.ones((kernel, kernel), dtype=bool)
    for i in range(seg_pred_bin.shape[0]):
        for r in range(seg_pred_bin.shape[1]):
            m = seg_pred_bin[i, r].astype(bool)
            if do_close:
                m = binary_closing(m, structure=struct, iterations=1)
            lab, n = cc_label(m)
            if n == 0:
                continue
            keep = np.zeros_like(m, dtype=bool)
            for k in range(1, n + 1):
                cc = (lab == k)
                if cc.sum() >= min_size:
                    keep |= cc
            out[i, r] = keep.astype(seg_pred_bin.dtype)
    return out


def plot_postproc_examples(seg_gt: np.ndarray, seg_pred_bin: np.ndarray,
                           seg_pred_pp: np.ndarray, out_path: Path,
                           n: int = 5):
    """후처리로 dice 가 크게 개선된 슬라이스를 5쌍 비교 시각화."""
    import matplotlib.pyplot as plt
    def _d(g, p):
        g = (g > 0.5).astype(np.float32); p = (p > 0.5).astype(np.float32)
        if g.sum() == 0: return np.nan
        inter = (g * p).sum()
        return 2 * inter / max(g.sum() + p.sum(), 1e-8)
    deltas = []
    for i in range(seg_gt.shape[0]):
        d0 = _d(seg_gt[i, 0], seg_pred_bin[i, 0])
        d1 = _d(seg_gt[i, 0], seg_pred_pp[i, 0])
        if np.isnan(d0) or np.isnan(d1):
            deltas.append(-np.inf)
        else:
            deltas.append(d1 - d0)
    deltas = np.array(deltas)
    order = np.argsort(deltas)[::-1][:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for row, idx in enumerate(order):
        axes[row, 0].imshow(seg_gt[idx, 0], cmap="gray"); axes[row, 0].set_title(f"GT (idx={idx})")
        axes[row, 1].imshow(seg_pred_bin[idx, 0], cmap="gray"); axes[row, 1].set_title("Pred (raw)")
        axes[row, 2].imshow(seg_pred_pp[idx, 0], cmap="gray")
        axes[row, 2].set_title(f"Pred (PP) Δd={deltas[idx]:+.3f}")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── P5: Bootstrap 95% CI (C-3) ──────────────────────────
def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """percentile bootstrap. fn(y, p) 가 float 을 반환해야 함."""
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = np.empty(n_boot, dtype=np.float64)
    idx = np.arange(N)
    for b in range(n_boot):
        s = rng.choice(idx, N, replace=True)
        try:
            vals[b] = float(fn(y[s], p[s]))
        except Exception:
            vals[b] = np.nan
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# ─── P6: ECE / Brier / Reliability diagram (C-2) ─────────
def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = (idx == b)
        if m.sum() == 0:
            conf.append(np.nan); acc.append(np.nan); cnt.append(0); continue
        conf.append(p[m].mean()); acc.append(y[m].mean()); cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    conf, acc, cnt = reliability_bins(y, p, n_bins)
    N = cnt.sum()
    if N == 0:
        return float("nan")
    mask = cnt > 0
    return float(np.sum(cnt[mask] / N * np.abs(conf[mask] - acc[mask])))


def plot_reliability(labels: np.ndarray, probs_pre: np.ndarray,
                     probs_post: np.ndarray, T: float, out_path: Path,
                     n_bins: int = 10):
    """temp scaling 전/후 reliability diagram 1장."""
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, p, title in [
        (axes[0], probs_pre, f"Pre-Temp (ECE={ece_score(labels, probs_pre):.4f})"),
        (axes[1], probs_post, f"Post-Temp T={T:.3f} (ECE={ece_score(labels, probs_post):.4f})"),
    ]:
        conf, acc, cnt = reliability_bins(labels, p, n_bins)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        mask = cnt > 0
        ax.plot(conf[mask], acc[mask], "o-", lw=2, label="model")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── P7: HD95 / per-volume dice / sens-spec (C-1) ────────
def hd95_2d(g: np.ndarray, p: np.ndarray) -> float:
    """2D HD95 (외부 라이브러리 없이 distance_transform 만 사용)."""
    if not _HAS_SCIPY:
        return float("nan")
    g = g.astype(bool); p = p.astype(bool)
    if g.sum() == 0 or p.sum() == 0:
        return float("nan")
    dg = distance_transform_edt(~g)
    dp = distance_transform_edt(~p)
    d_gp = dg[p]
    d_pg = dp[g]
    if len(d_gp) == 0 or len(d_pg) == 0:
        return float("nan")
    return float(max(np.percentile(d_gp, 95), np.percentile(d_pg, 95)))


def sens_spec(g: np.ndarray, p: np.ndarray):
    g = (g > 0.5); p = (p > 0.5)
    tp = int(((g) & (p)).sum())
    fn = int(((g) & (~p)).sum())
    tn = int(((~g) & (~p)).sum())
    fp = int(((~g) & (p)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return float(sens), float(spec)


def _subject_id_from_fname(fname: str) -> str:
    """BraTS-GLI-XXXXX-000_slice123.png → BraTS-GLI-XXXXX-000."""
    base = Path(fname).stem
    # _slice 또는 마지막 _ 기준 분리
    if "_slice" in base:
        return base.split("_slice")[0]
    if "_" in base:
        return "_".join(base.split("_")[:-1])
    return base


def compute_seg_extra(seg_gt: np.ndarray, seg_pred_bin: np.ndarray,
                      filenames: list):
    """region 별 HD95, sens/spec, per-volume(subject) dice 산출."""
    extra = {}
    # 슬라이스 단위 indexing 을 subject 로 묶음
    subj_idx = {}
    if filenames is not None and len(filenames) == seg_gt.shape[0]:
        for i, fn in enumerate(filenames):
            sid = _subject_id_from_fname(str(fn))
            subj_idx.setdefault(sid, []).append(i)
    for r, name in enumerate(["WT", "TC", "ET"]):
        hd_list, sens_list, spec_list = [], [], []
        for i in range(seg_gt.shape[0]):
            g = (seg_gt[i, r] > 0.5)
            p = (seg_pred_bin[i, r] > 0.5)
            if g.sum() == 0:
                continue
            hd = hd95_2d(g, p)
            if not np.isnan(hd):
                hd_list.append(hd)
            s, sp = sens_spec(g, p)
            sens_list.append(s); spec_list.append(sp)
        # per-volume dice (subject 단위로 슬라이스 합산)
        per_vol = []
        for sid, idxs in subj_idx.items():
            g_sum = 0.0; p_sum = 0.0; inter = 0.0
            for i in idxs:
                g = (seg_gt[i, r] > 0.5).astype(np.float32)
                p = (seg_pred_bin[i, r] > 0.5).astype(np.float32)
                inter += float((g * p).sum())
                g_sum += float(g.sum()); p_sum += float(p.sum())
            if g_sum == 0:
                continue
            per_vol.append(2.0 * inter / max(g_sum + p_sum, 1e-8))
        extra[name] = {
            "hd95_mean":    float(np.mean(hd_list)) if hd_list else float("nan"),
            "hd95_median":  float(np.median(hd_list)) if hd_list else float("nan"),
            "sens_mean":    float(np.mean(sens_list)) if sens_list else float("nan"),
            "spec_mean":    float(np.mean(spec_list)) if spec_list else float("nan"),
            "per_volume_dice_mean":   float(np.mean(per_vol)) if per_vol else float("nan"),
            "per_volume_dice_median": float(np.median(per_vol)) if per_vol else float("nan"),
            "n_volumes": int(len(per_vol)),
        }
    return extra


# ─── P8: Failure case 30+30 시각화 (D-3) ─────────────────
def save_failure_panels(seg_gt: np.ndarray, seg_pred_bin: np.ndarray,
                        filenames: list, out_path: Path,
                        n: int = 30, mode: str = "worst"):
    """slice-wise WT dice 기준 worst/best n 개를 4-panel 로 저장."""
    import matplotlib.pyplot as plt
    # 양성 슬라이스만 대상
    dices = []
    for i in range(seg_gt.shape[0]):
        g = (seg_gt[i, 0] > 0.5).astype(np.float32)
        p = seg_pred_bin[i, 0].astype(np.float32)
        if g.sum() == 0:
            dices.append(np.nan); continue
        d = 2 * (g * p).sum() / max(g.sum() + p.sum(), 1e-8)
        dices.append(float(d))
    dices = np.array(dices, dtype=np.float64)
    valid = np.where(~np.isnan(dices))[0]
    if len(valid) == 0:
        print(f"  [WARN] failure panel: 양성 슬라이스 0건 → skip")
        return
    sub = dices[valid]
    order_local = np.argsort(sub) if mode == "worst" else np.argsort(sub)[::-1]
    sel = valid[order_local[:min(n, len(valid))]]

    # 이미지는 step27 경로에서 직접 로드 (forward 0회)
    try:
        from step27_sota_dataset import T1CE_DIR, FLAIR_DIR, _imread_gray
        load_imgs = True
    except Exception as e:
        print(f"  [WARN] step27 import 실패 → 이미지 패널 생략: {e}")
        load_imgs = False

    rows = len(sel)
    cols = 4 if load_imgs else 2
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row, idx in enumerate(sel):
        col = 0
        if load_imgs and filenames is not None:
            fn = filenames[idx]
            t1ce = _imread_gray(T1CE_DIR / fn)
            flair = _imread_gray(FLAIR_DIR / fn)
            if t1ce is None: t1ce = np.zeros((224, 224), dtype=np.uint8)
            if flair is None: flair = np.zeros((224, 224), dtype=np.uint8)
            axes[row, col].imshow(t1ce, cmap="gray"); axes[row, col].set_title("T1ce"); col += 1
            axes[row, col].imshow(flair, cmap="gray"); axes[row, col].set_title("FLAIR"); col += 1
        axes[row, col].imshow(seg_gt[idx, 0], cmap="gray")
        axes[row, col].set_title(f"GT (WT) d={dices[idx]:.3f}")
        col += 1
        axes[row, col].imshow(seg_pred_bin[idx, 0], cmap="gray")
        axes[row, col].set_title("Pred (WT)")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()


# ─── P9: Manifest helpers (D-2) ──────────────────────────
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception as e:
        return f"error:{e}"


def git_short_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_DIR), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


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
def _safe_torch_load(path, map_location):
    """torch 버전에 따라 weights_only 인자 유무가 다른 문제 회피."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_args():
    p = argparse.ArgumentParser(description="Step 31 SOTA Evaluation")
    p.add_argument("--batch-size", type=int, default=None,
                   help="평가 배치 크기 (기본: cuda=2, cpu=16). OOM 시 더 줄이세요.")
    p.add_argument("--cpu", action="store_true",
                   help="CUDA 가능하더라도 CPU 로 강제 실행 (OOM 회피용).")
    p.add_argument("--no-tta", action="store_true",
                   help="TTA(4× forward) 끄기. VRAM 부족할 때 유용.")
    p.add_argument("--no-amp", action="store_true",
                   help="FP16 autocast 끄기. 정밀도 우선 시 사용.")
    p.add_argument("--no-swa", action="store_true",
                   help="P10: SWA 체크포인트 추가 평가 건너뛰기.")
    p.add_argument("--no-failure-fig", action="store_true",
                   help="P8: failure case PNG 생성 건너뛰기 (이미지 로딩 시간 절감).")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="P5: bootstrap CI 횟수 (기본 1000).")
    return p.parse_args()


def _load_model_to(ckpt_path: Path, device: torch.device,
                   use_amp_flag: bool):
    """ckpt → model → device 안전 적재. (model, device, use_amp) 반환.

    device 가 cuda 인데 OOM 이면 CPU 로 폴백하면서 use_amp 도 False 로 강등.
    """
    use_amp = bool(use_amp_flag)
    try:
        model = create_sota_model(pretrained=False, in_ch=3, seg_classes=3)
        ckpt = _safe_torch_load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt
        gc.collect()
        if device.type == "cuda":
            _free_cuda()
            model = model.to(device)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            _free_cuda()
        model.eval()
        return model, device, use_amp
    except _CUDA_OOM_TYPES as e:
        msg = str(e).lower()
        if device.type == "cuda" and ("out of memory" in msg or "cuda" in msg
                                       or "accelerator" in msg):
            print(f"  [WARN] 모델 GPU 적재 중 OOM → CPU 로 폴백: {e}")
            _free_cuda()
            device = torch.device("cpu")
            use_amp = False
            model = create_sota_model(pretrained=False, in_ch=3, seg_classes=3)
            ckpt = _safe_torch_load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            del ckpt
            model.eval()
            return model, device, use_amp
        raise


def evaluate_one_ckpt(ckpt_path: Path, device: torch.device,
                      val_l, test_l,
                      use_tta: bool, use_amp_flag: bool,
                      tag: str = "best"):
    """한 ckpt 에 대해 val→T 적합→test 예측을 1 회씩 forward.

    반환: dict — labels/probs_tta/probs_tscaled/logits/seg_gt/seg_pred/T/val_pack.
    GPU 사용은 본 함수 안에서만, 종료 시 _free_cuda().
    """
    print(f"\n  loading {ckpt_path}  (tag={tag})")
    model, device, use_amp = _load_model_to(ckpt_path, device, use_amp_flag)
    _free_cuda()

    print(f"  [{tag}/val] collecting raw logits (no TTA) ...")
    val_pack = collect_predictions(model, val_l, device,
                                   use_tta=False, return_logits=True,
                                   use_amp=use_amp)
    val_logits = torch.tensor(val_pack["logits"], dtype=torch.float32)
    val_labels = torch.tensor(val_pack["labels"], dtype=torch.float32)
    T = fit_temperature(val_logits, val_labels)
    print(f"  [{tag}] fitted Temperature T = {T:.4f}")

    print(f"  [{tag}/test] collecting predictions with TTA={use_tta} ...")
    test_pack = collect_predictions(model, test_l, device,
                                    use_tta=use_tta, return_logits=True,
                                    use_amp=use_amp)
    labels = test_pack["labels"]
    raw_logits = test_pack["logits"]
    probs_tta = test_pack["probs"]
    probs_tscaled = 1.0 / (1.0 + np.exp(-(raw_logits / T)))

    # model 해제 → 다음 ckpt 평가 (또는 후처리) 직전 GPU 비우기.
    del model
    _free_cuda()

    return {
        "tag": tag,
        "T": float(T),
        "labels": labels,
        "probs_tta": probs_tta,
        "probs_tscaled": probs_tscaled,
        "logits": raw_logits,
        "seg_gt": test_pack["seg_gt"],
        "seg_pred": test_pack["seg_pred"],
        "val_pack": val_pack,
        "val_logits": val_logits.numpy(),
        "val_labels": val_pack["labels"],
    }


def compute_cls_metrics_with_ci(labels: np.ndarray, probs: np.ndarray,
                                threshold: float = 0.5,
                                n_boot: int = 1000) -> dict:
    """P5: cls 지표 + bootstrap 95% CI."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    out = {
        "threshold": float(threshold),
        "f1":    float(f1_score(labels, preds)),
        "auroc": float(auc(*roc_curve(labels, probs)[:2])),
        "auprc": float(average_precision_score(labels, probs)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if n_boot and n_boot > 0:
        out["ci"] = {
            "auroc": bootstrap_ci(labels, probs, roc_auc_score,
                                  n_boot=n_boot, seed=0),
            "auprc": bootstrap_ci(labels, probs, average_precision_score,
                                  n_boot=n_boot, seed=1),
            "f1":    bootstrap_ci(labels,
                                  (probs >= threshold).astype(int),
                                  f1_score, n_boot=n_boot, seed=2),
            "n_boot": int(n_boot),
        }
    return out


def postprocess_block(eval_pack: dict,
                      val_seg_gt: np.ndarray, val_seg_pred: np.ndarray,
                      val_labels: np.ndarray, val_probs_pre_T: np.ndarray,
                      filenames: list, fig_dir: Path,
                      n_boot: int = 1000,
                      do_failure_fig: bool = True,
                      tag: str = "best") -> dict:
    """P1·P2·P3·P5·P6·P7·P8 — CPU/numpy 후처리 일습. forward 0회."""
    labels = eval_pack["labels"]
    probs_tta = eval_pack["probs_tta"]
    probs_tscaled = eval_pack["probs_tscaled"]
    seg_gt = eval_pack["seg_gt"]
    seg_pred = eval_pack["seg_pred"]
    T = eval_pack["T"]

    # ── cls 지표 + bootstrap CI ────────────────────────────
    cls_metrics = {
        "tta":         compute_cls_metrics_with_ci(labels, probs_tta,
                                                   threshold=0.5, n_boot=n_boot),
        "temp_scaled": compute_cls_metrics_with_ci(labels, probs_tscaled,
                                                   threshold=0.5, n_boot=n_boot),
    }

    # cls val F1-optimal threshold (A-4 부록)
    cls_thr = sweep_threshold_cls(val_labels, val_probs_pre_T)
    cls_metrics["tta_thrF1"] = compute_cls_metrics_with_ci(
        labels, probs_tta, threshold=cls_thr["threshold"], n_boot=n_boot,
    )
    cls_metrics["tta_thrF1"]["val_f1_at_best_thr"] = cls_thr["val_f1"]

    # ── P6: calibration (ECE/Brier/Reliability) ───────────
    calibration = {
        "pre_temp":  {
            "ece":   ece_score(labels, probs_tta),
            "brier": float(brier_score_loss(labels, probs_tta)),
        },
        "post_temp": {
            "ece":   ece_score(labels, probs_tscaled),
            "brier": float(brier_score_loss(labels, probs_tscaled)),
        },
        "T": float(T),
    }
    plot_reliability(labels, probs_tta, probs_tscaled, T,
                     fig_dir / f"reliability_pre_post_{tag}.png")

    # ── seg @ threshold 0.5 (baseline) ────────────────────
    seg_metrics_05 = compute_seg_metrics(seg_gt, seg_pred, threshold=0.5)

    # ── P2: per-region threshold sweep on val ─────────────
    print(f"  [{tag}] P2 — per-region threshold sweep on val ...")
    best_thr = sweep_threshold_per_region(val_seg_gt, val_seg_pred)
    plot_threshold_sweep(val_seg_gt, val_seg_pred, fig_dir)
    seg_metrics_swept = {}
    seg_pred_bin = np.zeros_like(seg_pred, dtype=np.uint8)
    for r, name in enumerate(["WT", "TC", "ET"]):
        t = best_thr[name]["threshold"]
        seg_pred_bin[:, r] = (seg_pred[:, r] > t).astype(np.uint8)
        sub = compute_seg_metrics(seg_gt[:, [r]],
                                  seg_pred[:, [r]],
                                  threshold=t)
        # compute_seg_metrics 는 첫 region 을 "WT" 라벨로 반환 — 값만 꺼냄
        first = list(sub.values())[0]
        seg_metrics_swept[name] = {**first, "threshold": float(t)}

    # ── P3: postproc (CC + closing) ───────────────────────
    print(f"  [{tag}] P3 — postprocessing (CC + closing) ...")
    seg_pred_pp = postproc_seg(seg_pred_bin, min_size=10,
                               do_close=True, kernel=3)
    seg_metrics_pp = compute_seg_metrics(seg_gt,
                                         seg_pred_pp.astype(np.float32),
                                         threshold=0.5)
    plot_postproc_examples(seg_gt, seg_pred_bin, seg_pred_pp,
                           fig_dir / f"postproc_examples_{tag}.png", n=5)

    # ── P7: HD95 / per-volume / sens-spec ─────────────────
    print(f"  [{tag}] P7 — HD95 / per-volume / sens-spec ...")
    seg_extra = compute_seg_extra(seg_gt, seg_pred_pp, filenames)

    # ── P8: failure case 30 + 30 ──────────────────────────
    if do_failure_fig:
        print(f"  [{tag}] P8 — failure worst30 / best30 ...")
        save_failure_panels(seg_gt, seg_pred_pp, filenames,
                            fig_dir / f"failure_worst30_{tag}.png",
                            n=30, mode="worst")
        save_failure_panels(seg_gt, seg_pred_pp, filenames,
                            fig_dir / f"failure_best30_{tag}.png",
                            n=30, mode="best")

    return {
        "cls":            cls_metrics,
        "calibration":    calibration,
        "seg":            seg_metrics_05,
        "seg_swept":      seg_metrics_swept,
        "seg_postproc":   seg_metrics_pp,
        "seg_extra":      seg_extra,
        "thresholds": {
            "cls_f1_opt": cls_thr,
            "seg_region": best_thr,
        },
        "_arrays_for_diag": {
            "seg_pred_bin": seg_pred_bin,
            "seg_pred_pp":  seg_pred_pp,
        },
    }


def main():
    args = _parse_args()

    print("=" * 60)
    print("  Step 31 (Day 7 / sota): SOTA Evaluation")
    print("=" * 60)

    # CUDA 사용 가능해도 가벼운 sanity check 후 device 결정
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"  device: {device}")
    else:
        device = torch.device("cuda")
        try:
            # 시작 전에 캐시 비우기
            torch.cuda.empty_cache()
            torch.zeros(1, device=device) + 1
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"  device: {device}  ({torch.cuda.get_device_name(0)})  "
                  f"free={free_b/1024**3:.2f}GB / total={total_b/1024**3:.2f}GB")
            if free_b < 1.5 * 1024**3:
                print(f"  [WARN] free VRAM < 1.5GB → CPU 폴백 권장. 그래도 GPU로 진행합니다.")
        except Exception as e:
            print(f"  [WARN] CUDA sanity check 실패 → CPU 폴백: {e}")
            device = torch.device("cpu")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 평가는 TTA 로 4× forward + seg 4개 누적이 들어가므로 학습 때보다 batch 를 작게.
    # 좁은 VRAM(예: 8GB 미만, 다른 프로세스가 점유 중) 환경에서도 안정적으로
    # 돌도록 기본값을 매우 보수적으로 잡는다. CLI 로 override 가능.
    if args.batch_size is not None:
        eval_batch_size = max(1, int(args.batch_size))
    else:
        eval_batch_size = 2 if device.type == "cuda" else 16
    use_tta = not args.no_tta
    use_amp_flag = (not args.no_amp) and (device.type == "cuda")
    train_l, val_l, test_l, _ = create_sota_dataloaders(
        batch_size=eval_batch_size, num_workers=0,
        use_weighted_sampler=False, channel_mode="t1ce_flair_diff",
    )
    print(f"  eval batch_size: {eval_batch_size}  |  TTA: {use_tta}  |  AMP(fp16): {use_amp_flag}")

    # test split 의 파일명 리스트 (P7 per-volume / P8 failure 시각화 용).
    # forward 0 회 — splits.csv 만 한 번 더 읽음.
    test_filenames = None
    try:
        import pandas as pd
        from step27_sota_dataset import SPLITS_CSV
        _df = pd.read_csv(SPLITS_CSV)
        test_filenames = _df[_df["split"] == "test"]["filename"].tolist()
        print(f"  test filenames: {len(test_filenames)} (from {SPLITS_CSV.name})")
    except Exception as e:
        print(f"  [WARN] test filenames 로딩 실패 → per-volume / failure 패널 부분 제한: {e}")

    ckpt_path = CKPT_DIR / "sota_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint 없음: {ckpt_path}")
        return

    # ───────────────────────────────────────────────────────
    # 1) BEST 평가 (forward 1회 / val+test)
    # ───────────────────────────────────────────────────────
    best_pack = evaluate_one_ckpt(
        ckpt_path, device, val_l, test_l,
        use_tta=use_tta, use_amp_flag=use_amp_flag, tag="best",
    )
    # 최종 device 는 evaluate_one_ckpt 안에서 폴백됐을 수 있음 (use_amp 도 동일).
    # 다만 후처리는 CPU 만 사용하므로 별도 영향 없음.
    T_best = best_pack["T"]

    # ── P1: raw npz 직렬화 ────────────────────────────────
    raw_test_path = LOG_DIR / "sota_test_raw.npz"
    np.savez_compressed(
        raw_test_path,
        labels=best_pack["labels"],
        probs_tta=best_pack["probs_tta"],
        probs_tscaled=best_pack["probs_tscaled"],
        logits=best_pack["logits"],
        seg_gt=best_pack["seg_gt"],
        seg_pred=best_pack["seg_pred"],
        T=np.array([T_best], dtype=np.float32),
    )
    raw_val_path = LOG_DIR / "sota_val_raw.npz"
    np.savez_compressed(
        raw_val_path,
        labels=best_pack["val_labels"],
        logits=best_pack["val_logits"],
        seg_gt=best_pack["val_pack"]["seg_gt"],
        seg_pred=best_pack["val_pack"]["seg_pred"],
    )
    print(f"  [P1] raw saved: {raw_test_path.name} "
          f"({raw_test_path.stat().st_size/1024/1024:.1f} MB), "
          f"{raw_val_path.name} "
          f"({raw_val_path.stat().st_size/1024/1024:.1f} MB)")

    # ── P2/P3/P5/P6/P7/P8: 후처리 일습 (CPU only) ─────────
    val_probs_pre_T = 1.0 / (1.0 + np.exp(-best_pack["val_logits"]))
    best_post = postprocess_block(
        best_pack,
        val_seg_gt=best_pack["val_pack"]["seg_gt"],
        val_seg_pred=best_pack["val_pack"]["seg_pred"],
        val_labels=best_pack["val_labels"],
        val_probs_pre_T=val_probs_pre_T,
        filenames=test_filenames,
        fig_dir=FIG_DIR,
        n_boot=int(args.n_boot),
        do_failure_fig=not args.no_failure_fig,
        tag="best",
    )

    print("\n  [Seg metrics — best @ thr=0.5]")
    for r, m in best_post["seg"].items():
        print(f"    {r}: dice={m['dice_mean']:.4f}  iou={m['iou_mean']:.4f}  "
              f"n_pos={m['n_positive_slices']}")
    print("  [Seg metrics — best @ val-best threshold (P2)]")
    for r, m in best_post["seg_swept"].items():
        print(f"    {r}: dice={m['dice_mean']:.4f}  thr={m['threshold']:.3f}")
    print("  [Seg metrics — best + postproc (P3)]")
    for r, m in best_post["seg_postproc"].items():
        print(f"    {r}: dice={m['dice_mean']:.4f}  iou={m['iou_mean']:.4f}")

    # ── Conformal (기존) — best 만 적용 ────────────────────
    try:
        val_probs_for_cp = 1.0 / (1.0 + np.exp(-(best_pack["val_logits"] / T_best)))
        cp = try_conformal(val_probs_for_cp,
                           best_pack["val_labels"],
                           best_pack["probs_tscaled"], alpha=0.1)
        if "q" in cp:
            covered = 0
            for true_y, included in zip(best_pack["labels"], cp["sets"]):
                if int(true_y) in included:
                    covered += 1
            cp["coverage"] = covered / len(best_pack["labels"])
            cp["sets"] = cp["sets"][:50]
    except Exception as e:
        cp = {"error": str(e)}

    # ── 기존 4-panel diagnostics ──────────────────────────
    plot_diagnostics(best_pack["labels"], best_pack["probs_tta"],
                     best_pack["seg_gt"], best_pack["seg_pred"], FIG_DIR)

    # ───────────────────────────────────────────────────────
    # 2) (선택) SWA 평가 — P10
    # ───────────────────────────────────────────────────────
    swa_block = None
    swa_path = CKPT_DIR / "sota_swa.pth"
    if (not args.no_swa) and swa_path.exists():
        print(f"\n  [P10] also evaluating SWA: {swa_path}")
        try:
            swa_pack = evaluate_one_ckpt(
                swa_path, device, val_l, test_l,
                use_tta=use_tta, use_amp_flag=use_amp_flag, tag="swa",
            )
            val_probs_pre_T_swa = 1.0 / (1.0 + np.exp(-swa_pack["val_logits"]))
            swa_post = postprocess_block(
                swa_pack,
                val_seg_gt=swa_pack["val_pack"]["seg_gt"],
                val_seg_pred=swa_pack["val_pack"]["seg_pred"],
                val_labels=swa_pack["val_labels"],
                val_probs_pre_T=val_probs_pre_T_swa,
                filenames=test_filenames,
                fig_dir=FIG_DIR,
                n_boot=int(args.n_boot),
                do_failure_fig=False,  # 시간 절감 — best 와 큰 차이 없음
                tag="swa",
            )

            # best vs swa diagnostics overlay
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                # ROC
                fpr_b, tpr_b, _ = roc_curve(best_pack["labels"], best_pack["probs_tta"])
                fpr_s, tpr_s, _ = roc_curve(swa_pack["labels"], swa_pack["probs_tta"])
                axes[0].plot(fpr_b, tpr_b, lw=2,
                             label=f"best AUROC={auc(fpr_b, tpr_b):.4f}")
                axes[0].plot(fpr_s, tpr_s, lw=2, ls="--",
                             label=f"swa  AUROC={auc(fpr_s, tpr_s):.4f}")
                axes[0].plot([0, 1], [0, 1], "k--", lw=1)
                axes[0].set_title("ROC — best vs SWA"); axes[0].legend(); axes[0].grid(alpha=0.3)
                # PR
                p_b, r_b, _ = precision_recall_curve(best_pack["labels"], best_pack["probs_tta"])
                p_s, r_s, _ = precision_recall_curve(swa_pack["labels"], swa_pack["probs_tta"])
                axes[1].plot(r_b, p_b, lw=2,
                             label=f"best AUPRC={average_precision_score(best_pack['labels'], best_pack['probs_tta']):.4f}")
                axes[1].plot(r_s, p_s, lw=2, ls="--",
                             label=f"swa  AUPRC={average_precision_score(swa_pack['labels'], swa_pack['probs_tta']):.4f}")
                axes[1].set_title("PR — best vs SWA"); axes[1].legend(); axes[1].grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(FIG_DIR / "best_vs_swa_diagnostics.png",
                            dpi=150, bbox_inches="tight")
                plt.close()
            except Exception as e:
                print(f"  [WARN] best_vs_swa plot 실패: {e}")

            # raw 도 저장 (분석 재실행 용)
            np.savez_compressed(
                LOG_DIR / "sota_test_raw_swa.npz",
                labels=swa_pack["labels"], probs_tta=swa_pack["probs_tta"],
                probs_tscaled=swa_pack["probs_tscaled"], logits=swa_pack["logits"],
                seg_gt=swa_pack["seg_gt"], seg_pred=swa_pack["seg_pred"],
                T=np.array([swa_pack["T"]], dtype=np.float32),
            )

            swa_block = {
                "temperature":  swa_pack["T"],
                "cls":          swa_post["cls"],
                "calibration":  swa_post["calibration"],
                "seg":          swa_post["seg"],
                "seg_swept":    swa_post["seg_swept"],
                "seg_postproc": swa_post["seg_postproc"],
                "seg_extra":    swa_post["seg_extra"],
                "thresholds":   swa_post["thresholds"],
                "ckpt_sha256":  sha256_of(swa_path),
            }
        except Exception as e:
            print(f"  [WARN] SWA 평가 실패 → 건너뜀: {e}")
            swa_block = {"error": str(e)}
    elif args.no_swa:
        print(f"\n  [P10] --no-swa 플래그 → SWA 평가 건너뜀")
    else:
        print(f"\n  [P10] {swa_path} 없음 → SWA 평가 건너뜀")

    # ───────────────────────────────────────────────────────
    # 3) P9 — manifest (재현성 메타데이터)
    # ───────────────────────────────────────────────────────
    meta = {
        "ckpt_path":   str(ckpt_path),
        "ckpt_sha256": sha256_of(ckpt_path),
        "swa_ckpt_path":   str(swa_path) if swa_path.exists() else None,
        "swa_ckpt_sha256": sha256_of(swa_path) if swa_path.exists() else None,
        "git_rev": git_short_rev(),
        "tta_modes": list(("identity", "hflip", "vflip", "rot180"))
            if use_tta else ["identity"],
        "tta_aggregation_cls": "logit_mean",   # P4 적용 후
        "tta_aggregation_seg": "sigmoid_mean", # seg 는 그대로
        "temperature_T_best": float(T_best),
        "device":  str(device),
        "amp":     bool(use_amp_flag),
        "eval_batch_size": int(eval_batch_size),
        "n_boot":  int(args.n_boot),
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "python":  sys.version.split()[0],
        "torch":   torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "scipy_available": bool(_HAS_SCIPY),
    }

    # ───────────────────────────────────────────────────────
    # 4) 비교 블록 — Day5 / Day6 / Day7(best/swa)
    # ───────────────────────────────────────────────────────
    print(f"\n  [Multi-way 비교]")
    comparison = {}
    mtl_candidates = [MTL_PREV_LOG / "mt_test_metrics.json",
                      MTL_PREV_LOG / "mt_evaluation_results.json"]
    mtl_path = next((p for p in mtl_candidates if p.exists()), mtl_candidates[0])
    for tag, path in [("Day5_MTL", mtl_path),
                      ("Day6_MMMT", MMMT_PREV_LOG / "mmmt_test_metrics.json")]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                comparison[tag] = prev
                print(f"    {tag}: loaded from {path.name}")
            except Exception as e:
                print(f"  [WARN] {tag} 비교 실패: {e}")
                comparison[tag] = {"error": str(e)}
    comparison["Day7_SOTA_best"] = {
        "cls_tta":         best_post["cls"]["tta"],
        "cls_temp_scaled": best_post["cls"]["temp_scaled"],
        "seg":             best_post["seg"],
        "seg_postproc":    best_post["seg_postproc"],
    }
    if swa_block and "error" not in swa_block:
        comparison["Day7_SOTA_swa"] = {
            "cls_tta":      swa_block["cls"]["tta"],
            "seg":          swa_block["seg"],
            "seg_postproc": swa_block["seg_postproc"],
        }
    print(f"    Day7 SOTA   : {best_post['cls']['tta']}")

    # ───────────────────────────────────────────────────────
    # 5) 통합 json 저장
    # ───────────────────────────────────────────────────────
    final = {
        "n_test": int(len(best_pack["labels"])),
        "meta": meta,                                # P9
        "temperature": float(T_best),                # 기존 호환
        "cls":           best_post["cls"],           # P5 (ci 포함)
        "calibration":   best_post["calibration"],   # P6
        "seg":           best_post["seg"],
        "seg_swept":     best_post["seg_swept"],     # P2
        "seg_postproc":  best_post["seg_postproc"],  # P3
        "seg_extra":     best_post["seg_extra"],     # P7
        "thresholds":    best_post["thresholds"],
        "conformal":     cp,
        "swa":           swa_block,                  # P10
        "comparison":    comparison,
        "raw_files": {
            "test": str(raw_test_path),
            "val":  str(raw_val_path),
            "swa_test": str(LOG_DIR / "sota_test_raw_swa.npz")
                if (swa_block and "error" not in (swa_block or {})) else None,
        },
    }
    with open(LOG_DIR / "sota_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  metrics saved: {LOG_DIR / 'sota_test_metrics.json'}")
    print(f"  figures:       {FIG_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
