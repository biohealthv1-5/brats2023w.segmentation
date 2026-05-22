# -*- coding: utf-8 -*-
"""
Step 29 (Day 7 / sota): SOTA Loss & TumorCP 모듈
=================================================
v2ways.md SOTA Loss 진화 사다리 (§3.1~§3.5) 전부 통합:

- FocalTverskyLoss (Abraham & Khan, ISBI 2019)
- BoundaryLoss (Kervadec et al. MIDL 2019, soft 근사 — 자체 SDF distance map)
- CompoundSegLoss = w_tv * Focal-Tversky + w_bce * BCE + w_b * Boundary
- SOTAMultiTaskLoss : Uncertainty Weighting (Kendall 2018) + Deep Supervision
- TumorCP collate (v2ways §4.3 — 소종양 직접 carpe-paste augmentation)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────
#  Focal-Tversky Loss
# ──────────────────────────────────────────────────────────
class FocalTverskyLoss(nn.Module):
    """
    FN을 강하게 패널티: alpha=0.7 (FN), beta=0.3 (FP), gamma=4/3
    pred는 logit, target은 {0,1} binary (B, C, H, W) 또는 (B, H, W)
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0 / 3.0, smooth=1.0,
                 reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        if pred_logits.shape != target.shape:
            target = target.expand_as(pred_logits)
        p = torch.sigmoid(pred_logits)
        dims = (0, 2, 3) if p.dim() == 4 else (0, 1, 2)
        TP = (p * target).sum(dim=dims if p.dim() == 4 else None)
        FN = ((1 - p) * target).sum(dim=dims if p.dim() == 4 else None)
        FP = (p * (1 - target)).sum(dim=dims if p.dim() == 4 else None)
        tversky = (TP + self.smooth) / (TP + self.alpha * FN
                                        + self.beta * FP + self.smooth)
        loss = (1.0 - tversky) ** self.gamma
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────
#  Boundary Loss (Kervadec 2019)
# ──────────────────────────────────────────────────────────
def compute_sdf(mask_np: np.ndarray) -> np.ndarray:
    """Signed Distance Function (음수: 내부, 양수: 외부) — (H, W)"""
    posmask = mask_np.astype(bool)
    if posmask.any() and not posmask.all():
        negmask = ~posmask
        dist_out = distance_transform_edt(negmask)
        dist_in = distance_transform_edt(posmask)
        sdf = dist_out - dist_in
    else:
        sdf = np.zeros_like(mask_np, dtype=np.float32)
    return sdf.astype(np.float32)


def compute_sdf_batch(masks: torch.Tensor) -> torch.Tensor:
    """masks: (B, C, H, W) {0,1} → 동일 shape SDF tensor (음수=내부)"""
    masks_np = masks.detach().cpu().numpy()
    B, C, H, W = masks_np.shape
    sdf = np.zeros_like(masks_np, dtype=np.float32)
    for b in range(B):
        for c in range(C):
            sdf[b, c] = compute_sdf(masks_np[b, c])
    return torch.from_numpy(sdf).to(masks.device)


class BoundaryLoss(nn.Module):
    """L_b = mean( sigmoid(pred) * SDF(target) ) — 음수 영역(내부)에 prob 높을수록 ↓"""

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        if pred_logits.shape != target.shape:
            target = target.expand_as(pred_logits)
        p = torch.sigmoid(pred_logits)
        with torch.no_grad():
            sdf = compute_sdf_batch(target)
        return (p * sdf).mean()


# ──────────────────────────────────────────────────────────
#  Compound Seg Loss (per region)
# ──────────────────────────────────────────────────────────
class CompoundSegLoss(nn.Module):
    def __init__(self, w_tv=1.0, w_bce=0.5, w_b=0.3,
                 alpha=0.7, beta=0.3, gamma=4.0 / 3.0):
        super().__init__()
        self.tv = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss()
        self.bd = BoundaryLoss()
        self.w_tv, self.w_bce, self.w_b = w_tv, w_bce, w_b

    def forward(self, pred_logits, target):
        l_tv = self.tv(pred_logits, target)
        l_bce = self.bce(pred_logits, target)
        l_b = self.bd(pred_logits, target)
        return (self.w_tv * l_tv + self.w_bce * l_bce + self.w_b * l_b,
                l_tv.detach(), l_bce.detach(), l_b.detach())


# ──────────────────────────────────────────────────────────
#  Multi-Task Loss (Uncertainty Weighting + Deep Supervision)
# ──────────────────────────────────────────────────────────
class SOTAMultiTaskLoss(nn.Module):
    """
    Loss =  exp(-s_cls) * L_cls + 0.5 * s_cls
          + exp(-s_seg) * L_seg + 0.5 * s_seg
    """

    def __init__(self, pos_weight=None,
                 w_tv=1.0, w_bce=0.5, w_b=0.3,
                 ds_weights=(0.125, 0.25, 0.5)):
        super().__init__()
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.seg = CompoundSegLoss(w_tv=w_tv, w_bce=w_bce, w_b=w_b)
        self.ds_w = ds_weights

    def forward(self, cls_logits, seg_main_logits, seg_aux_list,
                cls_target, seg_target, log_var_cls, log_var_seg):
        l_cls = self.bce_cls(cls_logits.squeeze(1), cls_target)
        l_seg_main_total = 0.0
        for r in range(seg_main_logits.shape[1]):
            l_r, _, _, _ = self.seg(seg_main_logits[:, r:r + 1],
                                    seg_target[:, r:r + 1])
            l_seg_main_total = l_seg_main_total + l_r
        l_seg_aux_total = 0.0
        if seg_aux_list is not None:
            wt_target = seg_target[:, 0:1]
            for w, aux in zip(self.ds_w, seg_aux_list):
                l_a, _, _, _ = self.seg(aux, wt_target)
                l_seg_aux_total = l_seg_aux_total + w * l_a
        l_seg = l_seg_main_total + l_seg_aux_total
        prec_cls = torch.exp(-log_var_cls)
        prec_seg = torch.exp(-log_var_seg)
        total = (prec_cls * l_cls + 0.5 * log_var_cls
                 + prec_seg * l_seg + 0.5 * log_var_seg).squeeze()
        return total, l_cls.detach(), l_seg.detach()


# ──────────────────────────────────────────────────────────
#  TumorCP — Tumor Copy-Paste (Yang et al. MICCAI 2022, 단순화)
# ──────────────────────────────────────────────────────────
def tumor_copy_paste(batch_imgs: torch.Tensor, batch_masks: torch.Tensor,
                     p: float = 0.5):
    """배치 내 종양 슬라이스의 패치를 다른 슬라이스에 paste"""
    if np.random.rand() > p:
        return batch_imgs, batch_masks
    B, C, H, W = batch_imgs.shape
    wt_sum = batch_masks[:, 0].view(B, -1).sum(dim=1)
    tumor_idx = (wt_sum > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
    if len(tumor_idx) == 0:
        return batch_imgs, batch_masks
    imgs_new = batch_imgs.clone()
    masks_new = batch_masks.clone()
    for tgt in range(B):
        if np.random.rand() > 0.5:
            continue
        src = int(np.random.choice(tumor_idx))
        if src == tgt:
            continue
        wt_src = batch_masks[src, 0].cpu().numpy()
        ys, xs = np.where(wt_src > 0.5)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        patch_img = batch_imgs[src, :, y0:y1, x0:x1]
        patch_mask = batch_masks[src, :, y0:y1, x0:x1]
        ph, pw = patch_img.shape[1], patch_img.shape[2]
        if H - ph <= 0 or W - pw <= 0:
            continue
        py = np.random.randint(0, H - ph)
        px = np.random.randint(0, W - pw)
        paste_region = patch_mask[0:1].expand(C, ph, pw) > 0.5
        imgs_new[tgt, :, py:py + ph, px:px + pw][paste_region] = \
            patch_img[paste_region]
        masks_new[tgt, :, py:py + ph, px:px + pw] = torch.maximum(
            masks_new[tgt, :, py:py + ph, px:px + pw], patch_mask
        )
    return imgs_new, masks_new


if __name__ == "__main__":
    B, C, H, W = 4, 3, 64, 64
    pred = torch.randn(B, C, H, W, requires_grad=True)
    target = (torch.rand(B, C, H, W) > 0.85).float()
    print(f"  FocalTversky: {FocalTverskyLoss()(pred, target).item():.4f}")
    cs = CompoundSegLoss()
    l, *_ = cs(pred, target)
    print(f"  CompoundSeg:  {l.item():.4f}")

    cls_logit = torch.randn(B, 1, requires_grad=True)
    cls_y = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    aux = [torch.randn(B, 1, H, W) for _ in range(3)]
    log_var_c = torch.zeros(1, requires_grad=True)
    log_var_s = torch.zeros(1, requires_grad=True)
    mml = SOTAMultiTaskLoss(pos_weight=torch.tensor([1.5]))
    total, lc, ls = mml(cls_logit, pred, aux, cls_y, target, log_var_c, log_var_s)
    print(f"  SOTAMultiTask total: {total.item():.4f}  cls={lc.item():.4f}  seg={ls.item():.4f}")

    imgs = torch.randn(B, 3, H, W)
    masks = (torch.rand(B, 3, H, W) > 0.95).float()
    masks[0] = 0
    new_imgs, new_masks = tumor_copy_paste(imgs, masks, p=1.0)
    print(f"  TumorCP: in mask sum {masks.sum().item():.0f} -> new {new_masks.sum().item():.0f}")
