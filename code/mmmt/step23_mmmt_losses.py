# -*- coding: utf-8 -*-
"""
Step 23 (Day 6 / mmmt): Loss 모듈
==================================
Day 6 Loss (revise_analysis §3.1 권장 — "Tversky 추가"):

  L_total = L_cls + beta_seg * L_seg
  L_cls   = BCE(pos_weight)
  L_seg   = 0.5 * Dice(WT) + 0.5 * Tversky(WT, alpha=0.7, beta=0.3)

* FN 가중치(α=0.7)는 본 Day 6의 *유일한 Loss 변화* — Focal-Tversky / Boundary /
  Uncertainty Weighting / Deep Supervision 등은 Day 7 SOTA 에서 도입.
* β_seg(=0.5) 는 multitask Day 5 와 동일한 수동 가중.
"""

import sys
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ──────────────────────────────────────────────────────────
#  Dice Loss (multitask/step18 과 동일 — 단일 채널 WT)
# ──────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        p = torch.sigmoid(pred_logits)
        if p.shape != target.shape:
            target = target.expand_as(p)
        # spatial sum per (B, C)
        dims = (2, 3) if p.dim() == 4 else (1, 2)
        inter = (p * target).sum(dim=dims)
        denom = p.sum(dim=dims) + target.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


# ──────────────────────────────────────────────────────────
#  Tversky Loss (Salehi et al. 2017 — FN 가중)
# ──────────────────────────────────────────────────────────
class TverskyLoss(nn.Module):
    """
    Tversky = TP / (TP + α·FN + β·FP)
    FN 패널티: α=0.7, FP 패널티: β=0.3   (revise §3.1 권장 설정)
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        p = torch.sigmoid(pred_logits)
        if p.shape != target.shape:
            target = target.expand_as(p)
        dims = (2, 3) if p.dim() == 4 else (1, 2)
        TP = (p * target).sum(dim=dims)
        FN = ((1 - p) * target).sum(dim=dims)
        FP = (p * (1 - target)).sum(dim=dims)
        tv = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP
                                   + self.smooth)
        return (1.0 - tv).mean()


# ──────────────────────────────────────────────────────────
#  Combined Seg Loss: 0.5 * Dice + 0.5 * Tversky
# ──────────────────────────────────────────────────────────
class DiceTverskyLoss(nn.Module):
    def __init__(self, w_dice: float = 0.5, w_tversky: float = 0.5,
                 alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.tv = TverskyLoss(alpha=alpha, beta=beta)
        self.w_dice = w_dice
        self.w_tv = w_tversky

    def forward(self, pred_logits, target):
        return self.w_dice * self.dice(pred_logits, target) \
               + self.w_tv * self.tv(pred_logits, target)


# ──────────────────────────────────────────────────────────
#  Multi-Task Loss (수동 가중 β_seg = 0.5)
# ──────────────────────────────────────────────────────────
class MMMTLoss(nn.Module):
    """
    L_total = L_cls + beta_seg * L_seg
    """

    def __init__(self, pos_weight: torch.Tensor = None,
                 beta_seg: float = 0.5,
                 alpha_tv: float = 0.7, beta_tv: float = 0.3):
        super().__init__()
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.seg = DiceTverskyLoss(alpha=alpha_tv, beta=beta_tv)
        self.beta_seg = beta_seg

    def forward(self, cls_logits, seg_logits, cls_target, seg_target):
        l_cls = self.bce_cls(cls_logits.squeeze(1), cls_target)
        l_seg = self.seg(seg_logits, seg_target)
        total = l_cls + self.beta_seg * l_seg
        return total, l_cls.detach(), l_seg.detach()


if __name__ == "__main__":
    # 빠른 동작 확인
    B, H, W = 4, 64, 64
    pred = torch.randn(B, 1, H, W, requires_grad=True)
    target = (torch.rand(B, 1, H, W) > 0.85).float()
    print(f"  Dice    : {DiceLoss()(pred, target).item():.4f}")
    print(f"  Tversky : {TverskyLoss()(pred, target).item():.4f}")
    print(f"  DT      : {DiceTverskyLoss()(pred, target).item():.4f}")

    cls_logit = torch.randn(B, 1, requires_grad=True)
    cls_y = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    mml = MMMTLoss(pos_weight=torch.tensor([1.5]))
    total, lc, ls = mml(cls_logit, pred, cls_y, target)
    print(f"  MMMT total: {total.item():.4f}  cls={lc.item():.4f}  seg={ls.item():.4f}")
