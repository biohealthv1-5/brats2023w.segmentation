# -*- coding: utf-8 -*-
"""
Step 28 (Day 7 / sota): SOTA Multi-Modal Multi-Task Model
==========================================================
v2ways.md §2.2 (Deep Supervision) + §3.4 (3-region) + §3.5 (Uncertainty
Weighting) 통합 모델.

입력  : (B, 3, 224, 224)  ← [T1ce, FLAIR, |T1ce-FLAIR|]
출력  :
  - cls_out      (B, 1)              : tumor 유무 (BCE)
  - seg_main     (B, 3, 224, 224)    : WT, TC, ET 3-region (sigmoid-ready)
  - seg_aux_list [aux2, aux3, aux4]  : Deep Supervision 보조 출력 (224×224, WT 채널)
  - log_var_cls, log_var_seg         : Uncertainty Weighting 학습 파라미터

본 모델은 Day 6 mmmt MMMTBrainNet 의 *확장*:
  + seg_classes 1 → 3 (WT/TC/ET)
  + Deep Supervision aux heads (3개)
  + Uncertainty Weighting 학습 가능 log_var 파라미터 2개
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SOTAMultiTaskNet(nn.Module):
    """
    Shared Encoder (ResNet-18, 3ch ImageNet pretrained)
      ├─ Classification Head
      └─ Segmentation Decoder
          - main head     : 3ch (WT, TC, ET) @ 224×224
          - aux head x3   : 1ch (WT) @ d2/d3/d4 → upsample to 224 (Deep Supervision)

    Uncertainty Weighting (Kendall et al. CVPR 2018):
      - log_var_cls / log_var_seg : 학습 가능 파라미터
    """

    def __init__(self, pretrained=True, in_ch=3, seg_classes=3,
                 encoder_channels=(64, 64, 128, 256, 512),
                 decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.in_ch = in_ch
        self.seg_classes = seg_classes

        # ── Encoder (ResNet-18) ─────────────────────────────
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4   # Grad-CAM 타겟

        # ── Classification Head ─────────────────────────────
        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_dropout = nn.Dropout(0.5)
        self.cls_fc = nn.Linear(encoder_channels[4], 1)

        # ── Segmentation Decoder ────────────────────────────
        self.up4 = UpBlock(encoder_channels[4], encoder_channels[3], decoder_channels[0])
        self.up3 = UpBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1])
        self.up2 = UpBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2])
        self.up1 = UpBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.seg_head = nn.Conv2d(decoder_channels[3], seg_classes, kernel_size=1)

        # ── Deep Supervision: WT aux heads ──────────────────
        self.aux_head_d4 = nn.Conv2d(decoder_channels[0], 1, kernel_size=1)
        self.aux_head_d3 = nn.Conv2d(decoder_channels[1], 1, kernel_size=1)
        self.aux_head_d2 = nn.Conv2d(decoder_channels[2], 1, kernel_size=1)

        # ── Uncertainty Weighting (Kendall 2018) ───────────
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_seg = nn.Parameter(torch.zeros(1))

        self._init_decoder()

    def _init_decoder(self):
        for module in [self.up4, self.up3, self.up2, self.up1,
                       self.seg_head, self.aux_head_d2, self.aux_head_d3,
                       self.aux_head_d4]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_aux=True):
        H, W = x.shape[2], x.shape[3]
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Classification
        cls_feat = self.cls_gap(e5).flatten(1)
        cls_out = self.cls_fc(self.cls_dropout(cls_feat))

        # Decoder
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        seg_main = self.seg_head(self.final_up(d1))

        if not return_aux:
            return cls_out, seg_main, None

        a4 = F.interpolate(self.aux_head_d4(d4), size=(H, W),
                           mode="bilinear", align_corners=True)
        a3 = F.interpolate(self.aux_head_d3(d3), size=(H, W),
                           mode="bilinear", align_corners=True)
        a2 = F.interpolate(self.aux_head_d2(d2), size=(H, W),
                           mode="bilinear", align_corners=True)
        return cls_out, seg_main, [a2, a3, a4]


def create_sota_model(pretrained=True, in_ch=3, seg_classes=3):
    return SOTAMultiTaskNet(pretrained=pretrained, in_ch=in_ch,
                            seg_classes=seg_classes)


def freeze_encoder(model: SOTAMultiTaskNet):
    for name, p in model.named_parameters():
        if name.startswith(("enc1", "enc2", "enc3", "enc4", "enc5", "pool")):
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"  [FREEZE] encoder 동결  ({n_train:,}/{n_tot:,} trainable)")


def unfreeze_encoder(model: SOTAMultiTaskNet):
    for p in model.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"  [UNFREEZE] 전체 학습  ({n_train:,}/{n_tot:,} trainable)")


def get_model_summary(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith(("enc1", "enc2", "enc3", "enc4", "enc5")))
    cls = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith("cls_"))
    seg = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith(("up", "seg_", "final", "aux_")))
    print(f"  Total params:     {total:,}")
    print(f"  Trainable:        {trainable:,}")
    print(f"  Encoder:          {enc:,} ({enc/total*100:.1f}%)")
    print(f"  Cls head:         {cls:,} ({cls/total*100:.1f}%)")
    print(f"  Seg dec + aux:    {seg:,} ({seg/total*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("  Step 28 (Day 7 / sota): SOTAMultiTaskNet forward 테스트")
    print("=" * 60)
    model = create_sota_model(pretrained=True, in_ch=3, seg_classes=3)
    get_model_summary(model)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cls_out, seg_main, aux = model(x)
    print(f"  cls_out:   {cls_out.shape}")
    print(f"  seg_main:  {seg_main.shape}")     # (2, 3, 224, 224)
    for i, a in enumerate(aux):
        print(f"  aux[{i}]:    {a.shape}")
    print(f"  log_var_cls init: {model.log_var_cls.item():.3f}")
    print(f"  log_var_seg init: {model.log_var_seg.item():.3f}")
    print("=" * 60)
