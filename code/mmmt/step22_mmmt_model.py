# -*- coding: utf-8 -*-
"""
Step 22 (Day 6 / mmmt): Multi-Modal Multi-Task Model
=====================================================
입력  : (B, 3, 224, 224)  ← [T1ce, FLAIR, |T1ce-FLAIR|]
출력  :
  - cls_out      (B, 1)              : tumor 유무 (BCE)
  - seg_out      (B, 1, 224, 224)    : WT 1-region (sigmoid-ready)

Day 6는 입력 표현 (단일 모달 → 멀티모달) 확장에 *집중* — 모델 구조는 기존
`multitask/step17_multitask_model.py` 의 MultiTaskBrainNet 과 동일하게 유지하여
순수 멀티모달 효과만 분리 평가합니다.

3-region (WT/TC/ET) head, Deep Supervision, Uncertainty Weighting 등은 Day 7
SOTA 패키지에서 도입 (`code/sota/`).

기본 백본: ImageNet pretrained ResNet-18.
첫 번째 conv1 (3ch in) 가중치는 그대로 재사용 가능 (입력 채널 수 동일).
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# ─── Decoder Block (step17과 동일 구조) ──────────────────────
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


# ─── MM-MTL Model ────────────────────────────────────────────
class MMMTBrainNet(nn.Module):
    """
    Shared Encoder (ResNet-18, 3ch ImageNet pretrained) +
      ├─ Classification Head: GAP → Dropout → FC(512 → 1)
      └─ Segmentation Decoder (4-stage UNet-style)
          - main head     : 1ch (WT) @ 224×224
    """

    def __init__(self, pretrained=True, in_ch=3, seg_classes=1,
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
        # in_ch == 3 이므로 conv1 그대로 재사용

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

        self._init_decoder()

    def _init_decoder(self):
        for module in [self.up4, self.up3, self.up2, self.up1, self.seg_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)             # (B, 64, 112, 112)
        e2 = self.enc2(self.pool(e1)) # (B, 64,  56,  56)
        e3 = self.enc3(e2)            # (B, 128, 28,  28)
        e4 = self.enc4(e3)            # (B, 256, 14,  14)
        e5 = self.enc5(e4)            # (B, 512,  7,   7)

        # Classification
        cls_feat = self.cls_gap(e5).flatten(1)
        cls_out = self.cls_fc(self.cls_dropout(cls_feat))   # (B,1)

        # Decoder
        d4 = self.up4(e5, e4)         # (B, 256, 14, 14)
        d3 = self.up3(d4, e3)         # (B, 128, 28, 28)
        d2 = self.up2(d3, e2)         # (B,  64, 56, 56)
        d1 = self.up1(d2, e1)         # (B,  32, 112,112)
        seg_out = self.seg_head(self.final_up(d1))          # (B, 1, 224, 224)

        return cls_out, seg_out


# ─── 헬퍼 ───────────────────────────────────────────────────
def create_mm_model(pretrained=True, in_ch=3, seg_classes=1):
    return MMMTBrainNet(pretrained=pretrained, in_ch=in_ch,
                        seg_classes=seg_classes)


def freeze_encoder(model: MMMTBrainNet):
    for name, p in model.named_parameters():
        if name.startswith(("enc1", "enc2", "enc3", "enc4", "enc5", "pool")):
            p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"  [FREEZE] encoder 동결  ({n_train:,}/{n_tot:,} trainable)")


def unfreeze_encoder(model: MMMTBrainNet):
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
              if n.startswith(("up", "seg_", "final")))
    print(f"  Total params:     {total:,}")
    print(f"  Trainable:        {trainable:,}")
    print(f"  Encoder:          {enc:,} ({enc/total*100:.1f}%)")
    print(f"  Cls head:         {cls:,} ({cls/total*100:.1f}%)")
    print(f"  Seg decoder:      {seg:,} ({seg/total*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("  Step 22 (Day 6): MMMTBrainNet forward 테스트")
    print("=" * 60)
    model = create_mm_model(pretrained=True, in_ch=3, seg_classes=1)
    get_model_summary(model)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cls_out, seg_out = model(x)
    print(f"  cls_out:   {cls_out.shape}")
    print(f"  seg_out:   {seg_out.shape}")
    print("=" * 60)
