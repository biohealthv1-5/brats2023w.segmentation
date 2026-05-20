# -*- coding: utf-8 -*-
"""
Step 17: Multi-Task 모델 정의 — MultiTaskBrainNet
===================================================
ResNet-18 Encoder를 공유하여 분류(Classification)와
세분화(Segmentation)를 동시에 수행하는 Multi-Task 모델.

아키텍처:
    Shared Encoder (ResNet-18, ImageNet Pretrained)
    ├── Classification Head: GAP → Dropout → FC(512→1)
    └── Segmentation Decoder: 4-stage upsampling with skip connections

사용법:
    from step17_multitask_model import create_multitask_model
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
    """
    Segmentation Decoder의 업샘플링 블록.
    Upsample → Concat(skip) → Conv → BN → ReLU → Conv → BN → ReLU
    """

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
        # skip과 크기 맞추기 (혹시 모를 1~2픽셀 차이 대응)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiTaskBrainNet(nn.Module):
    """
    ResNet-18 기반 Multi-Task 모델

    - Task A (Classification): 슬라이스 이진 분류 (종양 유무)
    - Task B (Segmentation): 픽셀 단위 종양 세분화 (보조 태스크)

    구조:
        입력 (B, 3, 224, 224)
            │
        ┌───┴───────────────────────┐
        │   Shared Encoder          │
        │   e1: (B, 64, 112, 112)   │  ← conv1+bn1+relu
        │   e2: (B, 64,  56,  56)   │  ← maxpool+layer1
        │   e3: (B, 128, 28,  28)   │  ← layer2
        │   e4: (B, 256, 14,  14)   │  ← layer3
        │   e5: (B, 512,  7,   7)   │  ← layer4  (Grad-CAM 타겟)
        └───┬───────────────────────┘
            │
       ┌────┴────┐
       │         │
    [CLS Head]  [SEG Decoder]
    GAP→FC→1     up4→up3→up2→up1→final
                 (B,1,224,224)
    """

    def __init__(self, pretrained=True, encoder_channels=[64, 64, 128, 256, 512],
                 decoder_channels=[256, 128, 64, 32]):
        super().__init__()

        # ═══ Shared Encoder (ResNet-18) ═══
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        else:
            resnet = models.resnet18(weights=None)

        # Encoder stages
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2 → 64ch
        self.pool = resnet.maxpool                                         # /4
        self.enc2 = resnet.layer1   # 64ch,  /4
        self.enc3 = resnet.layer2   # 128ch, /8
        self.enc4 = resnet.layer3   # 256ch, /16
        self.enc5 = resnet.layer4   # 512ch, /32  ← Grad-CAM 타겟

        # ═══ Classification Head ═══
        self.cls_gap = nn.AdaptiveAvgPool2d(1)
        self.cls_dropout = nn.Dropout(0.5)
        self.cls_fc = nn.Linear(encoder_channels[4], 1)

        # ═══ Segmentation Decoder ═══
        # up4: 512 + 256(skip from enc4) → 256
        self.up4 = UpBlock(encoder_channels[4], encoder_channels[3], decoder_channels[0])
        # up3: 256 + 128(skip from enc3) → 128
        self.up3 = UpBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1])
        # up2: 128 + 64(skip from enc2) → 64
        self.up2 = UpBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2])
        # up1: 64 + 64(skip from enc1) → 32
        self.up1 = UpBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])

        # Final upsample /2 → /1 + 1×1 conv
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.seg_head = nn.Conv2d(decoder_channels[3], 1, kernel_size=1)

        # Decoder weight 초기화
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Decoder 부분만 Kaiming 초기화"""
        for module in [self.up4, self.up3, self.up2, self.up1, self.seg_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # ─── Encoder (공유) ───────────────────────────────────
        e1 = self.enc1(x)              # (B, 64, 112, 112)
        e2 = self.enc2(self.pool(e1))  # (B, 64,  56,  56)
        e3 = self.enc3(e2)             # (B, 128, 28,  28)
        e4 = self.enc4(e3)             # (B, 256, 14,  14)
        e5 = self.enc5(e4)             # (B, 512,  7,   7)

        # ─── Classification Head ─────────────────────────────
        cls_feat = self.cls_gap(e5)            # (B, 512, 1, 1)
        cls_feat = cls_feat.flatten(1)          # (B, 512)
        cls_out = self.cls_fc(self.cls_dropout(cls_feat))  # (B, 1)

        # ─── Segmentation Decoder ────────────────────────────
        d4 = self.up4(e5, e4)          # (B, 256, 14, 14)
        d3 = self.up3(d4, e3)          # (B, 128, 28, 28)
        d2 = self.up2(d3, e2)          # (B, 64,  56, 56)
        d1 = self.up1(d2, e1)          # (B, 32, 112, 112)
        seg_out = self.seg_head(self.final_up(d1))  # (B, 1, 224, 224)

        return cls_out, seg_out


# ─── 모델 생성 헬퍼 함수 ─────────────────────────────────────
def create_multitask_model(pretrained=True):
    """MultiTaskBrainNet 모델을 생성합니다."""
    return MultiTaskBrainNet(pretrained=pretrained)


def freeze_encoder(model: MultiTaskBrainNet):
    """Encoder(ResNet-18)를 동결합니다. (Phase 1)"""
    for name, param in model.named_parameters():
        if name.startswith(("enc1", "enc2", "enc3", "enc4", "enc5", "pool")):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [FREEZE] Encoder 동결 완료")
    print(f"    학습 가능: {trainable:,} / 전체: {total:,} ({trainable/total*100:.1f}%)")


def unfreeze_encoder(model: MultiTaskBrainNet):
    """전체 모델을 학습 가능으로 설정합니다. (Phase 2)"""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [UNFREEZE] 전체 모델 학습 가능 설정")
    print(f"    학습 가능: {trainable:,} / 전체: {total:,} ({trainable/total*100:.1f}%)")


def get_model_summary(model: nn.Module):
    """모델 요약 정보를 출력합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 부분별 파라미터 수
    encoder_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith(("enc1", "enc2", "enc3", "enc4", "enc5"))
    )
    cls_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("cls_")
    )
    seg_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith(("up", "seg_", "final"))
    )

    print(f"\n{'─' * 55}")
    print(f"  모델 요약: MultiTaskBrainNet")
    print(f"{'─' * 55}")
    print(f"  전체 파라미터:     {total_params:,}")
    print(f"  학습 가능:         {trainable_params:,}")
    print(f"{'─' * 55}")
    print(f"  Encoder (ResNet-18): {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  Classification Head: {cls_params:,} ({cls_params/total_params*100:.1f}%)")
    print(f"  Segmentation Decoder: {seg_params:,} ({seg_params/total_params*100:.1f}%)")
    print(f"{'─' * 55}")


# ─── 직접 실행 시 모델 구조 확인 ─────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 17: MultiTaskBrainNet 모델 정의")
    print("=" * 60)

    # 모델 생성
    model = create_multitask_model(pretrained=True)
    get_model_summary(model)

    # Phase 1: Encoder 동결
    print()
    freeze_encoder(model)

    # Phase 2: 전체 학습
    print()
    unfreeze_encoder(model)

    # Forward pass 테스트
    print(f"\n{'─' * 55}")
    print("  Forward pass 테스트...")
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        cls_out, seg_out = model(dummy_input)

    print(f"  입력 shape:   {dummy_input.shape}")
    print(f"  분류 출력:    {cls_out.shape}  → {cls_out.squeeze().tolist()}")
    print(f"  세분화 출력:  {seg_out.shape}")
    print(f"  분류 확률:    {torch.sigmoid(cls_out).squeeze().tolist()}")
    print(f"  세분화 범위:  [{seg_out.min():.3f}, {seg_out.max():.3f}]")

    # Grad-CAM 타겟 확인
    print(f"\n{'─' * 55}")
    print("  Grad-CAM 타겟 레이어: model.enc5 (= ResNet-18 layer4)")
    print(f"{'=' * 60}")
