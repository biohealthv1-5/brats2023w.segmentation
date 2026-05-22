# -*- coding: utf-8 -*-
"""
Step 10: 패치 분류 경량 CNN 모델 정의
======================================
64×64 패치에 최적화된 경량 CNN (~200K 파라미터).
ResNet-18(11M)의 1/50 크기로 빠른 학습이 가능합니다.

사용법:
    from step10_patch_model import PatchCNN, create_patch_model
"""

import sys
import torch
import torch.nn as nn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


class PatchCNN(nn.Module):
    """
    경량 패치 분류 CNN

    구조:
        Conv(3→32) → BN → ReLU → MaxPool → 32×32
        Conv(32→64) → BN → ReLU → MaxPool → 16×16
        Conv(64→128) → BN → ReLU → AdaptiveAvgPool → 1×1
        Dropout(0.5) → Linear(128→1)

    입력: (B, 3, 64, 64)
    출력: (B, 1) — raw logit (BCEWithLogitsLoss 사용)
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 64→32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32→16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 16→AdaptivePool→1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (B, 128)
        x = self.classifier(x)      # (B, 1)
        return x

    # Grad-CAM 타겟 레이어 (Step 13에서 사용)
    def get_gradcam_target_layer(self):
        """features[-4] = 마지막 Conv2d(64→128) 레이어"""
        return self.features[-4]  # Conv2d(64, 128, 3)


def create_patch_model(device=None):
    """모델 생성 + 파라미터 정보 출력"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatchCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  모델: PatchCNN")
    print(f"  파라미터: {total_params:,} (학습 가능: {trainable:,})")
    print(f"  디바이스: {device}")

    return model


# ─── 단독 테스트 ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 10: PatchCNN 모델 테스트")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_patch_model(device)

    # 더미 입력 테스트
    dummy = torch.randn(4, 3, 64, 64).to(device)
    out = model(dummy)
    print(f"\n  입력: {dummy.shape}")
    print(f"  출력: {out.shape}")
    print(f"  출력값: {out.squeeze().tolist()}")

    # Grad-CAM 타겟
    target = model.get_gradcam_target_layer()
    print(f"\n  Grad-CAM 타겟: {target}")

    print(f"\n{'='*60}")
    print(f"  ✓ 모델 정상!")
    print(f"{'='*60}")
