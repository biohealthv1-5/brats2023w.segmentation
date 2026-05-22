# -*- coding: utf-8 -*-
"""
Step 4: 모델 정의 - ResNet-18 기반 이진 분류 모델
===================================================
ImageNet 사전학습된 ResNet-18의 FC layer를 수정하여
뇌 MRI 종양 슬라이스 이진 분류 모델을 정의합니다.

사용법:
    from step4_model import create_model, freeze_backbone, unfreeze_backbone
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 기반 이진 분류 모델을 생성합니다.

    Args:
        pretrained: ImageNet 사전학습 가중치 사용 여부

    Returns:
        수정된 ResNet-18 모델

    모델 구조:
        ResNet-18 (Pretrained)
        ├── conv1 → bn1 → relu → maxpool
        ├── layer1 (BasicBlock × 2, 64ch)
        ├── layer2 (BasicBlock × 2, 128ch)
        ├── layer3 (BasicBlock × 2, 256ch)
        ├── layer4 (BasicBlock × 2, 512ch)  ← Grad-CAM 타겟
        ├── AdaptiveAvgPool2d (1×1)
        ├── Dropout(0.5)
        └── Linear(512 → 1)  ← 이진 분류 헤드
    """
    # 사전학습 가중치 로드
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # FC layer 수정: 512 → 1 (이진 분류)
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, 1),
    )

    return model


def freeze_backbone(model: nn.Module):
    """
    Feature extractor(backbone)를 동결합니다.
    FC layer만 학습 가능하도록 설정합니다.
    (Phase 1: Transfer Learning)
    """
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # 학습 가능한 파라미터 수 확인
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [FREEZE] Backbone 동결 완료")
    print(f"    학습 가능: {trainable:,} / 전체: {total:,} "
          f"({trainable/total*100:.1f}%)")


def unfreeze_backbone(model: nn.Module):
    """
    전체 모델을 학습 가능하도록 설정합니다.
    (Phase 2: Fine-tuning)
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [UNFREEZE] 전체 모델 학습 가능 설정")
    print(f"    학습 가능: {trainable:,} / 전체: {total:,} "
          f"({trainable/total*100:.1f}%)")


def get_model_summary(model: nn.Module):
    """모델 요약 정보를 출력합니다."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"\n{'─' * 50}")
    print(f"  모델 요약: ResNet-18 (Binary Classification)")
    print(f"{'─' * 50}")
    print(f"  전체 파라미터: {total_params:,}")
    print(f"  학습 가능:     {trainable_params:,}")
    print(f"  동결:          {non_trainable_params:,}")
    print(f"{'─' * 50}")


# ─── 직접 실행 시 모델 구조 확인 ─────────────────────────────
if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  Step 4: ResNet-18 모델 정의")
    print("=" * 60)

    # 모델 생성
    model = create_model(pretrained=True)
    get_model_summary(model)

    # Phase 1: Backbone 동결
    print()
    freeze_backbone(model)

    # Phase 2: 전체 학습
    print()
    unfreeze_backbone(model)

    # 더미 입력으로 forward pass 테스트
    print(f"\n{'─' * 50}")
    print("  Forward pass 테스트...")
    dummy_input = torch.randn(4, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  입력 shape: {dummy_input.shape}")
    print(f"  출력 shape: {output.shape}")
    print(f"  출력 값: {output.squeeze().tolist()}")
    print(f"  Sigmoid 확률: {torch.sigmoid(output).squeeze().tolist()}")

    # Grad-CAM 타겟 레이어 확인
    print(f"\n{'─' * 50}")
    print("  Grad-CAM 타겟 레이어: model.layer4")
    print(f"  layer4 출력 채널: 512")
    print(f"{'=' * 60}")
