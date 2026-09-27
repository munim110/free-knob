"""
Shared ResNet-based encoder for all models.
Uses pretrained ResNet-50 backbone with FPN-style multi-scale features.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-50 encoder that extracts multi-scale features.
    Returns skip connections at 4 scales for the decoder.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Stage 0: conv1 + bn + relu + maxpool -> 1/4 scale, 64 channels
        self.stage0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        # Stage 1: layer1 -> 1/4 scale, 256 channels
        self.stage1 = resnet.layer1
        # Stage 2: layer2 -> 1/8 scale, 512 channels
        self.stage2 = resnet.layer2
        # Stage 3: layer3 -> 1/16 scale, 1024 channels
        self.stage3 = resnet.layer3
        # Stage 4: layer4 -> 1/32 scale, 2048 channels
        self.stage4 = resnet.layer4

    def forward(self, x):
        """
        Returns features at 4 scales:
        s1: 1/4, 256ch  |  s2: 1/8, 512ch  |  s3: 1/16, 1024ch  |  s4: 1/32, 2048ch
        """
        x0 = self.stage0(x)   # (B, 64, H/4, W/4)
        s1 = self.stage1(x0)  # (B, 256, H/4, W/4)
        s2 = self.stage2(s1)  # (B, 512, H/8, W/8)
        s3 = self.stage3(s2)  # (B, 1024, H/16, W/16)
        s4 = self.stage4(s3)  # (B, 2048, H/32, W/32)
        return s1, s2, s3, s4

    @property
    def channel_sizes(self):
        return [256, 512, 1024, 2048]
