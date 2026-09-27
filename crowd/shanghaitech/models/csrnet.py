"""
CSRNet: Dilated Convolutional Neural Networks for Understanding the
Highly Congested Scenes (Li et al., CVPR 2018).

VGG-16 frontend + dilated convolution backend.
Standard crowd counting architecture baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CSRNet(nn.Module):
    """
    CSRNet for crowd density estimation.
    Frontend: VGG-16 first 10 layers (up to pool3, 1/8 scale).
    Backend: 6 dilated conv layers maintaining 1/8 scale.
    Output upsampled to full resolution.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Frontend: VGG-16 features up to pool3 (first 23 layers)
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None)
        features = list(vgg.features.children())
        # VGG16_BN up to pool3: conv-bn-relu x2 (64), pool,
        #                        conv-bn-relu x2 (128), pool,
        #                        conv-bn-relu x3 (256), pool
        # That's 23 layers (indices 0-22), output is 1/8 scale, 256 channels
        self.frontend = nn.Sequential(*features[:23])

        # Backend: dilated convolutions
        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x
