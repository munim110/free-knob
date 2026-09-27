"""
Decoder blocks shared across baseline and DualDecoder architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Upsample + skip connection + conv block."""

    def __init__(self, in_ch, skip_ch, out_ch, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention = ChannelSpatialAttention(out_ch) if use_attention else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class ChannelSpatialAttention(nn.Module):
    """Lightweight channel + spatial attention (CBAM-style)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


class DensityDecoder(nn.Module):
    """
    Full decoder: 4 upsampling stages from 1/32 to 1/4 scale,
    then bilinear upsample to full resolution.
    """

    def __init__(self, encoder_channels=(256, 512, 1024, 2048),
                 use_attention=False):
        super().__init__()
        # Decoder goes from deepest to shallowest
        # s4 (2048) + s3 (1024) -> 512
        self.up4 = DecoderBlock(2048, 1024, 512, use_attention=use_attention)
        # 512 + s2 (512) -> 256
        self.up3 = DecoderBlock(512, 512, 256, use_attention=use_attention)
        # 256 + s1 (256) -> 128
        self.up2 = DecoderBlock(256, 256, 128, use_attention=use_attention)
        # Final 1x1 conv to density output
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, s1, s2, s3, s4):
        """
        Args:
            s1-s4: encoder features at 1/4, 1/8, 1/16, 1/32 scale
        Returns:
            density map at 1/4 scale (needs final upsampling)
        """
        x = self.up4(s4, s3)  # 1/16
        x = self.up3(x, s2)   # 1/8
        x = self.up2(x, s1)   # 1/4
        x = self.head(x)      # 1/4, 1 channel
        return x
