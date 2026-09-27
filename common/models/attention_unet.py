"""Attention U-Net baseline used for the AR downscaling comparison.

Reproduces the architecture from UNet_Retraining.ipynb verbatim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedAttention(nn.Module):
    """Channel + spatial attention applied at the U-Net bottleneck."""

    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, input_channels=4, base_channels=64, depth=4, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.depth = depth
        self.channels = [base_channels * min(2 ** i, 8) for i in range(depth)]

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = input_channels
        for i, out_ch in enumerate(self.channels):
            self.encoders.append(self._conv_block(in_ch, out_ch))
            if i < len(self.channels) - 1:
                self.downsamplers.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        bottleneck_ch = self.channels[-1]
        self.bottleneck = self._conv_block(self.channels[-1], bottleneck_ch)
        if self.use_attention:
            self.attention = SimplifiedAttention(bottleneck_ch)

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_ch if i == depth - 1 else self.channels[i + 1]
            out_ch = self.channels[i]
            self.upsamplers.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            self.decoders.append(self._conv_block(out_ch * 2, out_ch))

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0] // 2, 3, padding=1),
            nn.BatchNorm2d(self.channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels[0] // 2, 1, 1),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skips = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            skips.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        x = self.bottleneck(x)
        if self.use_attention:
            x = self.attention(x)

        for i, (up, dec) in enumerate(zip(self.upsamplers, self.decoders)):
            x = up(x)
            skip = skips[len(skips) - 1 - i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.final_conv(x)
