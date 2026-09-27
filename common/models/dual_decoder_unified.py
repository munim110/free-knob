"""DualDecoder with the unified decomposition.

Differences from model.py:
  - Background head learns the FULL field (no clipping), its target is just `y`
  - Extreme head learns a NON-NEGATIVE critical residual via a final ReLU
  - The model returns (bg_pred, ext_pred), the combined prediction
    `y_hat = bg_pred ± beta · ext_pred` is computed at train/eval time,
    not inside forward, so β stays an inference-time hyperparameter.

Architecture (encoder, decoders, attention) is unchanged from model.py so
that the only experimental variable here is the decomposition itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class SimplifiedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResBlock(out_channels + skip_channels, out_channels)
        self.attention = SimplifiedAttention(out_channels) if use_attention else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return self.attention(x)


class DualDecoderUnified(nn.Module):
    """Dual-decoder downscaler with unified-decomposition semantics.

    Heads:
      bg_pred, predicts the full field (any sign)
      ext_pred, predicts the non-negative critical residual (after ReLU)

    The model does NOT compute a fused output. The combined prediction
        y_hat = bg_pred - beta * ext_pred   (low-value critical, e.g. cold cloud tops)
        y_hat = bg_pred + beta * ext_pred   (high-value critical, e.g. crowd density)
    is the caller's responsibility, this keeps beta as a clean inference knob.
    """

    def __init__(self, n_channels=4, base_c=32):
        super().__init__()
        # Shared encoder (identical to DualDecoderDirect)
        self.enc1 = ResBlock(n_channels, base_c)
        self.enc2 = ResBlock(base_c, base_c * 2)
        self.enc3 = ResBlock(base_c * 2, base_c * 4)
        self.enc4 = ResBlock(base_c * 4, base_c * 8)
        self.down = nn.MaxPool2d(2)
        self.bottleneck = ResBlock(base_c * 8, base_c * 16)

        # Background head decoder, learns the FULL field
        self.up1_bg = DecoderBlock(base_c * 16, base_c * 8, base_c * 8)
        self.up2_bg = DecoderBlock(base_c * 8, base_c * 4, base_c * 4)
        self.up3_bg = DecoderBlock(base_c * 4, base_c * 2, base_c * 2)
        self.up4_bg = DecoderBlock(base_c * 2, base_c, base_c)
        self.out_bg = nn.Conv2d(base_c, 1, kernel_size=1)

        # Extreme head decoder, learns the non-negative critical residual
        self.up1_ext = DecoderBlock(base_c * 16, base_c * 8, base_c * 8)
        self.up2_ext = DecoderBlock(base_c * 8, base_c * 4, base_c * 4, use_attention=True)
        self.up3_ext = DecoderBlock(base_c * 4, base_c * 2, base_c * 2)
        self.up4_ext = DecoderBlock(base_c * 2, base_c, base_c)
        self.out_ext = nn.Conv2d(base_c, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.down(s1))
        s3 = self.enc3(self.down(s2))
        s4 = self.enc4(self.down(s3))
        b = self.bottleneck(self.down(s4))

        # Background head, full field, unconstrained sign
        cb4 = self.up1_bg(b, s4)
        cb3 = self.up2_bg(cb4, s3)
        cb2 = self.up3_bg(cb3, s2)
        cb1 = self.up4_bg(cb2, s1)
        bg_pred = self.out_bg(cb1)

        # Extreme head, non-negative critical residual
        e4 = self.up1_ext(b, s4)
        e3 = self.up2_ext(e4, s3)
        e2 = self.up3_ext(e3, s2)
        e1 = self.up4_ext(e2, s1)
        ext_pred = F.relu(self.out_ext(e1))

        return bg_pred, ext_pred
