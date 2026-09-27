"""RainNet-DualDecoder, dual-decoder variant of RainNet for the unified decomposition.

The ONLY change vs `rainnet.RainNet` is the decoder count:
  - Same encoder (enc1..enc5 + two max-pool-bracketed dropouts)
  - TWO parallel decoder stacks with the same structure, one per head
  - bg head output: linear (full-field prediction, any sign)
  - ext head output: ReLU (non-negative critical residual)

This is the architectural falsification lever for the DualDecoder thesis:
  dual-decoder reduces bias at equal CSI vs single-decoder, holding
  backbone / loss / data / sampler constant.

The model does NOT fuse internally, it returns (bg_pred, ext_pred). The
training and eval loops combine them as

    y_hat = bg_pred + sign * beta * ext_pred

with sign = +1 for high-value critical (e.g. heavy rain, RY product)
and sign = -1 for low-value critical (e.g. cold cloud tops, TBB).

Parameter count is roughly 2x the decoder of RainNet and encoder is shared,
so the total grows from ~31.4M to ~44M, similar to the original DualDecoder's
11.35M (base_c=32) vs Attention U-Net's 21.41M asymmetry.
"""

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _RainNetDecoder(nn.Module):
    """One RainNet-style decoder: 4 upsample+concat+double-conv blocks,
    then the 64->2->1 bridge. Kept as a module so we can instantiate it twice."""

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = _DoubleConv(1024 + 512, 512)
        self.dec3 = _DoubleConv(512 + 256, 256)
        self.dec2 = _DoubleConv(256 + 128, 128)
        self.dec1 = _DoubleConv(128 + 64, 64)
        self.bridge = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, b, s4, s3, s2, s1):
        d4 = self.dec4(torch.cat([self.up(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), s1], dim=1))
        return self.out(self.bridge(d1))


class RainNetDualDecoder(nn.Module):
    """Dual-decoder RainNet with unified decomposition semantics."""

    def __init__(self, in_channels: int = 4, ext_activation: str = "relu"):
        super().__init__()
        # ext head activation. 'relu' is correct for the unified decomposition,
        # whose ext target is a non-negative excess. The frequency decomposition's
        # residual y - G_sigma*y is SIGNED, and a ReLU head cannot represent its
        # negative part. On radar that means the model could never correct the
        # smoothed background DOWNWARD in the gaps between rain cells, inflating
        # false alarms exactly where they hurt. The crowd DualDecoder, which is where
        # frequency decomposition works, uses a linear head for this reason.
        if ext_activation not in ("relu", "linear"):
            raise ValueError(f"ext_activation must be 'relu' or 'linear', got {ext_activation!r}")
        self.ext_activation = ext_activation
        # Shared encoder, identical to RainNet
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)
        self.drop4 = nn.Dropout2d(0.5)
        self.enc5 = _DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(2)

        # Two independent decoders
        self.dec_bg = _RainNetDecoder()
        self.dec_ext = _RainNetDecoder()

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.drop4(self.enc4(self.pool(s3)))
        b = self.drop5(self.enc5(self.pool(s4)))

        bg_pred = self.dec_bg(b, s4, s3, s2, s1)
        ext_pred = self.dec_ext(b, s4, s3, s2, s1)
        if self.ext_activation == "relu":
            ext_pred = torch.relu(ext_pred)
        return bg_pred, ext_pred


if __name__ == "__main__":
    m = RainNetDualDecoder(in_channels=4)
    n = sum(p.numel() for p in m.parameters())
    print(f"RainNetDualDecoder params: {n:,}")
    with torch.no_grad():
        bg, ext = m(torch.zeros(1, 4, 64, 64))
    print(f"bg shape: {tuple(bg.shape)}  ext shape: {tuple(ext.shape)}")
