"""RainNet-DualHead, single-decoder variant with TWO output heads.

Architectural interpolation between vanilla RainNet (one decoder, one head)
and full RainNet-DualDecoder (two independent decoders). The body, encoder plus
decoder, is exactly vanilla RainNet. Only the final bridge + 1x1 output
conv is duplicated into a bg head (linear) and an ext head (ReLU).

Purpose: separate the "two outputs / unified-decomposition loss" effect
from the "two decoders" effect in the DualDecoder thesis. If DualHead recovers
most of DualDecoder's calibration benefit, the full decoder duplication is not
load-bearing, only having two outputs is.

Shared with RainNet-DualDecoder:
  - same in/output semantics: returns (bg_pred, ext_pred)
  - ext_pred is ReLU-gated (non-negative)
  - intended to be trained with the three-term DualDecoder loss (bg MSE + ext
    importance-weighted + combined MSE @ beta)

Parameter count: ~31.38 M (vanilla RainNet) + ~1.2 k (second bridge+out head)
                 ≈ 31.38 M, essentially identical to vanilla, a dramatic
                 contrast with RainNet-DualDecoder's 43.92 M.
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


def _make_head(in_c: int = 64):
    return nn.Sequential(
        nn.Conv2d(in_c, 2, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(2, 1, kernel_size=1),
    )


class RainNetDualHead(nn.Module):
    """Vanilla RainNet body (shared encoder + shared decoder) with two
    terminal output heads. Returns (bg_pred, ext_pred) to match the
    RainNet-DualDecoder API so the same training loop works unchanged."""

    @staticmethod
    def dec_channels_for(dec_width: float):
        """The four decoder widths implied by a multiplier, as explicit ints.

        Recovering the multiplier from a checkpoint by dividing one layer's
        channel count does NOT round-trip: dec_width=1.622 gives dec1 = 104, and
        104/64 = 1.625 re-derives dec3 as 416 instead of 415. So the widths are
        computed once here and passed around as integers.
        """
        return tuple(max(1, int(round(n * dec_width))) for n in (512, 256, 128, 64))

    def __init__(self, in_channels: int = 4, dec_width: float = 1.0,
                 dec_channels=None):
        """dec_width scales the WIDTH of the single shared decoder.

        RainNetDualDecoder is this network with the decoder duplicated: identical
        encoder (18.844M), exactly 2x the decoder (25.074M vs 12.537M). So at
        dec_width=1 a DualDecoder-vs-dualhead comparison confounds "two decoders" with
        "40% more parameters", so the gain may instead reflect capacity.
        Setting dec_width ~= 1.43 makes the two nets the same size
        with the same encoder and the same three-term loss, leaving the decoder
        COUNT as the only difference. Default 1.0 leaves every existing
        checkpoint loadable.
        """
        super().__init__()
        c4, c3, c2, c1 = (dec_channels if dec_channels is not None
                          else self.dec_channels_for(dec_width))

        # Encoder, deliberately NOT scaled, so it stays byte-identical to DualDecoder's
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)
        self.drop4 = nn.Dropout2d(0.5)
        self.enc5 = _DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(2)

        # Single shared decoder
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = _DoubleConv(1024 + 512, c4)
        self.dec3 = _DoubleConv(c4 + 256, c3)
        self.dec2 = _DoubleConv(c3 + 128, c2)
        self.dec1 = _DoubleConv(c2 + 64, c1)

        # Dual output heads, the only architectural split
        self.head_bg = _make_head(c1)
        self.head_ext = _make_head(c1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.drop4(self.enc4(self.pool(s3)))
        b = self.drop5(self.enc5(self.pool(s4)))

        d4 = self.dec4(torch.cat([self.up(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), s1], dim=1))

        bg_pred = self.head_bg(d1)
        ext_pred = torch.relu(self.head_ext(d1))
        return bg_pred, ext_pred


if __name__ == "__main__":
    m = RainNetDualHead(in_channels=4)
    n = sum(p.numel() for p in m.parameters())
    print(f"RainNetDualHead params: {n:,}")
    with torch.no_grad():
        bg, ext = m(torch.zeros(1, 4, 64, 64))
    print(f"bg: {tuple(bg.shape)}   ext: {tuple(ext.shape)}   ext.min: {ext.min().item():.4f}")
