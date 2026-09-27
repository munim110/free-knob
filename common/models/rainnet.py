"""RainNet (single-decoder baseline), PyTorch port of Ayzel 2020.

Faithful 1-to-1 port of the Keras model at
    https://github.com/hydrogo/rainnet/blob/master/rainnet.py

Structure:
  - U-Net-family encoder-decoder (no attention, no residuals, no BN)
  - 5 encoder levels (64 / 128 / 256 / 512 / 1024), each = two 3x3 conv + ReLU
  - 4 max-pool downsamples (between levels 1-4)
  - Dropout(0.5) after the level-4 double-conv and the bottleneck level-5
  - 4 decoder levels (upsample + concat skip + two 3x3 conv + ReLU)
  - Bridge conv (64 -> 2 channels, 3x3, ReLU) then 1x1 output conv (linear)
  - ~31.4 M parameters, identical to the Keras version

Input shape: (B, C_in, H, W) where H, W are multiples of 16 (four 2x pools).
For the RY product the canonical size is 928x928 (900 mirror-padded).
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


class RainNet(nn.Module):
    """Vanilla RainNet, single output head, regression mode."""

    def __init__(self, in_channels: int = 4, mode: str = "regression"):
        super().__init__()
        if mode not in ("regression", "segmentation"):
            raise ValueError(f"mode must be 'regression' or 'segmentation', got {mode!r}")
        self.mode = mode

        # Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)
        self.drop4 = nn.Dropout2d(0.5)
        self.enc5 = _DoubleConv(512, 1024)
        self.drop5 = nn.Dropout2d(0.5)
        self.pool = nn.MaxPool2d(2)

        # Decoder (Keras UpSampling2D == nearest-neighbour 2x, no params)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = _DoubleConv(1024 + 512, 512)
        self.dec3 = _DoubleConv(512 + 256, 256)
        self.dec2 = _DoubleConv(256 + 128, 128)
        self.dec1 = _DoubleConv(128 + 64, 64)

        # Bridge (Keras: Conv2D(2, 3, activation='relu'))
        self.bridge = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(2, 1, kernel_size=1)

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

        y = self.out(self.bridge(d1))
        if self.mode == "segmentation":
            y = torch.sigmoid(y)
        return y


if __name__ == "__main__":
    m = RainNet(in_channels=4)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"RainNet params: {n_params:,}  (Keras ref: 31,388,673)")
    with torch.no_grad():
        z = m(torch.zeros(1, 4, 928, 928))
    print(f"Output shape: {tuple(z.shape)}")
