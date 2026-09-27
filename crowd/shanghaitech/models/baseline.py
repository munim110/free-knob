"""
Baseline: ResNet-50 encoder + single decoder.
Standard single-decoder architecture for crowd density estimation.
Optionally uses FDS (Feature Distribution Smoothing) on the bottleneck features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ResNetEncoder
from .decoder import DensityDecoder
from .lds_fds import FDSLayer


class BaselineNet(nn.Module):
    """
    ResNet-50 + single decoder baseline.
    Predicts density map at 1/4 scale, bilinearly upsampled to full resolution.

    If use_fds=True, an FDSLayer is inserted on the deepest encoder features
    (s4, 2048 channels) before passing to the decoder.
    """

    def __init__(self, pretrained=True, use_fds=False, num_bins=100, fds_sigma=2.0,
                 fds_apply_in_training=False):
        super().__init__()
        self.use_fds = use_fds
        self.encoder = ResNetEncoder(pretrained=pretrained)
        if use_fds:
            self.fds = FDSLayer(feature_dim=2048, num_bins=num_bins, sigma=fds_sigma,
                                apply_in_training=fds_apply_in_training)
        self.decoder = DensityDecoder(
            encoder_channels=self.encoder.channel_sizes,
            use_attention=False
        )

    def forward(self, x, bin_indices=None):
        input_size = x.shape[2:]
        s1, s2, s3, s4 = self.encoder(x)
        if self.use_fds:
            s4 = self.fds(s4, bin_indices=bin_indices)
        pred = self.decoder(s1, s2, s3, s4)
        # Upsample to input resolution
        pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=False)
        return pred
