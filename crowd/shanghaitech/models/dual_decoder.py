"""
DualDecoder (Dual Architecture for Regression Tasks) for crowd density estimation.

Shared ResNet-50 encoder feeding two separate decoders:
- Background decoder: learns smooth low-frequency component (y_bg = G_sigma * y)
- Extreme decoder: learns high-frequency residual (y_ext = y - y_bg)

Inference: y_hat = y_bg_hat + beta * y_ext_hat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
import numpy as np
from .encoder import ResNetEncoder
from .decoder import DensityDecoder
from .lds_fds import FDSLayer


class TieredWeightedMSELoss(nn.Module):
    """
    Per-pixel weighted MSE matching the atmospheric DualDecoder loss.

    Pixels are upweighted based on the ORIGINAL target density value rather than the
    residual. For crowd density, pixels with density above the tier thresholds
    receive higher weight (e.g., {0.01: 10x, 0.05: 25x}).

    This is the loss applied to the extreme-event head only. The continuity
    head and the final fused prediction use plain MSE.

    Args:
        thresholds: dict mapping density_value -> weight
                    e.g. {0.01: 10.0, 0.05: 25.0}
                    Pixels with original_density >= threshold get the weight.
                    Higher thresholds overwrite lower ones (sorted ascending).
    """

    def __init__(self, thresholds):
        super().__init__()
        # Sort ascending so higher density thresholds overwrite lower ones
        self.thresholds = sorted(thresholds.items(), key=lambda kv: kv[0])
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, prediction, target_residual, original_target):
        """
        Args:
            prediction: (B, 1, H, W) ext-head predictions
            target_residual: (B, 1, H, W) ext_target = max(y - tau, 0)
            original_target: (B, 1, H, W) original density map y
        """
        loss = self.mse(prediction, target_residual)
        with torch.no_grad():
            weights = torch.ones_like(original_target)
            for density_thresh, weight in self.thresholds:
                weights[original_target >= density_thresh] = weight
        return torch.mean(loss * weights)


class DualDecoderNet(nn.Module):
    """
    DualDecoder dual-decoder architecture for crowd density.

    The threshold-agnostic decomposition uses spatial frequency:
    - y_bg = GaussianSmooth(y, decomp_sigma)
    - y_ext = y - y_bg (high-frequency residual at density peaks)

    Optional FDS layer applied to the shared bottleneck features (s4, 2048ch)
    before the decoders branch.
    """

    def __init__(self, pretrained=True, decomp_sigma=15.0,
                 use_fds=False, num_bins=100, fds_sigma=2.0):
        super().__init__()
        self.decomp_sigma = decomp_sigma
        self.use_fds = use_fds

        self.encoder = ResNetEncoder(pretrained=pretrained)

        if use_fds:
            self.fds = FDSLayer(feature_dim=2048, num_bins=num_bins, sigma=fds_sigma)

        # Background decoder: learns smooth field
        self.bg_decoder = DensityDecoder(
            encoder_channels=self.encoder.channel_sizes,
            use_attention=False
        )

        # Extreme decoder: learns high-frequency residual with attention
        self.ext_decoder = DensityDecoder(
            encoder_channels=self.encoder.channel_sizes,
            use_attention=True
        )

    def forward(self, x, beta=1.0, bin_indices=None):
        """
        Args:
            x: (B, 3, H, W) input image
            beta: scaling factor for extreme decoder output at inference
            bin_indices: (B,) integer bin indices for FDS (train/inference)

        Returns:
            final_pred: y_bg_hat + beta * y_ext_hat
            bg_pred: background decoder output
            ext_pred: extreme decoder output
        """
        input_size = x.shape[2:]
        s1, s2, s3, s4 = self.encoder(x)

        if self.use_fds:
            s4 = self.fds(s4, bin_indices=bin_indices)

        bg_pred = self.bg_decoder(s1, s2, s3, s4)
        bg_pred = F.interpolate(bg_pred, size=input_size, mode='bilinear', align_corners=False)

        ext_pred = self.ext_decoder(s1, s2, s3, s4)
        ext_pred = F.interpolate(ext_pred, size=input_size, mode='bilinear', align_corners=False)

        final_pred = bg_pred + beta * ext_pred

        return final_pred, bg_pred, ext_pred

    @staticmethod
    def decompose_target(density_map, sigma):
        """
        Threshold-agnostic frequency decomposition of ground truth density map.

        Args:
            density_map: (H, W) numpy array, ground truth density
            sigma: smoothing sigma for background extraction

        Returns:
            bg_target: (H, W) smooth background component
            ext_target: (H, W) high-frequency residual (zero-mass)
        """
        bg_target = scipy.ndimage.gaussian_filter(
            density_map.astype(np.float64), sigma=sigma, mode='constant'
        ).astype(np.float32)
        ext_target = density_map - bg_target
        return bg_target, ext_target

    @staticmethod
    def decompose_target_threshold(density_map, tau):
        """
        Threshold-based decomposition matching the atmospheric DualDecoder setup.

        For atmospheric (extreme = COLD): bg = max(y, tau), ext = y - bg <= 0.
        For crowd density (extreme = HIGH): bg = min(y, tau), ext = y - bg >= 0.

        Args:
            density_map: (H, W) numpy/tensor, ground truth density
            tau: per-pixel density threshold (proxy for high-density region)

        Returns:
            bg_target: density capped at tau (smooth sub-critical field)
            ext_target: positive excess above tau (concentrated at peaks)
        """
        if isinstance(density_map, np.ndarray):
            bg_target = np.minimum(density_map, tau).astype(np.float32)
            ext_target = (density_map - bg_target).astype(np.float32)
        else:
            bg_target = torch.clamp(density_map, max=tau)
            ext_target = density_map - bg_target
        return bg_target, ext_target

    @staticmethod
    def decompose_target_unified(density_map, tau):
        """
        Unified decomposition: background head learns the FULL field, extreme
        head learns the non-negative critical residual.

        For crowd density (extreme = HIGH):
            bg_target = y                              (full field, no clipping)
            ext_target = max(y - tau, 0)               (non-negative excess above tau)

        For atmospheric (extreme = COLD), the analog would be:
            bg_target = y
            ext_target = max(tau - y, 0)               (non-negative depth below tau)
            inference uses ŷ = bg_pred - β · ext_pred

        Properties:
        - bg head's training task is identical to a single-decoder baseline
          (gradient descent's natural MSE-optimal solution)
        - ext head's gradient signal is non-zero ONLY at critical pixels
          → architectural gradient isolation by construction
        - At β=0: ŷ = bg_pred ≈ baseline solution
        - At β>0: amplifies above-threshold mass at critical locations
        """
        if isinstance(density_map, np.ndarray):
            bg_target = density_map.astype(np.float32).copy()
            ext_target = np.maximum(density_map - tau, 0).astype(np.float32)
        else:
            bg_target = density_map
            ext_target = torch.clamp(density_map - tau, min=0)
        return bg_target, ext_target

    @staticmethod
    def importance_weights(ext_target, mode='quadratic', floor=0.1):
        """
        Importance weighting for extreme decoder loss.
        Pixels with larger residual magnitude get higher weight.

        Args:
            ext_target: (B, 1, H, W) tensor, extreme residual
            mode: 'linear' (1 + |ext|/mean), 'quadratic' (w^2), or 'mask' (hard floor)
            floor: minimum weight in 'mask' mode

        Returns:
            weights: (B, 1, H, W) tensor
        """
        abs_ext = ext_target.abs()
        mean_abs = abs_ext.mean(dim=(2, 3), keepdim=True) + 1e-8
        normalized = abs_ext / mean_abs  # mean == 1

        if mode == 'linear':
            weights = 1.0 + normalized
        elif mode == 'quadratic':
            # Quadratic: strongly boosts high-residual pixels
            weights = 1.0 + normalized ** 2
        elif mode == 'mask':
            # Hard floor: high weight for pixels above mean, low for rest
            high = (normalized > 1.0).float()
            weights = floor + (1.0 - floor) * high + normalized * high
        else:
            raise ValueError(f"Unknown importance weighting mode: {mode}")

        return weights
