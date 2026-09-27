"""
LDS (Label Distribution Smoothing) and FDS (Feature Distribution Smoothing)
from Yang et al., "Delving into Deep Imbalanced Regression", ICML 2021.

Adapted for 2D density map regression:
- LDS: smooth the label distribution to compute per-sample weights
- FDS: smooth feature statistics across label bins to calibrate representations
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d


class LDSMixin:
    """
    Label Distribution Smoothing for density estimation.

    Computes per-sample weights by:
    1. Binning training samples by their integrated density (total count)
    2. Building a histogram of label distribution
    3. Smoothing the histogram with a Gaussian kernel
    4. Weighting each sample inversely to its bin's smoothed frequency
    """

    @staticmethod
    def compute_lds_weights(dataset, num_bins=100, sigma=2.0, max_weight=10.0):
        """
        Compute LDS weights for all samples in the dataset.

        Args:
            dataset: ShanghaiTechDataset
            num_bins: number of bins for the label histogram
            sigma: Gaussian kernel sigma for smoothing
            max_weight: cap on maximum weight

        Returns:
            weights: (N,) numpy array of per-sample weights
        """
        # Compute integrated density (head count) for each sample
        counts = []
        for _, dm_path in dataset.samples:
            dm = np.load(dm_path)
            counts.append(dm.sum())
        counts = np.array(counts)

        # Build histogram
        bin_edges = np.linspace(counts.min() - 1, counts.max() + 1, num_bins + 1)
        bin_indices = np.digitize(counts, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        hist = np.bincount(bin_indices, minlength=num_bins).astype(np.float64)

        # Smooth histogram
        hist_smooth = gaussian_filter1d(hist, sigma=sigma)
        hist_smooth = np.maximum(hist_smooth, 1e-8)

        # Inverse frequency weighting
        weights = 1.0 / hist_smooth[bin_indices]
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        # Cap
        weights = np.minimum(weights, max_weight)

        return weights.astype(np.float32)


class FDSLayer(nn.Module):
    """
    Feature Distribution Smoothing layer.

    Placed between encoder and decoder. Smooths running feature statistics
    (mean, variance) across label bins so that rare-count images get
    feature calibration from neighboring bins.
    """

    def __init__(self, feature_dim, num_bins=100, sigma=2.0, momentum=0.9,
                 apply_in_training=False):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.momentum = momentum
        # If True, apply calibration in the training forward pass (Yang et al.
        # behavior: decoder learns to consume calibrated features). Default
        # False preserves the original buggy no-op behavior of this codebase.
        self.apply_in_training = apply_in_training

        # Running statistics per bin: mean and var of features
        self.register_buffer('running_mean', torch.zeros(num_bins, feature_dim))
        self.register_buffer('running_var', torch.ones(num_bins, feature_dim))
        self.register_buffer('bin_counts', torch.zeros(num_bins))

        # Smoothed statistics
        self.register_buffer('smoothed_mean', torch.zeros(num_bins, feature_dim))
        self.register_buffer('smoothed_var', torch.ones(num_bins, feature_dim))

        self._calibrated = False

    def update_statistics(self, features, bin_indices):
        """
        Update running statistics during training.

        Args:
            features: (B, C) global-average-pooled features
            bin_indices: (B,) integer bin index for each sample
        """
        for i in range(features.shape[0]):
            b = bin_indices[i].item()
            if b < 0 or b >= self.num_bins:
                continue
            n = self.bin_counts[b].item()
            if n == 0:
                self.running_mean[b] = features[i].detach()
                self.running_var[b] = torch.zeros_like(features[i])
            else:
                self.running_mean[b] = (self.momentum * self.running_mean[b] +
                                        (1 - self.momentum) * features[i].detach())
                diff = features[i].detach() - self.running_mean[b]
                self.running_var[b] = (self.momentum * self.running_var[b] +
                                       (1 - self.momentum) * diff ** 2)
            self.bin_counts[b] += 1

    def calibrate(self):
        """Smooth running statistics across bins."""
        # Gaussian smooth along bin axis
        mean_np = self.running_mean.cpu().numpy()
        var_np = self.running_var.cpu().numpy()

        for c in range(mean_np.shape[1]):
            mean_np[:, c] = gaussian_filter1d(mean_np[:, c], sigma=self.sigma)
            var_np[:, c] = gaussian_filter1d(var_np[:, c], sigma=self.sigma)

        self.smoothed_mean.copy_(torch.from_numpy(mean_np))
        self.smoothed_var.copy_(torch.from_numpy(var_np).clamp(min=1e-6))
        self._calibrated = True

    def _apply_calibration(self, features, bin_indices):
        """Shift+scale features from raw bin stats to smoothed bin stats."""
        calibrated = features.clone()
        for i in range(features.shape[0]):
            b = bin_indices[i].item()
            b = max(0, min(b, self.num_bins - 1))

            raw_mean = self.running_mean[b]
            raw_std = self.running_var[b].sqrt()
            smooth_mean = self.smoothed_mean[b]
            smooth_std = self.smoothed_var[b].sqrt()

            scale = (smooth_std / (raw_std + 1e-6)).view(-1, 1, 1)
            calibrated[i] = (features[i] - raw_mean.view(-1, 1, 1)) * scale + smooth_mean.view(-1, 1, 1)

        return calibrated

    def forward(self, features, bin_indices=None):
        """
        Apply feature distribution smoothing.

        During training: update statistics; if apply_in_training and already
        calibrated, also shift features so the decoder learns calibrated inputs.
        At inference with bin_indices: shift features from raw to smoothed stats.
        Otherwise: pass through unchanged.

        Args:
            features: (B, C, H, W) feature maps
            bin_indices: (B,) bin indices (needed for training stats update and
                calibrated inference)
        """
        if self.training and bin_indices is not None:
            pooled = features.mean(dim=(2, 3))
            self.update_statistics(pooled, bin_indices)
            if self.apply_in_training and self._calibrated:
                features = self._apply_calibration(features, bin_indices)
            return features

        if self._calibrated and bin_indices is not None:
            return self._apply_calibration(features, bin_indices)

        return features
