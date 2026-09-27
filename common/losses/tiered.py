"""Loss functions for the U-Net baseline."""

import torch
import torch.nn as nn


class TieredWeightedMSELoss(nn.Module):
    """MSE with per-pixel weights derived from the original target temperature.

    Differs from the DualDecoder version in that it operates directly on the
    final prediction (no residual decomposition), so it takes only three
    arguments: the prediction, the target, and the stats dict.
    """

    def __init__(self, thresholds):
        super().__init__()
        self.thresholds = sorted(thresholds.items(), key=lambda kv: kv[0])
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, prediction_norm, target_norm, stats):
        loss = self.mse(prediction_norm, target_norm)
        with torch.no_grad():
            target_k = target_norm * (stats['target_std'] + 1e-8) + stats['target_mean']
            weights = torch.ones_like(target_k)
            for temp_k, weight in self.thresholds:
                weights[target_k <= temp_k] = weight
        return torch.mean(loss * weights)
