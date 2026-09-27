"""Loss functions for the unified-decomposition DualDecoder.

The decomposition is a clean additive partition:

    bg_target = y                              # full field
    ext_target = clamp(τ - y, min=0)            # low-value critical (atmosphere)
              = clamp(y - τ, min=0)            # high-value critical (crowds)

Inference combines them as

    y_hat = bg_pred - β · ext_pred              # low-value critical
          = bg_pred + β · ext_pred              # high-value critical

The bg head sees the full target and is trained with plain MSE.
The ext head sees the non-negative residual and is trained with an
importance-weighted MSE that places extra weight on pixels deeper in the
critical region (graded, not binary).
"""

import torch
import torch.nn as nn


def decompose_unified(target, threshold, mode='low'):
    """Compute (bg_target, ext_target) for the unified decomposition.

    Args:
        target:    tensor in any consistent units (Kelvin or normalized)
        threshold: scalar in the SAME units as target
        mode:      'low'  → critical region is target ≤ threshold (atmosphere)
                   'high' → critical region is target ≥ threshold (crowds)

    Returns:
        bg_target, equal to `target` (full field, no clipping)
        ext_target, non-negative magnitude of the critical residual
    """
    if mode == 'low':
        ext_target = torch.clamp(threshold - target, min=0.0)
    elif mode == 'high':
        ext_target = torch.clamp(target - threshold, min=0.0)
    else:
        raise ValueError(f"mode must be 'low' or 'high', got {mode!r}")
    return target, ext_target


class ImportanceWeightedMSELoss(nn.Module):
    """MSE loss with per-pixel importance weights based on the residual depth.

    For each pixel:
        w = 1 + λ · ext_target          (where ext_target ≥ 0)

    Pixels outside the critical region (ext_target == 0) get weight 1.
    Pixels deeper in the critical region get progressively higher weight.
    This is a graded version of the original tiered loss, no thresholds,
    just a monotone scaling.
    """

    def __init__(self, lambda_weight: float):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, ext_pred, ext_target):
        loss = self.mse(ext_pred, ext_target)
        weights = 1.0 + self.lambda_weight * ext_target  # ext_target is already ≥0
        return torch.mean(loss * weights)


# ---------------------------------------------------------------------------
# Frequency decomposition, the variant the crowd-density DualDecoder uses.
#
#     bg_target  = G_sigma * y
#     ext_target = y - G_sigma * y        (SIGNED)
#
# Unlike `decompose_unified`, ext_target is not a pointwise function of y: it is
# height relative to the LOCAL background, so the ext head solves a
# different problem from the bg head rather than the same one restricted to the
# tail.
#
# The signed residual requires a LINEAR ext head (RainNetDualDecoder(ext_activation=
# "linear")). With a ReLU head the negative part is unrepresentable, and the
# model can never pull the smoothed background back DOWN between rain cells,
# which is precisely where false alarms are generated.
# ---------------------------------------------------------------------------

def _gaussian_kernel1d(sigma: float, device, dtype):
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x, sigma: float):
    """Separable Gaussian blur over (B, C, H, W), reflect padding."""
    import torch.nn.functional as F
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def decompose_frequency(target, sigma: float):
    """(bg_target, ext_target) with a signed high-frequency residual.

    bg + 1.0 * ext == target exactly, so beta=1 is the reconstruction identity.
    """
    bg = gaussian_blur(target, sigma)
    return bg, target - bg


class SignedImportanceWeightedMSELoss(nn.Module):
    """IW-MSE for a signed residual, weighting only the critical direction.

        w = 1 + lambda * relu(+ext)   mode='high'  (heavy rain above local bg)
        w = 1 + lambda * relu(-ext)   mode='low'

    w = 1 + lambda*ext (the unified form) would go NEGATIVE on the other side.
    """

    def __init__(self, lambda_weight: float, mode: str = "high"):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.mode = mode
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, ext_pred, ext_target):
        crit = ext_target if self.mode == "high" else -ext_target
        w = 1.0 + self.lambda_weight * torch.clamp(crit, min=0.0)
        return torch.mean(self.mse(ext_pred, ext_target) * w)
