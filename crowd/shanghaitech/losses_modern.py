"""Modern imbalanced-regression / crowd-counting losses as baselines.

Why these three specifically. The SCDR impossibility argument covers
"pixel-aggregated losses with bounded, finite per-pixel reweighting", and
states what a loss must abandon to escape: either unbounded reweighting, or
per-pixel aggregation entirely. Two of the three losses here abandon per-pixel
aggregation, so they are the named escape route and the sharpest available
test of the scope claim; the third stays inside it and should therefore fail.

    DMCountLoss      optimal transport between predicted and target measures.
                     NOT per-pixel aggregated -> outside the theorem's scope.
    BayesianLoss     likelihood over annotation points, not pixels.
                     NOT per-pixel aggregated -> outside the theorem's scope.
    BalancedMSELoss  per-pixel MSE re-derived under a balanced prior. Still a
                     pixel-aggregated loss with finite reweighting -> INSIDE
                     the scope, so the theory predicts it cannot buy calibrated
                     detection. This is the confirmatory control.

Deviation from the published DM-Count, stated explicitly: the original uses
raw annotation points as the OT target measure. Here the target is the
Gaussian-smoothed density map that every other arm in this repo trains on.
The two differ by the annotation kernel width, and using the density map keeps
all arms on byte-identical targets, so the comparison isolates the loss rather
than confounding it with a change of supervision. The property that matters
for the scope test, that the OT term is not a per-pixel sum, is unaffected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinkhorn_potentials(a, b, C, reg: float, n_iter: int):
    """Dual potentials (f, g) of entropic OT, computed WITHOUT autograd.

    Differentiating through the Sinkhorn iterations is both unnecessary and
    ruinously expensive: each iteration materialises a (B,N,M) tensor, so 100
    iterations at B=16, N=M=1024 retains ~27 GB and OOMs. By the envelope
    theorem the gradient of the OT cost with respect to the source measure is
    the dual potential itself, so detached potentials give the exact gradient
    at a fraction of the memory. This is what the reference DM-Count
    implementation does (it backprops through `beta` from the Sinkhorn log).
    """
    with torch.no_grad():
        log_a = torch.log(a.clamp_min(1e-30))
        log_b = torch.log(b.clamp_min(1e-30))
        Kt = -C / reg
        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)
        for _ in range(n_iter):
            f = reg * (log_a - torch.logsumexp(Kt.unsqueeze(0) + (g / reg).unsqueeze(1), dim=2))
            g = reg * (log_b - torch.logsumexp(Kt.t().unsqueeze(0) + (f / reg).unsqueeze(1), dim=2))
    return f, g


def _ot_dual(a, b, C, reg: float, n_iter: int):
    """Entropic OT value, differentiable in `a` through the detached potentials."""
    f, g = _sinkhorn_potentials(a, b, C, reg, n_iter)
    return (f * a).sum(1) + (g * b).sum(1)


def _sinkhorn_log(a, b, C, reg: float, n_iter: int):
    """Primal transport cost <P,C>. Kept for the unit tests; not used in training."""
    B = a.shape[0]
    f, g = _sinkhorn_potentials(a, b, C, reg, n_iter)
    Kt = -C / reg
    P = torch.exp(Kt.unsqueeze(0) + (f / reg).unsqueeze(2) + (g / reg).unsqueeze(1))
    return (P * C.unsqueeze(0)).reshape(B, -1).sum(1)


class DMCountLoss(nn.Module):
    """Distribution Matching for Crowd Counting (Wang et al., NeurIPS 2020).

    L = |count_hat - count| + w_ot * OT(p_hat, p) + w_tv * TV(p_hat, p) * count
    with p_hat, p the mass-normalised density maps.

    The map is average-pooled by `downsample` before the OT term (as in the
    reference implementation) to keep the cost matrix tractable; pooling is
    mass-preserving here because we rescale by the pooling area.
    """

    def __init__(self, w_ot=0.1, w_tv=0.01, downsample=8, reg=0.1, n_iter=100,
                 debiased=True):
        super().__init__()
        self.w_ot, self.w_tv = w_ot, w_tv
        self.ds, self.reg, self.n_iter = downsample, reg, n_iter
        # Entropic OT is biased: OT_eps(a,a) > 0, growing with eps, so the raw
        # primal cost carries a large offset and a weak signal (measured:
        # OT(a,a)=0.343 at eps=10 on a cost matrix whose max is 2.0). The
        # debiased Sinkhorn divergence
        #     S(a,b) = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b)
        # is zero exactly at a=b and is far less sensitive to eps, so the
        # regulariser stops being a tuned hyperparameter.
        self.debiased = debiased
        self._cost_cache = {}

    def _cost(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._cost_cache:
            ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                                    torch.arange(w, device=device, dtype=dtype),
                                    indexing="ij")
            pts = torch.stack([ys.reshape(-1) / max(h - 1, 1),
                               xs.reshape(-1) / max(w - 1, 1)], dim=1)
            self._cost_cache[key] = torch.cdist(pts, pts).pow(2)
        return self._cost_cache[key]

    def forward(self, pred, target):
        pred = F.relu(pred)
        B = pred.shape[0]
        cnt_hat = pred.reshape(B, -1).sum(1)
        cnt = target.reshape(B, -1).sum(1)
        loss_count = (cnt_hat - cnt).abs().mean()

        # mass-preserving pooling
        ph = F.avg_pool2d(pred, self.ds) * (self.ds ** 2)
        th = F.avg_pool2d(target, self.ds) * (self.ds ** 2)
        h, w = ph.shape[-2:]
        a = ph.reshape(B, -1); b = th.reshape(B, -1)
        a = a / a.sum(1, keepdim=True).clamp_min(1e-8)
        b = b / b.sum(1, keepdim=True).clamp_min(1e-8)

        C = self._cost(h, w, pred.device, pred.dtype)
        ot = _ot_dual(a, b, C, self.reg, self.n_iter)
        if self.debiased:
            # d/da of the OT(b,b) term is zero, so it is omitted from the
            # differentiable path; the debiasing that matters is -0.5*OT(a,a).
            ot = ot - 0.5 * _ot_dual(a, a, C, self.reg, self.n_iter)
        loss_ot = ot.mean()

        loss_tv = (0.5 * (a - b).abs().sum(1) * cnt).mean()
        return loss_count + self.w_ot * loss_ot + self.w_tv * loss_tv


class BayesianLoss(nn.Module):
    """Bayesian Loss for crowd counting (Ma et al., ICCV 2019).

    Defined over ANNOTATION POINTS, not pixels: each point n gets a posterior
    over pixels, and the loss is |1 - sum_m p(n|m) yhat_m| summed over points
    (plus a background term in BL+). Requires point coordinates, so the dataset
    must supply them; see `collate_with_points` in dataset_points.py.

    points: list of (N_i, 2) tensors in (y, x) pixel coordinates.
    """

    def __init__(self, sigma=8.0, use_background=True, bg_ratio=1.0):
        super().__init__()
        self.sigma = sigma
        self.use_background = use_background
        self.bg_ratio = bg_ratio

    def forward(self, pred, points):
        pred = F.relu(pred)
        B, _, H, W = pred.shape
        dev, dt = pred.device, pred.dtype
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                                torch.arange(W, device=dev, dtype=dt), indexing="ij")
        grid = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1)     # (HW,2)

        total = pred.new_zeros(())
        for i in range(B):
            p = pred[i].reshape(-1)                                     # (HW,)
            pts = points[i].to(dev, dt)
            if pts.numel() == 0:
                total = total + p.sum()                                 # no people -> predict 0
                continue
            d = torch.cdist(pts, grid)                                  # (N,HW)
            logits = -d.pow(2) / (2 * self.sigma ** 2)
            if self.use_background:
                # a background "point" whose distance is the per-pixel nearest-point
                # distance scaled by bg_ratio, as in BL+
                dmin = d.min(dim=0).values                              # (HW,)
                bg = (-(dmin * self.bg_ratio).pow(2) / (2 * self.sigma ** 2)).unsqueeze(0)
                logits = torch.cat([logits, bg], dim=0)                 # (N+1,HW)
            post = torch.softmax(logits, dim=0)                         # over points
            e = (post * p.unsqueeze(0)).sum(1)                          # (N[+1],)
            tgt = torch.ones_like(e)
            if self.use_background:
                tgt[-1] = 0.0                                           # background expects 0
            total = total + (e - tgt).abs().sum()
        return total / B


def split_points_by_density(points, gt_dms, tau=10.0, patch=32):
    """Split each image's annotation points into (background, extreme) lists.

    A point is "extreme" when the evaluation patch containing it exceeds tau in
    the ground-truth density map, using the SAME tau and patch grid the CSI
    metric uses, so the split the network is asked to learn is exactly the split
    it is scored on. This is DualDecoder's threshold decomposition written in point
    coordinates instead of pixel values.

    points:  list of (N_i, 2) tensors in (y, x) pixel coordinates
    gt_dms:  (B, 1, H, W) ground-truth density maps
    """
    B, _, H, W = gt_dms.shape
    gh, gw = H // patch, W // patch
    bg_out, ext_out = [], []
    for i in range(B):
        p = points[i]
        if p.numel() == 0:
            bg_out.append(p)
            ext_out.append(p)
            continue
        # patch sums on the same grid compute_patch_metrics uses
        crop = gt_dms[i, 0, :gh * patch, :gw * patch]
        sums = crop.reshape(gh, patch, gw, patch).sum(dim=(1, 3))      # (gh, gw)
        pi = (p[:, 0].long() // patch).clamp(0, gh - 1)
        pj = (p[:, 1].long() // patch).clamp(0, gw - 1)
        is_ext = sums[pi, pj] > tau
        ext_out.append(p[is_ext])
        bg_out.append(p[~is_ext])
    return bg_out, ext_out


class BalancedMSELoss(nn.Module):
    """Balanced MSE, BMC variant (Ren et al., CVPR 2022).

    Re-derives squared error under a balanced test prior by treating regression
    as classification against the other targets in the batch. Applied here to a
    random subset of pixels per step, since a dense map has too many.

    This remains a pixel-aggregated loss with finite per-pixel weighting, so it
    sits INSIDE the impossibility argument's scope: it is the control that
    should not buy calibrated detection.
    """

    def __init__(self, init_noise_sigma=None, n_pixels=2048, learn_sigma=True,
                 sigma_scale=1.0):
        super().__init__()
        # The published default (1.0) is tuned for scalar regression targets of
        # order 1. Crowd density pixels are ~1e-3, and at sigma >> the target
        # spread every pairwise logit collapses to zero, the softmax goes
        # uniform, and the loss becomes numerically IDENTICAL for a good
        # prediction, a bad one, and predicting zero everywhere (verified:
        # 16.635529 for all three at sigma=1.0). That is a dead loss that still
        # trains and reports plausible numbers, so it must not be the default.
        # Passing None self-scales sigma to the first batch's target std.
        self.learn_sigma = learn_sigma
        self.sigma_scale = sigma_scale
        self.n_pixels = n_pixels
        if init_noise_sigma is None:
            self.log_noise_sigma = nn.Parameter(torch.zeros(())) if learn_sigma \
                else torch.zeros(())
            self._initialised = False
        else:
            v = torch.tensor(float(init_noise_sigma)).log()
            self.log_noise_sigma = nn.Parameter(v) if learn_sigma else v
            self._initialised = True

    def forward(self, pred, target):
        p = pred.reshape(-1)
        t = target.reshape(-1)
        if not self._initialised:
            sd = t.std().clamp_min(1e-8) * self.sigma_scale
            with torch.no_grad():
                self.log_noise_sigma.copy_(sd.log().to(self.log_noise_sigma.device))
            self._initialised = True
        n = min(self.n_pixels, p.numel())
        idx = torch.randperm(p.numel(), device=p.device)[:n]
        p, t = p[idx].unsqueeze(1), t[idx].unsqueeze(1)          # (n,1)
        noise_var = (self.log_noise_sigma.to(p.device).exp()) ** 2
        logits = -0.5 * (p - t.t()).pow(2) / noise_var.clamp_min(1e-8)
        labels = torch.arange(n, device=p.device)
        loss = F.cross_entropy(logits, labels)
        return loss * (2 * noise_var).detach()


class DMCountOfficialLoss(nn.Module):
    """DM-Count with the AUTHORS' own OT code, points as the target measure.

    DMCountLoss above deviates from the paper in one documented way: it matches
    against the Gaussian-smoothed density map rather than the raw annotation
    points. That deviation keeps every arm on identical targets, but it is also
    a material implementation choice, since the OT term is the whole reason
    DM-Count is the named escape route in Section 4. So this
    arm removes it: `dmcount_official/ot_loss.py` is the authors' file copied
    verbatim, the target measure is uniform mass over the annotation points,
    and the composition, weights and regulariser are the published defaults
    (wot=0.1, wtv=0.01, reg=10, 100 Sinkhorn iterations, unnormalised
    coordinates, stride 8). If the two arms agree, the deviation was harmless;
    if they disagree, this one is the number to report.

    The prediction is mass-preservingly pooled to stride 8 because the authors'
    network emits density at 1/8 resolution while BaselineNet emits full scale.
    Pooling, not interpolation, so the predicted count is untouched.
    """

    def __init__(self, crop_size=256, stride=8, w_ot=0.1, w_tv=0.01,
                 reg=10.0, n_iter=100, norm_cood=0):
        super().__init__()
        from dmcount_official import OT_Loss
        self.crop_size, self.stride = crop_size, stride
        self.w_ot, self.w_tv = w_ot, w_tv
        self._ot_args = (crop_size, stride, norm_cood, n_iter, reg)
        self._ot = None
        self._OT_Loss = OT_Loss
        self.grid = crop_size // stride

    def _ot_module(self, device):
        # OT_Loss caches its coordinate grid on a fixed device at construction
        if self._ot is None or self._ot.device != device:
            c, s, nc, ni, rg = self._ot_args
            self._ot = self._OT_Loss(c, s, nc, device, ni, rg)
        return self._ot

    def _discrete(self, points, device, dtype):
        """Point counts binned onto the stride-8 grid (the authors' gt_discrete)."""
        g = self.grid
        out = torch.zeros(len(points), 1, g, g, device=device, dtype=dtype)
        for i, p in enumerate(points):
            if p.numel() == 0:
                continue
            idx = (p.to(device) / self.stride).long().clamp_(0, g - 1)   # p is (y, x)
            flat = idx[:, 0] * g + idx[:, 1]
            out[i, 0].view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=dtype))
        return out

    def forward(self, pred, points):
        pred = F.relu(pred)
        # full-res -> stride-8 density, mass preserved
        d = F.avg_pool2d(pred, self.stride) * (self.stride ** 2)
        B = d.shape[0]
        dev, dt = d.device, d.dtype

        cnt = d.reshape(B, -1).sum(1)
        gd = torch.tensor([float(len(p)) for p in points], device=dev, dtype=dt)
        d_normed = d / (cnt.view(B, 1, 1, 1) + 1e-6)

        # OT_Loss indexes points as (x, y); the dataset yields (y, x)
        pts_xy = [p.to(dev, dt).flip(1) if p.numel() else p.to(dev, dt) for p in points]
        ot, _, _ = self._ot_module(dev)(d_normed, d, pts_xy)

        gtd = self._discrete(points, dev, dt)
        gtd_normed = gtd / (gtd.reshape(B, -1).sum(1).view(B, 1, 1, 1) + 1e-6)
        tv = ((d_normed - gtd_normed).abs().reshape(B, -1).sum(1) * gd).mean()

        count = (cnt - gd).abs().mean()
        return count + self.w_ot * ot.squeeze() + self.w_tv * tv


def build_loss(name, **kw):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "dmcount":
        return DMCountLoss(**kw)
    if name == "dmcount_official":
        return DMCountOfficialLoss(**kw)
    if name == "bayesian":
        return BayesianLoss(**kw)
    if name == "balancedmse":
        return BalancedMSELoss(**kw)
    raise ValueError(f"unknown loss {name!r}")


NEEDS_POINTS = {"bayesian", "dmcount_official"}
