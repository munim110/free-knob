"""Post-hoc calibration control for single-decoder baselines.

The calibration concern
-----------------------
DualDecoder exposes an inference-time knob beta that traces a CSI/Bias frontier from one
checkpoint. The single-decoder baselines are quoted at a single operating point.
The control asks the same question: if the baselines were given an *analogous*
post-hoc knob (monotone rescaling, temperature scaling, or simply moving the
decision threshold) close the Bias gap at matched CSI, and thereby
explain DualDecoder's advantage as amplitude scaling rather than decomposition?

This script answers it. No model is retrained. Each converged single decoder is
run once per split; three post-hoc monotone knobs are then applied to the cached
predictions, each fitted on VALIDATION and reported on TEST:

  shift  y' = y - delta, i.e. classify the prediction at tau + delta while the
         ground truth stays at tau. The textbook decision-threshold knob, and
         the most favourable to the baseline because it moves the operating
         point at zero cost to the field. Implemented as a shift of the field
         rather than of the threshold argument, because calculate_metrics
         thresholds prediction and target with the same value, passing
         tau + delta there would silently re-define the task at a less sparse
         threshold and inflate CSI.
  gain   y' = mu_f + g * (y - mu_f) per frame. Expands the predicted dynamic
         range about the frame mean, the closest amplitude analogue of beta:
         it pushes cold anomalies colder without touching the frame mean.
  qmap   monotone quantile mapping of the pooled predicted distribution onto
         the pooled target distribution, fitted on validation. A proper
         distribution calibrator, the strongest of the three in principle.

A fourth knob, enabled by --include-sharpen, goes beyond the monotone control:

  sharpen  y' = y + k (y - GaussianBlur(y, sigma)), an unsharp mask, optionally
           composed with a shift to re-calibrate. Unlike the three above this is
           NOT a monotone pointwise transform: it raises the field's effective
           spatial Lipschitz constant, and so attacks the premise K_eq < g_true
           on which Theorem 4 rests rather than merely moving along the frontier
           the theorem describes. It is the cheap empirical stand-in for the
           question about Lipschitz-shaping architectures: if
           post-hoc sharpening reached DualDecoder's frontier, the architectural claim
           would be in trouble. If it cannot, because sharpening amplifies
           background noise as fast as critical structure, the claim is
           considerably strengthened. Reported separately, never mixed into the
           monotone-knob tables.

The decisive comparison is not the knob's best CSI but the shape of the frontier:
at the CSI the baseline's knob can reach, what Bias does it pay, and at Bias ~ 1
what CSI survives? Theorem 4 predicts the baseline frontier cannot reach
(CSI > 0, Bias ~ 1) however the knob is set, because a monotone pointwise
transform of a converged field cannot raise its spatial Lipschitz constant.

Usage
-----
    python src/posthoc_calibration_control.py \
        --data-dir data/processed/B08 --band 8 --threshold-k 220 \
        --unet-glob 'runs/unet_B08_s*/attention_unet_B08.pth' \
        --dual_decoder-glob 'runs/dual_decoder_B08_unified_dual_b2.0_s*/dual_decoder_unified_B08_best_csi.pth' \
        --out results/posthoc_control_B08.json
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC / "data"))
sys.path.insert(0, str(SRC / "models"))

# Dataset and model definitions are vendored under common/ so this runs from a
# clean checkout.
import sys as _s2
from pathlib import Path as _P2
_C = _P2(__file__).resolve().parents[1] / "common"
_s2.path[:0] = [str(_C), str(_C / "data"), str(_C / "models")]
from dataset import MultiVariableARDataset, resolve_variable_indices  # noqa: E402
from attention_unet import AttentionUNet  # noqa: E402
from dual_decoder_unified import DualDecoderUnified  # noqa: E402

# knob grids. deliberately generous at the top end so the baseline is never
# denied a plausible operating point.
# the single decoders over-predict cold extent (Bias ~ 4-6), so driving Bias
# toward 1 needs a warm shift: the grid runs well negative.
SHIFTS = [-25, -20, -16, -13, -10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 4, 6]   # Kelvin
GAINS = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]        # dimensionless
BETAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
SHARPEN_K = [0.0, 0.5, 1.0, 2.0, 4.0]                               # unsharp strength
SHARPEN_SIGMA = 2.0                                                  # pixels


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--band", type=int, default=8)
    p.add_argument("--threshold-k", type=float, default=220.0)
    p.add_argument("--variables", type=str, default="T500,T850,RH700,W500")
    p.add_argument("--target-shape", type=str, default="256x256")
    p.add_argument("--filter-percent", type=float, default=1.0)
    p.add_argument("--unet-glob", type=str, required=True,
                   help="glob over single-decoder checkpoints (one per seed)")
    p.add_argument("--dual_decoder-glob", type=str, default=None,
                   help="glob over DualDecoder checkpoints, for the reference frontier")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--unet-depth", type=int, default=4)
    p.add_argument("--dual_decoder-base-c", type=int, default=32)
    p.add_argument("--dual_decoder-beta", type=float, default=2.0,
                   help="beta at which DualDecoder is held while the four post-hoc knobs "
                        "are applied, matching the single-decoder treatment. "
                        "2.0 is the published value.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--include-sharpen", action="store_true",
                   help="add the non-monotone unsharp-mask knob (raises K_eq); "
                        "reported separately from the monotone knobs")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------- geometry ---
def crop_or_pad_with_mask(arr, target_shape):
    """Verbatim from src/eval_unet.py so the metric is identical."""
    h, w = arr.shape
    th, tw = target_shape
    valid_mask = np.ones(target_shape, dtype=bool)
    if h == th and w == tw:
        return arr, valid_mask
    start_h, start_w = max(0, (h - th) // 2), max(0, (w - tw) // 2)
    end_h, end_w = min(h, start_h + th), min(w, start_w + tw)
    cropped = arr[start_h:end_h, start_w:end_w]
    h_c, w_c = cropped.shape
    if h_c < th or w_c < tw:
        padded = np.zeros(target_shape, dtype=arr.dtype)
        padded[:h_c, :w_c] = cropped
        valid_mask[h_c:, :] = False
        valid_mask[:, w_c:] = False
        return padded, valid_mask
    return cropped, valid_mask


def calculate_metrics(pred_k, gt_k, valid_mask, threshold):
    """CSI/POD/FAR/Bias verbatim from src/eval_unet.py, plus intensity metrics.

    CSI and Bias are both *area* statistics: they depend only on which pixels
    cross tau, not on how deep the prediction goes past it. A monotone
    recalibration can therefore buy calibrated area by suppressing amplitude,
    relabelling which pixels cross tau without reconstructing intensity. The
    intensity metrics below separate the two, and they are the axis on which
    Theorem 4(A)'s premise (y_hat(s_c) >= y_e, i.e. peak recovery) actually lives.

      mae_crit   MAE over pixels where the TARGET is critical. "When it really is
                 convective, how close is the predicted temperature?"
      mae_bg     MAE over background pixels, as the reconstruction complement.
      p01_pred   1st percentile of the predicted field (a robust stand-in for the
                 cold peak; low-value-critical domain, so the peak is a minimum).
      p01_gt     the same statistic on the target.
      p01_err    p01_pred - p01_gt. Positive means the prediction is too warm,
                 i.e. it fails to reach the observed intensity.
    """
    pv, gv = pred_k[valid_mask], gt_k[valid_mask]
    pm, gm = pv <= threshold, gv <= threshold
    hits = int(np.sum(pm & gm))
    misses = int(np.sum(~pm & gm))
    fa = int(np.sum(pm & ~gm))
    denom = hits + misses + fa
    ae = np.abs(pv - gv)
    p01_p = float(np.percentile(pv, 1.0))
    p01_g = float(np.percentile(gv, 1.0))
    return {
        "csi": hits / denom if denom else 0.0,
        "pod": hits / (hits + misses) if (hits + misses) else 0.0,
        "far": fa / (hits + fa) if (hits + fa) else 0.0,
        "bias": (hits + fa) / (hits + misses) if (hits + misses) else 0.0,
        "actual_percent": float(np.mean(gm) * 100),
        # predicted critical area, label-free: the only quantity a deployable
        # rule may calibrate against without seeing the evaluation targets.
        "pred_percent": float(np.mean(pm) * 100),
        "_h": hits, "_m": misses, "_f": fa, "_n": int(pv.size),
        "mae_crit": float(ae[gm].mean()) if gm.any() else float("nan"),
        "mae_bg": float(ae[~gm].mean()) if (~gm).any() else float("nan"),
        "p01_pred": p01_p,
        "p01_gt": p01_g,
        "p01_err": p01_p - p01_g,
    }


def aggregate(preds, gts, masks, threshold, filter_percent):
    """Per-frame means plus pooled counts over the retained frame cohort."""
    ms = []
    for p, g, m in zip(preds, gts, masks):
        r = calculate_metrics(p, g, m, threshold)
        if r["actual_percent"] > filter_percent:
            ms.append(r)
    if not ms:
        return None
    keys = ("csi", "pod", "far", "bias", "mae_crit", "mae_bg",
            "p01_pred", "p01_gt", "p01_err", "actual_percent", "pred_percent")
    out = {k: float(np.nanmean([m[k] for m in ms])) for k in keys}

    # Pooled contingency alongside the per-frame means.
    #
    # The per-frame Bias is (hits+fa)/(hits+misses); on a frame holding only a
    # handful of critical pixels that denominator is tiny and the ratio is
    # heavy-tailed, so a mean over frames is dominated by the sparsest ones.
    # That is the dominant noise source when an operating point is selected by
    # |Bias - 1| on a small validation split, which is the failure
    # measured on both crowd and atmosphere. The pooled forms below are ratios
    # of sums rather than means of ratios, and are the stable statistic to
    # select on.
    H = sum(m["_h"] for m in ms); M = sum(m["_m"] for m in ms)
    F = sum(m["_f"] for m in ms); N = sum(m["_n"] for m in ms)
    out["csi_pooled"] = H / (H + M + F) if (H + M + F) else 0.0
    out["bias_pooled"] = (H + F) / (H + M) if (H + M) else 0.0
    out["pred_area_frac"] = (H + F) / N if N else 0.0
    out["true_area_frac"] = (H + M) / N if N else 0.0
    out["n_eventful"] = len(ms)
    return out


# ------------------------------------------------------------- inference ----
def cache_split(model, dataset, var_idx, target_shape, device, kind):
    """Return cached Kelvin fields for one split. kind in {'unet','dual_decoder'}."""
    stats = dataset.stats
    mean, std = stats["target_mean"], stats["target_std"] + 1e-8
    out = {"gt": [], "mask": []}
    if kind == "dual_decoder":
        out["bg"], out["ext"] = [], []
    else:
        out["pred"] = []

    for i in range(len(dataset)):
        predictor, target_norm, _ = dataset[i]
        x = predictor[var_idx, :, :].unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x)
        gt_native = target_norm.numpy().squeeze() * std + mean
        gt_k, m_gt = crop_or_pad_with_mask(gt_native, target_shape)

        if kind == "dual_decoder":
            bg, ext = y
            bg_native = (bg.cpu().numpy().squeeze() * std + mean).astype(np.float64)
            # ext lives in normalized units; to Kelvin is a pure scaling
            ext_native = (ext.cpu().numpy().squeeze() * std).astype(np.float64)
            bg_k, m_b = crop_or_pad_with_mask(bg_native, target_shape)
            ext_k, m_e = crop_or_pad_with_mask(ext_native, target_shape)
            out["bg"].append(bg_k); out["ext"].append(ext_k)
            out["mask"].append(m_b & m_e & m_gt)
        else:
            p_native = (y.cpu().numpy().squeeze() * std + mean).astype(np.float64)
            p_k, m_p = crop_or_pad_with_mask(p_native, target_shape)
            out["pred"].append(p_k)
            out["mask"].append(m_p & m_gt)
        out["gt"].append(gt_k)
    return out


# ------------------------------------------------------------------ knobs ---
def apply_shift(preds, delta):
    """y' = y - delta. Equivalent to testing y <= tau + delta with gt fixed at tau."""
    return [p - delta for p in preds]


def apply_gain(preds, g):
    """y' = mu_f + g (y - mu_f), per frame. Monotone, mean-preserving."""
    return [p.mean() + g * (p - p.mean()) for p in preds]


def fit_qmap(val_preds, val_gts, val_masks, n_q=1001):
    """Monotone quantile map from pooled predicted to pooled target distribution.

    Fitted on validation only. Returned as a pair of quantile vectors for
    np.interp, which is monotone by construction.
    """
    pv = np.concatenate([p[m] for p, m in zip(val_preds, val_masks)])
    gv = np.concatenate([g[m] for g, m in zip(val_gts, val_masks)])
    qs = np.linspace(0.0, 1.0, n_q)
    return np.quantile(pv, qs), np.quantile(gv, qs)


def apply_qmap(preds, src_q, dst_q):
    return [np.interp(p, src_q, dst_q) for p in preds]


def apply_sharpen(preds, k, sigma=SHARPEN_SIGMA):
    """Unsharp mask: y' = y + k (y - blur(y)). NOT monotone; raises K_eq."""
    from scipy.ndimage import gaussian_filter
    if k == 0.0:
        return list(preds)
    return [p + k * (p - gaussian_filter(p, sigma)) for p in preds]


# ------------------------------------------------------------- frontiers ----
def frontier_at_matched(curve, target_csi):
    """Bias on the knob curve at the point whose CSI is closest to target_csi.

    Reported alongside the achieved CSI so a miss is visible rather than hidden.
    """
    best = min(curve, key=lambda r: abs(r["csi"] - target_csi))
    return {"knob": best["knob"], "csi": best["csi"], "bias": best["bias"],
            "csi_gap_to_target": best["csi"] - target_csi}


def frontier_at_calibrated(curve):
    """CSI on the knob curve at the point whose Bias is closest to 1."""
    best = min(curve, key=lambda r: abs(r["bias"] - 1.0))
    return {"knob": best["knob"], "csi": best["csi"], "bias": best["bias"]}


def main():
    args = parse_args()
    target_shape = tuple(int(x) for x in args.target_shape.lower().split("x"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    var_idx = resolve_variable_indices([v.strip() for v in args.variables.split(",")])
    tau, fp = args.threshold_k, args.filter_percent

    splits = {s: MultiVariableARDataset(args.data_dir, s) for s in ("val", "test")}
    print(f"band B{args.band:02d}  tau={tau}K  device={device}  "
          f"val={len(splits['val'])} test={len(splits['test'])} frames")

    report = {"band": args.band, "threshold_k": tau, "filter_percent": fp,
              "frame_cohort": ("all" if fp < 0 else
                               f"observed coverage > {fp}%"),
              "shifts": SHIFTS, "gains": GAINS, "betas": BETAS,
              "single_decoder": {}, "dual_decoder": {}}

    # ---------------- single decoders, three knobs each --------------------
    for ck in sorted(glob.glob(args.unet_glob)):
        tag = Path(ck).parent.name
        model = AttentionUNet(input_channels=len(var_idx),
                             base_channels=args.base_channels,
                             depth=args.unet_depth, use_attention=True).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        cached = {s: cache_split(model, splits[s], var_idx, target_shape, device, "unet")
                  for s in ("val", "test")}

        arm = {}
        # -- shift: move the decision threshold only (as a shift of the field)
        for split in ("val", "test"):
            c = cached[split]
            arm.setdefault("shift", {})[split] = [
                {"knob": d, **aggregate(apply_shift(c["pred"], d), c["gt"], c["mask"], tau, fp)}
                for d in SHIFTS]
        # -- gain: expand dynamic range about the frame mean
        for split in ("val", "test"):
            c = cached[split]
            arm.setdefault("gain", {})[split] = [
                {"knob": g, **aggregate(apply_gain(c["pred"], g), c["gt"], c["mask"], tau, fp)}
                for g in GAINS]
        # -- qmap: fitted on val, applied to both
        src_q, dst_q = fit_qmap(cached["val"]["pred"], cached["val"]["gt"],
                                cached["val"]["mask"])
        for split in ("val", "test"):
            c = cached[split]
            arm.setdefault("qmap", {})[split] = [
                {"knob": "qmap",
                 **aggregate(apply_qmap(c["pred"], src_q, dst_q), c["gt"], c["mask"], tau, fp)}]
        # qmap composed with a shift, the strongest baseline recipe available
        for split in ("val", "test"):
            c = cached[split]
            mapped = apply_qmap(c["pred"], src_q, dst_q)
            arm.setdefault("qmap+shift", {})[split] = [
                {"knob": d, **aggregate(apply_shift(mapped, d), c["gt"], c["mask"], tau, fp)}
                for d in SHIFTS]

        # -- sharpen (+ shift): non-monotone, raises K_eq. Kept in its own key.
        if args.include_sharpen:
            for split in ("val", "test"):
                c = cached[split]
                rows = []
                for k in SHARPEN_K:
                    sharp = apply_sharpen(c["pred"], k)
                    for d in SHIFTS:
                        rows.append({"knob": f"k={k},d={d}", "k": k, "delta": d,
                                     **aggregate(apply_shift(sharp, d), c["gt"],
                                                 c["mask"], tau, fp)})
                arm.setdefault("sharpen+shift", {})[split] = rows

        report["single_decoder"][tag] = arm
        base = arm["shift"]["test"][SHIFTS.index(0)]
        print(f"  {tag:24s} baseline test CSI={base['csi']:.4f} Bias={base['bias']:.4f}")

    # ---------------- DualDecoder reference frontier ------------------------------
    if args.dual_decoder_glob:
        for ck in sorted(glob.glob(args.dual_decoder_glob)):
            tag = Path(ck).parent.name
            model = DualDecoderUnified(n_channels=len(var_idx), base_c=args.dual_decoder_base_c).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()
            cached = {s: cache_split(model, splits[s], var_idx, target_shape, device, "dual_decoder")
                      for s in ("val", "test")}
            def comb_at(c, b):
                return [bg - b * ext for bg, ext in zip(c["bg"], c["ext"])]

            arm = {}
            # (i) beta frontier. Kept under its own key for reference. This is
            #     what the published analysis used, and beta is an architectural
            #     blend weight, and therefore outside the post-hoc knob family.
            for split in ("val", "test"):
                c = cached[split]
                arm.setdefault("beta", {})[split] = [
                    {"knob": b, **aggregate(comb_at(c, b), c["gt"], c["mask"], tau, fp)}
                    for b in BETAS]

            # (ii) the matched control: DualDecoder held at its published beta, then given
            #      exactly the four knobs the single decoder receives. Without this
            #      the comparison calibrates one side only.
            cb = {s: comb_at(cached[s], args.dual_decoder_beta) for s in ("val", "test")}
            for split in ("val", "test"):
                c, p = cached[split], cb[split]
                arm.setdefault("shift", {})[split] = [
                    {"knob": d, **aggregate(apply_shift(p, d), c["gt"], c["mask"], tau, fp)}
                    for d in SHIFTS]
                arm.setdefault("gain", {})[split] = [
                    {"knob": g, **aggregate(apply_gain(p, g), c["gt"], c["mask"], tau, fp)}
                    for g in GAINS]
            src_q, dst_q = fit_qmap(cb["val"], cached["val"]["gt"], cached["val"]["mask"])
            for split in ("val", "test"):
                c, p = cached[split], cb[split]
                mapped = apply_qmap(p, src_q, dst_q)
                arm.setdefault("qmap", {})[split] = [
                    {"knob": "qmap",
                     **aggregate(mapped, c["gt"], c["mask"], tau, fp)}]
                arm.setdefault("qmap+shift", {})[split] = [
                    {"knob": d, **aggregate(apply_shift(mapped, d), c["gt"], c["mask"], tau, fp)}
                    for d in SHIFTS]

            report["dual_decoder"][tag] = arm
            print(f"  {tag:24s} DualDecoder cached: beta frontier + 4 matched knobs "
                  f"at beta={args.dual_decoder_beta}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
