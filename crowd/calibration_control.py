"""Post-hoc calibration control on crowd density (ShanghaiTech Part A).

Why this domain matters
-----------------------
Measured on the test split, the crowd domain's critical fraction at the paper's
headline metric (CSI@10, patches of 32x32) is R = 0.652% -- two orders of
magnitude sparser than the atmospheric domain at 220 K (R = 20.27%) and within a
factor of three of radar at 2.5 mm (R = 0.243%). It is therefore a *second*
domain sitting in the regime Proposition 2 describes, and the natural place to
ask whether the atmospheric result replicates or reverses.

On atmosphere, at R = 20%, a monotone recalibration of a single decoder matches
the dual decoder and the architectural claim does not survive. If the same
control run here, at R = 0.65% with the same protocol and knob family, fails to
close the gap, the crossover is not an artifact of one domain's evaluation.

Knobs
-----
A density map is non-negative and the critical criterion is a patch *sum*
exceeding a count, so the monotone family differs from the atmospheric one:

    gain    y_hat -> g * y_hat            rescales total predicted mass
    shift   y_hat -> relu(y_hat + d)      adds uniform density, then clips
    qmap    val-fitted quantile map of the pixel-density CDF onto the truth's

`gain` is the one that matters: patch sums scale linearly with it, so it moves
the detection rate directly and is the crowd analogue of the shift that
recalibrated the atmospheric U-Net. All three are fitted on the held-out
validation split (30 images, a deterministic split from the training set, never
test) and reported on test.

DualDecoder is swept over beta on the same frames, giving the two systems a comparable
one-parameter family.

Metrics come from the crowd codebase's own `compute_patch_metrics`, unmodified,
so CSI / Bias / R / mae_high are exactly the paper's definitions. `mae_high` --
patch-count error on critical patches, is the crowd analogue of
MAE_c and is the axis on which a monotone knob is not expected to help.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

CROWD = Path(str(P.CROWD))
# Dataset and model definitions are vendored under crowd/shanghaitech so this
# runs from a clean checkout. A user with their own working copy can point
# CROWD_DIR at it and that copy takes precedence.
_v = _Path(__file__).resolve().parent / "shanghaitech"
for _p in ((CROWD / "src"), _v):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from dataset import ShanghaiTechDataset            # noqa: E402
from evaluate import compute_patch_metrics         # noqa: E402
from models.baseline import BaselineNet            # noqa: E402
from models.csrnet import CSRNet                   # noqa: E402
from models.dual_decoder import DualDecoderNet                    # noqa: E402
import config                                      # noqa: E402

GAINS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
SHIFTS = [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03]
BETAS = [0.0, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
NQ = 512


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--thresholds", type=str, default="2,3,4,5,7,10,15",
                   help="persons per 32x32 patch. This grid is an R-sweep: measured "
                        "test R = 12.10, 7.37, 4.79, 3.20, 1.58, 0.652, 0.215%%. "
                        "It is CONFOUND-FREE, unlike a threshold sweep on the "
                        "atmospheric models: no crowd arm evaluated here has an "
                        "evaluation threshold anywhere in its training. The "
                        "single-decoder arms are plain MSE / LDS / FDS / CSRNet, and "
                        "the DualDecoder arm uses the frequency decomposition, which is "
                        "parameterised by decomp_sigma alone. (dual_decoder_threshold is the "
                        "one arm that does use thresholds, and it is excluded by "
                        "default for exactly that reason.) Stops at 15 because only "
                        "20 of 182 test images hold any critical patch beyond it.")
    p.add_argument("--single-arms", type=str,
                   default="baseline,baseline_fds,baseline_lds,baseline_lds_fds,csrnet",
                   help="paper-era single-decoder run prefixes (2026-04-10 artifacts)")
    p.add_argument("--dual_decoder-arms", type=str, default="dual_decoder")
    p.add_argument("--dual_decoder-beta", type=float, default=2.0,
                   help="beta at which the dual decoder is held while the "
                        "same post-hoc knobs as the single decoder are applied")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--outputs-dir", type=Path, default=CROWD / "outputs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def build(arm):
    if arm.startswith("csrnet"):
        return CSRNet(pretrained=False), "single"
    if arm.startswith("dual_decoder"):
        return DualDecoderNet(pretrained=False, decomp_sigma=config.DECOMP_SIGMA), "dual_decoder"
    return BaselineNet(pretrained=False, use_fds="fds" in arm), "single"


def cache(model, loader, kind, device, beta_list=None):
    """Run the model once and keep the raw fields; knobs are applied afterwards."""
    out = []
    model.eval()
    with torch.no_grad():
        for img, gt in loader:
            img = img.to(device)
            if kind == "dual_decoder":
                # cache the two heads so beta can be swept without re-running
                _, bg, ext = model(img, beta=1.0)
                out.append((bg.squeeze().cpu().numpy().astype(np.float64),
                            ext.squeeze().cpu().numpy().astype(np.float64),
                            gt.squeeze().numpy().astype(np.float64)))
            else:
                pred = model(img)
                out.append((np.maximum(pred.squeeze().cpu().numpy(), 0).astype(np.float64),
                            None, gt.squeeze().numpy().astype(np.float64)))
    return out


def fit_qmap(cached):
    """Match the predicted pixel-density CDF to the truth's, fitted on val only."""
    src = np.concatenate([c[0].ravel() for c in cached])
    dst = np.concatenate([c[2].ravel() for c in cached])
    qs = np.linspace(0, 1, NQ)
    return np.quantile(src, qs), np.quantile(dst, qs)


def apply_qmap(a, sq, dq):
    return np.interp(a, sq, dq)


def patch_sums(field, P):
    """Non-overlapping patch sums, identical to the loop in compute_patch_metrics.

    That function walks `range(0, h - P + 1, P)`, i.e. floor(h/P) x floor(w/P)
    whole patches with any remainder dropped. Truncate-then-reshape gives the
    same patches in the same order, vectorised. Equivalence is asserted in
    tests/test_crowd_scoring.py against the original on real density maps.
    """
    h, w = field.shape
    nh, nw = h // P, w // P
    if nh == 0 or nw == 0:
        return np.zeros(0)
    return field[:nh * P, :nw * P].reshape(nh, P, nw, P).sum(axis=(1, 3)).ravel()


def metrics_from_sums(pred_p, gt_p, threshold):
    """Same definitions as the crowd repo's compute_patch_metrics, from patch sums."""
    pe, ge = pred_p > threshold, gt_p > threshold
    H = int(np.sum(pe & ge)); M = int(np.sum(~pe & ge)); F = int(np.sum(pe & ~ge))
    denom, bden = H + M + F, H + M
    hi, lo = ge, ~ge
    return {
        "hits": H, "misses": M, "false_alarms": F,
        "csi": H / denom if denom else float("nan"),
        "bias": (H + F) / bden if bden else float("nan"),
        "R": (int(np.sum(ge)) / len(gt_p)) if len(gt_p) else 0.0,
        "n_high": int(np.sum(hi)), "n_low": int(np.sum(lo)),
        "mae_high": float(np.mean(np.abs(pred_p[hi] - gt_p[hi]))) if hi.any() else float("nan"),
        "mae_low": float(np.mean(np.abs(pred_p[lo] - gt_p[lo]))) if lo.any() else float("nan"),
    }


def per_image_rows(fields, gts_patch, thresholds):
    """Per-image contingency at each threshold, for the fixed-tau stratification.

    The threshold sweep varies R by moving tau, which confounds sparsity with
    event severity: a rarer threshold is also a more extreme one. Holding tau
    fixed and stratifying images by their OWN critical fraction breaks that,
    the criterion is identical across strata and only R moves. These rows are
    what that analysis consumes.
    """
    P = config.PATCH_SIZE
    preds_patch = [patch_sums(f, P) for f in fields]
    out = []
    for pp, gp in zip(preds_patch, gts_patch):
        # the per-image critical fraction is threshold-specific and is carried in
        # each threshold's "R" field; there is no threshold-free version of it.
        rec = {"n_patches": int(len(gp))}
        for t in thresholds:
            m = metrics_from_sums(pp, gp, t)
            rec[str(t)] = {k: m[k] for k in
                           ("hits", "misses", "false_alarms", "n_high", "mae_high", "R")}
        out.append(rec)
    return out


def score(fields, gts_patch, thresholds):
    """Mean patch metrics over images, per threshold.

    `gts_patch` is the pre-computed ground-truth patch sums, which never change
    across knobs or thresholds, so they are computed once for the whole run.
    Patch sums for the prediction are computed once per image and reused across
    every threshold, which is where the cost went in the naive version: the
    reference implementation recomputes them per threshold, making the sweep
    7x more expensive than it needs to be for no change in the numbers.
    """
    P = config.PATCH_SIZE
    preds_patch = [patch_sums(f, P) for f in fields]
    res = {}
    for t in thresholds:
        rows = [metrics_from_sums(pp, gp, t) for pp, gp in zip(preds_patch, gts_patch)]
        keep = [r for r in rows if r["n_high"] > 0]      # images holding a critical patch
        agg = {}
        for k in ("csi", "bias", "mae_high", "mae_low", "R"):
            v = [r[k] for r in keep if np.isfinite(r[k])]
            agg[k] = float(np.mean(v)) if v else float("nan")
        # pooled contingency, so images with no critical patch still contribute their FAs
        H = sum(r["hits"] for r in rows)
        M = sum(r["misses"] for r in rows)
        F = sum(r["false_alarms"] for r in rows)
        agg["csi_pooled"] = H / (H + M + F) if (H + M + F) else float("nan")
        agg["bias_pooled"] = (H + F) / (H + M) if (H + M) else float("nan")
        agg["n_images_with_critical"] = len(keep)
        res[str(t)] = agg
    return res


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    thresholds = [float(t) for t in args.thresholds.split(",")]
    seeds = [s.strip() for s in args.seeds.split(",")]

    splits = {}
    for name, kw in (("val", dict(split="val", use_train_val_split=True)),
                     ("test", dict(split="test_data"))):
        ds = ShanghaiTechDataset(config.DATA_ROOT, config.PART, augment=False,
                                 crop_size=config.CROP_SIZE, **kw)
        splits[name] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
        print(f"{name}: {len(ds)} images")

    # ground-truth patch sums never change across arms, knobs or thresholds
    gt_patch = {}
    for name, loader in splits.items():
        gt_patch[name] = [patch_sums(g.squeeze().numpy().astype(np.float64),
                                     config.PATCH_SIZE) for _, g in loader]
        print(f"{name}: cached GT patch sums for {len(gt_patch[name])} images")

    report = {"thresholds": thresholds, "gains": GAINS, "shifts": SHIFTS,
              "betas": BETAS, "single_decoder": {}, "dual_decoder": {}}

    for arm in [a.strip() for a in args.single_arms.split(",")] + \
               [a.strip() for a in args.dual_decoder_arms.split(",")]:
        for seed in seeds:
            tag = f"{arm}_s{seed}"
            ck = args.outputs_dir / tag / "best_model.pth"
            if not ck.exists():
                print(f"  [missing] {tag}")
                continue
            model, kind = build(arm)
            sd = torch.load(ck, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
            model.to(device)

            cached = {s: cache(model, splits[s], kind, device) for s in ("val", "test")}

            if kind == "dual_decoder":
                entry = {}
                for s in ("val", "test"):
                    gts = gt_patch[s]
                    # clipped at zero exactly as the crowd repo's own evaluation does
                    entry[s] = [{"knob": b,
                                 **score([np.maximum(bg + b * ext, 0)
                                          for bg, ext, _ in cached[s]],
                                         gts, thresholds)}
                                for b in BETAS]
                # per-image rows at every beta, test split only, for stratification
                entry["per_image_test"] = {
                    str(b): per_image_rows([np.maximum(bg + b * ext, 0)
                                            for bg, ext, _ in cached["test"]],
                                           gt_patch["test"], thresholds)
                    for b in BETAS}

                # Matched control. beta is an architectural blend weight and therefore not a
                # post-hoc knob: without this the single decoder is calibrated
                # and the dual decoder is not, and the comparison measures the
                # difference in treatment.
                b0 = args.dual_decoder_beta
                comb = {s: [np.maximum(bg + b0 * ext, 0)
                            for bg, ext, _ in cached[s]] for s in ("val", "test")}
                qs = np.linspace(0, 1, NQ)
                sq = np.quantile(np.concatenate([p.ravel()
                                                 for p in comb["val"]]), qs)
                dq = np.quantile(np.concatenate([c[2].ravel()
                                                 for c in cached["val"]]), qs)
                for s in ("val", "test"):
                    gts = gt_patch[s]
                    preds = comb[s]
                    mapped = [apply_qmap(p, sq, dq) for p in preds]
                    entry.setdefault("gain", {})[s] = [
                        {"knob": g, **score([p * g for p in preds], gts,
                                            thresholds)}
                        for g in GAINS]
                    entry.setdefault("shift", {})[s] = [
                        {"knob": d, **score([np.maximum(p + d, 0) for p in preds],
                                            gts, thresholds)}
                        for d in SHIFTS]
                    entry.setdefault("qmap+gain", {})[s] = [
                        {"knob": g, **score([m * g for m in mapped], gts,
                                            thresholds)}
                        for g in GAINS]

                report["dual_decoder"][tag] = entry
                print(f"  {tag:28s} DualDecoder: beta frontier + 3 matched knobs "
                      f"at beta={b0}")
            else:
                sq, dq = fit_qmap(cached["val"])
                entry = {}
                for s in ("val", "test"):
                    gts = gt_patch[s]
                    preds = [c[0] for c in cached[s]]
                    mapped = [apply_qmap(p, sq, dq) for p in preds]
                    entry.setdefault("gain", {})[s] = [
                        {"knob": g, **score([p * g for p in preds], gts, thresholds)}
                        for g in GAINS]
                    entry.setdefault("shift", {})[s] = [
                        {"knob": d, **score([np.maximum(p + d, 0) for p in preds],
                                            gts, thresholds)}
                        for d in SHIFTS]
                    entry.setdefault("qmap+gain", {})[s] = [
                        {"knob": g, **score([m * g for m in mapped], gts, thresholds)}
                        for g in GAINS]
                # per-image rows for the untouched model, test split only
                entry["per_image_test"] = per_image_rows(
                    [c[0] for c in cached["test"]], gt_patch["test"], thresholds)
                report["single_decoder"][tag] = entry
                base = entry["gain"]["test"][GAINS.index(1.0)]
                t0 = str(thresholds[min(1, len(thresholds) - 1)])
                print(f"  {tag:28s} untouched test CSI@{t0}={base[t0]['csi']:.4f} "
                      f"Bias={base[t0]['bias']:.4f}")

            del model
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
