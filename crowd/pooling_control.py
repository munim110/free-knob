"""The pooling confound outside radar: crowd density under the same sweep.

The SEVIR result is that a max-reduction neighbourhood convention drives the
pooled frequency bias of a mean-error-trained model far below one, and that a
free monotone knob buys back the bias deficit rather than any placement error.
If that is a property of the metric rather than of precipitation, it has to
reproduce on data that is not weather, with an architecture that is not a
nowcaster, in a benchmark whose authors never pooled with a max.

ShanghaiTech Part A is that benchmark. Its published protocol scores CSI on
32x32 patch sums exceeding a persons-per-patch count. That is a sum reduction,
the crowd analogue of average pooling. The domain therefore arrives with the
convention SEVIR finds safe, and the one SEVIR finds harmful can be added with
nothing else changed.

Pooling here is applied to the grid of patch counts, so `none` reproduces the
published protocol exactly and the thresholds stay in the paper's own units.

Registered predictions
----------------------
Written before the run, and reported as-is whether or not they hold.

P1  Under `none` (the published protocol) the mean-error models sit near
    bias 1 and the global quantile map buys little, because a sum reduction
    does not induce the deficit.
P2  Under `max` reductions the same models' pooled bias falls below one and the
    same knob's payoff rises, with the payoff ordered by the deficit.
P3  `gt_shift` is the ground-truth density map translated bodily by two
    patches, making it the crowd analogue of persistence: a real field with the
    observation's exact sharpness statistics, put in the wrong place. It should
    hold bias near one under every pooling and gain nothing from the knob,
    despite scoring badly. A confound driven by being wrong rather than by being
    smooth would show up in this arm.
P4  `gt_blur{s}` is the ground truth convolved with a Gaussian of width s, with
    placement left perfect. It should show bias falling monotonically with s
    under max reductions while staying near one under sum and average
    reductions, because a Gaussian blur conserves mass but destroys peaks. This
    is the mechanism with placement error held at zero, which no released
    checkpoint can provide.

P3 and P4 move sharpness and placement independently, which the trained models
confound. They are what make this a test rather than a measurement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, shift as ndshift
from torch.utils.data import DataLoader

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

CROWD = Path(str(P.CROWD))

# The dataset and model definitions are vendored under crowd/shanghaitech so this
# runs from a clean checkout. A user with their own working copy can point
# CROWD_DIR at it and that copy takes precedence.
_vendored = Path(__file__).resolve().parent / "shanghaitech"
for _p in ((CROWD / "src"), _vendored):
    if _p.is_dir():
        sys.path.insert(0, str(_p))

from dataset import ShanghaiTechDataset            # noqa: E402
from models.baseline import BaselineNet            # noqa: E402
from models.csrnet import CSRNet                   # noqa: E402
import config                                      # noqa: E402


PATCH = 32
NQ = 1001
THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
# (label, reduction, block) over the grid of patch counts. `none` is the
# published protocol; the rest are the neighbourhood conventions a nowcasting
# paper would reach for, applied to exactly the same counts.
POOLINGS = [("none", None, 1), ("avg2", "avg", 2), ("max2", "max", 2),
            ("avg4", "avg", 4), ("max4", "max", 4)]
BLURS = [1.0, 2.0, 4.0, 8.0]
# local contrast compression factors: 1.0 is identity, lower attenuates more
FLATS = [0.7, 0.5, 0.3, 0.1]
SHIFT_PATCHES = 2


def patch_grid(field, P=PATCH):
    """Patch sums as a 2-D grid, dropping the remainder like the crowd repo does."""
    h, w = field.shape
    nh, nw = h // P, w // P
    if nh == 0 or nw == 0:
        return None
    return field[:nh * P, :nw * P].reshape(nh, P, nw, P).sum(axis=(1, 3))


def pool(grid, how, k):
    """Non-overlapping reduction over the patch-count grid, stride = kernel."""
    if how is None or k == 1:
        return grid
    nh, nw = grid.shape[0] // k, grid.shape[1] // k
    if nh == 0 or nw == 0:
        return None
    b = grid[:nh * k, :nw * k].reshape(nh, k, nw, k)
    return b.mean(axis=(1, 3)) if how == "avg" else b.max(axis=(1, 3))


def counts(pred_grids, gt_grids, how, k, tau):
    """Pooled hits / false alarms / misses accumulated over the whole split."""
    H = F = M = N = P = 0
    for pg, gg in zip(pred_grids, gt_grids):
        pp, gp = pool(pg, how, k), pool(gg, how, k)
        if pp is None or gp is None:
            continue
        pe, ge = pp > tau, gp > tau
        H += int(np.sum(pe & ge))
        F += int(np.sum(pe & ~ge))
        M += int(np.sum(~pe & ge))
        N += pe.size
        P += int(np.sum(ge))
    denom, bden = H + F + M, H + M
    return {"hits": H, "false_alarms": F, "misses": M,
            "csi": H / denom if denom else float("nan"),
            "bias": (H + F) / bden if bden else float("nan"),
            "R": P / N if N else float("nan"), "n_blocks": N}


def build(arm):
    if arm.startswith("csrnet"):
        return CSRNet(pretrained=False)
    return BaselineNet(pretrained=False, use_fds="fds" in arm)


def run_model(arm, seed, loader, device, outputs_dir):
    ck = outputs_dir / f"{arm}_s{seed}" / "best_model.pth"
    if not ck.exists():
        return None
    model = build(arm)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.to(device).eval()
    preds, gts = [], []
    with torch.no_grad():
        for img, gt in loader:
            p = model(img.to(device))
            preds.append(np.maximum(p.squeeze().cpu().numpy(), 0).astype(np.float64))
            gts.append(gt.squeeze().numpy().astype(np.float64))
    return preds, gts


def raw_fields(loader):
    return [g.squeeze().numpy().astype(np.float64) for _, g in loader]


def synth(gts, kind, param):
    """Arms derived from the truth, so sharpness and placement move separately."""
    if kind == "shift":
        d = param * PATCH
        return [ndshift(g, (d, d), order=1, mode="constant", cval=0.0) for g in gts]
    if kind == "flat":
        # Local contrast compression: pull every pixel toward its own local mean.
        # This attenuates the amplitude a patch sum actually reads, unlike a
        # Gaussian blur, which conserves mass and so is nearly free here. It is
        # deliberately outside G: the compression is local, so no global
        # monotone map inverts it and any recovery the knob achieves is not
        # tautological. This arm tests the corrected mechanism directly rather
        # than by inference from the other two.
        out = []
        for g in gts:
            m = gaussian_filter(g, sigma=PATCH / 2.0, mode="constant", cval=0.0)
            out.append(np.maximum(m + param * (g - m), 0.0))
        return out
    # a Gaussian blur conserves total mass, so patch sums barely move while
    # local peaks are destroyed. Sharpness changes and placement does not.
    return [gaussian_filter(g, sigma=param, mode="constant", cval=0.0) for g in gts]


def fit_qmap(pred_fields, gt_fields):
    """One global map, pixel-density CDF onto the truth's, fitted on val only."""
    qs = np.linspace(0, 1, NQ)
    return (np.quantile(np.concatenate([p.ravel() for p in pred_fields]), qs),
            np.quantile(np.concatenate([g.ravel() for g in gt_fields]), qs))


# The criterion here is a patch *sum* exceeding a count, so the monotone family
# the domain calls for is not the one SEVIR uses. A pixel-wise quantile map
# matches marginals but does not conserve mass, which is what a sum reduction
# reads; a gain scales every patch sum linearly and is the knob that moves the
# operating point. We carry all three and let the val split choose, with
# identity (gain 1, shift 0) always available so the family can decline to act.
GAINS = [0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]
SHIFTS = [-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004, 0.008]


def knobs(sq, dq):
    """The family G, as (label, callable). Monotone and non-decreasing throughout."""
    fam = [("identity", lambda x: x)]
    fam += [(f"gain={g:g}", (lambda g: lambda x: g * x)(g))
            for g in GAINS if g != 1.0]
    fam += [(f"shift={s:+g}", (lambda s: lambda x: np.maximum(x + s, 0.0))(s))
            for s in SHIFTS if s != 0.0]
    fam += [("qmap", lambda x: np.interp(x, sq, dq))]
    return fam


def bias_cost(fields, gt_grids, fn):
    """Mean |Bias - 1| over every (pooling, threshold) cell, on the split given.

    Selecting on bias rather than on CSI is the conservative choice: it is the
    quantity the mechanism predicts, and it cannot tune directly for the score
    we then report.
    """
    grids = [patch_grid(fn(f)) for f in fields]
    keep = [i for i, g in enumerate(gt_grids) if g is not None
            and grids[i] is not None]
    gp = [grids[i] for i in keep]
    gg = [gt_grids[i] for i in keep]
    cs = []
    for lbl, how, k in POOLINGS:
        for tau in THRESHOLDS:
            b = counts(gp, gg, how, k, tau)["bias"]
            if np.isfinite(b):
                cs.append(abs(b - 1.0))
    return float(np.mean(cs)) if cs else float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="baseline,csrnet")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--outputs-dir", type=Path, default=CROWD / "outputs")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS / "crowd_pooling.json")))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    loaders = {}
    for name, kw in (("val", dict(split="val", use_train_val_split=True)),
                     ("test", dict(split="test_data"))):
        ds = ShanghaiTechDataset(config.DATA_ROOT, config.PART, augment=False, **kw)
        loaders[name] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    gts = {s: raw_fields(loaders[s]) for s in ("val", "test")}
    print(f"val {len(gts['val'])} images, test {len(gts['test'])} images")

    arms = {}
    for arm in [x.strip() for x in a.arms.split(",")]:
        got = {s: run_model(arm, a.seed, loaders[s], dev, a.outputs_dir)
               for s in ("val", "test")}
        if got["val"] is None or got["test"] is None:
            print(f"  !! {arm}_s{a.seed}: checkpoint missing, skipped")
            continue
        arms[arm] = {s: got[s][0] for s in ("val", "test")}
        print(f"  loaded {arm}_s{a.seed}")
    arms[f"gt_shift{SHIFT_PATCHES}"] = {
        s: synth(gts[s], "shift", SHIFT_PATCHES) for s in ("val", "test")}
    for b in BLURS:
        arms[f"gt_blur{b:g}"] = {s: synth(gts[s], "blur", b)
                                 for s in ("val", "test")}
    for a_ in FLATS:
        arms[f"gt_flat{a_:g}"] = {s: synth(gts[s], "flat", a_)
                                  for s in ("val", "test")}

    gt_val = [patch_grid(x) for x in gts["val"]]
    gt_test = [patch_grid(x) for x in gts["test"]]
    keep = [i for i, g in enumerate(gt_test) if g is not None]
    g_gt = [gt_test[i] for i in keep]

    rows, chosen = [], {}
    for name, f in arms.items():
        sq, dq = fit_qmap(f["val"], gts["val"])
        fam = knobs(sq, dq)
        # ONE member of G per arm, chosen on the val split by mean |Bias - 1|
        # over all cells and then held fixed across every pooling and every
        # threshold. This is the restrictive variant: it cannot be re-picked
        # per metric cell, and it is never fitted on anything we score.
        costs = {lbl: bias_cost(f["val"], gt_val, fn) for lbl, fn in fam}
        best = min(costs, key=costs.get)
        gfn = dict(fam)[best]
        chosen[name] = {"knob": best, "val_cost": costs[best],
                        "val_cost_identity": costs["identity"]}
        print(f"  {name:14s} knob={best:12s} "
              f"val mean|bias-1| {costs['identity']:.3f} -> {costs[best]:.3f}")

        g_id = [patch_grid(f["test"][i]) for i in keep]
        g_k = [patch_grid(gfn(f["test"][i])) for i in keep]
        # per-cell variant: the same family, but re-chosen for each (pooling,
        # threshold) on val. Still never fitted on test, still selected on bias
        # rather than on CSI. Reported alongside because one global transform
        # cannot serve an arm whose bias varies several-fold across cells.
        grids_val = {lbl: [patch_grid(fn(x)) for x in f["val"]] for lbl, fn in fam}
        grids_test = {lbl: [patch_grid(fn(f["test"][i])) for i in keep]
                      for lbl, fn in fam}
        kv = [i for i, g in enumerate(gt_val) if g is not None]
        gg_val = [gt_val[i] for i in kv]

        for lbl, how, k in POOLINGS:
            for tau in THRESHOLDS:
                # Two selection rules, both on val, both never touching test.
                #   bias : minimise |Bias - 1|. Conservative for our claim,
                #          since it cannot tune for the score we report. It is
                #          not neutral toward an arm whose bias is long, where
                #          pulling bias to one costs CSI.
                #   csi  : maximise val CSI. Each arm's best shot, so no arm can
                #          be said to have been handed a knob that hurt it.
                # We report both; a conclusion that needs a particular selection
                # rule is not a conclusion.
                pick, pcost = "identity", float("inf")
                pickc, pcsi = "identity", -1.0
                for klbl, _ in fam:
                    m = counts([grids_val[klbl][i] for i in kv], gg_val,
                               how, k, tau)
                    if np.isfinite(m["bias"]) and abs(m["bias"] - 1.0) < pcost:
                        pick, pcost = klbl, abs(m["bias"] - 1.0)
                    if np.isfinite(m["csi"]) and m["csi"] > pcsi:
                        pickc, pcsi = klbl, m["csi"]
                pc = counts(grids_test[pick], g_gt, how, k, tau)
                cc = counts(grids_test[pickc], g_gt, how, k, tau)
                u = counts(g_id, g_gt, how, k, tau)
                c = counts(g_k, g_gt, how, k, tau)
                gain = ((c["csi"] - u["csi"]) / u["csi"]
                        if np.isfinite(u["csi"]) and u["csi"] > 0
                        else float("nan"))
                gain_pc = ((pc["csi"] - u["csi"]) / u["csi"]
                           if np.isfinite(u["csi"]) and u["csi"] > 0
                           else float("nan"))
                npos = int(round(u["R"] * u["n_blocks"]))
                rows.append({"arm": name, "pooling": lbl, "threshold": tau,
                             "knob": best, "knob_percell": pick,
                             "knob_valcsi": pickc,
                             "csi_valcsi": cc["csi"], "bias_valcsi": cc["bias"],
                             "csi_uncal": u["csi"], "bias_uncal": u["bias"],
                             "csi_knob": c["csi"], "bias_knob": c["bias"],
                             "rel_gain": gain,
                             "csi_percell": pc["csi"], "bias_percell": pc["bias"],
                             "rel_gain_percell": gain_pc, "R": u["R"],
                             "n_blocks": u["n_blocks"], "n_pos": npos,
                             # a CSI over a handful of blocks is not a number we
                             # are willing to read; flag rather than silently drop
                             "usable": npos >= 30})

    a.out.write_text(json.dumps(
        {"patch": PATCH, "thresholds": THRESHOLDS,
         "poolings": [p[0] for p in POOLINGS], "seed": a.seed,
         "n_val": len(gts["val"]), "n_test": len(gts["test"]),
         "chosen_knob": chosen, "table": rows}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
