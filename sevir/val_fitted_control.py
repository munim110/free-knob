"""Refit the global quantile map on a prior-period validation split.

The knob everywhere else in the paper is fitted on one index-parity half of the
test split. That is the right control for the confound claim, because both
halves are drawn from one distribution and what is at stake is the comparison
between arms rather than the absolute size of the gain. It invites a second
reading the paper does not intend, that the transform is free at deployment
time, where the fit would come from earlier data and would have to survive
whatever drift lies between.

This script tests that reading. `make_val_subset.py` carves the 2019-01-01 to
2019-04-30 window out of the SEVIR nowcast training file, which is the window
both EarthFormer and CasCast hold out from training and which ends six weeks
before the test period begins. One global quantile map per arm is fitted there
and applied, unchanged, to the whole test evaluation half. The two fits are
scored side by side:

    qmap_test   fitted on the test calibration half, the paper's default
    qmap_val    fitted on the earlier validation window

Nothing else moves. If the payoff survives, the deployment reading holds; if it
shrinks, the paper says so, since the confound claim rests on the symmetric
control and not on this one.
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

import sys as _s2
from pathlib import Path as _P2
_s2.path.insert(0, str(_P2(__file__).resolve().parents[1] / "tools"))
_s2.path.insert(0, str(_P2(__file__).resolve().parent))
import paths as P                                          # noqa: E402
from pooling_control import (POOLINGS, THRESHOLDS, accumulate,   # noqa: E402
                             csi_bias, fit_global_qmap)

ARMS = ["persistence", "pysteps", "earthformer", "cascast_det",
        "cascast_cascade", "cascast_cascade_cfg1"]


def qmap_fn(src, dst, device):
    s = torch.from_numpy(src).to(device)
    t = torch.from_numpy(dst).to(device)

    def _q(x):
        flat = x.reshape(-1)
        i = torch.searchsorted(s, flat).clamp(1, len(s) - 1)
        x0, x1, y0, y1 = s[i - 1], s[i], t[i - 1], t[i]
        w = torch.where(x1 > x0, (flat - x0) / (x1 - x0 + 1e-12),
                        torch.zeros_like(flat))
        return (y0 + w * (y1 - y0)).reshape(x.shape)
    return _q


def dev(b):
    return abs(math.log(b)) if b and b > 0 else None


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
            if sx and sy else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=P.RESULTS)
    ap.add_argument("--gt", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--val-gt", type=Path,
                    default=P.SEVIR_DATA / "nowcast_val_subset.h5")
    ap.add_argument("--val-prefix", default="sevir_valpred_")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_val_fitted.json"))
    ap.add_argument("--out-md", type=Path,
                    default=Path(str(P.RESULTS) + "/sevir_val_fitted.md"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P.require(args.val_gt, "validation subset (run sevir/make_val_subset.py)")
    with h5py.File(args.gt, "r") as f:
        N = f["OUT_vil"].shape[0]
    cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)
    with h5py.File(args.val_gt, "r") as f:
        n_val = f["OUT_vil"].shape[0]
    val_idx = np.arange(n_val)

    per_arm = {}
    for a in ARMS:
        test_pred = args.results / f"sevir_pred_{a}.h5"
        val_pred = args.results / f"{args.val_prefix}{a}.h5"
        if not test_pred.exists() or not val_pred.exists():
            print(f"  {a}: skipped, missing prediction cache")
            continue
        print(f"{a}: fitting both quantile maps")
        knobs = {
            "identity": lambda x: x,
            "qmap_test": qmap_fn(*fit_global_qmap(test_pred, args.gt, cal),
                                 device),
            "qmap_val": qmap_fn(*fit_global_qmap(val_pred, args.val_gt, val_idx),
                                device),
        }
        A = accumulate(test_pred, args.gt, ev, knobs, device)
        cells = {}
        for pl in POOLINGS:
            for t in THRESHOLDS:
                row = {}
                for k in knobs:
                    c, b = csi_bias(*A[pl][k][t])
                    row[k] = {"csi": c, "bias": b}
                cells[f"{pl}|{t}"] = row
        per_arm[a] = cells
        h = cells["max16|219"]
        print(f"  max16 tau=219  identity {h['identity']['csi']:.4f} "
              f"(bias {h['identity']['bias']:.3f})  "
              f"test-fit {h['qmap_test']['csi']:.4f}  "
              f"val-fit {h['qmap_val']['csi']:.4f}")

    names = list(per_arm)
    stats = {}
    for fit in ("qmap_test", "qmap_val"):
        gaps, dis, flips, n = [], [], 0, 0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for pl in POOLINGS:
                    for t in THRESHOLDS:
                        key = f"{pl}|{t}"
                        ra, rb = per_arm[a][key], per_arm[b][key]
                        da = dev(ra["identity"]["bias"])
                        db = dev(rb["identity"]["bias"])
                        if da is None or db is None:
                            continue
                        raw = rb["identity"]["csi"] - ra["identity"]["csi"]
                        cal_ = rb[fit]["csi"] - ra[fit]["csi"]
                        gaps.append(da - db)
                        dis.append(raw - cal_)
                        flips += (raw > 0) != (cal_ > 0)
                        n += 1
        stats[fit] = {"n": n, "pearson": pearson(gaps, dis), "flips": flips}

    def share(fit, a, b, pl="max16", t=219):
        ra, rb = per_arm[a][f"{pl}|{t}"], per_arm[b][f"{pl}|{t}"]
        raw = (rb["identity"]["csi"] - ra["identity"]["csi"]) / ra["identity"]["csi"]
        cl = (rb[fit]["csi"] - ra[fit]["csi"]) / ra[fit]["csi"]
        return raw, cl, (1 - cl / raw) if raw > 0 else None

    out = {"n_val_events": int(n_val), "n_eval_events": int(len(ev)),
           "arms": names, "cells": per_arm, "stats": stats,
           "headline": {fit: dict(zip(("raw", "cal", "erased"),
                                      share(fit, "cascast_det",
                                            "cascast_cascade")))
                        for fit in ("qmap_test", "qmap_val")}}
    args.out.write_text(json.dumps(out, indent=2))

    L = ["# The knob refitted on a prior-period validation split\n",
         f"One global quantile map per arm, fitted on {n_val} events from "
         f"2019-01-01 to 2019-04-30 and applied unchanged to the {len(ev)} "
         "test evaluation events. `qmap_test` is the paper's default fit, on "
         "the test calibration half.\n",
         "## Per arm at max 16x16 pooling, tau = 219\n",
         "| arm | bias | CSI | CSI (test-fit) | CSI (val-fit) | gain test-fit | "
         "gain val-fit |",
         "|---|---|---|---|---|---|---|"]
    for a in names:
        c = per_arm[a]["max16|219"]
        u = c["identity"]["csi"]
        gt_ = (c["qmap_test"]["csi"] - u) / u if u > 0 else float("nan")
        gv = (c["qmap_val"]["csi"] - u) / u if u > 0 else float("nan")
        L.append(f"| {a} | {c['identity']['bias']:.3f} | {u:.4f} | "
                 f"{c['qmap_test']['csi']:.4f} | {c['qmap_val']['csi']:.4f} | "
                 f"{100*gt_:+.1f}% | {100*gv:+.1f}% |")
    L += ["\n## The headline decomposition under each fit\n",
          "| fit | raw | calibrated | erased |", "|---|---|---|---|"]
    for fit in ("qmap_test", "qmap_val"):
        r, c, e = out["headline"][fit]["raw"], out["headline"][fit]["cal"], \
            out["headline"][fit]["erased"]
        L.append(f"| {fit} | {100*r:+.1f}% | {100*c:+.1f}% | "
                 + (f"{100*e:.0f}% |" if e is not None else "n/a |"))
    L += ["\n## All pairwise contrasts under each fit\n",
          "| fit | cells | Pearson(gap, distortion) | sign flips |",
          "|---|---|---|---|"]
    for fit, s in stats.items():
        L.append(f"| {fit} | {s['n']} | {s['pearson']:+.3f} | {s['flips']} |")
    args.out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L[-8:]))
    print(f"\nwrote {args.out} and {args.out_md}")


if __name__ == "__main__":
    main()
