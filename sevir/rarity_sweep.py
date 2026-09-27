"""How much threshold skill does a free monotone recalibration buy, as a
function of event rarity?

SEVIR shows the gain concentrated at the rarest thresholds, but there rarity is
confounded with everything else that changes when you raise a threshold on a
fixed model. Here rarity is the only quantity that moves: each cell was RETRAINED
at its own threshold (tau_train = tau_eval), so R is a controlled variable.

Setup. Attention U-Net, trained under a weighted mean squared error on
geostationary infrared band B08 (6.2 um water vapour) for spatial downscaling.
The event is a COLD brightness temperature, y <= tau K, so rarity rises as tau
falls, which is the reverse of the direction a rain-rate threshold runs in. Ten
runs per threshold: five seeds crossed with the two single-decoder loss variants
(tiered breakpoints and a graded importance weight). No pooling is applied in
this domain; the operating point is the threshold alone.

For each (threshold, run) we take the single decoder and compare
  identity   the model's own output, no knob
  recal      the knob chosen on validation by pooled bias -> 1

The relative gain divides by the identity CSI, so it is undefined for a run that
scores exactly zero uncalibrated. Those runs are excluded from the relative
column and from it alone; `n` counts the runs that enter it, `n_all` counts every
run trained at that threshold, and the absolute and median columns are computed
over all `n_all`. The excluded runs are listed under the table with what the
knob does to them, since a run going from zero to a positive CSI has an
unbounded relative gain and dropping it can only understate the effect.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
from stats import spearman                                       # noqa: E402

KNOBS = ("shift", "gain", "qmap", "qmap+shift")

# The two single-decoder loss variants crossed with the five seeds. `tiered`
# places hard breakpoints on the target and weights the resulting bands;
# `graded` weights each pixel continuously by its distance past the threshold.
# Named here rather than left as an unexplained factor, because the pooled
# ordering averages over them and a reader is entitled to ask whether it
# survives within each.
VARIANTS = ("graded", "tiered")


def variant_of(tag):
    for v in VARIANTS:
        if v in tag:
            return v
    return None


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else None


def pick_identity(arm):
    """The un-knobbed prediction: shift with knob 0."""
    for r in arm["shift"]["test"]:
        if r["knob"] == 0:
            return r
    return None


def pick_recal(arm, key="csi"):
    """Best knob chosen on val by pooled bias -> 1, read off test."""
    best, bestdev = None, None
    for kn in KNOBS:
        if kn not in arm:
            continue
        for i, rv in enumerate(arm[kn]["val"]):
            b = rv.get("bias_pooled", rv.get("bias"))
            if b is None or not np.isfinite(b) or b <= 0:
                continue
            dev = abs(b - 1.0)
            if bestdev is None or dev < bestdev:
                bestdev, best = dev, arm[kn]["test"][i]
    return best


def perm_p(x, y):
    """Exact two-sided permutation p for Spearman on a handful of points.

    Six rarity levels is 720 permutations, so the exact null is cheaper than
    an asymptotic approximation and does not need scipy, which the rest of this
    tier does without.
    """
    from itertools import permutations
    obs = abs(spearman(x, y))
    perms = list(permutations(range(len(y))))
    hits = sum(1 for p in perms
               if abs(spearman(x, [y[i] for i in p])) >= obs - 1e-12)
    return hits / len(perms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", nargs="+", type=Path, required=True)
    ap.add_argument("--r-test", type=Path, required=True)
    ap.add_argument(
        "--metric", choices=("csi", "csi_pooled"), default="csi",
        help="score field to report; csi_pooled is the all-scene sensitivity",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rtest = json.loads(args.r_test.read_text())
    L = []
    W = L.append
    metric_label = "pooled CSI" if args.metric == "csi_pooled" else "CSI"
    W(f"# What a free monotone recalibration buys, versus event rarity ({metric_label})\n")
    W("Attention U-Net under a weighted MSE on geostationary infrared band B08. "
      "The event is `y <= tau` K, so rarity rises as tau falls. Ten runs per "
      "threshold: five seeds by two single-decoder loss variants. `identity` is "
      "the model's own output; `recal` is the knob chosen on validation by "
      "pooled bias -> 1. Each threshold was retrained at its own tau.\n")
    if args.metric == "csi_pooled":
        W("Every validation and test scene enters one pooled contingency table; "
          "there is no event-coverage frame filter.\n")
    W("`n` counts the runs entering the relative column, which is undefined for "
      "a run at CSI 0; `n_all` counts every run, and the absolute and median "
      "columns use all of them.\n")
    W(f"| tau | R (test) | {metric_label} identity | {metric_label} recal | "
      f"d{metric_label} | rel gain | n | "
      "n_all | dCSI (all) | median dCSI (all) |")
    W("|---|---|---|---|---|---|---|---|---|---|")

    xs, ys, notes, byvar = [], [], [], []
    for path in sorted(args.controls,
                       key=lambda p: -float(re.search(r"t(\d+)", p.name).group(1))):
        d = json.loads(path.read_text())
        tau = str(int(d["threshold_k"]))
        R = rtest.get(tau)
        if R is None:
            continue
        ident, recal, all_i, all_r, dropped = [], [], [], [], []
        per = {v: {"i": [], "r": [], "ai": [], "ar": []} for v in VARIANTS}
        for tag, arm in d["single_decoder"].items():
            i, r = pick_identity(arm), pick_recal(arm)
            if i is None or r is None:
                continue
            ci, cr = i.get(args.metric), r.get(args.metric)
            if ci is None or cr is None or not np.isfinite(ci):
                continue
            all_i.append(ci)
            all_r.append(cr)
            v = variant_of(tag)
            if v:
                per[v]["ai"].append(ci)
                per[v]["ar"].append(cr)
                if ci > 0:
                    per[v]["i"].append(ci)
                    per[v]["r"].append(cr)
            if ci <= 0:
                dropped.append((tag, ci, cr))
                continue
            ident.append(ci)
            recal.append(cr)
        if len(ident) < 2:
            continue
        byvar.append((tau, R, per))
        mi, mr = float(np.mean(ident)), float(np.mean(recal))
        rel = (mr - mi) / mi
        ai, ar = np.array(all_i), np.array(all_r)
        xs.append(np.log10(R))
        ys.append(rel)
        W(f"| {tau} | {R*100:.3f}% | {mi:.4f} | {mr:.4f} | {mr-mi:+.4f} | "
          f"{rel:+.1%} | {len(ident)} | {len(ai)} | "
          f"{ar.mean()-ai.mean():+.4f} | {np.median(ar-ai):+.4f} |")
        for tag, ci, cr in dropped:
            notes.append(f"- tau {tau}: `{tag}` scores {ci:.4f} uncalibrated "
                         f"and {cr:.4f} recalibrated, so its relative gain is "
                         f"unbounded and it is excluded from that column only.")

    if len(xs) >= 3:
        rho = spearman(xs, ys)
        W(f"\nSpearman(relative gain, log10 R) = {rho:+.3f}, "
          f"p = {perm_p(xs, ys):.4f}")
        W(f"\nRange: {min(ys):+.1%} at the most common threshold to "
          f"{max(ys):+.1%} at the rarest.")

    # The pooled row averages the two loss variants. That hides a factor, so the
    # same statistic is recomputed inside each variant: if the ordering only
    # exists after averaging, it should be disclosed explicitly. It does not
    # only exist after averaging, but it is not perfect within
    # either variant, and both facts are printed.
    if byvar:
        W("\n## The same sweep within each loss variant\n")
        W("The pooled table above averages the five seeds of both single-decoder "
          "loss variants. Averaging can manufacture an ordering, so here is the "
          "same statistic computed inside each variant separately.\n")
        W("| variant | " + " | ".join(f"tau {t}" for t, _, _ in byvar)
          + " | Spearman vs log10 R | two-sided p |")
        W("|---" * (len(byvar) + 3) + "|")
        for v in VARIANTS:
            vx, vy, cells = [], [], []
            for tau, R, per in byvar:
                p_ = per[v]
                if len(p_["i"]) < 2:
                    cells.append("n/a")
                    continue
                mi, mr = float(np.mean(p_["i"])), float(np.mean(p_["r"]))
                rel = (mr - mi) / mi
                vx.append(np.log10(R))
                vy.append(rel)
                cells.append(f"{rel:+.1%} ({len(p_['i'])}/{len(p_['ai'])})")
            W(f"| {v} | " + " | ".join(cells)
              + f" | {spearman(vx, vy):+.3f} | {perm_p(vx, vy):.4f} |")
        W("\nCells give the relative gain with the count entering it out of the "
          "runs trained at that threshold. The p is the same exact two-sided "
          "permutation test used on the pooled row, so all three are comparable. "
          "Six levels give the test almost no resolution: the smallest two-sided "
          "p attainable is 2/720, and only a near-perfect ordering clears a "
          "conventional threshold, which is why the claim rests on the size of "
          "the gains and where they sit rather than on this test.")


    if notes:
        W("\n## Runs excluded from the relative column\n")
        W("Each is listed with its uncalibrated and recalibrated CSI, so the "
          "table is self-contained and the exclusion can be checked here rather "
          "than against a working file.\n")
        L.extend(notes)

    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
