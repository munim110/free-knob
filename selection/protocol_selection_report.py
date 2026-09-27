"""How the operating point is chosen decides the answer. Three rules, compared.

The problem
-----------
DualDecoder exposes beta; a recalibrated single decoder exposes one knob. Both trace a
curve through (Bias, CSI): neither can move one without the other. So a
comparison is only meaningful once both are placed at a comparable point on
their own curve, and how that placement is done turns out to matter more than
anything else measured in this programme.

On atmosphere at tau = 210 the same models give a margin of -0.0364 when the
operating point is selected one way and +0.0101 when the two curves are compared
at matched achieved Bias. On crowd, isolating the two changes showed that giving
the baselines a knob cost DualDecoder ~0.009 while making DualDecoder select its own beta cost
~0.043. The binding constraint is selection, not representation.

Why the published rule is noisy
-------------------------------
Protocol A selects the knob minimising |Bias - 1| on validation, where Bias is
the *mean over frames* of (hits + false alarms) / (hits + misses). On a frame
holding a handful of critical pixels that denominator is tiny and the ratio is
heavy-tailed, so the mean is dominated by the sparsest frames, which are the
frames carrying least information. At tau = 197 the atmospheric validation
split holds 14 eventful frames; the crowd split holds 30 images in total.

The three rules
---------------
    A  per-frame-mean Bias on val closest to 1        (published; high variance)
    C  POOLED Bias on val closest to 1                (same information, stable:
                                                       a ratio of sums rather
                                                       than a mean of ratios)
    D  pooled PREDICTED area on val closest to the    (label-free: uses no
       climatological critical fraction from TRAIN     validation targets at all)

D is the deployable limit case. It never looks at an evaluation label, using
only the model's own predicted field and a rate estimated once from training
data. It cannot be accused of borrowing supervision, and its variance does not
depend on how many eventful frames the validation split happens to contain.

All three report on test, and all three are applied identically to DualDecoder's beta
and to every baseline knob, so the comparison stays symmetric.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

KNOBS = ("shift", "gain", "qmap", "qmap+shift")


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return m.group(1) if m else tag


def choose(rows_val, rule, r_clim):
    """Index of the operating point picked on validation under `rule`."""
    best, bd = None, np.inf
    for i, r in enumerate(rows_val):
        if r is None:
            continue
        if rule == "A":
            v = r.get("bias")
            d = abs(v - 1.0) if v is not None and np.isfinite(v) else np.inf
        elif rule == "C":
            v = r.get("bias_pooled")
            d = abs(v - 1.0) if v is not None and np.isfinite(v) else np.inf
        elif rule == "D":
            v = r.get("pred_area_frac")
            d = abs(v - r_clim) if v is not None and np.isfinite(v) else np.inf
        else:
            raise ValueError(rule)
        if d < bd:
            best, bd = i, d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", nargs="+", type=Path, required=True)
    ap.add_argument("--r-clim", type=Path, required=True)
    ap.add_argument("--r-test", type=Path, required=True)
    ap.add_argument("--key", type=str, default="csi",
                    help="csi | csi_pooled | mae_crit")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clim = json.load(open(args.r_clim))
    rtest = json.load(open(args.r_test))
    better = -1 if args.key.startswith("mae") else +1

    L = [f"# Operating-point selection: three rules, same models\n",
         f"Metric: `{args.key}`. Margin is DualDecoder minus the per-seed best "
         f"recalibrated single decoder"
         + (" (lower is better, so margin is baseline minus DualDecoder)." if better < 0 else "."),
         "\nA = per-frame-mean Bias on val -> 1 (published). "
         "C = pooled Bias on val -> 1. "
         "D = pooled predicted area on val -> train climatology (label-free).\n"]
    W = L.append
    rows = {}

    for path in sorted(args.controls, key=lambda p: -float(re.search(r"_t([\d.]+)\.json", p.name).group(1))):
        d = json.load(open(path))
        tau = d["threshold_k"]
        key_t = str(int(tau))
        rc = clim.get(key_t)
        R = rtest.get(key_t)
        if rc is None or R is None:
            continue
        sd, dual_decoder = d["single_decoder"], d["dual_decoder"]
        seeds = sorted({seed_of(t) for t in dual_decoder})
        rows[key_t] = {"R": R}

        for rule in ("A", "C", "D"):
            dv, bv = [], []
            for s in seeds:
                dtag = next((t for t in dual_decoder if seed_of(t) == s), None)
                if dtag is None:
                    continue
                # DualDecoder gets exactly the treatment the baseline gets: a per-seed
                # argmax over the same four post-hoc knobs. Older control files
                # carry only a flat beta frontier; fall back to it so those stay
                # readable, but that path is NOT a matched comparison.
                darm = dual_decoder[dtag]
                if isinstance(darm, dict) and any(kn in darm for kn in KNOBS):
                    dcand = []
                    for kn in KNOBS:
                        if kn not in darm:
                            continue
                        i = choose(darm[kn]["val"], rule, rc)
                        if i is None:
                            continue
                        v = darm[kn]["test"][i].get(args.key)
                        if v is not None and np.isfinite(v):
                            dcand.append(v)
                    if not dcand:
                        continue
                    x = max(dcand) if better > 0 else min(dcand)
                else:
                    i = choose(darm["val"], rule, rc)
                    if i is None:
                        continue
                    x = darm["test"][i].get(args.key)
                cand = []
                for t, arm in sd.items():
                    if seed_of(t) != s:
                        continue
                    for kn in KNOBS:
                        if kn not in arm:
                            continue
                        j = choose(arm[kn]["val"], rule, rc)
                        if j is None:
                            continue
                        y = arm[kn]["test"][j].get(args.key)
                        if y is not None and np.isfinite(y):
                            cand.append(y)
                if not cand or x is None or not np.isfinite(x):
                    continue
                dv.append(x)
                bv.append(max(cand) if better > 0 else min(cand))
            if len(dv) >= 2:
                dv, bv = np.array(dv), np.array(bv)
                m = (dv - bv) if better > 0 else (bv - dv)
                rows[key_t][rule] = {"dual_decoder": dv.mean(), "base": bv.mean(),
                                     "margin": m.tolist()}

    W("\n| tau | R (test) | rule | DualDecoder | baseline | margin | seeds | p |")
    W("|---|---|---|---|---|---|---|---|")
    for k in sorted(rows, key=lambda x: -float(x)):
        r = rows[k]
        for rule in ("A", "C", "D"):
            if rule not in r:
                continue
            e = r[rule]
            m = np.array(e["margin"])
            t, p = stats.ttest_1samp(m, 0.0)
            W(f"| {k} | {r['R']*100:.3f}% | {rule} | {e['dual_decoder']:.4f} | {e['base']:.4f} | "
              f"**{m.mean():+.4f}** | {int((m > 0).sum())}/{len(m)} | {p:.3f} |")

    W("\n## Trend of the margin against log10 R, by rule\n")
    W("| rule | mean margin | thresholds favouring DualDecoder | Spearman rho | p |")
    W("|---|---|---|---|---|")
    for rule in ("A", "C", "D"):
        xs, ys = [], []
        for k, r in rows.items():
            if rule in r:
                xs.append(np.log10(r["R"])); ys.append(np.mean(r[rule]["margin"]))
        if len(ys) >= 3:
            rho, p = stats.spearmanr(xs, ys)
            W(f"| {rule} | {np.mean(ys):+.4f} | {int(np.sum(np.array(ys) > 0))}/{len(ys)} | "
              f"{rho:+.3f} | {p:.4f} |")

    md = "\n".join(L)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=float))
    print(md)


if __name__ == "__main__":
    main()
