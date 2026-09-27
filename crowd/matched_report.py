"""Crowd domain: dual decoder vs single decoder, with the SAME knobs on both.

The earlier control gave the single decoder {gain, shift, qmap+gain} and the
dual decoder only its architectural blend weight beta. This reads the corrected
file, in which both sides carry the same three knobs, and reports the margin
under a validation-selected operating point (|bias - 1| minimised on val).

`beta_only` reproduces the old asymmetric comparison for contrast.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

KNOBS = ("gain", "shift", "qmap+gain")


def seed_of(tag):
    m = re.search(r"_s(\d+)$", tag)
    return int(m.group(1)) if m else None


def pick(rows_val, rows_test, thr, key="csi"):
    """Choose the row whose val bias is closest to 1; return its test value."""
    best, bestdev = None, None
    for rv, rt in zip(rows_val, rows_test):
        cell = rv.get(thr, {})
        # pooled bias, matching rule C used in the atmosphere analysis
        b = cell.get("bias_pooled", cell.get("bias"))
        if b is None or not np.isfinite(b):
            continue
        dev = abs(b - 1.0)
        if bestdev is None or dev < bestdev:
            bestdev, best = dev, rt.get(thr, {}).get(key)
    return best


def arm_best(entry, thr, knobs, key="csi"):
    """Best test value over the given knob families, each selected on val."""
    out = []
    for kn in knobs:
        if kn not in entry:
            continue
        v = pick(entry[kn]["val"], entry[kn]["test"], thr, key)
        if v is not None and np.isfinite(v):
            out.append(v)
    return max(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.control.read_text())
    thresholds = [str(t) for t in d["thresholds"]]
    sd, dual_decoder = d["single_decoder"], d["dual_decoder"]
    seeds = sorted({seed_of(t) for t in dual_decoder if seed_of(t) is not None})

    L = []
    W = L.append
    W("# Crowd domain: matched-knob comparison\n")
    W("Both sides receive {gain, shift, qmap+gain}; the operating point is "
      "chosen on validation by |bias - 1|. `beta only` is the old asymmetric "
      "comparison, in which the dual decoder got no post-hoc knob.\n")
    W("| threshold | R | DualDecoder matched | best single | margin | seeds | p | "
      "margin (beta only) |")
    W("|---|---|---|---|---|---|---|---|")

    for thr in thresholds:
        dm, dbeta, bs = [], [], []
        for s in seeds:
            dtag = next((t for t in dual_decoder if seed_of(t) == s), None)
            if dtag is None:
                continue
            e = dual_decoder[dtag]
            m = arm_best(e, thr, KNOBS)
            b = pick(e["val"], e["test"], thr)          # beta frontier
            cand = [arm_best(sd[t], thr, KNOBS) for t in sd if seed_of(t) == s]
            cand = [c for c in cand if c is not None and np.isfinite(c)]
            if m is None or not cand:
                continue
            dm.append(m); bs.append(max(cand))
            if b is not None and np.isfinite(b):
                dbeta.append(b)
        if len(dm) < 2:
            continue
        dm, bs = np.array(dm), np.array(bs)
        marg = dm - bs
        R = None
        for t in sd:
            r = sd[t]["gain"]["test"][0].get(thr, {}).get("R")
            if r is not None:
                R = r
                break
        try:
            p = stats.wilcoxon(dm, bs).pvalue if len(dm) >= 5 else float("nan")
        except Exception:
            p = float("nan")
        mb = (np.mean(np.array(dbeta) - bs[:len(dbeta)])
              if len(dbeta) == len(bs) else float("nan"))
        W(f"| {thr} | {R*100:.2f}% | {dm.mean():.4f} | {bs.mean():.4f} | "
          f"{marg.mean():+.4f} | {int((marg>0).sum())}/{len(marg)} | {p:.3f} | "
          f"{mb:+.4f} |")

    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
