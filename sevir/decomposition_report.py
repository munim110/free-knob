"""Decompose CasCast's reported cascade gain into calibration and residual skill.

CasCast (2024) reports that adding its diffusion cascade on top of its own
deterministic backbone improves CSI at the extreme SEVIR threshold by +91.8%.
Both models are run here from the authors' released checkpoints on the official
test split, so the comparison is architecture-vs-architecture under one pipeline.

Uncalibrated, the cascade plainly scores higher. This table answers a different
question: how much of that improvement is skill the backbone could not have
obtained for free. A 2x2 on one held-out half:

                     uncalibrated      + calibration
    deterministic         a                  b
    cascade               c                  d

    reported gain      = c/a - 1        (what the paper quotes)
    residual gain      = d/b - 1        (what survives a free monotone knob)
    calibration share  = 1 - (d/b - 1) / (c/a - 1)

Two calibration columns are reported, because the obvious objection to the first
is that the knob is chosen per (pooling, threshold) cell:

  * `knob`   best of {shift, mul} selected on the calibration half by pooled
             frequency bias -> 1, per cell. The strongest member of the family,
             and the one the objection applies to.
  * `qmapG`  a single global monotone quantile map fitted once on the calibration
             half of the raw field, then held fixed across every threshold and
             every pooling convention. Weaker, and immune to that objection.

Neither knob sees the evaluation half, and neither can reorder pixels.
"""
import argparse
import json
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P

POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]
THRESHOLDS = [16, 74, 133, 160, 181, 219]


def load(res, name):
    d = json.loads((res / f"sevir_poolingq_{name}.json").read_text())
    return {(r["pooling"], r["threshold"]): r for r in d["table"]}


def rel(new, old):
    return (new - old) / old if old > 0 else float("nan")


def share(reported, residual):
    """Fraction of the reported gain that a free knob on the baseline erases."""
    if reported <= 0:
        return float("nan")
    return 1.0 - residual / reported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(str(P.RESULTS)))
    ap.add_argument("--det", default="cascast_det")
    ap.add_argument("--cascade", default="cascast_cascade")
    ap.add_argument("--out", type=Path,
                    default=Path(str(P.RESULTS) + "/"
                                 "sevir_decomposition.md"))
    args = ap.parse_args()

    D = load(args.results, args.det)
    C = load(args.results, args.cascade)

    L = []
    W = L.append
    W("# Decomposing the CasCast cascade gain\n")
    W("Both models from the authors' released checkpoints, same test split, "
      "same pipeline. Calibration is fitted on a disjoint half of the events "
      "and never sees the evaluation half.\n")

    for tag, key in (("per-cell knob", "csi_recal"),
                     ("single global quantile map", "csi_qmap_global")):
        W(f"\n## Calibration = {tag}\n")
        W("| pooling | thr | det | cascade | reported | det+cal | casc+cal | "
          "residual | share erased |")
        W("|---|---|---|---|---|---|---|---|---|")
        for pl in POOLINGS:
            for t in THRESHOLDS:
                a = D[(pl, t)]["csi_uncal"]
                c = C[(pl, t)]["csi_uncal"]
                b = D[(pl, t)][key]
                d = C[(pl, t)][key]
                rep, res = rel(c, a), rel(d, b)
                W(f"| {pl} | {t} | {a:.4f} | {c:.4f} | {rep:+.1%} | "
                  f"{b:.4f} | {d:.4f} | {res:+.1%} | {share(rep, res):.0%} |")

    # headline cell
    pl, t = "max16", 219
    a, c = D[(pl, t)]["csi_uncal"], C[(pl, t)]["csi_uncal"]
    W("\n## Headline cell (POOL16-max, threshold 219)\n")
    W(f"- deterministic backbone: **{a:.4f}** "
      f"(frequency bias {D[(pl, t)]['bias_uncal']:.3f})")
    W(f"- + cascade: **{c:.4f}** "
      f"(bias {C[(pl, t)]['bias_uncal']:.3f}) -> **{rel(c, a):+.1%}**")
    for tag, key in (("best per-cell knob", "csi_recal"),
                     ("single global quantile map", "csi_qmap_global")):
        b, d = D[(pl, t)][key], C[(pl, t)][key]
        W(f"- backbone + {tag}: **{b:.4f}**"
          f"{'  (beats the cascade)' if b > c else ''}")
        W(f"  - cascade + same: **{d:.4f}**; residual cascade gain "
          f"**{rel(d, b):+.1%}**")
    W("\nThe cascade raises frequency bias from "
      f"{D[(pl, t)]['bias_uncal']:.3f} to {C[(pl, t)]['bias_uncal']:.3f}, i.e. "
      "its mechanical effect at this operating point is to move the model "
      "toward calibration. That is what a free monotone knob also does, which "
      "is why the two are largely non-additive.")

    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
