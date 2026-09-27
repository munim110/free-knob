"""Aggregate the crowd pooling sweep over seeds and emit the paper's table.

The two trained arms are seed-dependent and are reported as a mean over seeds
with the across-seed spread, so a reader can see the dose-response is not one
checkpoint's accident. The `gt_*` arms are derived from the ground truth and
carry no seed, so they are read from any one run and reported once.

Registered predictions from crowd_pooling_control.py, scored here:

  P1  the published protocol (`none`, a sum reduction) leaves the models near
      bias 1. FAILS for both arms, see below.
  P2  gain is ordered by the bias deficit. Held, rho reported.
  P3  gt_shift (real field, wrong place) holds bias near 1 and gains nothing.
  P4  gt_blur (perfect place, no peaks) depresses bias monotonically. Held in
      direction, but the effect is small because a Gaussian conserves mass and a
      patch sum reads mass. This failure is the informative one: what these
      metrics punish is amplitude attenuation rather than spatial spreading.
"""
import json
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P
from stats import spearman

RES = Path(str(P.RESULTS))
OUT = Path(str(P.PAPER))
TRAINED = ["baseline", "csrnet"]
SYNTH = ["gt_shift2", "gt_blur1", "gt_blur2", "gt_blur4", "gt_blur8"]
PRETTY = {"baseline": "CNN baseline (MSE)", "csrnet": "CSRNet",
          "gt_shift2": "truth, displaced 2 patches",
          "gt_blur1": "truth, blurred $\\sigma{=}1$",
          "gt_blur2": "truth, blurred $\\sigma{=}2$",
          "gt_blur4": "truth, blurred $\\sigma{=}4$",
          "gt_blur8": "truth, blurred $\\sigma{=}8$"}
TAUS = [2.0, 3.0, 5.0, 7.0, 10.0, 15.0]


def load():
    runs = {}
    for f in sorted(RES.glob("crowd_pooling_s*.json")):
        seed = f.stem.split("_s")[-1]
        runs[seed] = json.loads(f.read_text())
    return runs


# A relative gain measured off a base that has collapsed toward zero is a
# ratio of two small numbers and tells the reader nothing: CSRNet at tau=10
# scores 0.0017, so any recovery reads as +12000%. We carry the absolute
# change in CSI as the primary quantity and suppress the ratio below this floor.
CSI_FLOOR = 0.02


def cell(runs, arm, pooling, tau, key):
    """Values of `key` for this cell across every seed that has it."""
    vs = []
    for d in runs.values():
        for r in d["table"]:
            if not (r["arm"] == arm and r["pooling"] == pooling
                    and r["threshold"] == tau and r["usable"]):
                continue
            if key == "abs_gain_percell":
                v = r["csi_percell"] - r["csi_uncal"]
            elif key == "rel_gain_percell" and r["csi_uncal"] < CSI_FLOOR:
                continue                      # ratio off a collapsed base
            else:
                v = r.get(key, np.nan)
            if np.isfinite(v):
                vs.append(v)
    return vs


def fmt(vs, scale=1.0, dec=3, sign=False):
    if not vs:
        return "---"
    m = scale * np.mean(vs)
    s = f"{m:+.{dec}f}" if sign else f"{m:.{dec}f}"
    return s + (f" $\\pm$ {scale*np.std(vs):.{dec}f}" if len(vs) > 1 else "")


def main():
    runs = load()
    if not runs:
        print("no crowd_pooling_s*.json yet")
        return
    seeds = sorted(runs)
    print(f"seeds: {', '.join(seeds)}\n")

    lines = ["# Crowd density: does the confound exist outside radar?", "",
             f"ShanghaiTech Part A, {runs[seeds[0]]['n_test']} test images, "
             f"32x32 patch sums, the benchmark's own protocol. "
             f"Trained arms averaged over {len(seeds)} seeds ({', '.join(seeds)}).", ""]

    # ---- the headline: dose-response under the benchmark's published protocol
    lines += ["## Under the published protocol (`none`, a sum reduction)", "",
              "CSI is the uncalibrated score; dCSI is the absolute change from "
              "the per-cell knob. The relative column is suppressed where the "
              f"base CSI is below {CSI_FLOOR}, since a ratio off a collapsed "
              "base is not a readable number.", "",
              "| arm | tau | R | CSI | pooled Bias | dCSI | rel | seeds |",
              "|---|---|---|---|---|---|---|---|"]
    body = []
    for arm in TRAINED + SYNTH:
        for tau in TAUS:
            b = cell(runs, arm, "none", tau, "bias_uncal")
            if not b:
                continue
            c = cell(runs, arm, "none", tau, "csi_uncal")
            a = cell(runs, arm, "none", tau, "abs_gain_percell")
            g = cell(runs, arm, "none", tau, "rel_gain_percell")
            R = cell(runs, arm, "none", tau, "R")
            lines.append(
                f"| {arm} | {tau:g} | {100*np.mean(R):.2f}% | "
                f"{fmt(c, dec=4)} | {fmt(b)} | {fmt(a, dec=4, sign=True)} | "
                f"{fmt(g, 100, 1, True) + '%' if g else 'n/a'} | {len(b)} |")
        lines.append("")

    # ---- dose-response, pooled over every usable cell of every seed
    # correlate against the ABSOLUTE change, which is defined at every cell
    allc = [(r["bias_uncal"], r["csi_percell"] - r["csi_uncal"])
            for d in runs.values() for r in d["table"]
            if r["usable"] and np.isfinite(r["csi_percell"])
            and np.isfinite(r["csi_uncal"]) and np.isfinite(r["bias_uncal"])]
    under = [c for c in allc if c[0] < 1.0]
    over = [c for c in allc if c[0] >= 1.0]
    rho_all = -spearman([c[0] for c in allc], [c[1] for c in allc])
    rho_und = -spearman([c[0] for c in under], [c[1] for c in under])
    lines += ["## Dose-response", "",
              f"Spearman(bias deficit, absolute dCSI from the knob), all usable "
              f"cells over all "
              f"seeds: **{rho_all:+.3f}** over {len(allc)} cells.", "",
              f"Restricted to under-forecasting cells (Bias < 1), the regime "
              f"every SEVIR system occupies: **{rho_und:+.3f}** over "
              f"{len(under)} cells.", "",
              f"Over-forecasting cells (Bias >= 1, n={len(over)}) behave "
              f"oppositely: correcting bias toward one *costs* CSI, because at "
              f"a rare threshold scaling a too-massive field down removes hits "
              f"faster than false alarms. Median gain there is "
              f"{np.median([c[1] for c in over]):+.4f} CSI. The knob is not a "
              f"free lunch in both directions, and we say so.", ""]

    # ---- the two controls
    lines += ["## Controls: sharpness and placement moved separately", "",
              "| arm | what it holds fixed | Bias range | max abs gain |",
              "|---|---|---|---|"]
    for arm, what in (("gt_shift2", "real field, placement wrong by 2 patches"),
                      ("gt_blur8", "placement perfect, peaks destroyed")):
        bs, gs = [], []
        for tau in TAUS:
            for pl in runs[seeds[0]]["poolings"]:
                bs += cell(runs, arm, pl, tau, "bias_uncal")
                gs += cell(runs, arm, pl, tau, "abs_gain_percell")
        if bs:
            lines.append(f"| {arm} | {what} | [{min(bs):.3f}, {max(bs):.3f}] | "
                         f"{max(abs(np.array(gs))):.4f} CSI |")
    lines += ["",
              "`gt_shift` is badly wrong and perfectly calibrated: a monotone "
              "knob cannot recover a displacement and correctly declines to "
              "act. `gt_blur` conserves mass, so a patch sum barely notices it. "
              "The metric punishes amplitude attenuation, and amplitude is "
              "what the knob gives back. Position and spread are untouched.", ""]

    (RES / "crowd_pooling_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))
    print(f"\nwrote {RES/'crowd_pooling_report.md'}")

    # ---- paper table: bias and gain against threshold, published protocol
    for arm in TRAINED + ["gt_shift2", "gt_blur8"]:
        cells, gains = [], []
        for tau in (3.0, 7.0, 10.0):
            b = cell(runs, arm, "none", tau, "bias_uncal")
            g = cell(runs, arm, "none", tau, "abs_gain_percell")
            cells.append(fmt(b) if b else "---")
            gains.append(fmt(g, dec=4, sign=True) if g else "---")
        body.append(f"{PRETTY[arm]} & " + " & ".join(cells + gains) + " \\\\")
    (OUT / "tab_crowdpool.tex").write_text("\n".join(body) + "\n")
    print(f"wrote {OUT/'tab_crowdpool.tex'}")


if __name__ == "__main__":
    main()
