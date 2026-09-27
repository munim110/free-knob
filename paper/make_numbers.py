"""Emit paper/numbers.tex and paper/tab_*.tex directly from the result JSONs.

Every quantity quoted in the paper is defined here as a macro or generated as a
table body, so the manuscript cannot drift from the artefacts it reports. If a
run is re-done, this regenerates and the text follows. Nothing in main.tex is a
hand-typed number.
"""
import json
import re
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
import paths as P
from stats import spearman

RES = Path(str(P.RESULTS))
OUT = Path(str(P.PAPER))
THRESHOLDS = [16, 74, 133, 160, 181, 219]
POOLINGS = ["none", "avg4", "max4", "avg16", "max16"]
POOL_TEX = {"none": "none", "avg4": "avg $4^2$", "max4": "max $4^2$",
            "avg16": "avg $16^2$", "max16": "max $16^2$"}

macros = {}


def mac(name, value):
    macros[name] = value


def pct(x, d=1):
    return f"{100 * x:+.{d}f}\\%"


def thousands(n):
    """House rule: a separator at five digits and above, none below.

    The manuscript has four-digit counts that read as plain quantities (1872
    test events, 5000 images, 2000 replicates) and one five-digit count that
    does not (92,173 blocks). Applying a separator to all of them made the
    four-digit cases look like years with commas; applying it to none made the
    five-digit case unreadable. The rule is applied here so the two conventions
    cannot drift apart again.
    """
    return f"{n:,}".replace(",", "{,}") if n >= 10000 else str(n)


def load_pool(name):
    f = RES / f"sevir_poolingq_{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {(r["pooling"], r["threshold"]): r for r in d["table"]}, d


# ---------------------------------------------------------------- E1: rarity
def e1():
    txt = (RES / "calibration_sensitivity_vs_rarity.md").read_text()
    rows = re.findall(
        r"\| (\d+) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+) \| ([+-][\d.]+) \| "
        r"([+-][\d.]+)% \| (\d+) \| (\d+) \| ([+-][\d.]+) \| ([+-][\d.]+) \|",
        txt)
    body = []
    for tau, R, ci, cr, dc, rg, n, nall, dall, dmed in rows:
        body.append(f"{tau} & {R}\\% & {ci} & {cr} & {dc} & {rg}\\% & "
                    f"{n}/{nall} & {dall} & {dmed} \\\\")
    (OUT / "tab_rarity.tex").write_text("\n".join(body) + "\n")
    rho = re.search(r"Spearman\(relative gain, log10 R\) = ([-\d.]+), "
                    r"p = ([\d.]+)", txt)
    mac("EoneRho", rho.group(1))
    # exact permutation p over the six rarity levels. With six points the
    # smallest attainable two-sided p is 2/720, so this is quoted as a floor
    # rather than as a vanishing p-value.
    mac("EoneP", rho.group(2))
    # the same ordering inside each loss variant, so the pooled row cannot be
    # an artefact of averaging over a hidden factor
    for v, tag in (("graded", "Graded"), ("tiered", "Tiered")):
        m = re.search(r"\| " + v + r" \|.*\| ([-+][\d.]+) \| ([\d.]+) \|", txt)
        if m:
            mac(f"EoneRho{tag}", m.group(1))
            mac(f"EoneP{tag}", m.group(2))
    vr = re.findall(r"^\| ([a-z]+) \| .* \| [-+][\d.]+ \| [\d.]+ \|\s*$",
                    txt, re.M)
    if len(vr) == 2:
        mac("EoneVariants", " and ".join(vr))
        mac("EoneNVariants", str(len(vr)))
    mac("EoneMin", rows[0][5] + "\\%")
    mac("EoneMax", rows[-1][5] + "\\%")
    mac("EoneRmax", rows[0][1] + "\\%")
    mac("EoneRmin", rows[-1][1] + "\\%")
    mac("EoneNtau", str(len(rows)))
    mac("EoneNRuns", rows[0][7])
    mac("EoneNRunsTotal", str(sum(int(r[7]) for r in rows)))
    # the survivorship check: absolute and median change over EVERY run, so the
    # relative column's exclusion of zero-CSI runs cannot be carrying the trend
    absall = [float(r[8]) for r in rows]
    mac("EoneAbsMax", f"{max(absall, key=abs):+.4f}")
    mac("EoneAbsLo", f"{min(absall):+.4f}")
    mac("EoneAbsRarest", rows[-1][8])
    mac("EoneMedRarest", rows[-1][9])
    mac("EoneCsiFold", f"{float(rows[0][2]) / float(rows[-1][2]):.0f}")
    # runs whose uncalibrated CSI is exactly zero: excluded from the relative
    # column only, and every one of them is lifted off zero by the knob
    exc = re.findall(r"scores 0\.0000 uncalibrated and ([\d.]+) recalibrated",
                     txt)
    mac("EoneExcluded", str(len(exc)))
    if exc:
        mac("EoneExclLo", f"{min(float(x) for x in exc):.4f}")
        mac("EoneExclHi", f"{max(float(x) for x in exc):.4f}")
    print(f"  E1: {len(rows)} rarity levels, rho={rho.group(1)}, "
          f"{len(exc)} zero-CSI runs excluded from the relative column")


# ------------------------------------------------- E6: pooling sets the size
def e6():
    T, _ = load_pool("cascast_det")
    gains = {pl: T[(pl, 219)]["rel_gain_qmap_global"] for pl in POOLINGS}
    lo_pl = min(gains, key=lambda k: gains[k])
    hi_pl = max(gains, key=lambda k: gains[k])
    plain = {"none": "no pooling", "avg4": "average $4{\\times}4$",
             "max4": "max $4{\\times}4$", "avg16": "average $16{\\times}16$",
             "max16": "max $16{\\times}16$"}
    mac("SixGainLo", pct(gains[lo_pl]))
    mac("SixGainHi", pct(gains[hi_pl]))
    mac("SixPoolLo", plain[lo_pl])
    mac("SixPoolHi", plain[hi_pl])
    # a ratio is meaningless when one end is negative; quote the spread in
    # percentage points of relative gain instead
    mac("SixSpread", f"{100 * (gains[hi_pl] - gains[lo_pl]):.0f}")
    # Growth along the max-pooling sequence (none, 4x4, 16x16) with the model
    # held fixed: the block-size dependence quoted in the mechanism section.
    for pl, tag in (("none", "None"), ("max4", "MaxFour"),
                    ("max16", "MaxSixteen")):
        mac(f"SixBias{tag}", f"{T[(pl, 219)]['bias_uncal']:.3f}")
        mac(f"SixGain{tag}", pct(gains[pl]))

    # Ranking reversal: pysteps (2019 advection) vs EarthFormer (2022).
    # Counted three ways, all with the SAME treatment applied to both models,
    # because the whole point of the paper is that asymmetric treatment
    # manufactures conclusions.
    Tp, _ = load_pool("pysteps")
    Te, _ = load_pool("earthformer")

    def wins(pl, key):
        """thresholds at which the 2019 advection scheme outranks the 2022 net"""
        return sum(1 for t in THRESHOLDS
                   if Tp[(pl, t)][key] > Te[(pl, t)][key])

    rev = {pl: wins(pl, "csi_uncal") for pl in POOLINGS}
    revq = {pl: wins(pl, "csi_qmap_global") for pl in POOLINGS}
    revk = {pl: wins(pl, "csi_recal") for pl in POOLINGS}
    mac("SixRevMaxSixteen", str(rev["max16"]))
    mac("SixRevOther", str(sum(v for k, v in rev.items() if k != "max16")))
    mac("SixRevQ", str(revq["max16"]))
    mac("SixRevK", str(revk["max16"]))

    body = []
    for pl in POOLINGS:
        r = T[(pl, 219)]
        body.append(f"{POOL_TEX[pl]} & {r['bias_uncal']:.3f} & "
                    f"{r['csi_uncal']:.4f} & {r['csi_qmap_global']:.4f} & "
                    f"{pct(r['rel_gain_qmap_global'])} & "
                    f"{rev[pl]} & {revq[pl]} & {revk[pl]} \\\\")
    (OUT / "tab_pooling.tex").write_text("\n".join(body) + "\n")
    print(f"  E6: gain range {gains[lo_pl]:+.1%} ({lo_pl}) to "
          f"{gains[hi_pl]:+.1%} ({hi_pl})")
    print(f"      advection outranks the transformer at {rev['max16']}/6 max16 "
          f"thresholds ({sum(v for k, v in rev.items() if k != 'max16')} under "
          f"all other poolings combined);")
    print(f"      survives global qmap: {revq['max16']}/6; "
          f"survives per-cell knob: {revk['max16']}/6")


# ------------------------------------------- mechanism: dose-response + control
def dose():
    order = ["persistence", "pysteps", "earthformer", "cascast_det",
             "cascast_cascade"]
    pretty = {"persistence": "persistence (a real radar field)",
              "pysteps": "pysteps LK (advection)",
              "earthformer": "EarthFormer",
              "cascast_det": "CasCast backbone",
              "cascast_cascade": "CasCast cascade"}
    body, xs, ys = [], [], []
    ax, ay = [], []
    for n in order:
        L = load_pool(n)
        if not L:
            continue
        T, _ = L
        cells = " & ".join(
            f"{T[('max16', t)]['bias_uncal']:.3f}" for t in [133, 181, 219])
        gains = " & ".join(
            f"{pct(T[('max16', t)]['rel_gain_qmap_global'], 1)}"
            for t in [133, 181, 219])
        body.append(f"{pretty[n]} & {cells} & {gains} \\\\")
        for t in THRESHOLDS:
            xs.append(T[("max16", t)]["bias_uncal"])
            ys.append(T[("max16", t)]["rel_gain_qmap_global"])
        for p in POOLINGS:
            for t in THRESHOLDS:
                ax.append(T[(p, t)]["bias_uncal"])
                ay.append(T[(p, t)]["rel_gain_qmap_global"])
    (OUT / "tab_dose.tex").write_text("\n".join(body) + "\n")

    # Two counts that both happen to equal 30 and mean different things, so they
    # get different macros rather than one reused one. A *cell* is a (pooling,
    # threshold) pair, of which there are 5 x 6 per arm; the dose-response
    # figure reads one pooling convention across arms, so its points are
    # arm x threshold. Emitting both from the data keeps the coincidence from
    # turning into a wrong sentence if either grid changes.
    mac("DoseCells", str(len(POOLINGS) * len(THRESHOLDS)))
    mac("DosePts", str(len(xs)))
    mac("DoseNAll", str(len(ax)))
    mac("DoseRho", f"{-spearman(xs, ys):.3f}")
    mac("DoseRhoAll", f"{-spearman(ax, ay):.3f}")
    # the dose-response uses the five primary arms; the unguided cascade is a
    # reference arm and enters only the pairwise contrasts, so the two counts
    # differ and both are emitted rather than written by hand
    mac("DoseNArms", str(len(body)))

    # The global quantile map's own settings, read from the function that fits
    # it rather than restated in the manuscript, so a change to the sampling cap
    # moves the text with it. The defaults are parsed from the source rather
    # than imported: pooling_control pulls in torch and h5py at module level,
    # and Tier 1 has to run on numpy and matplotlib alone.
    import ast
    src = (_Path(__file__).resolve().parents[1]
           / "sevir" / "pooling_control.py").read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "fit_global_qmap")
    names = [a.arg for a in fn.args.args][-len(fn.args.defaults):]
    d = dict(zip(names, [ast.literal_eval(v) for v in fn.args.defaults]))
    per = max(1, d["sub"] // (384 * 384 * 12))
    mac("QmapQuantiles", thousands(d["n_q"]))
    mac("QmapEvents", str(per))
    mac("QmapPixels", f"{per * 384 * 384 * 12 / 1e7:.1f}\\times10^{{7}}")

    # The guidance arm. Classifier-free guidance weight is a sampling-time
    # scalar, not architecture and not training, and between the unguided and
    # guided cascade it supplies most of the remaining amplitude. Emitted so the
    # three-point ladder (backbone, unguided, guided) can be quoted from data.
    G = load_pool("cascast_cascade_cfg1")
    if G:
        Tg, _ = G
        g = Tg[("max16", 219)]
        mac("GuidBias", f"{g['bias_uncal']:.3f}")
        mac("GuidCsi", f"{g['csi_uncal']:.4f}")
        mac("GuidGain", pct(g["rel_gain_qmap_global"]))
        # The backbone -> unguided -> guided ladder splits into two legs, and
        # they must not be confused: the DIFFUSION leg is backbone to unguided,
        # the GUIDANCE leg is unguided to guided. Both are emitted under names
        # that say which is which, on both scales, because an earlier version
        # printed the diffusion leg under a sentence about guidance.
        import math
        b0 = load_pool("cascast_det")[0][("max16", 219)]["bias_uncal"]
        b1 = g["bias_uncal"]
        b2 = load_pool("cascast_cascade")[0][("max16", 219)]["bias_uncal"]
        d0, d1, d2 = (abs(math.log(x)) for x in (b0, b1, b2))
        mac("LegDiffLog", f"{100 * (d0 - d1) / (d0 - d2):.0f}\\%")
        mac("LegCfgLog", f"{100 * (d1 - d2) / (d0 - d2):.0f}\\%")
        mac("LegDiffLin", f"{100 * (b1 - b0) / (b2 - b0):.0f}\\%")
        mac("LegCfgLin", f"{100 * (b2 - b1) / (b2 - b0):.0f}\\%")

    T, _ = load_pool("persistence")
    b = [T[("max16", t)]["bias_uncal"] for t in THRESHOLDS]
    g = [abs(T[("max16", t)]["rel_gain_qmap_global"]) for t in THRESHOLDS]
    mac("PersBiasLo", f"{min(b):.3f}")
    mac("PersBiasHi", f"{max(b):.3f}")
    mac("PersGainMax", f"{100 * max(g):.1f}\\%")
    print(f"  dose: rho={macros['DoseRho']} over {len(xs)} arm-threshold points "
          f"at max16, {macros['DoseRhoAll']} over all {len(ax)} cells; "
          f"persistence bias [{min(b):.3f},{max(b):.3f}]")


# ------------------------------------------------------ E8: CasCast decomposition
def e8():
    Ld, Lc = load_pool("cascast_det"), load_pool("cascast_cascade")
    if not Lc:
        print("  E8: cascade not yet available, skipping")
        return
    D, _ = Ld
    C, _ = Lc
    pl, t = "max16", 219
    a, c = D[(pl, t)]["csi_uncal"], C[(pl, t)]["csi_uncal"]
    mac("EightDet", f"{a:.4f}")
    mac("EightCasc", f"{c:.4f}")
    mac("EightDetBias", f"{D[(pl, t)]['bias_uncal']:.3f}")
    mac("EightDetQBias", f"{D[(pl, t)]['bias_qmap_global']:.3f}")
    mac("EightDetGain", pct(D[(pl, t)]["rel_gain_qmap_global"], 1))
    mac("EightCascGain", pct(C[(pl, t)]["rel_gain_qmap_global"], 1))
    mac("EightCascBias", f"{C[(pl, t)]['bias_uncal']:.3f}")
    mac("EightReported", pct((c - a) / a, 1))
    for tag, key in (("Q", "csi_qmap_global"), ("K", "csi_recal")):
        b, d = D[(pl, t)][key], C[(pl, t)][key]
        mac(f"EightDet{tag}", f"{b:.4f}")
        mac(f"EightCasc{tag}", f"{d:.4f}")
        mac(f"EightResid{tag}", pct((d - b) / b, 1))
        mac(f"EightShare{tag}",
            f"{100 * (1 - ((d - b) / b) / ((c - a) / a)):.0f}\\%")

    # Absolute form of the same decomposition, quoted in the abstract: the
    # cascade-over-backbone CSI gap before and after the global map.
    gq = C[(pl, t)]["csi_qmap_global"] - D[(pl, t)]["csi_qmap_global"]
    mac("EightGapRaw", f"{c - a:.4f}")
    mac("EightGapCal", f"{gq:.4f}")
    mac("EightGapReduction", f"{100 * (1 - gq / (c - a)):.1f}\\%")
    mac("EightReportedAbs", f"{100 * (c - a) / a:.1f}\\%")
    rq = gq / D[(pl, t)]["csi_qmap_global"]
    mac("EightResidQAbs", f"{100 * rq:.1f}\\%")
    mac("EightRelReduction", f"{100 * (1 - rq / ((c - a) / a)):.1f}\\%")

    # the per-cell knob the backbone was given, quoted so the reader can see how
    # blunt it is: one scalar, fitted on the calibration half, applied to the other
    mac("EightKnob", D[(pl, t)]["knob"].replace("shift=", "$+$")
                                       .replace("mul=", "$\\times$"))

    # our cascade reproduction exceeds the published value, so the decomposition
    # credits the cascade with a larger gain than its authors claimed
    pub_casc = 0.2841
    mac("EightCascPub", f"{pub_casc:.4f}")
    mac("EightCascPubDev", pct((C[(pl, t)]["csi_uncal"] - pub_casc) / pub_casc, 1))

    # The erased share above is a fraction of OUR reproduction of the gain.
    # Against the figure CasCast prints, +91.8%, it is a slightly smaller
    # fraction, and a reader will do that arithmetic, so both are emitted.
    pub_rel = 0.918
    mac("EightReportedPub", f"{100 * pub_rel:+.1f}\\%")
    resid_q = (D[(pl, t)]["csi_qmap_global"], C[(pl, t)]["csi_qmap_global"])
    mac("EightSharePub",
        f"{100 * (1 - ((resid_q[1] - resid_q[0]) / resid_q[0]) / pub_rel):.0f}\\%")

    # under average pooling the cascade is not ahead at all
    mac("EightAvgDet", f"{D[('avg16', t)]['csi_uncal']:.4f}")
    mac("EightAvgCasc", f"{C[('avg16', t)]['csi_uncal']:.4f}")
    mac("EightAvgNBetter", str(sum(
        C[("avg16", s)]["csi_uncal"] > D[("avg16", s)]["csi_uncal"]
        for s in THRESHOLDS)))

    # The per-threshold decomposition now appears as the first block of
    # tab_contrasts, emitted by contrasts(). Only the erased shares are needed
    # here, for the EightShare* macros.
    shares = {}
    for s in THRESHOLDS:
        a, c = D[(pl, s)]["csi_uncal"], C[(pl, s)]["csi_uncal"]
        b, d = D[(pl, s)]["csi_qmap_global"], C[(pl, s)]["csi_qmap_global"]
        rep = (c - a) / a
        res = (d - b) / b
        # a share is only meaningful where there is a reported gain to erase
        if rep > 0.01:
            shares[s] = 1 - res / rep
    ks = sorted(shares)
    mac("EightShareLoTau", str(ks[0]))
    mac("EightShareLo", f"{100 * shares[ks[0]]:.0f}\\%")
    mac("EightShareHi", f"{100 * shares[ks[-1]]:.0f}\\%")
    mac("EightShareMono",
        "increases monotonically" if all(
            shares[x] < shares[y] for x, y in zip(ks, ks[1:])) else "increases")
    print(f"  E8: reported {macros['EightReported']}, "
          f"residual after global qmap {macros['EightResidQ']} "
          f"(share {macros['EightShareQ']}), per-cell knob {macros['EightShareK']}")


# --------------------------------------------------- reproduction fidelity
def repro():
    pub = {("CSI-M", "none"): 0.4310, ("CSI-M", "max16"): 0.4351,
           ("CSI-219", "none"): 0.1448, ("CSI-219", "max16"): 0.1481}
    T, _ = load_pool("cascast_det")
    body, worst = [], 0.0
    for (metric, pl), p in pub.items():
        v = (sum(T[(pl, t)]["csi_uncal"] for t in THRESHOLDS) / 6
             if metric == "CSI-M" else T[(pl, 219)]["csi_uncal"])
        rel = (v - p) / p
        worst = max(worst, abs(rel))
        body.append(f"{metric} & {POOL_TEX[pl]} & {p:.4f} & {v:.4f} & "
                    f"{pct(rel)} \\\\")
    (OUT / "tab_repro.tex").write_text("\n".join(body) + "\n")
    mac("ReproWorst", f"{100 * worst:.1f}\\%")
    print(f"  repro: worst deviation {100 * worst:.1f}%")


# ------------------------------------------------------------------ crowd
def valfitboot():
    """Intervals on the residual under the prior-period fit (Appendix J).

    Skips silently until sevir/valfit_bootstrap.py has been run, which needs a
    per-event cache carrying the qmap_val knob. Emitting the macros here means
    the manuscript sentence can be written once and will fill itself in.
    """
    f = RES / "sevir_valfit_bootstrap.json"
    if not f.exists():
        print("  valfitboot: not available, skipping "
              "(run sevir/valfit_bootstrap.py)")
        return
    d = json.loads(f.read_text())
    head = next(r for r in d["table"] if r["threshold"] == 219)
    for key, tag in (("residual_val", "Val"), ("residual_test", "Test")):
        c = head["contrasts"][key]
        mac(f"VBoot{tag}Diff", f"{c['diff']:+.4f}")
        mac(f"VBoot{tag}Lo", f"{c['ci95'][0]:+.4f}")
        mac(f"VBoot{tag}Hi", f"{c['ci95'][1]:+.4f}")
    n = sum(r["contrasts"]["residual_val"]["excludes_zero"] for r in d["table"])
    mac("VBootNExcl", str(n))
    mac("VBootNThr", str(len(d["table"])))
    mac("VBootN", thousands(d["n_boot"]))
    # the sentence Appendix J needs differs depending on the answer, so the
    # direction is emitted rather than left to be written by hand and forgotten
    mac("VBootVerdict",
        "excludes zero at every threshold" if n == len(d["table"]) else
        f"excludes zero at {n} of {len(d['table'])} thresholds")
    print(f"  valfitboot: val-fit residual at 219 "
          f"{macros['VBootValDiff']} [{macros['VBootValLo']}, "
          f"{macros['VBootValHi']}]; {macros['VBootVerdict']}")


def ets():
    """Does switching to a chance-corrected score remove the confound?

    ETS subtracts the hits a random forecast of the same frequency would get,
    which is the standard answer to "CSI depends on the base rate". It is not an
    answer to "CSI depends on frequency bias", and rather than assert that we
    compute it: the per-event cache carries h, f, m for both knobs, and N is the
    pooled block count, so ETS follows in closed form.
    """
    import numpy as np
    f = RES / "sevir_perevent_counts.npz"
    if not f.exists():
        print("  ets: no per-event cache, skipping")
        return
    z = np.load(f, allow_pickle=True)
    c, arms = z["counts"], [str(a) for a in z["arms"]]
    pi = POOLINGS.index("max16")
    ai, ci = arms.index("cascast_det"), arms.index("cascast_cascade")
    # (384/16)^2 blocks per frame, 12 frames, over the evaluation events
    N = (384 // 16) ** 2 * 12 * c.shape[0]

    def sc(arm, kn, ti):
        h, fa, m = c[:, arm, pi, ti, kn, :].sum(0).astype(float)
        hr = (h + fa) * (h + m) / N
        return h / (h + fa + m), (h - hr) / (h + fa + m - hr)

    shares, rows = {}, []
    for ti, t in enumerate(THRESHOLDS):
        out, cells = {}, []
        for j, key in ((0, "csi"), (1, "ets")):
            d0, c0 = sc(ai, 0, ti)[j], sc(ci, 0, ti)[j]
            d1, c1 = sc(ai, 1, ti)[j], sc(ci, 1, ti)[j]
            raw, cal = (c0 - d0) / d0, (c1 - d1) / d1
            out[key] = (1 - cal / raw) if raw > 0 else None
            cells.append(f"{pct(raw)} & {pct(cal)} & "
                         + (f"{100 * out[key]:.0f}\\%" if out[key] is not None
                            else "---"))
        shares[t] = out
        rows.append(f"{t} & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_ets.tex").write_text("\n".join(rows) + "\n")
    h219 = shares[219]
    mac("EtsErasedQ", f"{100 * h219['ets']:.0f}\\%")
    mac("EtsErasedCsiQ", f"{100 * h219['csi']:.0f}\\%")
    gap = max(abs(v["ets"] - v["csi"]) for v in shares.values()
              if v["ets"] is not None and v["csi"] is not None)
    mac("EtsMaxGap", f"{100 * gap:.0f}")
    # The erased share is only defined where the raw gain is positive; at the
    # lowest threshold the cascade does not lead, so that row carries no share.
    defined = [v["ets"] for v in shares.values() if v["ets"] is not None]
    mac("EtsNThr", str(len(defined)))
    mac("EtsThrLo", str(min(t for t, v in shares.items() if v["ets"] is not None)))
    nog = [t for t, v in shares.items() if v["ets"] is None]
    mac("EtsThrNoGain", ", ".join(str(t) for t in nog) or "none")
    lo, hi = min(defined), max(defined)
    mac("EtsErasedLo", f"{100 * lo:.0f}\\%")
    mac("EtsErasedHi", f"{100 * hi:.0f}\\%")
    print(f"  ets: erased share at 219 is {macros['EtsErasedQ']} under ETS "
          f"against {macros['EtsErasedCsiQ']} under CSI; "
          f"largest gap over thresholds {macros['EtsMaxGap']} points")


def crowd():
    f = RES / "crowd_matched_report.md"
    if not f.exists():
        return
    txt = f.read_text()
    rows = re.findall(
        r"\| ([\d.]+) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+) \| ([+-][\d.]+) \| "
        r"(\d+)/(\d+) \| ([\d.nan]+) \| ([+-][\d.]+) \|", txt)
    if not rows:
        return
    body = [f"{a} & {b}\\% & {c} & {d} & {e} & {f_}/{g} & {h} & {i} \\\\"
            for a, b, c, d, e, f_, g, h, i in rows]
    (OUT / "tab_crowd.tex").write_text("\n".join(body) + "\n")
    swings = [float(r[4]) - float(r[8]) for r in rows]
    k = max(range(len(rows)), key=lambda i: abs(swings[i]))
    mac("CrowdSwing", f"{swings[k]:+.3f}")
    mac("CrowdThr", rows[k][0])
    mac("CrowdMatched", rows[k][4])
    mac("CrowdBeta", rows[k][8])
    mac("CrowdSeeds", f"{rows[k][5]}/{rows[k][6]}")
    print(f"  crowd: largest swing {swings[k]:+.3f} at threshold {rows[k][0]}")


# ------------------------------------------------------------------ radar
def radar():
    f = RES / "radar_matched_report.md"
    if not f.exists():
        return
    rows = re.findall(
        r"\| ([\d.]+mm) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| "
        r"([+-][\d.]+) \| ([+-][\d.]+) \|", f.read_text())
    if not rows:
        return
    (OUT / "tab_radar.tex").write_text(
        "\n".join(f"{a} & {b} & {c} & {d} & {e} & {g} \\\\"
                  for a, b, c, d, e, g in rows) + "\n")
    swings = [abs(float(r[5]) - float(r[4])) for r in rows]
    k = max(range(len(rows)), key=lambda i: swings[i])
    others = [swings[i] for i in range(len(rows)) if i != k]
    mac("RadarThr", rows[k][0].replace("mm", "\\,mm"))
    mac("RadarBeta", rows[k][4])
    mac("RadarMatched", rows[k][5])
    mac("RadarOtherMax", f"{max(others):.3f}")
    mac("RadarNThr", str(len(rows)))
    print(f"  radar: largest swing at {rows[k][0]} "
          f"({rows[k][4]} -> {rows[k][5]}); others <= {max(others):.3f}")


# ------------------------------------------------- atmosphere (self-check)
def atmos():
    f = RES / "selection_matched_csi.md"
    if not f.exists():
        return
    txt = f.read_text()
    rows = re.findall(r"\| (\d+) \| ([\d.]+)% \| C \| [\d.]+ \| [\d.]+ \| "
                      r"\*\*([+-][\d.]+)\*\* \|", txt)
    if not rows:
        return
    neg = sum(1 for r in rows if float(r[2]) < 0)
    mac("AtmosBehind", str(neg))
    mac("AtmosTotal", str(len(rows)))
    print(f"  atmosphere: dual decoder behind at {neg}/{len(rows)} thresholds "
          f"under the matched control (rule C)")

    # the per-threshold table behind that count, so the claim is not a bare
    # number in the text with nothing under it
    j = RES / "selection_matched_csi.json"
    if not j.exists():
        return
    d = json.loads(j.read_text())
    body, mags = [], []
    for tau in sorted(d, key=lambda t: -int(t)):
        r = d[tau]
        a, c = r["A"], r["C"]
        m = c["dual_decoder"] - c["base"]
        won = sum(1 for x in c["margin"] if x > 0)
        mags.append(abs(m))
        body.append(f"{tau} & {100 * r['R']:.2f}\\% & {c['dual_decoder']:.4f} & "
                    f"{c['base']:.4f} & {m:+.4f} & {won}/{len(c['margin'])} & "
                    f"{a['dual_decoder'] - a['base']:+.4f} \\\\")
    (OUT / "tab_atmos.tex").write_text("\n".join(body) + "\n")
    mac("AtmosSeeds", str(len(d["220"]["C"]["margin"])))
    mac("AtmosWorst", f"{-max(mags):+.4f}")
    mac("AtmosBest",
        f"{max(d[t]['C']['dual_decoder'] - d[t]['C']['base'] for t in d):+.4f}")


def atmos_allscene():
    """Compact manuscript macros for the no-frame-filter robustness check."""
    rarity = RES / "allscene_pooled" / "atmospheric_rarity_allscene_pooled.md"
    selection = (RES / "allscene_pooled" /
                 "atmospheric_selection_allscene_pooled.json")
    if not rarity.exists() or not selection.exists():
        return

    txt = rarity.read_text()
    rows = re.findall(
        r"\| (\d+) \| ([\d.]+)% \| [\d.]+ \| [\d.]+ \| [+-][\d.]+ \| "
        r"([+-][\d.]+)% \|", txt)
    rho = re.search(r"Spearman\(relative gain, log10 R\) = ([-\d.]+), "
                    r"p = ([\d.]+)", txt)
    if not rows or not rho:
        return
    by_tau = {tau: gain for tau, _, gain in rows}
    d = json.loads(selection.read_text())
    behind = sum(d[t]["C"]["dual_decoder"] < d[t]["C"]["base"] for t in d)
    mac("AtmosAllRho", rho.group(1))
    mac("AtmosAllP", rho.group(2))
    mac("AtmosAllGainCommon", by_tau["220"] + "\\%")
    mac("AtmosAllGainRarest", by_tau["197"] + "\\%")
    mac("AtmosAllBehind", str(behind))
    mac("AtmosAllTotal", str(len(d)))
    print(f"  atmosphere all-scene: rho={rho.group(1)}; dual decoder behind "
          f"at {behind}/{len(d)} thresholds under rule C")


# ------------------------------------------------------------- base rates
def rates():
    f = RES / "sevir_base_rates.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    R = d["R"]
    mac("RareTop", f"{100 * R['219']:.3f}\\%")
    mac("RareTopOne", f"{1 / R['219']:.0f}")
    mac("RareMid", f"{100 * R['181']:.3f}\\%")
    mac("RateEvents", str(d["n_events"]))
    print(f"  base rates: tau=219 R={100*R['219']:.3f}% (1 in {1/R['219']:.0f})")


# -------------------------------------------------------------- bootstrap
def crowdpool():
    """Crowd density under its own protocol: the generality claim, from artefacts."""
    import numpy as np
    runs = [json.loads(f.read_text())
            for f in sorted(RES.glob("crowd_pooling_s*.json"))]
    if not runs:
        print("  crowdpool: not yet available, skipping")
        return
    cells = [(r["bias_uncal"], r["csi_percell"] - r["csi_uncal"])
             for d in runs for r in d["table"]
             if r["usable"] and all(np.isfinite(r[k])
                                    for k in ("bias_uncal", "csi_percell",
                                              "csi_uncal"))]
    def rho(cs):
        return -spearman([c[0] for c in cs], [c[1] for c in cs])

    # Over the trained arms, which is the analogue of the SEVIR statistic
    # (that one ranges over systems). Reported alongside the all-cell value,
    # which is diluted: the constructed controls are 800 of the cells and sit in
    # a tie at exactly zero gain, which Spearman resolves by giving every member
    # of the tie the same averaged rank.
    trained = [(r["bias_uncal"], r["csi_percell"] - r["csi_uncal"])
               for d in runs for r in d["table"]
               if r["usable"] and r["arm"] in ("baseline", "csrnet")
               and all(np.isfinite(r[k]) for k in ("bias_uncal", "csi_percell",
                                                   "csi_uncal"))]
    mac("CrowdRho", f"{rho(trained):.3f}")
    mac("CrowdN", str(len(trained)))
    mac("CrowdRhoAll", f"{rho(cells):.3f}")
    mac("CrowdNAll", str(len(cells)))
    mac("CrowdSeeds", str(len(runs)))

    # The seed spread: same architecture, same data, same recipe, one number
    # changed. At a rare threshold the score is not a stable property of the
    # architecture, and what it tracks is the calibration state.
    rs = [r for d in runs for r in d["table"]
          if r["arm"] == "csrnet" and r["pooling"] == "none"
          and r["threshold"] == 10.0 and r["usable"]]
    if len(rs) >= 3:
        c = [r["csi_uncal"] for r in rs]
        b = [r["bias_uncal"] for r in rs]
        mac("CrowdSeedTau", "10")
        mac("CrowdSeedCsiLo", f"{min(c):.4f}")
        mac("CrowdSeedCsiHi", f"{max(c):.4f}")
        mac("CrowdSeedBiasLo", f"{min(b):.3f}")
        mac("CrowdSeedBiasHi", f"{max(b):.3f}")
        mac("CrowdSeedFold", f"{max(c) / max(min(c), 1e-9):.0f}")
        mac("CrowdSeedRho", f"{-rho(list(zip(b, c))):.2f}")

        # A seed spread on a threshold score means nothing on its own: a run
        # that scores CSI 0.0017 may simply be broken. The same records carry a
        # count-level error over the high-density patches, which is what the
        # benchmark is about, so the two ranges can be quoted side by side and
        # the reader can see which one the metric is amplifying. Read from
        # posthoc_crowd_matched.json, whose csi_pooled/bias_pooled reproduce
        # the csi_uncal/bias_uncal above exactly, so this is one measurement.
        mf = RES / "posthoc_crowd_matched.json"
        if mf.exists():
            sd = json.loads(mf.read_text())["single_decoder"]
            mae = []
            for k in sorted(sd):
                if not k.startswith("csrnet_"):
                    continue
                rec = next((x for x in sd[k]["shift"]["test"]
                            if x["knob"] == 0), None)
                cell_ = rec.get("10.0") if rec else None
                if cell_ and np.isfinite(cell_.get("mae_high", float("nan"))):
                    mae.append(cell_["mae_high"])
            if len(mae) >= 3:
                mac("CrowdSeedMaeLo", f"{min(mae):.1f}")
                mac("CrowdSeedMaeHi", f"{max(mae):.1f}")
                mac("CrowdSeedMaeFold", f"{max(mae) / min(mae):.1f}")
                mac("CrowdSeedNMae", str(len(mae)))
            # per-seed table, so the spread is inspectable rather than asserted
            srows = []
            for k in sorted(sd):
                if not k.startswith("csrnet_"):
                    continue
                rec = next((x for x in sd[k]["shift"]["test"]
                            if x["knob"] == 0), None)
                c_ = rec.get("10.0") if rec else None
                if not c_:
                    continue
                srows.append(f"{k.split('_')[-1]} & {c_['bias_pooled']:.3f} & "
                             f"{c_['csi_pooled']:.4f} & {c_['mae_high']:.2f} & "
                             f"{c_['mae_low']:.3f} \\\\")
            if srows:
                (OUT / "tab_crowdseed.tex").write_text("\n".join(srows) + "\n")
            # Does the count error order the score? If it does, the spread is
            # the metric amplifying a real but small difference rather than
            # separating working runs from broken ones.
            pairs = []
            for k in sorted(sd):
                if not k.startswith("csrnet_"):
                    continue
                rec = next((x for x in sd[k]["shift"]["test"]
                            if x["knob"] == 0), None)
                c_ = rec.get("10.0") if rec else None
                if c_:
                    pairs.append((c_["mae_high"], c_["csi_pooled"]))
            if len(pairs) >= 3:
                mac("CrowdSeedMaeRho",
                    f"{spearman([p[0] for p in pairs], [p[1] for p in pairs]):.3f}")

    # The displacement control: a real field put in the wrong place. Quoted
    # under the benchmark's published protocol, where every cell is scored over
    # ~92k blocks. The extra pooling variants we add on top are coarser, and
    # the avg 4x4 cells carry as few as 45 positive blocks, so their spread is
    # reported separately rather than folded into the headline range.
    sh = [r for d in runs for r in d["table"]
          if r["arm"].startswith("gt_shift") and r["usable"]
          and r["pooling"] == "none"]
    shall = [r for d in runs for r in d["table"]
             if r["arm"].startswith("gt_shift") and r["usable"]]
    if sh:
        b = [r["bias_uncal"] for r in sh]
        g = [abs(r["csi_percell"] - r["csi_uncal"]) for r in sh]
        mac("CrowdShiftBiasLo", f"{min(b):.3f}")
        mac("CrowdShiftBiasHi", f"{max(b):.3f}")
        mac("CrowdShiftGain", f"{max(g):.4f}")
        lo = min(sh, key=lambda r: r["csi_uncal"])
        mac("CrowdShiftCsiLo", f"{lo['csi_uncal']:.4f}")
        mac("CrowdShiftCsiLoTau", f"{lo['threshold']:g}")
        mac("CrowdShiftCsiHi", f"{max(r['csi_uncal'] for r in sh):.4f}")
        mac("CrowdShiftCsiHiTau",
            f"{max(sh, key=lambda r: r['csi_uncal'])['threshold']:g}")
        mac("CrowdShiftBlocks", thousands(sh[0]["n_blocks"]))
        ga = [abs(r["csi_percell"] - r["csi_uncal"]) for r in shall]
        mac("CrowdShiftGainAll", f"{max(ga):.4f}")
    # Table body. The reported knob is the one selected on |Bias - 1|, which is
    # the conservative choice here: for CSRNet it recovers strictly less than
    # selecting on validation CSI would, so we are not quoting the knob's best
    # case. The val-CSI figures go in the text as the robustness check.
    taus = [3.0, 7.0, 10.0]
    pretty = {"csrnet": "CSRNet", "baseline": "CNN baseline (MSE)",
              "gt_shift2": "target, displaced 2 blocks",
              "gt_blur4": "target, blurred (mass conserved)",
              "gt_flat0.3": "target, peaks compressed (mass conserved)"}

    def agg(arm, tau, key):
        v = []
        for d in runs:
            for r in d["table"]:
                if (r["arm"] == arm and r["pooling"] == "none"
                        and r["threshold"] == tau and r["usable"]):
                    v.append(r["csi_percell"] - r["csi_uncal"]
                             if key == "d" else r[key])
        return v

    body = []
    for arm in ("csrnet", "baseline", "gt_shift2", "gt_blur4",
                "gt_flat0.3"):
        cs = [agg(arm, t, "bias_uncal") for t in taus]
        ds = [agg(arm, t, "d") for t in taus]
        if not all(cs):
            continue
        body.append(pretty[arm] + " & "
                    + " & ".join(f"{np.mean(c):.3f}" for c in cs) + " & "
                    + " & ".join(f"{np.mean(d):+.4f}" for d in ds) + " \\\\")
    (OUT / "tab_crowdpool.tex").write_text("\n".join(body) + "\n")

    # val-CSI selection: each arm's best shot, so no arm is handed a knob that
    # hurt it. Quoted in the text because the conclusion must not need a rule.
    for arm, tag in (("csrnet", "Csr"), ("baseline", "Base")):
        w = [r["csi_valcsi"] - r["csi_uncal"] for d in runs for r in d["table"]
             if r["arm"] == arm and r["pooling"] == "none" and r["usable"]
             and "csi_valcsi" in r and np.isfinite(r["csi_valcsi"])]
        b = [r["csi_percell"] - r["csi_uncal"] for d in runs for r in d["table"]
             if r["arm"] == arm and r["pooling"] == "none" and r["usable"]]
        if w:
            mac(f"Crowd{tag}ValMax", f"{max(w):+.4f}")
            mac(f"Crowd{tag}ValMin", f"{min(w):+.4f}")
        if b:
            mac(f"Crowd{tag}BiasMax", f"{max(b):+.4f}")
            mac(f"Crowd{tag}BiasMin", f"{min(b):+.4f}")
    for a_, tag in (("gt_flat0.7", "Mild"), ("gt_flat0.3", "Hard")):
        rs = [r for d in runs for r in d["table"]
              if r["arm"] == a_ and r["pooling"] == "none"
              and r["threshold"] == 10.0 and r["usable"]]
        if rs:
            mac(f"CrowdFlat{tag}Bias", f"{np.mean([r['bias_uncal'] for r in rs]):.3f}")
            mac(f"CrowdFlat{tag}Gain",
                f"{np.mean([r['csi_percell'] - r['csi_uncal'] for r in rs]):+.4f}")
    rs = [r for d in runs for r in d["table"]
          if r["arm"] == "gt_blur4" and r["pooling"] == "none"
          and r["threshold"] == 10.0 and r["usable"]]
    if rs:
        mac("CrowdBlurBias", f"{np.mean([r['bias_uncal'] for r in rs]):.3f}")
        mac("CrowdBlurGain",
            f"{np.mean([r['csi_percell'] - r['csi_uncal'] for r in rs]):+.4f}")
        mac("CrowdBlurCsi", f"{np.mean([r['csi_uncal'] for r in rs]):.4f}")
    print(f"  crowdpool: rho={macros['CrowdRho']} over {len(trained)} trained cells, "
          f"{len(runs)} seeds")


def segpool():
    """COCO + DeepLabV3: the same statistic under the name IoU.

    The registered prediction was that the knob's payoff rises with class
    rarity, as it does in the other two domains. It does not. This model is
    trained with cross-entropy, a proper scoring rule, and sits near calibration
    whatever the class frequency, so rarity alone leaves the knob nothing to
    take. The relationship that does hold, more cleanly here than in either other
    domain, is
    that the payoff tracks the bias deficit. Both numbers are emitted and the
    manuscript reports the failure as prominently as the success.
    """
    import numpy as np

    # reproduction anchor: its own artefact, read independently of the sweep
    rf = RES / "seg_repro.json"
    if rf.exists():
        rp = json.loads(rf.read_text())
        mac("SegReproOurs", f"{rp['miou']:.4f}")
        mac("SegReproPub", f"{rp['published']:.3f}")
        mac("SegReproDev", f"{100 * rp['rel_dev']:+.1f}\\%")
        mac("SegReproN", str(rp["n_images"]))
        # the two protocol differences behind the mIoU gap, named in the
        # artefact rather than retyped into the appendix
        for i, s in enumerate(rp.get("known_differences", [])[:2]):
            mac(f"SegDiff{'AB'[i]}", s)
        mac("SegNDiff", str(len(rp.get("known_differences", []))))

    f = RES / "seg_pooling.json"
    if not f.exists():
        print("  segpool: sweep not yet available, skipping")
        return
    d = json.loads(f.read_text())
    rows = [r for r in d["table"] if r["pooling"] == "none" and r["usable"]]
    if len(rows) < 4:
        print(f"  segpool: only {len(rows)} usable classes, skipping")
        return
    rows.sort(key=lambda r: -r["R"])

    sp = spearman

    R = [r["R"] for r in rows]
    dl = [r["iou_bias"] - r["iou"] for r in rows]
    dv = [r["iou_valiou"] - r["iou"] for r in rows]
    bias = [r["bias"] for r in rows]
    mac("SegRhoRarity", f"{-sp(R, dl):+.3f}")
    mac("SegRhoBias", f"{-sp(bias, dl):+.3f}")
    # The visible counterexample to "near calibration implies an inert knob":
    # the class inside the near-one band that loses the most IoU. Named with
    # the validation-selected figure beside it, since the gap between the two
    # is what identifies the cause as calibration-half noise.
    near = [r for r in rows if abs(r["bias"] - 1) <= 0.06]
    if near:
        w = min(near, key=lambda r: r["iou_bias"] - r["iou"])
        mac("SegCtrCls", w["cls"])
        mac("SegCtrBias", f"{w['bias']:.3f}")
        mac("SegCtrR", f"{100 * w['R']:.3f}\\%")
        mac("SegCtrGain", f"{w['iou_bias'] - w['iou']:+.4f}")
        mac("SegCtrGainVal", f"{w['iou_valiou'] - w['iou']:+.4f}")
        # The general statement the counterexample is an instance of: at bias ~ 1
        # the fitted shift is noise about zero, so its expected value at a
        # near-optimal operating point is non-positive. Emitted as a count and a
        # mean over the whole band so the claim is checkable, not asserted from
        # one row.
        nd = [r["iou_bias"] - r["iou"] for r in near]
        mac("SegNearMeanGain", f"{np.mean(nd):+.4f}")
        mac("SegNearHurt", str(sum(1 for x in nd if x < -0.001)))
        mac("SegNearHelped", str(sum(1 for x in nd if x > 0.001)))
        mac("SegNearFlat", str(sum(1 for x in nd if abs(x) <= 0.001)))
    mac("SegMeanGain", f"{np.mean(dl):+.4f}")
    mac("SegMeanGainVal", f"{np.mean(dv):+.4f}")
    mac("SegHelped", str(sum(1 for x in dl if x > 0.001)))
    mac("SegHurt", str(sum(1 for x in dl if x < -0.001)))
    # classes the knob leaves alone. Without this count the helped and hurt
    # counts do not sum to the class count and the table looks inconsistent.
    mac("SegFlat", str(sum(1 for x in dl if abs(x) <= 0.001)))
    mac("SegFlatThr", "0.001")
    mac("SegMaxGain", f"{max(dl):+.4f}")
    mac("SegNCls", str(len(rows)))
    mac("SegNImg", str(d["n_images"]))
    mac("SegArch", d["arch"].replace("_", "-"))
    mac("SegRLo", f"{100 * min(R):.3f}\\%")
    mac("SegRHi", f"{100 * max(R):.2f}\\%")

    # how many classes this cross-entropy model already places near calibration,
    # and what the knob manages to buy them
    near_thr = 0.06
    near = [r for r in rows if abs(r["bias"] - 1) <= near_thr]
    mac("SegNearOne", str(len(near)))
    mac("SegNearThr", f"{near_thr:g}")
    mac("SegNearGainMax",
        f"{max(abs(r['iou_bias'] - r['iou']) for r in near):.4f}")

    # the deficit classes, where the knob does pay
    dfc = sorted([r for r in rows if r["bias"] < 0.85], key=lambda r: r["bias"])
    mac("SegDefN", str(len(dfc)))
    mac("SegDefCls", dfc[0]["cls"])
    mac("SegDefBias", f"{dfc[0]['bias']:.3f}")
    mac("SegDefGain", f"{dfc[0]['iou_bias'] - dfc[0]['iou']:+.4f}")
    best = max(rows, key=lambda r: r["iou_bias"] - r["iou"])
    mac("SegBestCls", best["cls"])
    mac("SegBestBias", f"{best['bias']:.3f}")
    mac("SegBestGain", f"{best['iou_bias'] - best['iou']:+.4f}")

    # the one long-bias class: matching bias costs it IoU, as in the crowd domain
    ovr = max(rows, key=lambda r: r["bias"])
    mac("SegOverCls", ovr["cls"])
    mac("SegOverBias", f"{ovr['bias']:.3f}")
    mac("SegOverGain", f"{ovr['iou_bias'] - ovr['iou']:+.4f}")

    # pooling strengthens the bias relationship, as the SEVIR mechanism predicts
    for pl, tag in (("max8", "PoolA"), ("max16", "PoolB")):
        rr = [r for r in d["table"] if r["pooling"] == pl and r["usable"]]
        if len(rr) >= 4:
            mac(f"SegRhoBias{tag}", f"{-sp([x['bias'] for x in rr], [x['iou_bias'] - x['iou'] for x in rr]):+.3f}")
            mac(f"SegN{tag}", str(len(rr)))

    # twenty classes is a tall table; pair them into two column-groups so every
    # class is still shown rather than a chosen subset
    def cell(r):
        return (f"{r['cls']} & {100*r['R']:.3f}\\% & {r['bias']:.3f} & "
                f"{r['iou']:.4f} & {r['iou_bias'] - r['iou']:+.4f}")
    half = (len(rows) + 1) // 2
    body = []
    for i in range(half):
        left = cell(rows[i])
        right = cell(rows[i + half]) if i + half < len(rows) else " & & & & "
        body.append(f"{left} & {right} \\\\")
    (OUT / "tab_seg.tex").write_text("\n".join(body) + "\n")
    print(f"  segpool: {len(rows)} classes / {d['n_images']} images; "
          f"rho(rarity)={macros['SegRhoRarity']} PREDICTION FAILED; "
          f"rho(bias)={macros['SegRhoBias']} holds; "
          f"helped {macros['SegHelped']} hurt {macros['SegHurt']}; "
          f"mean {macros['SegMeanGain']}")


# ------------------------------- published contrasts, before and after calibration
def contrasts():
    """Every pairwise contrast, and how far calibrating both arms moves it.

    Two things are emitted. First the predictive claim, that the pair's
    calibration gap orders the distortion. Then the consequences: the flips, the one published
    comparison whose real gain the raw metric *understates*, and the pair that is
    one architecture trained twice.
    """
    f = RES / "sevir_published_contrasts.json"
    d = json.loads(f.read_text())
    rows, st = d["rows"], d["stats"]

    mac("ContrastN", str(st["all"]["n"]))
    mac("ContrastNPairs", str(len({(r["a"], r["b"]) for r in rows})))
    mac("ContrastNArms", str(len({r["a"] for r in rows} | {r["b"] for r in rows})))
    for tag, key in (("all", "ContrastRAll"), ("max16", "ContrastRMax"),
                     ("extreme", "ContrastRExt"), ("published", "ContrastRPub")):
        mac(key, f"{st[tag]['pearson']:+.3f}")
        mac(key.replace("ContrastR", "ContrastRho"),
            f"{st[tag]['spearman']:+.3f}")
    mac("ContrastFlips", str(st["all"]["flips"]))
    mac("ContrastFlipsPct", f"{100*st['all']['flips']/st['all']['n']:.0f}\\%")
    big = [r for r in rows if r["flip"]
           and min(abs(r["raw"]), abs(r["cal"])) > 0.01]
    mac("ContrastFlipsBig", str(len(big)))
    mac("ContrastFlipsBigThr", "0.01")
    # How many reversals survive if the reference arm is dropped entirely. The
    # unguided cascade is the arm a reader is most likely to call a
    # non-comparison, so the count without it is emitted rather than defended.
    fl = [r for r in rows if r["flip"]]
    nog = [r for r in fl if "cfg1" not in r["a"] and "cfg1" not in r["b"]]
    mac("ContrastFlipsNoGuid", str(len(nog)))
    mac("ContrastFlipsNoGuidPct", f"{100 * len(nog) / len(fl):.0f}\\%")

    # Size of the reversing contrasts in the units papers quote: the raw
    # relative gain of the arm that leads before calibration.
    import statistics
    relmag = [abs(r["rel_raw"]) for r in fl]
    mac("ContrastFlipsRelMed", f"{100 * statistics.median(relmag):.1f}\\%")
    mac("ContrastFlipsRelFive", str(sum(x > 0.05 for x in relmag)))
    mac("ContrastFlipsRelTen", str(sum(x > 0.10 for x in relmag)))

    # The subset of pairs that a published paper itself compares.
    mac("ContrastNPub", str(st["published"]["n"]))
    mac("ContrastFlipsPub", str(st["published"]["flips"]))
    mac("ContrastNPubPairs", str(len({(r["a"], r["b"]) for r in rows
                                      if r["published"]})))

    def cell(a, b, pl, th):
        for r in rows:
            if (r["a"], r["b"], r["pooling"], r["threshold"]) == (a, b, pl, th):
                return r
        return None

    # The case that runs the other way: persistence is a real radar field and sits
    # at bias ~1, so the raw comparison penalises the attenuated model instead of
    # flattering it. This is the reason the paper does not say "papers overclaim".
    r = cell("persistence", "earthformer", "max16", 219)
    mac("DeflTau", "219")
    mac("DeflRaw", pct(r["rel_raw"]))
    mac("DeflCal", pct(r["rel_cal"]))
    mac("DeflFactor", f"{r['rel_cal']/r['rel_raw']:.1f}")
    mac("DeflBiasP", f"{r['bias_a']:.3f}")
    mac("DeflBiasM", f"{r['bias_b']:.3f}")
    rf = cell("persistence", "earthformer", "max16", 160)
    mac("DeflFlipTau", "160")
    mac("DeflFlipRaw", pct(rf["rel_raw"]))
    mac("DeflFlipCal", pct(rf["rel_cal"]))

    # One architecture, two independent trainings, two released checkpoints.
    sa = [r for r in rows if r.get("same_arch") and r["pooling"] == "max16"]
    sa.sort(key=lambda r: r["threshold"])
    mac("SameArchFlips", str(sum(1 for r in sa if r["flip"])))
    mac("SameArchN", str(len(sa)))
    hi = [r for r in sa if r["threshold"] == 219][0]
    mac("SameArchRaw", pct(hi["rel_raw"]))
    mac("SameArchCal", pct(hi["rel_cal"]))
    mac("SameArchBiasA", f"{hi['bias_a']:.3f}")
    mac("SameArchBiasB", f"{hi['bias_b']:.3f}")
    (OUT / "tab_samearch.tex").write_text("\n".join(
        f"{r['threshold']} & {r['bias_a']:.3f} & {r['bias_b']:.3f} & "
        f"{r['raw']:+.4f} & {r['cal']:+.4f} & {pct(r['rel_raw'])} & "
        f"{pct(r['rel_cal'])} \\\\" for r in sa) + "\n")

    # Cross-paper: CasCast's cascade against the EarthFormer release it reports
    # beating. A second published headline, decomposed the same way.
    ec = [r for r in rows if (r["a"], r["b"]) == ("earthformer", "cascast_cascade")
          and r["pooling"] == "max16"]
    ec.sort(key=lambda r: r["threshold"])
    hi = [r for r in ec if r["threshold"] == 219][0]
    mac("EFCascRaw", pct(hi["rel_raw"]))
    mac("EFCascCal", pct(hi["rel_cal"]))
    mac("EFCascErased", f"{100*(1 - hi['rel_cal']/hi['rel_raw']):.0f}\\%")

    # Table body: the three published pairs at the headline pooling.
    # One table for all three published contrasts, at the four rarest thresholds.
    # "erased" is 1 - calibrated/raw, shown only where there is a raw gain to erase.
    order = [("cascast_det", "cascast_cascade", "cascade vs.\\ own backbone"),
             ("earthformer", "cascast_cascade", "cascade vs.\\ EarthFormer"),
             ("persistence", "earthformer", "EarthFormer vs.\\ persistence")]
    body = []
    for i, (a, b, lab) in enumerate(order):
        # The first block runs from tau=74 rather than 133, because the low end
        # of the erased-share trend is quoted from it in the text and should be
        # a row the reader can check. The other two start at 133: extending them
        # costs a page of main text and no claim rests on those cells.
        lo = 74 if i == 0 else 133
        sub = sorted([r for r in rows if (r["a"], r["b"]) == (a, b)
                      and r["pooling"] == "max16" and r["threshold"] >= lo],
                     key=lambda r: r["threshold"])
        if i:
            body.append("\\midrule")
        for j, r in enumerate(sub):
            name = f"\\multirow{{{len(sub)}}}{{*}}{{{lab}}}" if j == 0 else ""
            # Only meaningful when there is a positive reported gain and the
            # calibrated one is smaller. Where calibration *raises* the gain the
            # ratio is unbounded off a near-zero base, so we print an arrow
            # instead of a number that would be read as precision.
            er = (f"{100*(1 - r['rel_cal']/r['rel_raw']):.0f}\\%"
                  if r["rel_raw"] and r["rel_raw"] > 0
                  and r["rel_cal"] < r["rel_raw"] else "$\\uparrow$")
            body.append(
                f"{name} & {r['threshold']} & {r['bias_a']:.3f} & "
                f"{r['bias_b']:.3f} & {pct(r['rel_raw'])} & "
                f"{pct(r['rel_cal'])} & {er} \\\\")
    (OUT / "tab_contrasts.tex").write_text("\n".join(body) + "\n")

    print(f"  contrasts: {st['all']['n']} cells over "
          f"{macros['ContrastNPairs']} pairs, r={macros['ContrastRMax']} (max16), "
          f"{st['all']['flips']} flips ({len(big)} substantive)")


def canon(arm, pooling, thr):
    """Point estimates from the pooling-control record the main text quotes.

    The bootstrap and prior-window records were computed from an earlier
    inference run of the CasCast backbone, which differs from the current one by
    at most 1.1e-4 CSI. Tables built from those records take their identity and
    test-half point estimates from here, so one number appears everywhere, and
    keep their own intervals and prior-window columns.
    """
    L = load_pool(arm)
    if not L:
        return None
    r = L[0][(pooling, thr)]
    return r["csi_uncal"], r["csi_qmap_global"]


def head_contrast(r):
    """The residual contrast, oriented as cascade+cal minus det+cal.

    The resampler emits each pair once in whichever order it enumerated them,
    so we take whichever key exists and flip the sign (and the interval ends)
    when it is the reversed one.
    """
    if "cascade+cal - det+cal" in r["contrasts"]:
        return r["contrasts"]["cascade+cal - det+cal"], 1
    return r["contrasts"].get("det+cal - cascade+cal"), -1


def boot():
    f = RES / "sevir_bootstrap_cascast.json"
    if not f.exists():
        print("  bootstrap: not yet available, skipping")
        return
    d = json.loads(f.read_text())
    rows = {(r["pooling"], r["threshold"]): r for r in d["table"]}
    body, pmax = [], 0.0
    for t in THRESHOLDS:
        r = rows[("max16", t)]
        # point estimates for the four arms, then the full interval on the two
        # contrasts that carry the argument -- per-arm intervals are wide
        # because events differ in difficulty, which the paired contrast removes
        du, dq = canon("cascast_det", "max16", t)
        cu, cq = canon("cascast_cascade", "max16", t)
        cells = [f"{v:.4f}" for v in (du, dq, cu, cq)]
        k = r["contrasts"].get("det - det+cal")
        cells.append(f"{dq - du:+.4f} [{-k['ci95'][1]:+.3f}, "
                     f"{-k['ci95'][0]:+.3f}]")
        c, sign = head_contrast(r)
        pmax = max(pmax, c["p"])
        body.append(" & ".join([str(t)] + cells) +
                    f" & {cq - dq:+.4f} "
                    f"[{sign * c['ci95'][sign < 0]:+.3f}, "
                    f"{sign * c['ci95'][sign > 0]:+.3f}] \\\\")
    (OUT / "tab_boot.tex").write_text("\n".join(body) + "\n")
    mac("BootPMax", "< 0.001" if pmax == 0 else f"= {pmax:.3f}")
    mac("BootN", str(d["n_boot"]))
    mac("BootEvents", str(d["n_eval"]))
    r = rows[("max16", 219)]
    c, sign = head_contrast(r)
    if c:
        du, dq = canon("cascast_det", "max16", 219)
        cu, cq = canon("cascast_cascade", "max16", 219)
        mac("BootHeadDiff", f"{cq - dq:+.4f}")
        mac("BootHeadLo", f"{sign * c['ci95'][sign < 0]:+.4f}")
        mac("BootHeadHi", f"{sign * c['ci95'][sign > 0]:+.4f}")
        # the resampler reports p as a proportion of replicates; 0 means no
        # replicate crossed zero, which at 2000 draws is a bound rather than a value
        mac("BootHeadP", "0.001" if c["p"] == 0 else f"{c['p']:.3f}")
        mac("BootHeadPRel", "<0.001" if c["p"] == 0 else f"= {c['p']:.3f}")
    # the knob's own effect on the backbone: the quantity the paper says is free
    k = r["contrasts"].get("det - det+cal")
    if k:
        mac("BootKnobDiff", f"{dq - du:+.4f}")
        mac("BootKnobLo", f"{-k['ci95'][1]:+.4f}")
        mac("BootKnobHi", f"{-k['ci95'][0]:+.4f}")
    if c:
        print(f"  bootstrap: cascade+cal - det+cal at 219 = {sign*c['diff']:+.4f} "
              f"[{sign*c['ci95'][sign<0]:+.4f}, {sign*c['ci95'][sign>0]:+.4f}], "
              f"p={c['p']:.4f} over {d['n_boot']} replicates")


def cboot():
    """Intervals on the pairwise-contrast claims, from resampling events.

    The 450 cells share arms and share events, so the correlation over them and
    the count of sign reversals among them are both computed from a dependent
    sample. Resampling the event, which is the independent unit, puts an
    interval on each. The pair-level figure repeats the correlation with the 15
    pairs as the sample, which is the conservative reading.
    """
    f = RES / "sevir_contrast_bootstrap.json"
    if not f.exists():
        print("  cboot: not available, skipping")
        return
    d = json.loads(f.read_text())
    mac("CBootN", thousands(d["n_boot"]))
    mac("CBootEvents", str(d["n_events"]))
    for key, tag, fmt in (("r_all", "RAll", "{:+.3f}"),
                          ("r_max16", "RMax", "{:+.3f}"),
                          ("r_extreme", "RExt", "{:+.3f}"),
                          ("r_pair", "RPair", "{:+.3f}"),
                          ("flips", "Flips", "{:.0f}"),
                          ("flips_big", "FlipsBig", "{:.0f}")):
        c = d["ci"][key]
        mac(f"CBoot{tag}Lo", fmt.format(c["lo"]))
        mac(f"CBoot{tag}Hi", fmt.format(c["hi"]))
        mac(f"CBoot{tag}", fmt.format(d["point"][key]))
    print(f"  cboot: r(max16) {d['point']['r_max16']:+.3f} "
          f"[{d['ci']['r_max16']['lo']:+.3f}, {d['ci']['r_max16']['hi']:+.3f}], "
          f"flips {d['point']['flips']} "
          f"[{d['ci']['flips']['lo']:.0f}, {d['ci']['flips']['hi']:.0f}], "
          f"pair-level r {d['point']['r_pair']:+.3f}")


def valfit():
    """The same knob fitted on a validation window that precedes the test set.

    Everything else in the paper fits on one index-parity half of the test split.
    That is the right control for a claim about a comparison between arms and the
    wrong one for a claim about deployment, so this refits on events both models
    held out from training and which end before the test period begins.
    """
    f = RES / "sevir_val_fitted.json"
    if not f.exists():
        print("  valfit: not available, skipping")
        return
    d = json.loads(f.read_text())
    mac("ValN", str(d["n_val_events"]))
    for fit, tag in (("qmap_test", "Test"), ("qmap_val", "Val")):
        h, s = d["headline"][fit], d["stats"][fit]
        mac(f"ValHead{tag}", f"{100 * h['cal']:+.1f}\\%")
        mac(f"ValErased{tag}",
            f"{100 * h['erased']:.0f}\\%" if h["erased"] is not None else "n/a")
        mac(f"ValR{tag}", f"{s['pearson']:+.3f}")
        mac(f"ValFlips{tag}", str(s["flips"]))
    mac("ValHeadRaw", f"{100 * d['headline']['qmap_test']['raw']:+.1f}\\%")
    mac("ValCells", str(d["stats"]["qmap_val"]["n"]))

    pretty = {"persistence": "persistence", "pysteps": "pysteps",
              "earthformer": "EarthFormer", "cascast_det": "CasCast backbone",
              "cascast_cascade": "CasCast cascade",
              "cascast_cascade_cfg1": "CasCast cascade (no guidance)"}
    body, gains = [], []
    for a in d["arms"]:
        c = d["cells"][a]["max16|219"]
        u, qt = canon(a, "max16", 219) or (c["identity"]["csi"],
                                           c["qmap_test"]["csi"])
        gt_ = (qt - u) / u if u > 0 else float("nan")
        gv = (c["qmap_val"]["csi"] - u) / u if u > 0 else float("nan")
        gains.append((a, gt_, gv, c["identity"]["bias"]))
        body.append(f"{pretty.get(a, a)} & {c['identity']['bias']:.3f} & "
                    f"{u:.4f} & {qt:.4f} & "
                    f"{c['qmap_val']['csi']:.4f} & {pct(gt_)} & {pct(gv)} \\\\")
    (OUT / "tab_valfit.tex").write_text("\n".join(body) + "\n")
    # how the val-fitted knob compares with the test-half fit, arm by arm, among
    # the arms the knob actually pays for. Reported as a ratio in both
    # directions rather than as a shortfall, because it is not always a
    # shortfall.
    pay = [g for g in gains if g[1] > 0.02]
    if pay:
        lo = min(pay, key=lambda x: x[2] / x[1])
        hi = max(pay, key=lambda x: x[2] / x[1])
        mac("ValRatioLo", f"{100 * lo[2] / lo[1]:.0f}\\%")
        mac("ValRatioHi", f"{100 * hi[2] / hi[1]:.0f}\\%")
        mac("ValRatioLoArm", pretty.get(lo[0], lo[0]))
        mac("ValRatioHiArm", pretty.get(hi[0], hi[0]))
        mac("ValNPay", str(len(pay)))
    # The arms the knob leaves alone, and the bound has to hold under BOTH fits
    # because that is what the sentence quoting it claims. Selected by name
    # rather than by pooled bias: the quantile map is fitted on the raw,
    # unpooled marginal, so pooled bias near one does not imply the map does
    # nothing (the cascade sits at pooled bias 1.037 and still moves -2.9%,
    # which is the point Section 3 makes and is reported separately below).
    flat = [g for g in gains if g[0] in ("persistence", "pysteps")]
    if flat:
        mac("ValFlatMax",
            f"{100 * max(max(abs(t), abs(v)) for _, t, v, _ in flat):.1f}\\%")
        mac("ValNFlat", str(len(flat)))
        mac("ValFlatArms", " and ".join(pretty.get(a, a) for a, *_ in flat))
    # the cascade: nearest to pooled bias 1 of any arm, yet moved by the map,
    # so it belongs in neither bucket above and is quoted on its own
    casc = next((g for g in gains if g[0] == "cascast_cascade"), None)
    if casc:
        mac("ValCascBias", f"{casc[3]:.3f}")
        mac("ValCascTest", pct(casc[1]))
        mac("ValCascVal", pct(casc[2]))
        # The comparison has to name the arm it is against. Persistence is
        # nearer to one than the cascade, so the contrast is with pysteps, and
        # the factor is emitted rather than asserted.
        oth = next((g for g in flat if g[0] == "pysteps"), None)
        if oth:
            import math
            mac("ValFlatOtherBias", f"{oth[3]:.3f}")
            mac("ValCascCloser",
                f"{abs(math.log(oth[3])) / abs(math.log(casc[3])):.1f}")
    print(f"  valfit: headline erased {macros.get('ValErasedTest')} test-fit vs "
          f"{macros.get('ValErasedVal')} val-fit; "
          f"r {macros.get('ValRTest')} vs {macros.get('ValRVal')}")



def lead():
    """The per-lead-time map of Section 9's tightening, measured.

    Two budgets. `matched` fits every map from the same calibration events, so a
    per-lead map sees 1/12 the pixels; `equal` gives each per-lead map as many
    pixels as the global one. The global column is identical in both by
    construction, which is what makes the per-lead columns comparable.
    """
    LR = RES / "leadtime"
    CASC = "cascade_tiled_s0"

    def cell(name, budget, pooling="max16", thr=219):
        d = json.loads((LR / f"{name}_{budget}.json").read_text())
        row = next(r for r in d["table"]
                   if r["pooling"] == pooling and r["threshold"] == thr)
        return row, d

    def gain(r, k):
        return (r[f"csi_{k}"] - r["csi_identity"]) / r["csi_identity"]

    det_m, dm = cell("cascast_det", "matched")
    det_e, _ = cell("cascast_det", "equal")
    mac("LeadNLead", str(dm["n_lead"]))
    mac("LeadGainG", pct(gain(det_m, "qmap_global")))
    mac("LeadGainLm", pct(gain(det_m, "qmap_lead")))
    mac("LeadGainLe", pct(gain(det_e, "qmap_lead")))

    prof = [r for r in dm["per_lead"] if r["threshold"] == 219]
    mac("LeadBiasFirst", f"{prof[0]['bias_identity']:.3f}")
    mac("LeadBiasLast", f"{prof[-1]['bias_identity']:.3f}")
    mac("LeadBiasFold", f"{prof[0]['bias_identity']/prof[-1]['bias_identity']:.0f}")

    for tag, budget in (("m", "matched"), ("e", "equal")):
        a, _ = cell("cascast_det", budget)
        b, _ = cell(CASC, budget)
        raw = (b["csi_identity"] - a["csi_identity"]) / a["csi_identity"]
        cg = (b["csi_qmap_global"] - a["csi_qmap_global"]) / a["csi_qmap_global"]
        cl = (b["csi_qmap_lead"] - a["csi_qmap_lead"]) / a["csi_qmap_lead"]
        if tag == "m":
            mac("LeadRaw", pct(raw))
            mac("LeadErasedG", f"{100 * (1 - cg / raw):.1f}\\%")
            mac("LeadBiasCasc", f"{b['bias_identity']:.3f}")
            mac("LeadBiasDet", f"{a['bias_identity']:.3f}")
        mac(f"LeadErasedL{tag}", f"{100 * (1 - cl / raw):.1f}\\%")

    # a second cascade sample, to show the direction is not a sampling accident
    ALT = "cascade_indep_s0"
    for tag, budget in (("m", "matched"), ("e", "equal")):
        a, _ = cell("cascast_det", budget)
        b, _ = cell(ALT, budget)
        raw = (b["csi_identity"] - a["csi_identity"]) / a["csi_identity"]
        cg = (b["csi_qmap_global"] - a["csi_qmap_global"]) / a["csi_qmap_global"]
        cl = (b["csi_qmap_lead"] - a["csi_qmap_lead"]) / a["csi_qmap_lead"]
        if tag == "m":
            mac("LeadErasedGb", f"{100 * (1 - cg / raw):.1f}\\%")
            mac("LeadRawb", pct(raw))
        mac(f"LeadErasedL{tag}b", f"{100 * (1 - cl / raw):.1f}\\%")

    # the profile table: bias and CSI by lead under each knob
    body = []
    for r in prof:
        body.append(f"{r['lead'] + 1} & {r['bias_identity']:.3f} & "
                    f"{r['bias_qmap_global']:.3f} & {r['bias_qmap_lead']:.3f} & "
                    f"{r['csi_identity']:.4f} & {r['csi_qmap_global']:.4f} & "
                    f"{r['csi_qmap_lead']:.4f} \\\\")
    (OUT / "tab_lead.tex").write_text("\n".join(body) + "\n")



def knobsevir():
    """Knob family and selection rule crossed, at the SEVIR headline cell.

    Naming matters here and does not match Appendix O's. ``global'' in the
    manuscript is the single quantile map held fixed across all cells. The rows
    below are the *discrete* family of shifts and gains, selected per cell:
    ``pooled-horizon'' selects one member per cell from a table summed over lead
    times, which is the manuscript's per-cell knob, and ``per-horizon'' selects
    one per (cell, lead). Neither is the global map.
    """
    KR = RES / "knobrule"
    CASC = "cascade_tiled_s0"

    def cell(name, pooling="max16", thr=219):
        d = json.loads((KR / f"{name}.json").read_text())
        return next(r for r in d["table"]
                    if r["pooling"] == pooling and r["threshold"] == thr), d

    det, dd = cell("cascast_det")
    mac("KnobDetUncal", f"{det['csi_uncal']:.4f}")
    mac("KnobDetPooledBias", f"{det['csi_global_bias']:.4f}")
    mac("KnobDetPooledCsi", f"{det['csi_global_csi']:.4f}")
    mac("KnobDetPerhBias", f"{det['csi_perlead_bias']:.4f}")
    mac("KnobDetPerhCsi", f"{det['csi_perlead_csi']:.4f}")
    mac("KnobDetKnob", det["knob_global_bias"].replace("=", "$=$"))

    # where selecting on validation CSI loses to selecting on bias, out of sample
    worst = None
    for r in dd["table"]:
        gap = r["csi_global_csi"] - r["csi_global_bias"]
        if worst is None or gap < worst[0]:
            worst = (gap, r)
    _, w = worst
    mac("KnobCsiLoseTau", str(w["threshold"]))
    mac("KnobCsiLosePool", POOL_TEX[w["pooling"]])
    mac("KnobCsiLoseBias", f"{w['csi_global_bias']:.4f}")
    mac("KnobCsiLoseCsi", f"{w['csi_global_csi']:.4f}")

    if not (KR / f"{CASC}.json").exists():
        print("  knobsevir: cascade arm pending, det-only macros written")
        return
    casc, _ = cell(CASC)
    raw = (casc["csi_uncal"] - det["csi_uncal"]) / det["csi_uncal"]
    mac("KnobRaw", pct(raw))
    for tag, key in (("Pooled", "global_bias"), ("PooledCsi", "global_csi"),
                     ("Perh", "perlead_bias"), ("PerhCsi", "perlead_csi")):
        resid = (casc[f"csi_{key}"] - det[f"csi_{key}"]) / det[f"csi_{key}"]
        mac(f"KnobResid{tag}", pct(resid))
        mac(f"KnobErased{tag}", f"{100 * (1 - resid / raw):.1f}\\%")
    mac("KnobCascPerh", f"{casc['csi_perlead_csi']:.4f}")
    mac("KnobCascPooled", f"{casc['csi_global_bias']:.4f}")

    ALT = "cascade_indep_s0"
    if (KR / f"{ALT}.json").exists():
        alt, _ = cell(ALT)
        rawb = (alt["csi_uncal"] - det["csi_uncal"]) / det["csi_uncal"]
        mac("KnobRawb", pct(rawb))
        for tag, key in (("Pooled", "global_bias"),
                         ("PerhCsi", "perlead_csi")):
            r = (alt[f"csi_{key}"] - det[f"csi_{key}"]) / det[f"csi_{key}"]
            mac(f"KnobResid{tag}b", pct(r))
            mac(f"KnobErased{tag}b", f"{100 * (1 - r / rawb):.1f}\\%")

    body = []
    LAB = [("pooled-horizon", "bias", "global_bias"),
           ("pooled-horizon", "val-CSI", "global_csi"),
           ("per-horizon", "bias", "perlead_bias"),
           ("per-horizon", "val-CSI", "perlead_csi")]
    for fam, rule, key in LAB:
        resid = (casc[f"csi_{key}"] - det[f"csi_{key}"]) / det[f"csi_{key}"]
        body.append(f"{fam} & {rule} & {det['csi_' + key]:.4f} & "
                    f"{casc['csi_' + key]:.4f} & {pct(resid)} & "
                    f"{100 * (1 - resid / raw):.1f}\\% \\\\")
    (OUT / "tab_knobsevir.tex").write_text("\n".join(body) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    for fn in (e1, e6, dose, e8, repro, contrasts, cboot, valfit, valfitboot,
               ets, crowd, crowdpool, segpool, radar, atmos, atmos_allscene,
               rates, boot,
               lead, knobsevir):
        try:
            fn()
        except Exception as exc:                       # keep the rest usable
            print(f"  !! {fn.__name__} failed: {exc}")
    # Any macro the manuscript references but no result file supports yet gets a
    # loud placeholder rather than a compile error, so a half-finished run is
    # visibly half-finished instead of silently missing a number.
    main = (OUT / "main.tex")
    stubbed = []
    if main.exists():
        used = set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", main.read_text()))
        latex_builtin = {"Delta", "Pi", "Sigma", "Omega", "Lambda", "Gamma"}
        for u in sorted(used - set(macros) - latex_builtin):
            macros[u] = r"\textbf{??}"
            stubbed.append(u)
    for tab in ("tab_repro", "tab_rarity", "tab_pooling", "tab_dose",
                "tab_crowd", "tab_radar", "tab_boot",
                "tab_crowdpool", "tab_seg", "tab_contrasts", "tab_samearch",
                "tab_atmos", "tab_valfit"):
        f = OUT / f"{tab}.tex"
        if not f.exists():
            f.write_text("\\multicolumn{1}{l}{\\textbf{?? pending}} \\\\\n")
            stubbed.append(tab)

    lines = ["% generated by paper/make_numbers.py. Do not edit."]
    for k, v in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'numbers.tex'} with {len(macros)} macros")
    if stubbed:
        print(f"PLACEHOLDERS still outstanding: {', '.join(stubbed)}")


if __name__ == "__main__":
    main()
