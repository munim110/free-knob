"""Pairwise contrasts for the AIFS / IFS precipitation audit (output of aifs_precip_audit.py).

As wb2_precip_report.py: for every arm pair, lead, pooling and threshold, the raw margin, the
margin after each arm's own calibrated threshold, the calibration share (raw - cal) / raw, and
sign reversals, with 95% intervals from a bootstrap over evaluation days in blocks of 4 (8
calendar days), thresholds held fixed. Margins are given for CSI (the paper's score) and for
the Peirce skill score (the score ECMWF reports for AIFS vs IFS, arXiv 2509.18994, Fig. 3).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/revision/aifs")
ORDER = ["ifs_hres", "aifs_single", "ifs_ens_m1", "aifs_ens_m1"]
LABEL = {"ifs_hres": "IFS HRES", "aifs_single": "AIFS Single", "ifs_ens_m1": "IFS ENS m1",
         "aifs_ens_m1": "AIFS ENS m1"}
PUBLISHED = {("ifs_hres", "aifs_single")}
B, BLOCK, SEED = 2000, 4, 0


def csi(c):  # c[..., 3] = hits, false alarms, misses
    return c[..., 0] / np.maximum(c.sum(-1), 1e-12)


def pss(c, n):  # n: total weight, broadcast over c[..., 0]
    h, f, m = c[..., 0], c[..., 1], c[..., 2]
    return h / np.maximum(h + m, 1e-12) - f / np.maximum(n - h - m, 1e-12)


def main():
    M = {m: json.loads((D / f"aifs_{m}.json").read_text()) for m in ORDER if (D / f"aifs_{m}.json").exists()}
    Z = {m: np.load(D / f"aifs_{m}.npz") for m in M}
    C = {m: Z[m]["counts"] for m in M}
    wtot = Z[next(iter(M))]["wtot"]
    names = list(M)
    meta = M[names[0]]
    leads, pools, thrs = meta["leads_h"], meta["poolings"], meta["thresholds_mm"]
    n_eval = C[names[0]].shape[0]
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n_eval / BLOCK))
    starts = rng.integers(0, n_eval - BLOCK + 1, size=(B, nb))
    idx = (starts[..., None] + np.arange(BLOCK)).reshape(B, -1)[:, :n_eval]
    Nw = wtot[None, :, None, None] * n_eval          # (1, P, 1, 1) -> broadcasts over (L, P, T, 2)
    score = {"csi": (lambda c: csi(c), lambda c: csi(c)),
             "pss": (lambda c: pss(c, Nw), lambda c: pss(c, Nw[None]))}
    point = {s: {m: f(C[m].sum(0)) for m in M} for s, (f, _) in score.items()}
    boot = {s: {m: g(C[m][idx].sum(1)) for m in M} for s, (_, g) in score.items()}

    def bias(m, lh, p, tau):
        return next(r["bias_raw"] for r in M[m]["table"] if (r["lead_h"], r["pooling"], r["threshold_mm"]) == (lh, p, tau))

    rows = []
    for a, b in itertools.combinations(names, 2):
        for j, lh in enumerate(leads):
            for pi, p in enumerate(pools):
                for ti, tau in enumerate(thrs):
                    ba, bb = bias(a, lh, p, tau), bias(b, lh, p, tau)
                    row = {"a": a, "b": b, "lead_h": lh, "pooling": p, "threshold_mm": tau,
                           "published": (a, b) in PUBLISHED, "bias_a": ba, "bias_b": bb,
                           "devgap": abs(np.log(ba)) - abs(np.log(bb)) if ba > 0 and bb > 0 else None}
                    for s in score:
                        raw = point[s][b][j, pi, ti, 0] - point[s][a][j, pi, ti, 0]
                        cal = point[s][b][j, pi, ti, 1] - point[s][a][j, pi, ti, 1]
                        braw = boot[s][b][:, j, pi, ti, 0] - boot[s][a][:, j, pi, ti, 0]
                        bcal = boot[s][b][:, j, pi, ti, 1] - boot[s][a][:, j, pi, ti, 1]
                        rci, cci = np.percentile(braw, [2.5, 97.5]), np.percentile(bcal, [2.5, 97.5])
                        row[s] = {"raw": float(raw), "cal": float(cal),
                                  "raw_ci": [float(x) for x in rci], "cal_ci": [float(x) for x in cci],
                                  "share_calibration": float((raw - cal) / raw) if abs(raw) > 1e-9 else None,
                                  "flip": bool((raw > 0) != (cal > 0)),
                                  "both_separated": bool((rci[0] > 0 or rci[1] < 0) and (cci[0] > 0 or cci[1] < 0))}
                    rows.append(row)

    L = ["# AIFS vs IFS: free-knob audit of 24-h precipitation (ECMWF open data)", "",
         f"Arms: {', '.join(LABEL[m] for m in names)}. Truth: ERA5 24-h precipitation. "
         f"Initialisations 00 UTC, {meta['dates'][0]} .. {meta['dates'][-1]}; {meta['n_calib']} calibration days, "
         f"{n_eval} evaluation days. Area-weighted counts. Calibration = each arm's own threshold matched to the "
         "observed event frequency on the calibration days, fitted after pooling. 95% intervals: bootstrap over "
         f"evaluation days in blocks of {BLOCK}, B = {B}.", "",
         "## Per arm (no pooling)", "",
         "| Arm | Lead | τ (mm) | B raw | CSI raw | CSI cal | PSS raw | PSS cal |", "|---|---|---|---|---|---|---|---|"]
    for m in names:
        for r in M[m]["table"]:
            if r["pooling"] == "none":
                L.append(f"| {LABEL[m]} | {r['lead_h']}h | {r['threshold_mm']:g} | {r['bias_raw']:.3f} | "
                         f"{r['csi_raw']:.4f} | {r['csi_cal']:.4f} | {r['pss_raw']:.4f} | {r['pss_cal']:.4f} |")
    for s in score:
        L += ["", f"## Pairwise {s.upper()} contrasts (no pooling)", "",
              "| b − a | Lead | τ | B a / b | raw [95% CI] | calibrated [95% CI] | calibration share | flip |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["pooling"] != "none":
                continue
            q = r[s]
            sh = "n/a" if q["share_calibration"] is None else f"{100 * q['share_calibration']:+.0f}%"
            L.append(f"| {LABEL[r['b']]} − {LABEL[r['a']]}{' (published)' if r['published'] else ''} | {r['lead_h']}h | "
                     f"{r['threshold_mm']:g} | {r['bias_a']:.2f} / {r['bias_b']:.2f} | "
                     f"{q['raw']:+.4f} [{q['raw_ci'][0]:+.4f}, {q['raw_ci'][1]:+.4f}] | "
                     f"{q['cal']:+.4f} [{q['cal_ci'][0]:+.4f}, {q['cal_ci'][1]:+.4f}] | {sh} | "
                     f"{'yes' + (' (both separated)' if q['both_separated'] else '') if q['flip'] else ''} |")
    L += ["", "## Summary over all contrasts (CSI)", "",
          "| subset | n | r(devgap, distortion) | ρ | flips | flips with both margins separated |",
          "|---|---|---|---|---|---|"]
    for tag, sub in [("all", rows)] + [(p, [r for r in rows if r["pooling"] == p]) for p in pools]:
        sub = [r for r in sub if r["devgap"] is not None]
        x = [r["devgap"] for r in sub]
        y = [r["csi"]["raw"] - r["csi"]["cal"] for r in sub]
        fl = [r for r in sub if r["csi"]["flip"]]
        L.append(f"| {tag} | {len(sub)} | {pearsonr(x, y).statistic:+.3f} | {spearmanr(x, y).statistic:+.3f} | "
                 f"{len(fl)} | {sum(r['csi']['both_separated'] for r in fl)} |")
    (D / "aifs_contrasts.json").write_text(json.dumps(rows, indent=1))
    (D / "AIFS_PRECIP_AUDIT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
