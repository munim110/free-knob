"""Pairwise contrasts for the WeatherBench 2 precipitation audit.

For every model pair, lead, pooling and threshold: raw CSI margin, margin
after each model's own calibrated threshold, the share of the raw margin that
is calibration ((raw - cal) / raw), and sign reversals. 95% intervals come
from a bootstrap over evaluation days in blocks of 4 (8 calendar days), with
the fitted thresholds held fixed.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/revision/wb2")
ORDER = ["hres", "ens_mean", "graphcast", "graphcast_oper", "fuxi"]
LABEL = {"hres": "IFS HRES", "ens_mean": "IFS ENS mean", "graphcast": "GraphCast (ERA5 init)",
         "graphcast_oper": "GraphCast (HRES init)", "fuxi": "FuXi"}
B, BLOCK, SEED = 2000, 4, 0


def csi(c):  # c[..., 3] = hits, false alarms, misses
    return c[..., 0] / np.maximum(c.sum(-1), 1e-12)


def main():
    M = {m: json.loads((D / f"wb2_{m}.json").read_text()) for m in ORDER if (D / f"wb2_{m}.json").exists()}
    C = {m: np.load(D / f"wb2_{m}.npz")["counts"] for m in M}
    names = list(M)
    meta = M[names[0]]
    leads, pools, thrs = meta["leads_h"], meta["poolings"], meta["thresholds_mm"]
    n_eval = C[names[0]].shape[0]
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(n_eval / BLOCK))
    starts = rng.integers(0, n_eval - BLOCK + 1, size=(B, nb))
    idx = (starts[..., None] + np.arange(BLOCK)).reshape(B, -1)[:, :n_eval]
    boot = {m: csi(C[m][idx].sum(1)) for m in M}  # (B, L, P, T, 2)
    point = {m: csi(C[m].sum(0)) for m in M}       # (L, P, T, 2)

    lines = ["# WeatherBench 2: free-knob audit of 24-h precipitation", "",
             f"Models: {', '.join(LABEL[m] for m in names)}. Truth: ERA5 24-h precipitation. "
             f"2020, 00 UTC initialisations; {meta['n_calib']} calibration days, {n_eval} evaluation days. "
             "Area-weighted counts. Calibration = each model's own threshold matched to the observed "
             "event frequency on the calibration days, fitted after pooling.", "",
             "## Per model (no pooling)", "",
             "| Model | Lead | τ (mm) | CSI raw | B raw | CSI cal | knob gain |", "|---|---|---|---|---|---|---|"]
    for m in names:
        for r in M[m]["table"]:
            if r["pooling"] == "none":
                lines.append(f"| {LABEL[m]} | {r['lead_h']}h | {r['threshold_mm']:.0f} | {r['csi_raw']:.4f} | "
                             f"{r['bias_raw']:.3f} | {r['csi_cal']:.4f} | {r['csi_cal'] - r['csi_raw']:+.4f} |")
    rows = []
    for a, b in itertools.combinations(names, 2):
        for j, lh in enumerate(leads):
            for pi, p in enumerate(pools):
                for ti, tau in enumerate(thrs):
                    raw = point[b][j, pi, ti, 0] - point[a][j, pi, ti, 0]
                    cal = point[b][j, pi, ti, 1] - point[a][j, pi, ti, 1]
                    braw = boot[b][:, j, pi, ti, 0] - boot[a][:, j, pi, ti, 0]
                    bcal = boot[b][:, j, pi, ti, 1] - boot[a][:, j, pi, ti, 1]
                    ba = next(r["bias_raw"] for r in M[a]["table"] if (r["lead_h"], r["pooling"], r["threshold_mm"]) == (lh, p, tau))
                    bb = next(r["bias_raw"] for r in M[b]["table"] if (r["lead_h"], r["pooling"], r["threshold_mm"]) == (lh, p, tau))
                    rows.append({"a": a, "b": b, "lead_h": lh, "pooling": p, "threshold_mm": tau,
                                 "bias_a": ba, "bias_b": bb, "raw": float(raw), "cal": float(cal),
                                 "raw_ci": [float(x) for x in np.percentile(braw, [2.5, 97.5])],
                                 "cal_ci": [float(x) for x in np.percentile(bcal, [2.5, 97.5])],
                                 "share_calibration": float((raw - cal) / raw) if abs(raw) > 1e-9 else None,
                                 "flip": bool((raw > 0) != (cal > 0)),
                                 "devgap": abs(np.log(ba)) - abs(np.log(bb))})
    sep = lambda ci: ci[0] > 0 or ci[1] < 0
    lines += ["", "## Pairwise contrasts (no pooling)", "",
              "| b − a | Lead | τ | B a / b | raw [95% CI] | calibrated [95% CI] | calibration share | flip |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["pooling"] != "none":
            continue
        sh = "n/a" if r["share_calibration"] is None else f"{100 * r['share_calibration']:+.0f}%"
        lines.append(f"| {LABEL[r['b']]} − {LABEL[r['a']]} | {r['lead_h']}h | {r['threshold_mm']:.0f} | "
                     f"{r['bias_a']:.2f} / {r['bias_b']:.2f} | {r['raw']:+.4f} [{r['raw_ci'][0]:+.4f}, {r['raw_ci'][1]:+.4f}] | "
                     f"{r['cal']:+.4f} [{r['cal_ci'][0]:+.4f}, {r['cal_ci'][1]:+.4f}] | {sh} | {'yes' if r['flip'] else ''} |")
    from scipy.stats import pearsonr, spearmanr
    lines += ["", "## Summary over all contrasts", "",
              "| subset | n | r(devgap, distortion) | ρ | flips | flips with both margins separated |", "|---|---|---|---|---|---|"]
    for tag, sub in [("all", rows)] + [(p, [r for r in rows if r["pooling"] == p]) for p in pools] + \
                    [(f"τ = {t:.0f} mm", [r for r in rows if r["threshold_mm"] == t]) for t in thrs]:
        x = [r["devgap"] for r in sub]
        y = [r["raw"] - r["cal"] for r in sub]
        fl = [r for r in sub if r["flip"]]
        lines.append(f"| {tag} | {len(sub)} | {pearsonr(x, y).statistic:+.3f} | {spearmanr(x, y).statistic:+.3f} | "
                     f"{len(fl)} | {sum(sep(r['raw_ci']) and sep(r['cal_ci']) for r in fl)} |")
    (D / "wb2_contrasts.json").write_text(json.dumps(rows, indent=1))
    (D / "WB2_PRECIP_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
