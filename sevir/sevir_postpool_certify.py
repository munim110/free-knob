"""Paired event-bootstrap intervals for the 450 SEVIR contrasts under the post-pooling control.

sevir_pool_then_calibrate.py fixed each arm's control on the calibration events (even
indices): the paper's pixel map for 'none' pooling, and the exact per-arm threshold on pooled
values ('post_exact_threshold') otherwise. Both are monotone, so each is a per-arm threshold.
Here the control is held fixed and one more pass over the evaluation events (odd indices)
records per-event contingency counts, raw and calibrated, for every arm, pooling and
threshold. Summed over events they must equal the stored counts exactly (checked).

The bootstrap resamples evaluation events with replacement (B = 2000, seed 0), the same
resample for every arm, so each contrast's raw and calibrated margins are paired. Intervals
are percentile 95%, as in nwp/wb2_precip_report.py (there with day blocks; SEVIR events are
separate storms, so single events are resampled).

Reads the six rerun prediction HDF5s, the SEVIR test HDF5 and results/revision/postpool/*.json.
Writes results/revision/postpool/sevir_postpool_certified.json and SEVIR_POSTPOOL_CERTIFIED.md.
"""
import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

from sevir_pool_then_calibrate import (POOLINGS, THRESHOLDS, GT_BLOCK, SUB, apply_map,
                                       fit_pre, pool)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import paths as P  # noqa: E402
RES = P.RESULTS / "revision" / "postpool"
ARMS = ["persistence", "pysteps", "earthformer", "cascast_det",
        "cascast_cascade", "cascast_cascade_cfg1"]
PUBLISHED = {("cascast_det", "cascast_cascade"), ("earthformer", "cascast_cascade"),
             ("persistence", "earthformer")}
B, SEED = 2000, 0


def csi(c):
    h, f, m = c[..., 0], c[..., 1], c[..., 2]
    d = h + f + m
    return np.where(d > 0, h / np.maximum(d, 1), 0.0)


def collect(pred_dir, gt_path):
    meta = {a: json.loads((RES / f"sevir_postpool_{a}.json").read_text()) for a in ARMS}
    thr = {a: {(r["pooling"], r["threshold"]): r.get("post_exact_threshold")
               for r in meta[a]["table"]} for a in ARMS}
    fg = h5py.File(gt_path, "r")
    G = fg["OUT_vil"]
    N = G.shape[0]
    fps = {a: h5py.File(pred_dir / f"sevir_pred_{a}.h5", "r") for a in ARMS}
    P = {a: f["pred_vil"] for a, f in fps.items()}
    cal, ev = np.arange(0, N, 2), np.arange(1, N, 2)
    t0 = time.time()
    pre = {a: fit_pre(P[a], G, cal) for a in ARMS}
    print(f"pixel maps refitted ({time.time() - t0:.0f}s)", flush=True)
    # C[event, arm, pooling, threshold, kind(raw, cal), (h, f, m)]
    C = np.zeros((len(ev), len(ARMS), len(POOLINGS), len(THRESHOLDS), 2, 3), np.int64)
    for b0 in range(0, N, GT_BLOCK):
        b1 = min(N, b0 + GT_BLOCK)
        gblk = G[b0:b1]
        for s0 in range(b0, b1, SUB):
            s1 = min(b1, s0 + SUB)
            pb = {a: P[a][s0:s1] for a in ARMS}
            for i in range(s0, s1):
                if i % 2 == 0:
                    continue
                e = i // 2
                g = gblk[i - b0:i - b0 + 1].astype(np.float32)
                gp = {m: pool(g, m) for m in POOLINGS}
                for ai, a in enumerate(ARMS):
                    p = pb[a][i - s0:i - s0 + 1]
                    q = apply_map(p, *pre[a])
                    for mi, m in enumerate(POOLINGS):
                        pr = pool(p, m)
                        pc = pool(q, m) if m == "none" else None
                        for ti, t in enumerate(THRESHOLDS):
                            gh = gp[m] >= t
                            for ki, ph in ((0, pr >= t),
                                           (1, pc >= t if m == "none" else pr >= thr[a][(m, t)])):
                                C[e, ai, mi, ti, ki] = (np.count_nonzero(ph & gh),
                                                        np.count_nonzero(ph & ~gh),
                                                        np.count_nonzero(~ph & gh))
        print(f"  {b1}/{N} events ({time.time() - t0:.0f}s)", flush=True)
    for f in fps.values():
        f.close()
    fg.close()
    # exact check against the stored aggregate counts
    worst = 0
    for ai, a in enumerate(ARMS):
        rows = {(r["pooling"], r["threshold"]): r for r in meta[a]["table"]}
        for mi, m in enumerate(POOLINGS):
            for ti, t in enumerate(THRESHOLDS):
                for ki, k in ((0, "counts_raw"), (1, "counts_post_exact")):
                    d = np.abs(C[:, ai, mi, ti, ki].sum(0) - np.array(rows[(m, t)][k])).max()
                    worst = max(worst, int(d))
    print(f"max |summed per-event counts - stored counts| = {worst}", flush=True)
    assert worst == 0, "per-event counts do not reproduce the stored post-pool counts"
    return C


def certify(C):
    tot = C.sum(0)                        # (A, P, T, 2, 3)
    point = csi(tot)                      # (A, P, T, 2)
    bias = (tot[..., 0] + tot[..., 1]) / np.maximum(tot[..., 0] + tot[..., 2], 1)
    rng = np.random.default_rng(SEED)
    n = C.shape[0]
    boot = np.empty((B,) + point.shape)
    for r in range(B):
        boot[r] = csi(C[rng.integers(0, n, n)].sum(0))
    rows = []
    for i, a in enumerate(ARMS):
        for j in range(i + 1, len(ARMS)):
            b = ARMS[j]
            for mi, m in enumerate(POOLINGS):
                for ti, t in enumerate(THRESHOLDS):
                    raw = point[j, mi, ti, 0] - point[i, mi, ti, 0]
                    cal = point[j, mi, ti, 1] - point[i, mi, ti, 1]
                    braw = boot[:, j, mi, ti, 0] - boot[:, i, mi, ti, 0]
                    bcal = boot[:, j, mi, ti, 1] - boot[:, i, mi, ti, 1]
                    rci = np.percentile(braw, [2.5, 97.5])
                    cci = np.percentile(bcal, [2.5, 97.5])
                    ba, bb = bias[i, mi, ti, 0], bias[j, mi, ti, 0]
                    gap = (abs(math.log(ba)) - abs(math.log(bb))) if ba > 0 and bb > 0 else None
                    rows.append({
                        "a": a, "b": b, "pooling": m, "threshold": t,
                        "published": (a, b) in PUBLISHED,
                        "raw": float(raw), "cal": float(cal), "distortion": float(raw - cal),
                        "calibration_gap": gap, "bias_a": float(ba), "bias_b": float(bb),
                        "raw_ci": [float(x) for x in rci], "cal_ci": [float(x) for x in cci],
                        "flip": bool((raw > 0) != (cal > 0)),
                        "raw_separated": bool(rci[0] > 0 or rci[1] < 0),
                        "cal_separated": bool(cci[0] > 0 or cci[1] < 0),
                    })
    return rows


def report(rows, n_eval):
    for r in rows:
        r["both_separated"] = r["raw_separated"] and r["cal_separated"]
    ok = [r for r in rows if r["calibration_gap"] is not None]
    rr = pearsonr([r["calibration_gap"] for r in ok], [r["distortion"] for r in ok])[0]
    rho = spearmanr([r["calibration_gap"] for r in ok], [r["distortion"] for r in ok])[0]
    flips = [r for r in rows if r["flip"]]
    sep = [r for r in flips if r["both_separated"]]
    L = ["# SEVIR, 450 contrasts under the post-pooling control: paired event-bootstrap intervals", "",
         f"Evaluation events {n_eval}; control fixed on the calibration events; B = {B} event resamples "
         f"(seed {SEED}), the same resample for every arm; percentile 95% intervals.", "",
         "| subset | n | r(gap, distortion) | ρ | flips | flips with both margins separated |",
         "|---|---|---|---|---|---|",
         f"| all | {len(rows)} | {rr:.3f} | {rho:.3f} | {len(flips)} | {len(sep)} |", "",
         "A margin is separated when its 95% interval excludes zero.", "",
         "## Flips", "",
         "| b − a | pooling | τ | B a / b | raw [95% CI] | calibrated [95% CI] | both separated |",
         "|---|---|---|---|---|---|---|"]
    for r in flips:
        L.append(f"| {r['b']} − {r['a']}{' (published)' if r['published'] else ''} | {r['pooling']} | "
                 f"{r['threshold']} | {r['bias_a']:.2f} / {r['bias_b']:.2f} | "
                 f"{r['raw']:+.4f} [{r['raw_ci'][0]:+.4f}, {r['raw_ci'][1]:+.4f}] | "
                 f"{r['cal']:+.4f} [{r['cal_ci'][0]:+.4f}, {r['cal_ci'][1]:+.4f}] | "
                 f"{'yes' if r['both_separated'] else ''} |")
    L += ["", "## Published contrasts", "",
          "| b − a | pooling | τ | raw [95% CI] | calibrated [95% CI] |", "|---|---|---|---|---|"]
    for r in rows:
        if r["published"]:
            L.append(f"| {r['b']} − {r['a']} | {r['pooling']} | {r['threshold']} | "
                     f"{r['raw']:+.4f} [{r['raw_ci'][0]:+.4f}, {r['raw_ci'][1]:+.4f}] | "
                     f"{r['cal']:+.4f} [{r['cal_ci'][0]:+.4f}, {r['cal_ci'][1]:+.4f}] |")
    return "\n".join(L) + "\n", {"r": rr, "rho": rho, "n_flips": len(flips), "n_both_separated": len(sep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=P.RESULTS)
    ap.add_argument("--gt", type=Path, default=P.SEVIR_H5)
    ap.add_argument("--counts-cache", type=Path, default=RES / "sevir_postpool_event_counts.npy")
    args = ap.parse_args()
    if args.counts_cache.exists():
        C = np.load(args.counts_cache)
    else:
        C = collect(args.pred_dir, args.gt)
        np.save(args.counts_cache, C)
    rows = certify(C)
    md, summ = report(rows, C.shape[0])
    (RES / "SEVIR_POSTPOOL_CERTIFIED.md").write_text(md, encoding="utf-8")
    (RES / "sevir_postpool_certified.json").write_text(
        json.dumps({"summary": summ, "B": B, "seed": SEED, "n_eval": int(C.shape[0]), "rows": rows}, indent=1))
    print(md[:1500])


if __name__ == "__main__":
    main()
