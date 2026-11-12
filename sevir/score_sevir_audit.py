"""Free-knob audit of a published SEVIR nowcasting system from its own released weights.

Systems:
  diffcast  h5 from infer_diffcast.py: gt, diffcast, backbone, last_input (N, 20, 128, 128);
            arms diffcast, backbone (its PhyDNet), persistence.
  prediff   h5 from infer_prediff.py: gt (N, 6, 128, 128), prediff (N, K, 6, 128, 128), last_input;
            arms prediff (each of the K samples scored separately and counts summed, as the
            released test_step does), prediff_mean (floor of the mean of the K samples, the
            paper's appendix convention; only when K > 1), persistence.

Persistence repeats the last input frame over the output frames.

Scoring follows both systems' evaluators (DiffCast's Evaluator and PreDiff's SEVIRSkillScore,
which agree): events are value >= tau on the 0-255 scale, tau in (16, 74, 133, 160, 181, 219);
counts pooled over all windows and frames; pooled CSI uses non-overlapping max pooling over
4x4 and 16x16 blocks. Average pooling (4x4, 16x16) is added for completeness. The published
tables average CSI over the six thresholds; that average is printed for the reproduction check.

Control: per (arm, pooling, tau), the forecast threshold whose exceedance count on the
calibration events is closest to the observed count, fitted on pooled block values (so it is
exactly a per-arm threshold). Calibration events are the even-indexed events, evaluation
events the odd-indexed ones; all windows of an event stay on one side (DiffCast: 5 windows per
event; PreDiff: 3 overlapping windows per event, window i belongs to event i // 3). Everything
is computed from per-event histograms of pooled forecast values split by observed outcome, so
the fit and a bootstrap over evaluation events are exact and cheap.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

THR = (16, 74, 133, 160, 181, 219)
POOLS = ("none", "max4", "max16", "avg4", "avg16")
AVG_RES = 4  # histogram bins per unit for average-pooled values

SYSTEMS = {
    "diffcast": {
        "title": "DiffCast (CVPR 2024) against its own PhyDNet backbone",
        "split": "SEVIR test events (2019-10-01 on)",
        "frames": 20,
        "published": {"diffcast": {"none": 0.2757, "max4": 0.3797, "max16": 0.5296}},  # DiffCast Table 1, SEVIR
        "pairs": [("backbone", "diffcast"), ("persistence", "diffcast"), ("persistence", "backbone")],
    },
    "prediff": {
        "title": "PreDiff (NeurIPS 2023) against persistence",
        "split": "SEVIR-LR test events (2019-06-01 on)",
        "frames": 6,
        # PreDiff Table 2, SEVIR (CSI, CSI-pool4, CSI-pool16)
        "published": {"prediff": {"none": 0.4100, "max4": 0.4624, "max16": 0.6244},
                      "persistence": {"none": 0.2613, "max4": 0.3702, "max16": 0.4690}},
        "pairs": [("persistence", "prediff"), ("persistence", "prediff_mean"), ("prediff", "prediff_mean")],
    },
}
PREDIFF_WINDOWS_PER_EVENT = 3  # 1 + (25 - 13) // 6


def pool(x, p):
    """x: (..., 128, 128) uint8 -> pooled array; values on an integer grid of nbins(p)."""
    if p == "none":
        return x.astype(np.int32)
    k = int(p[3:])
    b = x.reshape(*x.shape[:-2], 128 // k, k, 128 // k, k)
    if p.startswith("max"):
        return b.max(axis=(-3, -1)).astype(np.int32)
    return np.floor(b.mean(axis=(-3, -1), dtype=np.float64) * AVG_RES).astype(np.int32)


def nbins(p):
    return 256 * (AVG_RES if p.startswith("avg") else 1)


def scale(p):
    return AVG_RES if p.startswith("avg") else 1


def event_ids(system, f):
    if system == "diffcast":
        return f["event_id"][:]
    assert f.attrs["n_test_windows"] % PREDIFF_WINDOWS_PER_EVENT == 0
    return f["window_index"][:] // PREDIFF_WINDOWS_PER_EVENT


def forecasts(system, f, sl, arms, frames):
    """arm -> list of (n, frames, 128, 128) uint8 arrays; counts of all arrays in a list are summed."""
    fc = {"persistence": [np.repeat(f["last_input"][sl][:, None], frames, axis=1)]}
    if system == "diffcast":
        fc["diffcast"] = [f["diffcast"][sl]]
        fc["backbone"] = [f["backbone"][sl]]
    else:
        s = f["prediff"][sl]  # (n, K, frames, 128, 128)
        fc["prediff"] = [s[:, k] for k in range(s.shape[1])]
        if "prediff_mean" in arms:
            fc["prediff_mean"] = [np.floor(s.mean(axis=1, dtype=np.float64)).astype(np.uint8)]
    return {a: fc[a] for a in arms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=sorted(SYSTEMS))
    ap.add_argument("h5", nargs="+", help="one or more inference files (e.g. offsets of one stride)")
    ap.add_argument("--out-dir", default=None, help="default results/revision/<system>")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    S = SYSTEMS[args.system]
    out = Path(args.out_dir or f"results/revision/{args.system}")
    out.mkdir(parents=True, exist_ok=True)
    files = [h5py.File(p, "r") for p in args.h5]
    frames = S["frames"]
    arms = ["persistence"]
    if args.system == "diffcast":
        arms = ["diffcast", "backbone", "persistence"]
    else:
        K = {int(f.attrs["samples"]) for f in files}
        assert len(K) == 1, "all files must hold the same number of samples"
        K = K.pop()
        arms = ["prediff"] + (["prediff_mean"] if K > 1 else []) + ["persistence"]
    pairs = [(a, b) for a, b in S["pairs"] if a in arms and b in arms]

    ev = np.concatenate([event_ids(args.system, f) for f in files])
    order = {e: i for i, e in enumerate(dict.fromkeys(ev))}  # first-seen (catalogue) order
    ev_idx = np.array([order[e] for e in ev])
    n_ev = len(order)
    T = len(THR)
    # H[arm][p] : (n_ev, T, 2, nbins)  histogram of pooled forecast values for obs-negative / obs-positive blocks
    H = {a: {p: np.zeros((n_ev, T, 2, nbins(p)), dtype=np.int64) for p in POOLS} for a in arms}
    N = len(ev)
    done = 0
    for f in files:
        n_f = f["gt"].shape[0]
        for s in range(0, n_f, 50):
            sl = slice(s, min(n_f, s + 50))
            gt = f["gt"][sl]
            assert gt.shape[1] == frames
            fc = forecasts(args.system, f, sl, arms, frames)
            e_sl = ev_idx[done: done + gt.shape[0]]
            for p in POOLS:
                og = pool(gt, p)
                for a in arms:
                    for arr in fc[a]:
                        fp = pool(arr, p)
                        for w in range(og.shape[0]):
                            e = e_sl[w]
                            for ti, tau in enumerate(THR):
                                pos = og[w] >= tau * scale(p)
                                H[a][p][e, ti, 1] += np.bincount(fp[w][pos], minlength=nbins(p))
                                H[a][p][e, ti, 0] += np.bincount(fp[w][~pos], minlength=nbins(p))
            done += gt.shape[0]
            print(f"  histogrammed {done}/{N}", flush=True)

    calib = np.arange(n_ev) % 2 == 0
    evalm = ~calib

    def counts(h, t):  # h: (..., 2, nbins); forecast event = bin >= t
        hits = h[..., 1, t:].sum(-1)
        fa = h[..., 0, t:].sum(-1)
        miss = h[..., 1, :t].sum(-1)
        return hits, fa, miss

    def csi(h, fa, m):
        return h / np.maximum(h + fa + m, 1)

    res, table = {}, []
    for a in arms:
        res[a] = {}
        for p in POOLS:
            raw_t = [tau * scale(p) for tau in THR]
            hc = H[a][p][calib].sum(0)  # (T, 2, nbins)
            cal_t = []
            for ti in range(T):
                n_obs = hc[ti, 1].sum()
                exceed = np.cumsum(hc[ti].sum(0)[::-1])[::-1]  # forecast count >= t, t = 0..nbins-1
                cal_t.append(int(np.argmin(np.abs(exceed - n_obs))))
            res[a][p] = {"raw_t": raw_t, "cal_t": cal_t}
            for ti, tau in enumerate(THR):
                he = H[a][p][evalm].sum(0)[ti]
                full = H[a][p].sum(0)[ti]
                r = {"arm": a, "pooling": p, "threshold": tau, "cal_forecast_threshold": cal_t[ti] / scale(p)}
                for tag, hh in (("all", full), ("eval", he)):
                    for k, t in (("raw", raw_t[ti]), ("cal", cal_t[ti])):
                        h_, fa, m = counts(hh, t)
                        r[f"csi_{k}_{tag}"] = float(csi(h_, fa, m))
                        r[f"bias_{k}_{tag}"] = float((h_ + fa) / max(h_ + m, 1))
                table.append(r)

    attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in files[0].attrs.items()}
    lines = [f"# {S['title']}: free-knob audit", "",
             f"Windows: {N} from {n_ev} {S['split']}. "
             f"Calibration events {calib.sum()}, evaluation events {evalm.sum()}.",
             f"Inference attributes (first file): {attrs}", "",
             "## Reproduction (all events, raw, CSI averaged over the six thresholds)", "",
             "| arm | CSI | CSI-pool4 (max) | CSI-pool16 (max) | published |", "|---|---|---|---|---|"]
    for a in arms:
        m = {p: np.mean([r["csi_raw_all"] for r in table if r["arm"] == a and r["pooling"] == p])
             for p in ("none", "max4", "max16")}
        pub = S["published"].get(a)
        lines.append(f"| {a} | {m['none']:.4f} | {m['max4']:.4f} | {m['max16']:.4f} | "
                     + (f"{pub['none']:.4f} / {pub['max4']:.4f} / {pub['max16']:.4f}" if pub else "–") + " |")

    # contrasts on evaluation events, with event bootstrap
    rng = np.random.default_rng(0)
    eval_ids = np.nonzero(evalm)[0]
    bidx = rng.integers(0, len(eval_ids), size=(args.boot, len(eval_ids)))
    rows = []
    for a, b in pairs:
        for p in POOLS:
            for ti, tau in enumerate(THR):
                vals = {}
                for arm in (a, b):
                    he = H[arm][p][eval_ids][:, ti]  # (n_eval, 2, nbins)
                    for k in ("raw", "cal"):
                        t = res[arm][p][f"{k}_t"][ti]
                        h_, fa, m = counts(he, t)
                        vals[(arm, k)] = (float(csi(h_.sum(), fa.sum(), m.sum())),
                                          csi(h_[bidx].sum(1), fa[bidx].sum(1), m[bidx].sum(1)))
                raw = vals[(b, "raw")][0] - vals[(a, "raw")][0]
                cal = vals[(b, "cal")][0] - vals[(a, "cal")][0]
                braw = vals[(b, "raw")][1] - vals[(a, "raw")][1]
                bcal = vals[(b, "cal")][1] - vals[(a, "cal")][1]
                ra = next(r for r in table if (r["arm"], r["pooling"], r["threshold"]) == (a, p, tau))
                rb = next(r for r in table if (r["arm"], r["pooling"], r["threshold"]) == (b, p, tau))
                rows.append({"a": a, "b": b, "pooling": p, "threshold": tau,
                             "csi_a_raw": ra["csi_raw_eval"], "csi_b_raw": rb["csi_raw_eval"],
                             "bias_a": ra["bias_raw_eval"], "bias_b": rb["bias_raw_eval"],
                             "raw": raw, "cal": cal,
                             # relative margins only where the reference arm has a usable CSI
                             "raw_rel": raw / ra["csi_raw_eval"] if ra["csi_raw_eval"] >= 0.01 else None,
                             "cal_rel": cal / ra["csi_cal_eval"] if ra["csi_cal_eval"] >= 0.01 else None,
                             "raw_ci": np.percentile(braw, [2.5, 97.5]).tolist(),
                             "cal_ci": np.percentile(bcal, [2.5, 97.5]).tolist(),
                             "flip": bool((raw > 0) != (cal > 0))})
    rel = lambda x: "n/a" if x is None else f"{100 * x:+.1f}%"
    a0, b0 = pairs[0]
    lines += ["", f"## {b0} − {a0}, evaluation events", "",
              f"| pooling | τ | B {a0} / {b0} | raw margin (rel.) [95% CI] | calibrated margin (rel.) [95% CI] | flip |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        if (r["a"], r["b"]) == (a0, b0):
            lines.append(f"| {r['pooling']} | {r['threshold']} | {r['bias_a']:.2f} / {r['bias_b']:.2f} | "
                         f"{r['raw']:+.4f} ({rel(r['raw_rel'])}) [{r['raw_ci'][0]:+.4f}, {r['raw_ci'][1]:+.4f}] | "
                         f"{r['cal']:+.4f} ({rel(r['cal_rel'])}) [{r['cal_ci'][0]:+.4f}, {r['cal_ci'][1]:+.4f}] | "
                         f"{'yes' if r['flip'] else ''} |")
    lines += ["", "## Threshold-averaged margins (the papers' reporting convention), evaluation events", "",
              "| pair | pooling | raw: CSI a → b (rel.) | calibrated: CSI a → b (rel.) |", "|---|---|---|---|"]
    for a, b in pairs:
        for p in POOLS:
            def avg(arm, k):
                return np.mean([r[f"csi_{k}_eval"] for r in table if r["arm"] == arm and r["pooling"] == p])
            ra_, rb_, ca_, cb_ = avg(a, "raw"), avg(b, "raw"), avg(a, "cal"), avg(b, "cal")
            lines.append(f"| {b} − {a} | {p} | {ra_:.4f} → {rb_:.4f} ({100 * (rb_ / ra_ - 1):+.1f}%) | "
                         f"{ca_:.4f} → {cb_:.4f} ({100 * (cb_ / ca_ - 1):+.1f}%) |")
    (out / f"{args.system}_table.json").write_text(json.dumps({"table": table, "contrasts": rows}, indent=1))
    (out / f"{args.system.upper()}_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
