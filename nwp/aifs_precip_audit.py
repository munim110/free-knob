"""Free-knob audit of 24-h precipitation from operational AIFS and IFS (ECMWF open data).

Same event, weighting and control as wb2_precip_audit.py, on the arrays written by
aifs_fetch_tp.py (no inference): 24-h precipitation >= tau mm in a block (pixel, or 1 deg =
4x4 block under max or mean pooling), counts weighted by cos(latitude), truth ERA5.

Control: per (arm, lead, pooling, tau), the forecast threshold whose area-weighted exceedance on
the calibration days equals the observed one, fitted on pooled block values. Calibration days
are the even-indexed initialisation dates, evaluation days the odd-indexed ones. Dates with a
missing file in any arm are dropped from every arm, so all contrasts are paired.

Writes <out>/aifs_<arm>.json and aifs_<arm>.npz in the wb2 format (counts[day, lead, pooling,
tau, raw/cal, (h, f, m)]), plus wtot[day, pooling], the total weight per evaluation day, so
correct negatives (and the Peirce skill score ECMWF reports) can be recovered.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

POOLINGS = ("none", "max4", "avg4")
FLOOR = 0.2  # mm; calibration values below this are not stored (flagged if needed)
ARMS = ["aifs_single", "ifs_hres", "aifs_ens_m1", "ifs_ens_m1"]


def pool(x):
    """x: (720, 1440) mm -> dict pooling -> 2-D array."""
    b = x.reshape(180, 4, 360, 4)
    return {"none": x, "max4": b.max(axis=(1, 3)), "avg4": b.mean(axis=(1, 3))}


def lat_weights():
    lat = np.linspace(90, -90, 721)[:-1]   # drop the -90 row (weight 0) -> 720 rows
    w = np.cos(np.deg2rad(lat.astype(np.float64)))
    return {"none": w, "max4": w.reshape(180, 4).mean(1), "avg4": w.reshape(180, 4).mean(1)}


def field(A, i, j):
    return np.asarray(A[i, j, :720, :], dtype=np.float32)


def run_arm(arm, src, out, thresholds, cal_idx, ev_idx):
    meta = json.loads((src / "meta.json").read_text())
    leads = meta["leads_h"]
    A = np.load(src / f"{arm}.npy", mmap_mode="r")
    E = np.load(src / "era5.npy", mmap_mode="r")
    W = lat_weights()
    thr = np.array(thresholds, dtype=np.float32)
    L, P, T = len(leads), len(POOLINGS), len(thr)
    cal_vals = {(j, p): [] for j in range(L) for p in POOLINGS}
    cal_rows = {(j, p): [] for j in range(L) for p in POOLINGS}
    obs_w = np.zeros((L, P, T))
    fc_w_raw = np.zeros((L, P, T))
    t0 = time.time()
    for n, i in enumerate(cal_idx):
        for j in range(L):
            F, O = pool(field(A, i, j)), pool(field(E, i, j))
            for pi, p in enumerate(POOLINGS):
                f, o, w = F[p], O[p], W[p][:, None]
                m = f >= FLOOR
                cal_vals[(j, p)].append(f[m])
                cal_rows[(j, p)].append(np.nonzero(m)[0].astype(np.int16))
                for ti, tau in enumerate(thr):
                    obs_w[j, pi, ti] += (w * (o >= tau)).sum()
                    fc_w_raw[j, pi, ti] += (w * (f >= tau)).sum()
        if n % 25 == 0:
            print(f"{arm} calib {n}/{len(cal_idx)} {time.time() - t0:.0f}s", flush=True)
    thr_cal = np.zeros((L, P, T), dtype=np.float32)
    flag = np.zeros((L, P, T), dtype=bool)
    for j in range(L):
        for pi, p in enumerate(POOLINGS):
            v = np.concatenate(cal_vals.pop((j, p)))
            r = np.concatenate(cal_rows.pop((j, p)))
            order = np.argsort(-v, kind="stable")
            v, cw = v[order], np.cumsum(W[p][r[order]])
            for ti in range(T):
                k = int(np.searchsorted(cw, obs_w[j, pi, ti]))
                if k >= len(v):
                    flag[j, pi, ti], thr_cal[j, pi, ti] = True, FLOOR
                else:
                    thr_cal[j, pi, ti] = v[k]
            del v, cw, r, order

    counts = np.zeros((len(ev_idx), L, P, T, 2, 3))
    wtot = np.array([W[p].sum() * (1440 if p == "none" else 360) for p in POOLINGS])
    t0 = time.time()
    for n, i in enumerate(ev_idx):
        for j in range(L):
            F, O = pool(field(A, i, j)), pool(field(E, i, j))
            for pi, p in enumerate(POOLINGS):
                f, o, w = F[p], O[p], W[p][:, None]
                for ti, tau in enumerate(thr):
                    ob = o >= tau
                    for k, t in enumerate((tau, thr_cal[j, pi, ti])):
                        fb = f >= t
                        counts[n, j, pi, ti, k] = ((w * (fb & ob)).sum(), (w * (fb & ~ob)).sum(),
                                                   (w * (~fb & ob)).sum())
        if n % 25 == 0:
            print(f"{arm} eval {n}/{len(ev_idx)} {time.time() - t0:.0f}s", flush=True)

    tot = counts.sum(0)
    h, fa, mi = tot[..., 0], tot[..., 1], tot[..., 2]
    csi = h / np.maximum(h + fa + mi, 1e-12)
    bias = (h + fa) / np.maximum(h + mi, 1e-12)
    N = wtot[None, :, None, None] * len(ev_idx)
    pss = h / np.maximum(h + mi, 1e-12) - fa / np.maximum(N - h - mi, 1e-12)
    table = []
    for j, lh in enumerate(leads):
        for pi, p in enumerate(POOLINGS):
            for ti, tau in enumerate(thr):
                table.append({"lead_h": lh, "pooling": p, "threshold_mm": float(tau),
                              "csi_raw": float(csi[j, pi, ti, 0]), "bias_raw": float(bias[j, pi, ti, 0]),
                              "csi_cal": float(csi[j, pi, ti, 1]), "bias_cal": float(bias[j, pi, ti, 1]),
                              "pss_raw": float(pss[j, pi, ti, 0]), "pss_cal": float(pss[j, pi, ti, 1]),
                              "fc_threshold_cal_mm": float(thr_cal[j, pi, ti]),
                              "calib_bias_raw": float(fc_w_raw[j, pi, ti] / max(obs_w[j, pi, ti], 1e-12)),
                              "obs_rate_eval": float((h + mi)[j, pi, ti, 0] / (wtot[pi] * len(ev_idx))),
                              "floor_flag": bool(flag[j, pi, ti])})
    dates = meta["dates"]
    info = {"model": arm, "source": meta["arms"][arm], "truth": meta["era5"], "leads_h": leads,
            "poolings": POOLINGS, "thresholds_mm": list(thresholds),
            "dates": [dates[i] for i in sorted(cal_idx + ev_idx)],
            "n_calib": len(cal_idx), "n_eval": len(ev_idx)}
    (out / f"aifs_{arm}.json").write_text(json.dumps({**info, "table": table}, indent=1))
    np.savez_compressed(out / f"aifs_{arm}.npz", counts=counts, wtot=wtot,
                        eval_dates=np.array([dates[i] for i in ev_idx]))
    print(f"{arm}: done", flush=True)
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="aifs_tp")
    ap.add_argument("--out", default="results/revision/aifs")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[1, 5, 10, 20, 50])
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    dates, leads = meta["dates"], meta["leads_h"]
    bad = {d for arm in args.arms for d, _ in meta["missing"].get(arm, [])}
    E = np.load(src / "era5.npy", mmap_mode="r")
    for i, d in enumerate(dates):
        if not np.isfinite(E[i, :, ::40, ::40]).all():
            bad.add(d)
    keep = [i for i, d in enumerate(dates) if d not in bad]
    cal_idx = [i for i in keep if i % 2 == 0]
    ev_idx = [i for i in keep if i % 2 == 1]
    print(f"{len(dates)} dates, {len(bad)} dropped, {len(cal_idx)} calibration, {len(ev_idx)} evaluation",
          flush=True)
    with ProcessPoolExecutor(len(args.arms)) as ex:
        for a in ex.map(run_arm, args.arms, [src] * len(args.arms), [out] * len(args.arms),
                        [args.thresholds] * len(args.arms), [cal_idx] * len(args.arms),
                        [ev_idx] * len(args.arms)):
            pass
    (out / "aifs_dates.json").write_text(json.dumps({"dropped": sorted(bad), "calibration": [dates[i] for i in cal_idx],
                                                     "evaluation": [dates[i] for i in ev_idx]}, indent=1))


if __name__ == "__main__":
    main()
